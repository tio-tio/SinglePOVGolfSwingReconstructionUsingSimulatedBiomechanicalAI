"""LLM-as-event-detector pipeline.

This is structurally different from our pose adapters: instead of producing
17 landmarks per frame, the LLM directly predicts the 8 GolfDB swing event
frame indices from a few sampled video frames. The output IS the event
prediction — there's no separate heuristic detector layer.

The point: test whether a frontier multimodal model can match (or beat)
our specialized 23-pipeline benchmark on the metric the user-facing app
actually cares about (PCE@5 on the 8 swing events).

The adapter samples N frames evenly across the clip, arranges them in a
labeled grid, and asks the model to identify which frame each of the 8
events occurs at. The model receives the actual frame indices as labels
so it can return them directly.

Currently supports:
  - anthropic (Claude Sonnet/Opus with vision)
  - openai (GPT-4o/4-vision) — same interface, swap by SDK
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from eval_utils import (
    BaseAdapter,
    COCO17_IDX,
    COCO17_NAMES,
    InferenceResult,
    SWING_EVENTS,
    video_info,
)


_SYSTEM_PROMPT = """You are a golf swing analyst. You will be shown a grid of frames sampled evenly from a single golf swing video. Each frame is labeled with its frame number (e.g., "f=42").

Your task: identify the specific frame number where each of the 8 canonical swing events occurs. The 8 events, in order:

1. address - the golfer is set up over the ball, club resting behind it, body still
2. toe_up - the club shaft has rotated to horizontal during the backswing (shaft pointing back, toe of clubhead pointing up)
3. mid_backswing - halfway up; the lead arm is roughly parallel to the ground
4. top - the highest point of the backswing; wrists at maximum height, body fully wound
5. mid_downswing - halfway down; the lead arm is parallel to the ground again, descending
6. impact - the clubhead strikes the ball
7. mid_follow_through - halfway through the follow-through; mirrors mid_backswing
8. finish - the end of the swing; weight on lead foot, club over the shoulder

For each event, return the frame number of the closest labeled frame in the grid. The events MUST appear in monotonic order: address < toe_up < mid_backswing < top < mid_downswing < impact < mid_follow_through < finish.

Reply with ONLY a JSON object in this exact shape, no prose:
{"address": <int>, "toe_up": <int>, "mid_backswing": <int>, "top": <int>, "mid_downswing": <int>, "impact": <int>, "mid_follow_through": <int>, "finish": <int>}
"""


class LLMEventAdapter(BaseAdapter):
    """Frontier multimodal model directly predicting swing events.

    `predict()` returns an InferenceResult whose `landmarks` DataFrame is
    augmented with the predicted event frames so compute_metrics can
    score PCE directly without going through the heuristic detector.
    """

    family = "llm"
    native_skeleton = "event-only"

    def __init__(self,
                 provider: str = "anthropic",
                 model: str = "claude-sonnet-4-5",
                 n_grid_frames: int = 24,
                 grid_cols: int = 6,
                 cell_size: int = 200,
                 max_retries: int = 3):
        """
        Args:
          provider: "anthropic" or "openai"
          model: API model name
          n_grid_frames: number of frames to sample from the clip (more = better
            temporal resolution, more cost)
          grid_cols: how many columns in the frame grid
          cell_size: pixel size per cell (cell_size x cell_size each frame)
        """
        self.provider = provider
        self.model = model
        self.n_grid_frames = n_grid_frames
        self.grid_cols = grid_cols
        self.cell_size = cell_size
        self.max_retries = max_retries
        self.name = f"llm_{provider}_{model.replace('-', '_').replace('.', '_')}"

        if provider == "anthropic":
            import anthropic
            self.client = anthropic.Anthropic()
        elif provider == "openai":
            import openai
            self.client = openai.OpenAI()
        else:
            raise ValueError(f"unknown provider {provider!r}")

    # ---------------------------------------------------------------------
    # Frame sampling + grid composition
    # ---------------------------------------------------------------------

    def _sample_frame_indices(self, n_total: int) -> list[int]:
        """Pick n_grid_frames evenly across [0, n_total). Always includes
        first and last frame."""
        n = self.n_grid_frames
        if n >= n_total:
            return list(range(n_total))
        return [int(round(i * (n_total - 1) / (n - 1))) for i in range(n)]

    def _build_grid_image(self, frames: list[np.ndarray], indices: list[int]) -> bytes:
        """Compose a labeled grid of frames as a PNG. Each cell is the
        frame downsized to cell_size, with the frame index drawn in the
        top-left corner."""
        cell = self.cell_size
        cols = self.grid_cols
        rows = (len(frames) + cols - 1) // cols
        canvas = np.full((rows * cell, cols * cell, 3), 30, dtype=np.uint8)

        for i, (frame, idx) in enumerate(zip(frames, indices)):
            r, c = divmod(i, cols)
            y0 = r * cell
            x0 = c * cell
            # frame is HxWx3 RGB; resize to cell x cell, convert to BGR for cv2
            resized = cv2.resize(frame, (cell, cell), interpolation=cv2.INTER_AREA)
            bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
            canvas[y0:y0 + cell, x0:x0 + cell] = bgr
            # Frame label in top-left, with white background for readability
            label = f"f={idx}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(canvas, (x0 + 2, y0 + 2), (x0 + 6 + tw, y0 + 8 + th),
                          (0, 0, 0), -1)
            cv2.putText(canvas, label, (x0 + 4, y0 + 6 + th),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        ok, buf = cv2.imencode(".png", canvas)
        if not ok:
            raise RuntimeError("png encode failed")
        return bytes(buf)

    # ---------------------------------------------------------------------
    # API call
    # ---------------------------------------------------------------------

    def _ask(self, png_bytes: bytes) -> Optional[dict]:
        """Send the grid image, parse JSON response. Returns None on failure."""
        b64 = base64.standard_b64encode(png_bytes).decode("ascii")

        for attempt in range(self.max_retries):
            try:
                if self.provider == "anthropic":
                    resp = self.client.messages.create(
                        model=self.model,
                        max_tokens=400,
                        system=_SYSTEM_PROMPT,
                        messages=[{
                            "role": "user",
                            "content": [
                                {"type": "image",
                                 "source": {"type": "base64",
                                            "media_type": "image/png",
                                            "data": b64}},
                                {"type": "text",
                                 "text": "Identify the 8 events from this grid. Return JSON only."},
                            ],
                        }],
                    )
                    text = resp.content[0].text
                elif self.provider == "openai":
                    resp = self.client.chat.completions.create(
                        model=self.model,
                        max_tokens=400,
                        messages=[
                            {"role": "system", "content": _SYSTEM_PROMPT},
                            {"role": "user", "content": [
                                {"type": "image_url",
                                 "image_url": {"url": f"data:image/png;base64,{b64}"}},
                                {"type": "text",
                                 "text": "Identify the 8 events from this grid. Return JSON only."},
                            ]},
                        ],
                    )
                    text = resp.choices[0].message.content
                # Pull the JSON object out
                m = re.search(r"\{[\s\S]*\}", text)
                if not m:
                    print(f"  [llm] no JSON found in response: {text[:200]}")
                    continue
                events = json.loads(m.group())
                # Validate
                missing = set(SWING_EVENTS) - set(events.keys())
                if missing:
                    print(f"  [llm] missing events: {missing}")
                    continue
                return {k: int(events[k]) for k in SWING_EVENTS}
            except Exception as e:
                print(f"  [llm] attempt {attempt+1}/{self.max_retries} failed: {repr(e)[:200]}")
                time.sleep(2 ** attempt)
        return None

    # ---------------------------------------------------------------------
    # Adapter contract
    # ---------------------------------------------------------------------

    def predict(self, video_path) -> InferenceResult:
        from eval_utils import iter_frames as _iter
        info = video_info(video_path)
        frames = [f for _, f in _iter(video_path, rgb=True)]
        n_frames = len(frames)
        indices = self._sample_frame_indices(n_frames)
        sampled = [frames[i] for i in indices]
        png = self._build_grid_image(sampled, indices)

        t0 = time.perf_counter()
        events = self._ask(png)
        t1 = time.perf_counter()

        # Emit a minimal "landmarks" DataFrame: one synthetic row per frame
        # carrying the predicted event index in the "event_pred" column.
        # The metrics layer reads this directly instead of running the
        # heuristic detector.
        rows = []
        for fi in range(n_frames):
            for ki in range(17):
                rows.append({
                    "frame": fi, "kp_idx": ki, "kp_name": COCO17_NAMES[ki],
                    "x": 0.0, "y": 0.0, "conf": 0.0,
                })
        df = pd.DataFrame(rows)

        # Attach predicted events as side-channel metadata via a special row
        # marker. To stay compatible with existing parquet schema we use a
        # negative frame index for the "event prediction" row.
        if events is not None:
            for ev_name, ev_frame in events.items():
                df.loc[len(df)] = {
                    "frame": -1, "kp_idx": -1, "kp_name": f"EVENT::{ev_name}",
                    "x": float(ev_frame), "y": 0.0, "conf": 1.0,
                }

        return InferenceResult(
            landmarks=df,
            seconds_per_frame=(t1 - t0) / max(1, n_frames),
            n_frames=n_frames,
            n_frames_detected=n_frames if events else 0,
        )
