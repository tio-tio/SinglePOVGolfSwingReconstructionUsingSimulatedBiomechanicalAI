"""Dataset expansion pipeline — turn YouTube searches into GolfDB-compatible clips.

8-stage pipeline:
  01_search_youtube    queries -> candidate URL list
  02_download_videos   URLs    -> raw mp4 + metadata
  03_detect_swings     mp4     -> swing intervals via MediaPipe
  04_extract_clips     intervals -> 160x160 mp4 + GolfDB-style row
  05_label_with_llm    clip    -> 8 event frames via Codex
  06_consensus_filter  labels  -> filter unrealistic predictions
  07_validate_and_merge        -> combined dataset.pkl

Each stage is idempotent: re-running picks up where it stopped.
"""
