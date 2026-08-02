"""SQS-triggered processing worker — the container-Lambda backend for real uploads.

Wiring (full_app.yaml): S3 upload to 01_inputs/uploads/<job_id>/<file>  ->  EventBridge
rule  ->  SQS ingest queue  ->  this handler. One message = one uploaded swing.

Per message it:
  1. resolves the uploaded video (bucket/key from the S3 event) and its job_id,
  2. downloads it to /tmp,
  3. runs the SAME pipeline the local demo runs (2D -> 3D lift -> smoothing ->
     overlay + 3D replay; then event detection + coaching scorecard; then the
     grounded Claude eval; then web_artifacts.py for metrics/explanation JSON)
     as subprocesses — each step is torch-heavy and isolated,
  4. uploads the artifacts (overlay.mp4, replay_3d.json, metrics.json,
     explanation.json, pose_debug.mp4, pose_diag.json) to 03_outputs/<job_id>/,
  5. writes queryable rows (swings/analyses/indicators/swing_events) via the RDS
     Data API, mirroring deploy/db/load_data.py's backfill shape.

Runs as a CONTAINER image (needs torch/mediapipe/mixste + the model bundle) — this
is the one piece that can't ship as a zip. Gated in full_app.yaml behind
HasProcessingImage until the image is built (deploy/chat/buildspec.yml pattern +
the processing Dockerfile, still to be authored).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import boto3

ARTIFACTS_BUCKET = os.environ.get("ARTIFACTS_BUCKET", "")
DB_CLUSTER_ARN = os.environ.get("DB_CLUSTER_ARN", "")
DB_SECRET_ARN = os.environ.get("DB_SECRET_ARN", "")
DB_NAME = os.environ.get("DB_NAME", "motioncaddie")
APP_ROOT = Path(os.environ.get("APP_ROOT", "/var/task"))
SCRIPTS = APP_ROOT / "Scripts"
PY = sys.executable
BACKBONE = os.environ.get("POSE_BACKBONE", "mediapipe_lite")
LIFTER = os.environ.get("LIFTER", "golfpose3d")

_s3 = boto3.client("s3")
_rd = boto3.client("rds-data") if DB_CLUSTER_ARN else None

# The pipeline steps run as subprocesses, but the ingest bake happens inline.
# Guarded: this handler must never fail to LOAD over the orientation fix — the
# Scripts-side open_capture() still corrects orientation if this is missing.
sys.path.insert(0, str(SCRIPTS))
try:
    from video_orientation import normalize_video
except Exception as _e:  # pragma: no cover — image build regression only
    print(f"[proc] video_orientation unavailable ({_e}); relying on ORIENTATION_AUTO", flush=True)
    normalize_video = None

# artifacts the front end consumes, keyed by the suffix each pipeline step emits.
# pose_debug.mp4 / pose_diag.json are diagnostics (L/R-colored MediaPipe skeleton
# + swap/jitter metrics) — not consumed by the app UI but kept with every job so
# limb-crossing and jitter reports can be triaged from S3 alone.
ARTIFACTS = ["overlay.mp4", "replay_3d.json", "metrics.json", "explanation.json",
             "pose_debug.mp4", "pose_diag.json",
             # full scorecard: the chat Lambda reads it (via CloudFront) so the
             # coach can answer questions about uploaded swings too
             "scorecard.json",
             # measured ball track + physics-fit flight (chat tool + trajectory UI);
             # written with quality:"simulated" when no confident track exists
             "ball_3d.json",
             # session/context metadata (uploaded_at, session_id, club, notes …) —
             # copied from the video's x-amz-meta-mc-* fields; the chat coach and
             # the portal's session grouping read it. jobs_handler EXCLUDES this
             # file from its date aggregate so later save_context edits can't
             # shift a job's session.
             "job_meta.json"]
# the web app's pollJob() marks a job ready only once ALL of these exist — fail
# the message loudly (SQS retry) rather than leave a job that never completes
REQUIRED_ARTIFACTS = {"overlay.mp4", "replay_3d.json", "metrics.json", "explanation.json"}


# --------------------------------------------------------------------------
def _job_from_key(key: str) -> str:
    """01_inputs/uploads/<job_id>/<file>  ->  <job_id>."""
    parts = key.split("/")
    if len(parts) >= 3 and parts[0] == "01_inputs" and parts[1] == "uploads":
        return parts[2]
    return Path(key).stem  # fallback: filename stem


def _iter_s3_records(event: dict):
    """Yield (bucket, key) from an SQS batch whose bodies are S3/EventBridge events."""
    for rec in event.get("Records", []):
        try:
            body = json.loads(rec["body"])
        except (KeyError, json.JSONDecodeError):
            continue
        # EventBridge S3 "Object Created" shape
        if body.get("detail", {}).get("bucket"):
            yield body["detail"]["bucket"]["name"], body["detail"]["object"]["key"]
        # raw S3 notification shape (fallback)
        for r in body.get("Records", []):
            if "s3" in r:
                yield r["s3"]["bucket"]["name"], r["s3"]["object"]["key"]


def _run(cmd: list[str], desc: str, extra_env: dict | None = None) -> None:
    print(f"[proc] {desc}: {' '.join(cmd)}", flush=True)
    # Lambda: only /tmp is writable — torch/mediapipe/matplotlib all try to write caches
    env = {**os.environ, "HOME": "/tmp", "MPLCONFIGDIR": "/tmp/mpl", "XDG_CACHE_HOME": "/tmp/xdg",
           **(extra_env or {})}
    res = subprocess.run(cmd, cwd=str(SCRIPTS), capture_output=True, text=True, env=env)
    if res.returncode != 0:
        raise RuntimeError(f"{desc} failed (rc={res.returncode}): "
                           f"stderr={res.stderr[-1200:]} stdout={res.stdout[-400:]}")


def _probe_fps(video: Path) -> float | None:
    """Container frame rate of the upload (phone videos play in real time, so
    this is the motion rate too). None on failure -> hand-speed indicator is
    simply omitted, never wrong."""
    try:
        import cv2
        cap = cv2.VideoCapture(str(video))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        cap.release()
        return fps if 10 <= fps <= 120 else None
    except Exception as e:
        print(f"[proc] fps probe failed ({e}) — skipping hand-speed indicator", flush=True)
        return None


def _normalize_orientation(video: Path, job_id: str) -> None:
    """Bake any camera rotation into the pixels, in place, before step one.

    Phones store portrait clips as landscape frames plus a display-matrix
    rotation. Doing this once here is what makes the rest of the pipeline
    orientation-free: pose, overlay, diagnostics and ball tracking all just
    read an upright file, and so does anything added later.

    Rotation-free uploads cost a metadata probe and no transcode. Never fatal
    — Scripts' open_capture() is the second line of defence, so a failure here
    degrades to the old behaviour instead of losing the job.
    """
    if normalize_video is None:
        return
    try:
        upright, rotation = normalize_video(video, video.with_name(f"{video.stem}_upright.mp4"))
        if upright != video:
            # move it back over the original name so the pipeline's stem — and
            # therefore every artifact filename — is unchanged by the fix
            os.replace(str(upright), str(video))
            print(f"[proc] job {job_id}: baked {rotation}° camera rotation into pixels", flush=True)
    except Exception as e:
        print(f"[proc] job {job_id}: orientation normalise skipped "
              f"({type(e).__name__}: {e}) — falling back to ORIENTATION_AUTO", flush=True)


def _pipeline(video: Path, work: Path) -> dict:
    """Run the 3-step pipeline; return the paths of the produced artifacts."""
    stem = video.stem
    cache = work / "cache"  # pipeline defaults to <repo>/Data/eval_runs — read-only in Lambda
    _run([PY, str(SCRIPTS / "pipeline.py"), str(video),
          "--backbone", BACKBONE, "--lifter", LIFTER,
          "--out-dir", str(work), "--cache-dir", str(cache)],
         "2D->3D pipeline + overlay + replay",
         # adapters read the upstream-2D cache through this env override
         extra_env={"PIPELINE_CACHE_DIR": str(cache)})
    # 3D parquet lands under the lifter's cache subdir (possibly the "_lrfix"
    # variant when the L/R swap repair rewrote the 2D input) — prefer the
    # repaired lift, then any lifter output, then a bare rglob as last resort.
    cands = (sorted(cache.glob(f"{LIFTER}_from_*/{stem}.parquet"),
                    key=lambda p: "_lrfix" not in p.parent.name)
             or list(work.glob(f"*{stem}*3d*.parquet"))
             or list(cache.rglob(f"{stem}.parquet")))
    if not cands:
        raise RuntimeError(f"pipeline produced no 3D parquet for {stem} in {work} or {cache}")
    parquet_3d = cands[0]
    scorecard = work / f"{stem}_scorecard.json"
    sc_cmd = [PY, str(SCRIPTS / "scorecard_step.py"), "--parquet", str(parquet_3d),
              "--out-dir", str(work), "--stem", stem]
    fps = _probe_fps(video)
    if fps:
        sc_cmd += ["--fps", str(fps)]  # enables the time-based hand-speed indicator
    _run(sc_cmd, "event detection + scorecard")
    # ball tracking + flight fit: emits ball_3d.json and redraws the ball onto
    # overlay.mp4 — non-fatal (artifact carries quality:"simulated" on no-track,
    # and jobs must not fail over a missing ball)
    try:
        _run([PY, str(SCRIPTS / "ball_step.py"), str(video),
              "--landmarks", str(work / f"{stem}_landmarks_2d.csv"),
              "--scorecard", str(scorecard),
              "--replay", str(work / f"{stem}_replay_3d.json"),
              "--overlay", str(work / f"{stem}_overlay.mp4"),
              "--out", str(work / f"{stem}_ball_3d.json")],
             "ball tracking + flight fit")
    except RuntimeError as e:
        print(f"[proc] ball step skipped: {e}", flush=True)
    # grounded Claude eval (needs ANTHROPIC_API_KEY in env) — writes the LLM
    # explanation back INTO the scorecard JSON; non-fatal if it fails
    try:
        _run([PY, str(SCRIPTS / "coaching_explain.py"), "--scorecard", str(scorecard)],
             "coaching eval")
    except RuntimeError as e:
        print(f"[proc] coaching eval skipped: {e}", flush=True)
    # metrics.json + explanation.json from the (possibly LLM-augmented) scorecard —
    # two of the three JSONs the web app's pollJob() requires. --parquet-3d is the
    # backstop for replay_3d.json in case pipeline.py's own emit failed (non-fatal
    # there); skipped when the pipeline already wrote it.
    _run([PY, str(SCRIPTS / "web_artifacts.py"), "--scorecard", str(scorecard),
          "--out-dir", str(work), "--stem", stem, "--parquet-3d", str(parquet_3d)],
         "web artifacts (metrics + explanation)")
    return {"scorecard": scorecard, "work": work, "stem": stem}


def _upload_artifacts(work: Path, stem: str, job_id: str) -> list[str]:
    uploaded = []
    for name in ARTIFACTS:
        # pipeline writes some as <stem>_<name>; accept either exact or stem-prefixed
        for cand in (work / name, work / f"{stem}_{name}"):
            if cand.exists():
                dest = f"03_outputs/{job_id}/{name}"
                # explicit ContentType: upload_file defaults to octet-stream,
                # and the browser serves these straight from CloudFront
                ctype = ("video/mp4" if name.endswith(".mp4") else "application/json")
                _s3.upload_file(str(cand), ARTIFACTS_BUCKET, dest,
                                ExtraArgs={"ContentType": ctype})
                uploaded.append(dest)
                break
    return uploaded


def _sql(sql: str, params: list) -> None:
    # Aurora Serverless scales to zero (DbMinCapacity=0); the first Data API
    # call while it wakes throws DatabaseResumingException — wait it out
    # (resume is typically ~15-30s) instead of failing the whole job.
    import time
    from botocore.exceptions import ClientError
    for attempt in range(8):
        try:
            _rd.execute_statement(resourceArn=DB_CLUSTER_ARN, secretArn=DB_SECRET_ARN,
                                  database=DB_NAME, sql=sql, parameters=params)
            return
        except ClientError as e:
            if (e.response.get("Error", {}).get("Code") != "DatabaseResumingException"
                    or attempt == 7):
                raise
            print(f"[proc] DB resuming; retry {attempt + 1}/7 in 5s", flush=True)
            time.sleep(5)


def _p(name: str, **kv):
    (k, v), = kv.items()
    return {"name": name, "value": {k: v}}


def _write_db(job_id: str, scorecard: Path, raw_key: str) -> None:
    """Insert swings + analyses + indicators rows for the uploaded swing.

    Never raises: by this point the artifacts are uploaded and the job IS done
    for the player — a DB hiccup (schema not migrated yet, wake-up timeout)
    must not fail the SQS message and burn a retry on a finished job.
    """
    try:
        _write_db_rows(job_id, scorecard, raw_key)
    except Exception as e:
        print(f"[proc] DB row write skipped ({type(e).__name__}): {e}", flush=True)


def _write_db_rows(job_id: str, scorecard: Path, raw_key: str) -> None:
    if _rd is None:
        print("[proc] no DB configured; skipping row write", flush=True)
        return
    sc = json.loads(scorecard.read_text(encoding="utf-8"))
    _sql("""INSERT INTO swings (swing_id, source, raw_video_s3_key, status)
            VALUES (:sid::uuid, 'upload', :key, 'done')
            ON CONFLICT (swing_id) DO NOTHING""",
         [_p("sid", stringValue=_as_uuid(job_id)), _p("key", stringValue=raw_key)])
    _sql("""INSERT INTO analyses (analysis_id, swing_id, pipeline_version, lifter, scorecard_json)
            VALUES (:aid::uuid, :sid::uuid, :ver, :lifter, :blob::jsonb)
            ON CONFLICT (analysis_id) DO NOTHING""",
         [_p("aid", stringValue=_as_uuid(job_id + "a")), _p("sid", stringValue=_as_uuid(job_id)),
          _p("ver", stringValue="mixste+oneeuro@2026-07"), _p("lifter", stringValue=LIFTER),
          _p("blob", stringValue=json.dumps(sc))])
    # scorecard indicators are a dict keyed by indicator name (coaching_scorecard)
    for key, ind in (sc.get("indicators") or {}).items():
        _sql("""INSERT INTO indicators (analysis_id, indicator_key, value, percentile, confidence_tier)
                VALUES (:aid::uuid, :key, :val, :pct, :tier)
                ON CONFLICT (analysis_id, indicator_key) DO NOTHING""",
             [_p("aid", stringValue=_as_uuid(job_id + "a")), _p("key", stringValue=key),
              _p("val", doubleValue=float(ind.get("value") or 0)),
              _p("pct", doubleValue=float(ind.get("percentile") or 0)),
              _p("tier", stringValue=str(ind.get("confidence_tier") or "unknown"))])


def _as_uuid(seed: str) -> str:
    """Deterministic UUID from the job id so retries are idempotent (uuid5)."""
    import uuid
    return str(uuid.uuid5(uuid.NAMESPACE_URL, seed))


def _write_job_meta(bucket: str, key: str, work: Path, job_id: str) -> None:
    """Session/context metadata: the video's x-amz-meta-mc-* fields (stamped by
    the upload handler's presigned POST) -> work/job_meta.json, published with
    the other artifacts. Never fatal — a job without meta clusters by its S3
    date, exactly like every pre-chat-v2 job."""
    try:
        from session_meta import meta_from_s3_metadata
        head = _s3.head_object(Bucket=bucket, Key=key)
        meta = meta_from_s3_metadata(head.get("Metadata"))
        meta.setdefault("uploaded_at",
                        head["LastModified"].isoformat(timespec="seconds"))
        (work / "job_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    except Exception as e:
        print(f"[proc] job {job_id}: job_meta skipped ({type(e).__name__}: {e})", flush=True)


# --------------------------------------------------------------------------
def handler(event, _ctx=None):
    processed = []
    for bucket, key in _iter_s3_records(event):
        job_id = _job_from_key(key)
        print(f"[proc] job {job_id}: s3://{bucket}/{key}", flush=True)
        work = Path("/tmp") / job_id
        work.mkdir(parents=True, exist_ok=True)
        video = work / Path(key).name
        try:
            _s3.download_file(bucket, key, str(video))
        except Exception as e:  # deleted/expired upload (30d TTL) — drop, don't poison retries
            print(f"[proc] job {job_id}: source object gone ({type(e).__name__}) — skipping", flush=True)
            continue
        _write_job_meta(bucket, key, work, job_id)
        _normalize_orientation(video, job_id)
        out = _pipeline(video, work)
        uploaded = _upload_artifacts(work, out["stem"], job_id)
        missing = REQUIRED_ARTIFACTS - {Path(k).name for k in uploaded}
        if missing:
            raise RuntimeError(f"job {job_id}: required artifacts never produced: "
                               f"{sorted(missing)} — front end would poll forever")
        _write_db(job_id, out["scorecard"], key)
        print(f"[proc] job {job_id} done; artifacts: {uploaded}", flush=True)
        processed.append(job_id)
    return {"processed": processed}
