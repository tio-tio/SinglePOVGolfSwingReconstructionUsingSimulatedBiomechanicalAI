"""AWS Lambda `/upload-url` issuer for MotionCaddie user video uploads.

Returns a short-lived **presigned S3 POST** so the browser uploads the video
straight to S3 (never through compute — big files must not route through Lambda).
Presigned POST (not PUT) so we can enforce content-type + a hard size cap as
S3 policy conditions the client cannot override.

Flow: browser POSTs {filename, content_type} -> this issues {url, fields, job_id,
object_key}; browser does a multipart POST straight to S3; the S3 upload triggers
the EventBridge->SQS->processing path (full_app.yaml), which writes results to
03_outputs/<job_id>/ that the front end then polls.

The exec role's own s3:PutObject perms back the presign, so no user creds are
exposed. ALLOWED_ORIGINS / MAX_UPLOAD_MB / URL_TTL_SECONDS are env-tunable.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import re
import sys
import uuid
from pathlib import Path

import boto3
from botocore.config import Config

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "Scripts"))
try:
    from session_meta import clean_meta  # context whitelist (chat v2 sessions)
except ImportError:                       # zip built without Scripts/ — degrade
    clean_meta = lambda raw: {}

UPLOADS_BUCKET = os.environ["UPLOADS_BUCKET"]
UPLOAD_PREFIX = os.environ.get("UPLOAD_PREFIX", "01_inputs/uploads")
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "200"))
URL_TTL_SECONDS = int(os.environ.get("URL_TTL_SECONDS", "300"))
ALLOWED_CONTENT = set(json.loads(os.environ.get("ALLOWED_CONTENT", '["video/mp4","video/quicktime"]')))
CORS_ORIGIN = os.environ.get("CORS_ORIGIN", "*")

_s3 = boto3.client("s3", config=Config(signature_version="s3v4"))
_SAFE = re.compile(r"[^A-Za-z0-9._-]")


def _resp(code: int, body: dict) -> dict:
    return {"statusCode": code,
            "headers": {"content-type": "application/json",
                        "access-control-allow-origin": CORS_ORIGIN,
                        "access-control-allow-methods": "POST,OPTIONS",
                        "access-control-allow-headers": "content-type"},
            "body": json.dumps(body)}


def handler(event, _ctx=None):
    if (event.get("requestContext", {}).get("http", {}).get("method") == "OPTIONS"):
        return _resp(200, {})  # CORS preflight

    try:
        body = json.loads(event.get("body") or "{}")
    except json.JSONDecodeError:
        return _resp(400, {"error": "invalid JSON"})

    content_type = (body.get("content_type") or "").strip()
    if content_type not in ALLOWED_CONTENT:
        return _resp(400, {"error": f"content_type must be one of {sorted(ALLOWED_CONTENT)}"})

    raw_name = (body.get("filename") or "swing.mp4").strip()
    safe_name = _SAFE.sub("_", raw_name)[-80:] or "swing.mp4"
    job_id = uuid.uuid4().hex
    key = f"{UPLOAD_PREFIX}/{job_id}/{safe_name}"

    # session/context fields ride as object metadata on the video itself — a
    # sidecar object under 01_inputs/uploads/ would re-trigger the processing
    # pipeline (EventBridge fires on every Object Created). The processing
    # Lambda reads these via head_object and publishes 03_outputs/<job>/
    # job_meta.json. Whitelisted + length-capped in clean_meta.
    meta = clean_meta(body.get("session") or {})
    meta.setdefault("session_id", uuid.uuid4().hex[:12])
    meta.setdefault("uploaded_at",
                    _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"))
    meta_fields = {f"x-amz-meta-mc-{k.replace('_', '-')}": v for k, v in meta.items()}

    try:
        presigned = _s3.generate_presigned_post(
            Bucket=UPLOADS_BUCKET, Key=key,
            Fields={"Content-Type": content_type, **meta_fields},
            Conditions=[
                {"Content-Type": content_type},
                ["content-length-range", 1, MAX_UPLOAD_MB * 1024 * 1024],
                # every extra field must be covered by an exact-match condition
                # or S3 rejects the browser's POST
                *[{k: v} for k, v in meta_fields.items()],
            ],
            ExpiresIn=URL_TTL_SECONDS,
        )
    except Exception as e:  # never leak internals
        print(f"[upload] presign error: {type(e).__name__}: {e}")
        return _resp(502, {"error": "could not issue upload url"})

    return _resp(200, {
        "job_id": job_id,
        "session_id": meta["session_id"],
        "object_key": key,
        "url": presigned["url"],
        "fields": presigned["fields"],
        "max_mb": MAX_UPLOAD_MB,
        "expires_in": URL_TTL_SECONDS,
        # where results will appear once processing runs
        "result_prefix": f"03_outputs/{job_id}/",
    })
