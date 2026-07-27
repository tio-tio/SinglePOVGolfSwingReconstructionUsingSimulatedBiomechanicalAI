"""AWS Lambda `GET /jobs` — the dev account's shared swing library.

The portal's per-browser localStorage library can't show a teammate's uploads,
so this endpoint lists EVERY processed upload straight from S3: job prefixes
under 03_outputs/ in the artifacts bucket (ready = all four required artifacts
present), joined with the original filenames still sitting in the uploads
bucket (01_inputs/uploads/<job>/<name> — 30-day TTL, so older jobs fall back
to a short-id name).

PRIVATE: uploaded swings are of real people. Callers must present the dev
access code (?t= query or x-mc-access header) matching MC_RESULTS_TOKEN —
the same code the CloudFront mc-results-gate function enforces for the
artifacts themselves.
"""
from __future__ import annotations

import json
import os
import re

import boto3

ARTIFACTS_BUCKET = os.environ["ARTIFACTS_BUCKET"]
UPLOADS_BUCKET = os.environ.get("UPLOADS_BUCKET", "")
MC_RESULTS_TOKEN = os.environ.get("MC_RESULTS_TOKEN", "")
MAX_JOBS = int(os.environ.get("MAX_JOBS", "200"))

REQUIRED = {"metrics.json", "explanation.json", "replay_3d.json", "overlay.mp4"}
_JOB_ID = re.compile(r"^[0-9a-f]{32}$")

s3 = boto3.client("s3")


def _resp(code: int, body) -> dict:
    return {"statusCode": code,
            "headers": {"content-type": "application/json",
                        "cache-control": "no-store",
                        "access-control-allow-origin": "*",
                        "access-control-allow-headers": "content-type,x-mc-access"},
            "body": json.dumps(body)}


def _access_ok(event: dict) -> bool:
    if not MC_RESULTS_TOKEN:
        return True                       # gate not configured (dev/local)
    qs = event.get("queryStringParameters") or {}
    headers = event.get("headers") or {}
    supplied = qs.get("t") or headers.get("x-mc-access") or ""
    return supplied == MC_RESULTS_TOKEN


def _list_jobs() -> list[dict]:
    """One paginated sweep of 03_outputs/: aggregate per-job artifact presence
    and the newest LastModified as the job's date."""
    jobs: dict[str, dict] = {}
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=ARTIFACTS_BUCKET, Prefix="03_outputs/"):
        for obj in page.get("Contents", []):
            parts = obj["Key"].split("/")
            if len(parts) != 3 or not _JOB_ID.match(parts[1]):
                continue
            j = jobs.setdefault(parts[1], {"files": set(), "date": obj["LastModified"]})
            j["files"].add(parts[2])
            if obj["LastModified"] > j["date"]:
                j["date"] = obj["LastModified"]
    out = [{"job_id": jid, "date": j["date"].isoformat(),
            "ready": REQUIRED <= j["files"]}
           for jid, j in jobs.items()]
    out.sort(key=lambda x: x["date"], reverse=True)
    return out[:MAX_JOBS]


def _upload_names() -> dict[str, str]:
    """job_id -> original filename, while the upload still exists (30-day TTL)."""
    names: dict[str, str] = {}
    if not UPLOADS_BUCKET:
        return names
    paginator = s3.get_paginator("list_objects_v2")
    try:
        for page in paginator.paginate(Bucket=UPLOADS_BUCKET, Prefix="01_inputs/uploads/"):
            for obj in page.get("Contents", []):
                parts = obj["Key"].split("/")
                if len(parts) == 4 and _JOB_ID.match(parts[2]) and parts[3]:
                    names[parts[2]] = parts[3]
    except Exception as e:                # names are a nicety, never fatal
        print(f"[jobs] uploads listing failed: {e}")
    return names


def handler(event, _ctx=None):
    method = ((event.get("requestContext") or {}).get("http") or {}).get("method", "GET")
    if method == "OPTIONS":               # CORS preflight
        return _resp(204, {})
    if not _access_ok(event):
        return _resp(403, {"error": "uploaded swings are private — sign in on the portal"})
    names = _upload_names()
    jobs = [{**j, "name": names.get(j["job_id"], f"swing {j['job_id'][:8]}")}
            for j in _list_jobs()]
    return _resp(200, {"jobs": jobs, "count": len(jobs)})
