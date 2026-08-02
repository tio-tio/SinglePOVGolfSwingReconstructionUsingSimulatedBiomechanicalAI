"""AWS Lambda `GET /jobs` + `DELETE /jobs?id=<job>` — the dev account's
shared swing library.

The portal's per-browser localStorage library can't show a teammate's uploads,
so this endpoint lists EVERY processed upload straight from S3: job prefixes
under 03_outputs/ in the artifacts bucket (ready = all four required artifacts
present), joined with the original filenames still sitting in the uploads
bucket (01_inputs/uploads/<job>/<name> — 30-day TTL, so older jobs fall back
to a short-id name).

DELETE is a team-wide HARD delete: it removes every S3 object the job owns —
03_outputs/<job>/ and the TTS narration cache audio/<job>/ in the artifacts
bucket, plus the original upload 01_inputs/uploads/<job>/. The swings are of
real people, so "delete" must actually destroy the footage, not hide it.

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
                        "access-control-allow-methods": "GET,DELETE,OPTIONS",
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


def _delete_prefix(bucket: str, prefix: str) -> int:
    """Delete every object under bucket/prefix; returns how many went."""
    deleted = 0
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        keys = [{"Key": o["Key"]} for o in page.get("Contents", [])]
        if not keys:
            continue
        s3.delete_objects(Bucket=bucket, Delete={"Objects": keys, "Quiet": True})
        deleted += len(keys)
    return deleted


def _delete_job(event: dict):
    if not MC_RESULTS_TOKEN:
        # listing without a configured gate is a dev convenience; DESTROYING
        # footage of real people without one is a misconfiguration
        return _resp(403, {"error": "delete disabled: MC_RESULTS_TOKEN not configured"})
    qs = event.get("queryStringParameters") or {}
    job_id = (qs.get("id") or "").strip().lower()
    if not _JOB_ID.match(job_id):
        return _resp(400, {"error": "id must be a 32-hex job id"})
    # the trailing slash is load-bearing: without it job id "ab…" could sweep
    # a sibling prefix that merely starts with the same characters
    deleted = _delete_prefix(ARTIFACTS_BUCKET, f"03_outputs/{job_id}/")
    deleted += _delete_prefix(ARTIFACTS_BUCKET, f"audio/{job_id}/")
    if UPLOADS_BUCKET:
        deleted += _delete_prefix(UPLOADS_BUCKET, f"01_inputs/uploads/{job_id}/")
    if not deleted:
        return _resp(404, {"error": "no such job", "job_id": job_id})
    print(f"[jobs] hard-deleted job {job_id}: {deleted} objects")
    return _resp(200, {"job_id": job_id, "deleted": deleted})


def handler(event, _ctx=None):
    method = ((event.get("requestContext") or {}).get("http") or {}).get("method", "GET")
    if method == "OPTIONS":               # CORS preflight
        return _resp(204, {})
    if not _access_ok(event):
        return _resp(403, {"error": "uploaded swings are private — sign in on the portal"})
    if method == "DELETE":
        return _delete_job(event)
    names = _upload_names()
    jobs = [{**j, "name": names.get(j["job_id"], f"swing {j['job_id'][:8]}")}
            for j in _list_jobs()]
    return _resp(200, {"jobs": jobs, "count": len(jobs)})
