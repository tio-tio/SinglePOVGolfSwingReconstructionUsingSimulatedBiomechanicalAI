"""Session model + per-job context metadata (chat v2, deploy/CHAT_V2_PLAN.md §1-2).

A SESSION is a practice visit: a run of uploads close together in time. Two
layers, declared beating derived:

  - DERIVED: swings whose dates sit within SESSION_GAP_S (2 h) of the previous
    swing cluster into one session. Deterministic, computed from the job listing
    (S3 LastModified for pre-meta jobs), works retroactively.
  - DECLARED: a `session_id` stamped in the job's metadata at upload time. All
    swings sharing a declared id form one session regardless of the clock, and a
    user `session_label` names it.

Context metadata travels as S3 object metadata on the uploaded video itself
(`x-amz-meta-mc-*` presigned-POST fields) — NOT a sidecar object, because every
Object Created in the uploads bucket triggers the processing pipeline. The
processing Lambda copies it into 03_outputs/<job>/job_meta.json where the chat
Lambda (and later save_context updates) can read it through CloudFront.

Pure stdlib — imported by the chat Lambda, the upload/processing/jobs handlers,
and offline tests.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone

SESSION_GAP_S = 2 * 3600          # uploads within 2 h of the last one = same visit

# whitelist for user-supplied context fields (upload form + save_context tool).
# name -> (max_len, validator). Values are stored as strings in S3 metadata and
# typed on read. Anything not listed here is dropped, never stored.
_SETTINGS = ("range", "course", "sim")
_BALLS = ("range", "premium")
META_FIELDS: dict[str, int] = {
    "session_id": 32,
    "session_label": 60,
    "club": 24,
    "setting": 8,
    "ball": 8,
    "notes": 200,
    "uploaded_at": 32,
    "lat": 12,
    "lon": 12,
}

_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}")
_SESSION_ID = re.compile(r"^[0-9a-f]{6,32}$")
_NUM = re.compile(r"^-?\d{1,3}(?:\.\d{1,7})?$")


def clean_meta(raw: dict | None) -> dict[str, str]:
    """Validate + normalize user-supplied context fields. Returns only
    whitelisted keys with length-capped, printable-ASCII string values (S3
    object metadata must be ASCII, and these strings later enter LLM prompts
    via tool results — the existing trust boundary, but keep them tame)."""
    out: dict[str, str] = {}
    for key, cap in META_FIELDS.items():
        v = (raw or {}).get(key)
        if v is None:
            continue
        s = str(v).strip()
        s = "".join(ch for ch in s if 32 <= ord(ch) < 127)  # printable ASCII only
        s = re.sub(r"\s+", " ", s)[:cap]
        if not s:
            continue
        if key == "session_id" and not _SESSION_ID.match(s):
            continue
        if key == "uploaded_at" and not _ISO.match(s):
            continue
        if key == "setting" and s.lower() not in _SETTINGS:
            continue
        if key == "ball" and s.lower() not in _BALLS:
            continue
        if key in ("lat", "lon") and not _NUM.match(s):
            continue
        out[key] = s.lower() if key in ("setting", "ball") else s
    return out


def parse_date(s: str | None) -> datetime | None:
    """ISO-ish timestamp -> aware datetime (UTC assumed when naive)."""
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def cluster_sessions(jobs: list[dict]) -> list[dict]:
    """Group a job listing into sessions.

    `jobs`: [{job_id, date (ISO), name?, session_id?, session_label?, ...}]
    (the /jobs listing shape, optionally enriched from job_meta). Jobs with an
    unparseable date go last in a session of their own.

    Returns sessions NEWEST FIRST:
      [{session_id, label, date (YYYY-MM-DD), start, end, n_swings,
        swings: [job dicts, newest first]}]

    Declared session_ids group regardless of time; undeclared jobs cluster by
    the 2 h gap rule. A derived session's id is "s-" + its oldest job's first 8
    hex (stable across re-listings as long as that job exists).
    """
    declared: dict[str, list[dict]] = {}
    timed: list[tuple[datetime, dict]] = []
    undated: list[dict] = []
    for j in jobs:
        sid = j.get("session_id")
        if sid:
            declared.setdefault(sid, []).append(j)
            continue
        dt = parse_date(j.get("date"))
        if dt is None:
            undated.append(j)
        else:
            timed.append((dt, j))

    clusters: list[list[tuple[datetime, dict]]] = []
    for dt, j in sorted(timed, key=lambda t: t[0]):
        if clusters and (dt - clusters[-1][-1][0]).total_seconds() <= SESSION_GAP_S:
            clusters[-1].append((dt, j))
        else:
            clusters.append([(dt, j)])

    sessions: list[dict] = []
    for chunk in clusters:
        swings = [j for _, j in sorted(chunk, key=lambda t: t[0], reverse=True)]
        start, end = chunk[0][0], chunk[-1][0]
        sessions.append({
            "session_id": "s-" + str(swings[-1].get("job_id", ""))[:8],
            "label": None,
            "date": start.date().isoformat(),
            "start": start.isoformat(),
            "end": end.isoformat(),
            "n_swings": len(swings),
            "swings": swings,
        })
    for sid, group in declared.items():
        dts = [d for d in (parse_date(j.get("date")) for j in group) if d]
        start = min(dts) if dts else None
        swings = sorted(group, key=lambda j: j.get("date") or "", reverse=True)
        label = next((j.get("session_label") for j in swings if j.get("session_label")), None)
        sessions.append({
            "session_id": sid,
            "label": label,
            "date": start.date().isoformat() if start else None,
            "start": start.isoformat() if start else None,
            "end": max(dts).isoformat() if dts else None,
            "n_swings": len(swings),
            "swings": swings,
        })
    if undated:
        sessions.append({"session_id": "s-undated", "label": None, "date": None,
                         "start": None, "end": None, "n_swings": len(undated),
                         "swings": undated})
    sessions.sort(key=lambda s: s.get("start") or "", reverse=True)
    return sessions


def meta_from_s3_metadata(md: dict | None) -> dict[str, str]:
    """x-amz-meta-mc-* (boto3 strips the x-amz-meta- prefix) -> meta dict."""
    out = {}
    for k, v in (md or {}).items():
        if k.startswith("mc-"):
            out[k[3:].replace("-", "_")] = v
    return clean_meta(out)
