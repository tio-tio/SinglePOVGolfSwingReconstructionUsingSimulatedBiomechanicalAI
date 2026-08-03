"""Offline tests for session_meta.py — the chat-v2 session model. Run:
    .venv/Scripts/python.exe Scripts/test_session_meta.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from session_meta import (SESSION_GAP_S, clean_meta, cluster_sessions,
                          meta_from_s3_metadata, parse_date)

_PASS = _FAIL = 0


def check(name, cond, extra=""):
    global _PASS, _FAIL
    ok = bool(cond)
    _PASS += ok; _FAIL += (not ok)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + ("" if ok else f"  {extra}"))


print("\n[1] clean_meta: whitelist, caps, validation")
m = clean_meta({"club": "9 iron", "notes": "windy day at the range", "setting": "Range",
                "junk_key": "x", "session_label": "L" * 200})
check("whitelisted fields kept", m["club"] == "9 iron" and "windy" in m["notes"])
check("unknown keys dropped", "junk_key" not in m)
check("labels length-capped", len(m["session_label"]) == 60)
check("setting normalized to lowercase enum", m["setting"] == "range")
check("bad setting dropped", "setting" not in clean_meta({"setting": "moon"}))
check("bad session_id dropped", "session_id" not in clean_meta({"session_id": "NOT-HEX!"}))
check("good session_id kept", clean_meta({"session_id": "abc123def456"})["session_id"] == "abc123def456")
check("bad timestamp dropped", "uploaded_at" not in clean_meta({"uploaded_at": "yesterday"}))
check("ISO timestamp kept", "uploaded_at" in clean_meta({"uploaded_at": "2026-08-01T10:30:00+00:00"}))
check("non-ascii stripped", clean_meta({"notes": "wind→left café"})["notes"] == "windleft caf")
check("lat/lon validated", clean_meta({"lat": "43.65", "lon": "-79.38"}) ==
      {"lat": "43.65", "lon": "-79.38"} and "lat" not in clean_meta({"lat": "north"}))
check("empty input ok", clean_meta(None) == {})

print("\n[2] parse_date")
check("iso with Z", parse_date("2026-08-01T10:30:00Z") is not None)
check("naive becomes aware", parse_date("2026-08-01T10:30:00").tzinfo is not None)
check("garbage -> None", parse_date("not a date") is None and parse_date(None) is None)

print("\n[3] cluster_sessions: 2h gap rule")
J = lambda jid, iso, **kw: {"job_id": jid, "date": iso, "name": jid[:4], **kw}
jobs = [
    J("a" * 32, "2026-07-30T09:00:00+00:00"),
    J("b" * 32, "2026-07-30T09:45:00+00:00"),   # +45 min — same session
    J("c" * 32, "2026-07-30T13:00:00+00:00"),   # +3h15 — new session
    J("d" * 32, "2026-07-28T17:00:00+00:00"),   # different day
]
S = cluster_sessions(jobs)
check("three sessions", len(S) == 3, str([s["session_id"] for s in S]))
check("newest first", S[0]["date"] == "2026-07-30" and S[0]["swings"][0]["job_id"] == "c" * 32)
sess_ab = next(s for s in S if s["n_swings"] == 2)
check("morning pair clustered", {sw["job_id"] for sw in sess_ab["swings"]} == {"a" * 32, "b" * 32})
check("derived id is stable (oldest job prefix)", sess_ab["session_id"] == "s-" + "a" * 8)
check("swings newest-first inside a session", sess_ab["swings"][0]["job_id"] == "b" * 32)
check("boundary: exactly 2h gap still clusters",
      len(cluster_sessions([J("a" * 32, "2026-07-30T09:00:00+00:00"),
                            J("b" * 32, f"2026-07-30T{9 + SESSION_GAP_S // 3600}:00:00+00:00")])) == 1)

print("\n[4] cluster_sessions: declared ids beat the clock")
jobs = [
    J("a" * 32, "2026-07-30T09:00:00+00:00", session_id="deadbeef1234", session_label="driver work"),
    J("b" * 32, "2026-07-30T16:00:00+00:00", session_id="deadbeef1234"),  # 7h later, same declared id
    J("c" * 32, "2026-07-30T09:30:00+00:00"),                             # derived, alone
]
S = cluster_sessions(jobs)
declared = next(s for s in S if s["session_id"] == "deadbeef1234")
check("declared id groups across the gap", declared["n_swings"] == 2)
check("label surfaces", declared["label"] == "driver work")
check("derived job not swallowed", any(s["n_swings"] == 1 for s in S))

print("\n[5] cluster_sessions: edge cases")
S = cluster_sessions([])
check("empty listing -> no sessions", S == [])
S = cluster_sessions([J("a" * 32, None), J("b" * 32, "2026-07-30T09:00:00+00:00")])
check("undated jobs quarantined, dated job still clustered",
      any(s["session_id"] == "s-undated" for s in S) and any(s["date"] == "2026-07-30" for s in S))

print("\n[6] meta_from_s3_metadata (boto3 strips x-amz-meta-)")
m = meta_from_s3_metadata({"mc-session-id": "abc123def456", "mc-club": "driver",
                           "mc-uploaded-at": "2026-08-01T10:00:00+00:00",
                           "unrelated": "x"})
check("mc- keys mapped + validated", m == {"session_id": "abc123def456", "club": "driver",
                                           "uploaded_at": "2026-08-01T10:00:00+00:00"}, str(m))
check("empty metadata ok", meta_from_s3_metadata(None) == {})

print(f"\n{'=' * 46}\n  {_PASS} passed, {_FAIL} failed\n{'=' * 46}")
sys.exit(1 if _FAIL else 0)
