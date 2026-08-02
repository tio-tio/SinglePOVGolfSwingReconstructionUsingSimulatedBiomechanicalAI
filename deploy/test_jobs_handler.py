"""Offline tests for the /jobs Lambda — no AWS. Stubs the boto3 client. Run:
    .venv/Scripts/python.exe deploy/test_jobs_handler.py
"""
from __future__ import annotations

import datetime as dt
import io
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "deploy" / "jobs"))
sys.path.insert(0, str(ROOT / "Scripts"))
os.environ.setdefault("ARTIFACTS_BUCKET", "test-bucket")
import jobs_handler as J

_PASS = _FAIL = 0


def check(name, cond, extra=""):
    global _PASS, _FAIL
    ok = bool(cond)
    _PASS += ok; _FAIL += (not ok)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + ("" if ok else f"  {extra}"))


UTC = dt.timezone.utc
JOB_A, JOB_B = "a" * 32, "b" * 32
READY = ["metrics.json", "explanation.json", "replay_3d.json", "overlay.mp4"]


class StubS3:
    """Just enough of the S3 client for _list_jobs/_enrich_meta/_update_meta."""

    def __init__(self, objects, metas=None):
        self.objects = objects            # [(key, LastModified)]
        self.metas = dict(metas or {})    # job_id -> meta dict
        self.put_calls = []

    def get_paginator(self, _name):
        objs = self.objects
        class P:
            def paginate(self, Bucket=None, Prefix=None):
                yield {"Contents": [{"Key": k, "LastModified": t}
                                    for k, t in objs if k.startswith(Prefix or "")]}
        return P()

    def get_object(self, Bucket=None, Key=None):
        job = Key.split("/")[1]
        if job not in self.metas:
            raise KeyError(Key)
        return {"Body": io.BytesIO(json.dumps(self.metas[job]).encode())}

    def put_object(self, Bucket=None, Key=None, Body=None, ContentType=None):
        self.put_calls.append((Key, json.loads(Body)))
        self.metas[Key.split("/")[1]] = json.loads(Body)


def t(h, m=0):
    return dt.datetime(2026, 7, 30, h, m, tzinfo=UTC)


print("\n[1] listing: job_meta.json excluded from the date aggregate")
objs = [(f"03_outputs/{JOB_A}/{f}", t(9)) for f in READY]
objs.append((f"03_outputs/{JOB_A}/job_meta.json", t(15)))   # a much later meta edit
J.s3 = StubS3(objs)
jobs = J._list_jobs()
check("one ready job", len(jobs) == 1 and jobs[0]["ready"])
check("date ignores the meta edit", jobs[0]["date"] == t(9).isoformat(), jobs[0]["date"])

print("\n[2] listing: meta enrichment inlines session fields + prefers uploaded_at")
meta = {"session_id": "deadbeef1234", "club": "driver",
        "uploaded_at": "2026-07-30T08:55:00+00:00", "notes": "range day"}
J.s3 = StubS3(objs, metas={JOB_A: meta})
jobs = J._list_jobs()
check("session fields inlined", jobs[0].get("session_id") == "deadbeef1234"
      and jobs[0].get("club") == "driver", str(jobs[0]))
check("uploaded_at becomes the date", jobs[0]["date"] == meta["uploaded_at"])
check("has_meta flag consumed (not leaked)", "has_meta" not in jobs[0])

print("\n[3] listing: meta fetch failure is non-fatal")
J.s3 = StubS3(objs, metas={})             # has_meta says yes, fetch will fail
jobs = J._list_jobs()
check("job still listed with S3 date", len(jobs) == 1 and jobs[0]["date"] == t(9).isoformat())

print("\n[4] POST /jobs: merge + validation + gate")
J.s3 = StubS3(objs, metas={JOB_A: {"session_id": "deadbeef1234",
                                   "uploaded_at": "2026-07-30T08:55:00+00:00",
                                   "weather": {"temp_c": 21, "source": "open-meteo"}}})
J.MC_RESULTS_TOKEN = "sekret"
ev = lambda body, tok="sekret", method="POST": {
    "requestContext": {"http": {"method": method}},
    "headers": {"x-mc-access": tok}, "body": json.dumps(body)}
r = J.handler(ev({"id": JOB_A, "meta": {"club": "7 iron", "notes": "into the wind",
                                        "uploaded_at": "2020-01-01T00:00:00+00:00",
                                        "evil": "x"}}))
body = json.loads(r["body"])
check("200 + merged meta", r["statusCode"] == 200 and body["meta"]["club"] == "7 iron", str(body))
check("existing fields survive the merge", body["meta"]["session_id"] == "deadbeef1234")
check("uploaded_at not user-editable", body["meta"]["uploaded_at"] == "2026-07-30T08:55:00+00:00")
check("pipeline weather block not clobbered", body["meta"]["weather"]["source"] == "open-meteo")
check("non-whitelisted field dropped", "evil" not in body["meta"])
check("write actually issued", J.s3.put_calls and J.s3.put_calls[0][0].endswith("job_meta.json"))

r = J.handler(ev({"id": "nope", "meta": {"club": "x"}}))
check("bad job id -> 400", r["statusCode"] == 400)
r = J.handler(ev({"id": JOB_B, "meta": {"junk": "x"}}))
check("no valid fields -> 400", r["statusCode"] == 400)
r = J.handler(ev({"id": JOB_A, "meta": {"club": "x"}}, tok="wrong"))
check("wrong access code -> 403", r["statusCode"] == 403)

print("\n[5] upload handler: presigned POST carries mc-* metadata fields")
sys.path.insert(0, str(ROOT / "deploy" / "upload"))
os.environ.setdefault("UPLOADS_BUCKET", "uploads-bucket")
import upload_handler as U

captured = {}
def fake_presign(**kw):
    captured.update(kw)
    return {"url": "https://s3.test", "fields": kw["Fields"]}
U._s3.generate_presigned_post = fake_presign
r = U.handler({"requestContext": {"http": {"method": "POST"}},
               "body": json.dumps({"filename": "swing.mp4", "content_type": "video/mp4",
                                   "session": {"session_id": "deadbeef1234",
                                               "session_label": "tues range",
                                               "club": "driver", "evil": "x"}})})
body = json.loads(r["body"])
check("200 + echoes session_id", r["statusCode"] == 200 and body["session_id"] == "deadbeef1234")
f = captured["Fields"]
check("metadata fields in the form", f.get("x-amz-meta-mc-session-id") == "deadbeef1234"
      and f.get("x-amz-meta-mc-club") == "driver", str(f))
check("uploaded_at auto-stamped", "x-amz-meta-mc-uploaded-at" in f)
check("non-whitelisted field never signed", not any("evil" in k for k in f))
check("every meta field has an exact-match condition",
      all({k: v} in captured["Conditions"] for k, v in f.items() if k.startswith("x-amz-meta-")),
      str(captured["Conditions"]))

r = U.handler({"requestContext": {"http": {"method": "POST"}},
               "body": json.dumps({"filename": "swing.mp4", "content_type": "video/mp4"})})
body = json.loads(r["body"])
check("no session block: server mints session_id", len(body["session_id"]) == 12)

print(f"\n{'=' * 46}\n  {_PASS} passed, {_FAIL} failed\n{'=' * 46}")
sys.exit(1 if _FAIL else 0)
