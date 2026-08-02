# Upload-URL issuer Lambda

Issues short-lived **presigned S3 POST** URLs so the browser uploads swing videos
straight to S3 (never through compute). Presigned POST enforces content-type + a
hard size cap (`MAX_UPLOAD_MB`, default 200) as S3 conditions the client can't override.

## Deployed
- Function: `motion-caddie-upload-url` (python3.12, zip — boto3 is in the runtime, no deps)
- Role: `motion-caddie-app-upload-exec` (logs + `s3:PutObject` on the uploads prefix only)
- Env: `UPLOADS_BUCKET`, `UPLOAD_PREFIX=01_inputs/uploads`, `MAX_UPLOAD_MB=200`, `URL_TTL_SECONDS=300`

**Verified end-to-end**: issuer → presigned POST → browser-style multipart upload to
S3 (HTTP 204) → object lands at `01_inputs/uploads/<job_id>/<file>`. That PUT is what
triggers the EventBridge→SQS→processing path in `full_app.yaml`.

## Contract
`POST {filename, content_type, session?: {session_id?, session_label?, club?,
setting?, ball?, notes?, lat?, lon?}}` →
`{ job_id, session_id, object_key, url, fields, max_mb, expires_in, result_prefix }`
Client does a multipart POST of `fields` + the file to `url`; then polls
`result_prefix` (`03_outputs/<job_id>/`) for pipeline output.

Chat v2 sessions: the whitelisted `session` fields (validated by
`Scripts/session_meta.clean_meta`) are signed into the presigned POST as
`x-amz-meta-mc-*` object metadata — they ride on the video object itself so no
extra S3 write fires the processing trigger. `session_id` and `uploaded_at` are
auto-minted when the client omits them. The processing Lambda copies the
metadata to `03_outputs/<job_id>/job_meta.json`.

## Redeploy
```bash
# NB: session_meta.py must ship inside the zip (flat, next to the handler)
zip -j build/upload-lambda.zip deploy/upload/upload_handler.py Scripts/session_meta.py
aws lambda update-function-code --function-name motion-caddie-upload-url \
  --zip-file fileb://build/upload-lambda.zip --profile capstone
```

## Note
The issuer's public Function URL is subject to the same account SCP blocking
unauthenticated Function URLs (see `deploy/infra/PENDING_PERMISSIONS.md`). The S3
upload itself is direct-to-bucket and unaffected.
