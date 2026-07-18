# MotionCaddie infrastructure

`full_app.yaml` — the net-new tier for real user uploads (async processing + data +
auth), on top of the serverless demo stack. **cfn-lint clean (0 errors, 0 warnings).**
Designed around one constraint: **limited AWS credits → nothing idles, everything has a
ceiling.**

`web_hosting.yaml` — Phase 1 of `deploy/DEPLOYMENT_PLAN.md`: hosts the static
`deploy/web/` front end. See its own section below; it's a separate, independently
teardownable stack from `full_app.yaml`.

## GPU decision — you probably don't need one

The full pipeline is MediaPipe 2D (CPU) + **MixSTE 3D lift** (387 MB transformer) + a
1.1 MB event CNN + One-Euro + scorecard. All of it fits a **CPU Lambda container**
(10 GB RAM = 6 vCPU, 10 GB image, 15-min timeout). Per-swing ≈ **30–60 s on CPU**, which
is fine for an async job.

| Option | Latency / swing | $ / swing | Idle | When |
|---|---|---|---|---|
| **CPU Lambda (default)** | 30–60 s | ~$0.01 | **$0** | recommended — reuses the demo container, zero GPU spend |
| SageMaker Async, `ml.g4dn.xlarge` (T4 16 GB) | ~30–40 s | ~$0.008 | **$0** (scale-to-zero) | only if you need snappier latency |
| AWS Batch + g4dn spot | ~30 s | ~$0.004 | $0 | max thrift, more ops — overkill for a capstone |

So: **GPU = `ml.g4dn.xlarge` if you want one** — the cheapest inference GPU, one NVIDIA
T4. It's wired in `full_app.yaml` behind `EnableGpuWorker=true` as a **SageMaker Async
endpoint autoscaled to `MinCapacity: 0`** (a GPU that costs $0 when the queue is empty).
Default is off — start on CPU, flip it only if latency bites.

## Every resource has a timeout / TTL / cap

The user asked for timeouts on everything — here they are, one per resource:

| Resource | Guardrail |
|---|---|
| Aurora Serverless v2 | `MinCapacity: 0` + `SecondsUntilAutoPause: 900` → **auto-pauses to $0** after 15 min idle; `MaxCapacity: 2` caps active spend |
| Processing Lambda | `Timeout: 600 s` (hard); `ReservedConcurrentExecutions: 3` caps concurrent spend |
| SQS ingest queue | `VisibilityTimeout` = the Lambda timeout; `MessageRetentionPeriod: 4 d`; `maxReceiveCount: 3` → DLQ |
| SQS DLQ | `MessageRetentionPeriod: 14 d` |
| Lambda→SQS mapping | `MaximumConcurrency` cap |
| S3 uploads | `ExpirationInDays: 30` (raw video is PII → short TTL) + abort-incomplete-MPU |
| S3 artifacts | `ExpirationInDays: 90` (regenerable) |
| SageMaker Async (opt) | autoscale `Min 0 / Max 2`; `ScaleInCooldown: 300 s` → back to 0; `MaxConcurrentInvocationsPerInstance: 2` |
| Cognito tokens | 60-min access/id token validity |
| AWS Budgets | alert at 80% actual + 100% forecast of `MonthlyBudgetUSD` (default $40) |
| CloudWatch | alarms on Lambda `Errors` and DLQ depth → SNS email |

**Deliberately avoided** (they bill even when idle): NAT gateway (~$32/mo) and RDS Proxy
(~$15/mo) — sidestepped by using the **RDS Data API** (`EnableHttpEndpoint: true`), so
Lambda reaches Aurora over HTTP without a VPC attachment or a connection that pins the DB
awake. No provisioned concurrency. No always-on endpoint.

## Cost posture vs the $250 credits

- **Idle: ≈ $0/mo** — Aurora paused, Lambda at rest, SageMaker at 0 instances, S3/SQS
  request-priced.
- **Active: cents per swing** — Lambda compute + a brief Aurora burst + one Claude call.
- Order-of-magnitude: **thousands of swings fit inside $250.** The budget alarm + the
  Anthropic console cap are the backstops.

## Resource inventory (what `full_app.yaml` provisions)

Network (minimal VPC, no NAT/IGW) · Aurora SLv2 Postgres (Data API, scale-to-zero) ·
S3 uploads + artifacts (TTL'd, private) · SQS ingest + DLQ · EventBridge S3→SQS rule ·
processing Lambda (container) + least-priv role + SQS event source · Cognito user pool +
client · SNS alerts + 2 CloudWatch alarms + monthly Budget · **[optional]** SageMaker
Async GPU model/endpoint + autoscale-to-zero.

## Deploy & teardown

```bash
# validate offline (no AWS creds)
cfn-lint deploy/infra/full_app.yaml

# deploy (CPU default). ProcessingImageUri = the built processing container in ECR.
aws cloudformation deploy \
  --template-file deploy/infra/full_app.yaml \
  --stack-name motion-caddie-app \
  --capabilities CAPABILITY_IAM \
  --parameter-overrides \
      AlertEmail=you@example.com \
      ProcessingImageUri=<ecr-image-uri>

# turn the GPU on later:  EnableGpuWorker=true GpuModelImageUri=... GpuModelDataUrl=...

# teardown — every resource is DeletionPolicy: Delete / teardownable
aws s3 rm s3://motion-caddie-uploads-<acct> --recursive
aws s3 rm s3://motion-caddie-artifacts-<acct> --recursive
aws cloudformation delete-stack --stack-name motion-caddie-app
```

## Still to build (not infra — application code)
- The **processing container** (`ProcessingImageUri`): the SQS-triggered handler that runs
  `pipeline.py` on the uploaded mp4 and writes artifacts + rows via `load_data.py` accessors.
- The **upload-URL issuer** + a Cognito-authorized HTTP API route (thin; presigned S3 PUT).
- If GPU: package MixSTE as a SageMaker model artifact + inference container.

---

# Web hosting (`web_hosting.yaml`) — Phase 1

Puts the static `deploy/web/` site (vanilla HTML/CSS/JS, no build step) on a real
`https://` URL: a **private** S3 bucket (no static-website-hosting mode, no
public-read) fronted by **CloudFront** via **Origin Access Control (OAC)**. This is
the **web bucket**, distinct from `full_app.yaml`'s upload/artifact **data buckets** —
it never receives user video, only site files.

## Resource inventory
S3 bucket (private, SSE, `PublicAccessBlockConfiguration` fully on) · CloudFront OAC ·
CloudFront distribution (HTTP/2, gzip/br compression, `PriceClass_100` by default,
managed `CachingOptimized` policy, 403→`/index.html` fallback for bare-domain /
trailing-slash requests) · bucket policy scoped to `s3:GetObject` from this exact
distribution's ARN only.

## Deploy & sync

```bash
# validate offline (no AWS creds)
cfn-lint deploy/infra/web_hosting.yaml

aws cloudformation deploy \
  --template-file deploy/infra/web_hosting.yaml \
  --stack-name motion-caddie-web

# get the bucket name + public URL
aws cloudformation describe-stacks --stack-name motion-caddie-web \
  --query "Stacks[0].Outputs"

# ship the site (re-run after any deploy/web/ change)
# deploy/web/ is the publish root: EVERYTHING in it becomes public, so keep
# internal docs out of it (they live in deploy/). The exclude is only a guard
# against macOS Finder junk.
aws s3 sync deploy/web/ s3://<WebBucketName>/ --delete --exclude ".DS_Store"

# CloudFront caches aggressively — bust the cache after a sync
aws cloudfront create-invalidation --distribution-id <DistributionId> --paths "/*"
```

## Teardown
```bash
aws s3 rm s3://<WebBucketName> --recursive
aws cloudformation delete-stack --stack-name motion-caddie-web
```

## Open items (need Lawrence's sign-off per DEPLOYMENT_PLAN.md)
- Confirm `ProjectName` (default `motion-caddie` → bucket `motion-caddie-web-<acct>`)
  doesn't collide with anything already in his account.
- No custom domain / ACM cert wired yet — ships on the default
  `*.cloudfront.net` domain. Adding a custom domain later is a template addition
  (`Aliases` + `ViewerCertificate`), not a rebuild.
- H.264/yuv420p video assets under `deploy/web/assets/` sync and serve as plain
  objects — no special CloudFront behavior needed, `GET`/`HEAD` + compression is
  enough (video isn't re-compressed twice; compression only applies where CloudFront
  detects a compressible content-type).
