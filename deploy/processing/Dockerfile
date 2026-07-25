# Processing worker — container Lambda (the one piece that CAN'T be a zip: torch +
# mediapipe + the model bundle exceed the 250 MB zip limit).
#
# Build from repo root; the model bundle (~470 MB, gitignored) is pulled from S3
# into ./deploy/processing/models by the buildspec BEFORE this build. See buildspec.yml.
#   docker build -f deploy/processing/Dockerfile -t motion-caddie/processing .
#
# STATUS: authored, not yet built (no Docker/CodeBuild locally). Expect to iterate
# on native libs (mediapipe/opencv) on the first real build.

FROM public.ecr.aws/lambda/python:3.12

# native libs mediapipe/opencv need at runtime
RUN dnf install -y mesa-libGL libglvnd-glx && dnf clean all

COPY deploy/processing/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV APP_ROOT=/var/task
# pipeline scripts + adapters + the modules they import (whole Scripts/ is simplest & correct)
COPY Scripts/                 ${LAMBDA_TASK_ROOT}/Scripts/
# vendored GolfPose common/ package (MixSTE2 model code) — the adapter imports
# it from GOLFPOSE_REPO; locally that's the C:\dev\golfpose-repo clone
ENV GOLFPOSE_REPO=/var/task/golfpose_repo
COPY golfpose_repo/           ${LAMBDA_TASK_ROOT}/golfpose_repo/
COPY deploy/processing/processing_handler.py ${LAMBDA_TASK_ROOT}/processing_handler.py
# KB + tour reference bands + confidence tiers + gated feature configs — the
# small JSONs the scorecard / coaching eval / web-artifact steps read
COPY Data/coaching/indicator_kb.json ${LAMBDA_TASK_ROOT}/Data/coaching/indicator_kb.json
COPY Data/coaching/reference_bands.json ${LAMBDA_TASK_ROOT}/Data/coaching/reference_bands.json
COPY Data/coaching/indicator_confidence.json ${LAMBDA_TASK_ROOT}/Data/coaching/indicator_confidence.json
COPY Data/coaching/features/ ${LAMBDA_TASK_ROOT}/Data/coaching/features/
# Coaching personas — coaching_llm_summary_v2.load_persona() reads these by
# name at runtime (default "traditional"); missing dir raises a load error.
COPY Data/coaching/personas/ ${LAMBDA_TASK_ROOT}/Data/coaching/personas/
# model bundle (staged from S3 by buildspec into deploy/processing/models/)
COPY deploy/processing/models/ ${LAMBDA_TASK_ROOT}/Models/

CMD [ "processing_handler.handler" ]
