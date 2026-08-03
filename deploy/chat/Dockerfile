# Chat Lambda container — grounded coaching chatbot `/chat` endpoint.
#
# CPU-only, no GPU, no ML stack. This is the light/cached demo path: one HTTP
# request = one grounded chat turn over an already-cached scorecard. Deploys as a
# Lambda **container image** behind a Function URL (see full_app.yaml pattern).
#
# Build from the REPO ROOT (context = repo root, not this dir):
#   docker build -f deploy/chat/Dockerfile -t motion-caddie/chat .
#
# The dependency graph is: chat_handler -> Scripts/coaching_chat -> Scripts/coaching_llm_summary_v2
# (stdlib only) + the Anthropic SDK. KB + demo scorecards are baked so the image
# is self-contained; ANTHROPIC_API_KEY is injected at deploy time, never baked.

FROM public.ecr.aws/lambda/python:3.12

# --- deps (Anthropic SDK only) -------------------------------------------- #
COPY deploy/chat/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- app layout ------------------------------------------------------------ #
# chat_handler computes ROOT from APP_ROOT; coaching_llm_summary_v2 computes its
# PROJECT_ROOT as __file__.parent.parent. Both must resolve to /var/task, so the
# handler lives at the root and the modules under Scripts/ — mirroring the repo.
ENV APP_ROOT=/var/task

COPY deploy/chat_handler.py                 ${LAMBDA_TASK_ROOT}/chat_handler.py
COPY Scripts/coaching_chat.py               ${LAMBDA_TASK_ROOT}/Scripts/coaching_chat.py
COPY Scripts/coaching_llm_summary_v2.py     ${LAMBDA_TASK_ROOT}/Scripts/coaching_llm_summary_v2.py
COPY Scripts/coaching_persona.py            ${LAMBDA_TASK_ROOT}/Scripts/coaching_persona.py
COPY Scripts/ball_flight.py                 ${LAMBDA_TASK_ROOT}/Scripts/ball_flight.py
# chat v2 sessions: clustering + rollups + the evaluated delta comparator
COPY Scripts/session_meta.py                ${LAMBDA_TASK_ROOT}/Scripts/session_meta.py
COPY Scripts/coaching_sessions.py           ${LAMBDA_TASK_ROOT}/Scripts/coaching_sessions.py
COPY Scripts/coaching_progress.py           ${LAMBDA_TASK_ROOT}/Scripts/coaching_progress.py

# KB (indicator glossary/explainer cards) — loaded relative to the v2 module.
COPY Data/coaching/indicator_kb.json        ${LAMBDA_TASK_ROOT}/Data/coaching/indicator_kb.json
# Grounded-coaching drill corpus (get_drills serves ONLY these cards)
COPY Data/coaching/drill_cards.json         ${LAMBDA_TASK_ROOT}/Data/coaching/drill_cards.json
# Coaching personas — coaching_chat.load_persona() reads these by name at
# runtime (default "traditional"); missing dir raises PersonaLoadError.
COPY Data/coaching/personas/                ${LAMBDA_TASK_ROOT}/Data/coaching/personas/
# Phys-NN ball-flight weights (envelope-gated; ball_flight.py falls back to
# Phys-Q polynomials when absent)
COPY Data/coaching/ball_flight_nn.json      ${LAMBDA_TASK_ROOT}/Data/coaching/ball_flight_nn.json

# Cached demo scorecards the /chat path reads: Data/demo/<clip>/<clip>_scorecard.json
COPY Data/demo                              ${LAMBDA_TASK_ROOT}/Data/demo

CMD [ "chat_handler.handler" ]
