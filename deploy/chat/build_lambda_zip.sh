#!/usr/bin/env bash
# Build the chat Lambda as a ZIP package (no Docker, no CodeBuild needed).
# Cross-compiles Linux wheels from any OS, bakes handler + modules + KB + demo
# scorecards, emits chat-lambda.zip. Deploy with `aws lambda create-function
# --runtime python3.12 --handler chat_handler.handler --zip-file fileb://chat-lambda.zip`.
#
# Usage:  bash deploy/chat/build_lambda_zip.sh <out-dir>
set -euo pipefail
OUT="${1:-./build}"; ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PKG="$OUT/chat-pkg"; rm -rf "$PKG"; mkdir -p "$PKG"

# 1. Linux wheels: anthropic SDK + numpy (Phys-NN ball-flight forward pass)
python -m pip install "anthropic>=0.111,<1.0" "numpy>=1.26" \
  --platform manylinux2014_x86_64 --python-version 3.12 --only-binary=:all: \
  --target "$PKG" --quiet

# 2. handler + the Scripts modules it imports + KB (relative to v2 module)
cp "$ROOT/deploy/chat_handler.py" "$PKG/chat_handler.py"
mkdir -p "$PKG/Scripts" "$PKG/Data/coaching"
cp "$ROOT/Scripts/coaching_chat.py" "$ROOT/Scripts/coaching_llm_summary_v2.py" \
   "$ROOT/Scripts/coaching_persona.py" "$ROOT/Scripts/ball_flight.py" "$PKG/Scripts/"
cp "$ROOT/Data/coaching/indicator_kb.json" "$PKG/Data/coaching/"
cp "$ROOT/Data/coaching/ball_flight_nn.json" "$PKG/Data/coaching/"   # Phys-NN weights
# coaching_chat.load_persona() reads these by name at runtime (default
# "traditional"); missing dir raises PersonaLoadError on every /chat call.
cp -r "$ROOT/Data/coaching/personas" "$PKG/Data/coaching/personas"

# 3. cached demo scorecards only (skip mp4s/frames — not read by /chat)
find "$ROOT/Data/demo" -name "*_scorecard.json" | while read -r f; do
  rel="${f#"$ROOT"/Data/demo/}"; mkdir -p "$PKG/Data/demo/$(dirname "$rel")"
  cp "$f" "$PKG/Data/demo/$rel"
done

# 4. trim + zip
find "$PKG" -type d \( -name "*.dist-info" -o -name "__pycache__" \) -prune -exec rm -rf {} + 2>/dev/null || true
python -c "import shutil,sys; shutil.make_archive(sys.argv[1],'zip',sys.argv[2])" "$OUT/chat-lambda" "$PKG"
echo "built $OUT/chat-lambda.zip"
# Lambda env: APP_ROOT=/var/task, ALLOWED_CLIPS=[0,2,4,6,8,10,1292], ANTHROPIC_API_KEY=<secret>
