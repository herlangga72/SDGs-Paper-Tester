#!/bin/sh
# Deploy the SDG Paper Matcher on whichever port the platform assigns:
#   - Hugging Face Spaces exposes port 7860 (default)
#   - Render free tier injects $PORT (e.g. 10000)
set -e
PORT="${PORT:-7860}"
echo "[entrypoint] starting SDG Paper Matcher on 0.0.0.0:${PORT}"
exec python3 web/app.py --host 0.0.0.0 --port "${PORT}" --no-browser
