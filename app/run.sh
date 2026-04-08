#!/usr/bin/env bash
# Run SAGE locally
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load .env if present
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

# Prefer the user-local bin where pip3 installs scripts on macOS
export PATH="$HOME/Library/Python/3.9/bin:$PATH"

echo "Starting SAGE on http://localhost:8501 ..."
streamlit run app.py \
  --server.port 8501 \
  --server.headless false \
  --browser.gatherUsageStats false
