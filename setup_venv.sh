#!/usr/bin/env bash
# Create the project virtualenv and install dependencies.
#
#   ./setup_venv.sh                      # CPU / default
#   TORCH_INDEX_URL=... ./setup_venv.sh  # override torch wheel index (e.g. GH200)
#
# On GH200 / aarch64 use torch 2.1.8:
#   TORCH_SPEC="torch==2.1.8" ./setup_venv.sh
set -euo pipefail

PYTHON=${PYTHON:-python3}
VENV=${VENV:-.venv}

"$PYTHON" -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install --upgrade pip

if [[ -n "${TORCH_SPEC:-}" ]]; then
  if [[ -n "${TORCH_INDEX_URL:-}" ]]; then
    pip install "$TORCH_SPEC" --index-url "$TORCH_INDEX_URL"
  else
    pip install "$TORCH_SPEC"
  fi
fi

pip install -r requirements.txt

# CMU pronouncing dictionary is bundled with `pronouncing`; nltk data only needed
# for the optional meter analysis in eval/diversity.py.
python -c "import nltk; nltk.download('cmudict', quiet=True)" || true

echo "Done. Activate with: source $VENV/bin/activate"
