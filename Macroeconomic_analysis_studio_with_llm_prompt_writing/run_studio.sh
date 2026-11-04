#!/usr/bin/env bash
# Start the studio from the same folder as this launcher, regardless of the current directory.
set -euo pipefail
cd "$(dirname "$0")"
project_dir="$(pwd -P)"
py="$project_dir/.venv/bin/python"

if [[ ! -x "$py" ]]; then
    python_cmd=""
    for candidate in python3 python3.12 python3.11 python3.13 python3.14 python3.10; do
        if command -v "$candidate" >/dev/null && "$candidate" -c 'import sys; sys.exit(sys.version_info < (3, 10))'; then
            python_cmd="$candidate"
            break
        fi
    done
    if [[ -z "$python_cmd" ]]; then
        echo "Install Python 3.10 or newer for macOS, then run this launcher again." >&2
        exit 1
    fi
    echo "Creating the studio's Python environment…"
    "$python_cmd" -m venv .venv
fi
if [[ ! -x .venv/bin/econ422-studio || pyproject.toml -nt .venv/bin/econ422-studio ]] || ! "$py" -c 'import studio_core, dml, reportlab' >/dev/null 2>&1; then
    echo "Installing the studio and its Python dependencies…"
    "$py" -m pip install -e "$project_dir"
fi

model="llama3.1:8b"
export OLLAMA_HOST="http://127.0.0.1:11434"
ollama_cmd="$(command -v ollama || true)"
if [[ -z "$ollama_cmd" ]]; then
    for candidate in /Applications/Ollama.app/Contents/Resources/ollama "$HOME/Applications/Ollama.app/Contents/Resources/ollama"; do
        if [[ -x "$candidate" ]]; then
            ollama_cmd="$candidate"
            break
        fi
    done
fi
# Worked examples run in Python even when the local model is not installed.
if [[ -n "$ollama_cmd" ]]; then
    if ! curl --noproxy '*' -fsS --max-time 2 "$OLLAMA_HOST/api/tags" >/dev/null 2>&1; then
        if command -v open >/dev/null && [[ "$(uname -s)" == "Darwin" ]]; then
            open -a Ollama >/dev/null 2>&1 || true
        fi
    fi
else
    echo "Worked examples are ready without Ollama."
fi
echo "For optional Llama review, open Ollama and run: ollama pull $model"

echo "Opening Macroeconomic Analysis Studio with LLM prompt writing from $project_dir"
PYTHONPATH="$project_dir" exec "$py" -m studio_core app "$@"
