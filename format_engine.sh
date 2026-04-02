#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./format_engine.sh [engine_dir]

Formats supported files under the engine directory:
  - Python (*.py) via Black
  - JSON (*.json) via Python's json module

If no directory is provided, ./engine relative to this script is used.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="${1:-$SCRIPT_DIR/engine}"

if [[ ! -d "$TARGET_DIR" ]]; then
  echo "Error: directory not found: $TARGET_DIR" >&2
  exit 1
fi

PYTHON_BIN="$(command -v python3 || true)"
if [[ -z "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python || true)"
fi

if [[ -z "$PYTHON_BIN" ]]; then
  echo "Error: Python was not found in PATH." >&2
  exit 1
fi

BLACK_CMD=()
if command -v black >/dev/null 2>&1; then
  BLACK_CMD=(black)
elif "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import importlib.util
import sys

sys.exit(0 if importlib.util.find_spec("black") else 1)
PY
then
  BLACK_CMD=("$PYTHON_BIN" -m black)
else
  echo "Error: Black is not installed. Install it with 'pip install black' or in your conda environment." >&2
  exit 1
fi

python_file_count=$(find "$TARGET_DIR" -type f -name '*.py' | wc -l | tr -d ' ')
json_file_count=$(find "$TARGET_DIR" -type f -name '*.json' | wc -l | tr -d ' ')

if [[ "$python_file_count" -gt 0 ]]; then
  echo "Formatting $python_file_count Python files with Black..."
  "${BLACK_CMD[@]}" --line-length 200 --skip-magic-trailing-comma "$TARGET_DIR"
fi

if [[ "$json_file_count" -gt 0 ]]; then
  echo "Formatting $json_file_count JSON files..."
  "$PYTHON_BIN" - "$TARGET_DIR" <<'PY'
from pathlib import Path
import json
import sys

root = Path(sys.argv[1])

INDENT = "  "
MAX_INLINE_WIDTH = 120


def format_json(value, level=0):
  current_indent = INDENT * level
  next_indent = INDENT * (level + 1)

  if isinstance(value, dict):
    if not value:
      return "{}"

    items = []
    for key, item in value.items():
      formatted_value = format_json(item, level + 1)
      items.append(f'{next_indent}{json.dumps(key, ensure_ascii=False)}: {formatted_value}')

    return "{\n" + ",\n".join(items) + "\n" + current_indent + "}"

  if isinstance(value, list):
    if not value:
      return "[]"

    primitive_only = all(not isinstance(item, (dict, list)) for item in value)
    if primitive_only:
      inline = "[" + ", ".join(json.dumps(item, ensure_ascii=False) for item in value) + "]"
      if len(current_indent) + len(inline) <= MAX_INLINE_WIDTH:
        return inline

    items = [f"{next_indent}{format_json(item, level + 1)}" for item in value]
    return "[\n" + ",\n".join(items) + "\n" + current_indent + "]"

  return json.dumps(value, ensure_ascii=False)

for path in sorted(root.rglob("*.json")):
  with path.open("r", encoding="utf-8") as handle:
    data = json.load(handle)

  formatted = format_json(data) + "\n"
  path.write_text(formatted, encoding="utf-8")
  print(f"Formatted {path}")
PY
fi

echo "Done. Supported files under '$TARGET_DIR' were formatted."
echo "Other file types were left unchanged."