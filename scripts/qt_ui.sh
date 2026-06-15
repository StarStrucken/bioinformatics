#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"
cd "$REPO_ROOT"

if [[ "$#" -ne 1 ]]; then
  echo "usage: bash scripts/qt_ui.sh DATASET_ID" >&2
  exit 2
fi

python -m xenum.ui.graph_inspector.cli "$1"
