#!/usr/bin/env bash
# Run the speed-trap station on a Raspberry Pi 5 + Hailo HAT.
# Usage: scripts/run_on_pi.sh [config_yaml]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ ! -f .venv/bin/activate ]]; then
    echo "ERROR: .venv/bin/activate not found in $PROJECT_ROOT" >&2
    echo "Bootstrap first:" >&2
    echo "  python3 -m venv .venv --system-site-packages" >&2
    echo "  source .venv/bin/activate" >&2
    echo "  pip install -r requirements.txt && pip install -e ." >&2
    echo "  # picamera2 + hailo runtime come from apt, not pip" >&2
    exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate

CONFIG="${1:-config/station_a.yaml}"
exec python -m apps.run_station --config "$CONFIG"
