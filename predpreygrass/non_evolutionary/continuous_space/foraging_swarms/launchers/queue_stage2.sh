#!/usr/bin/env bash
# Stage 2 queue (README): run the given "<condition> <seed>" pairs one after
# another, each with launch.py; a failed run is logged and the queue goes on.
# Two lanes side by side (14 simulation workers each), started from the
# repository root, e.g.:
#   systemd-run --user --unit=sw-lane1 --working-directory=$PWD \
#       bash predpreygrass/.../launchers/queue_stage2.sh "B 1" "C 2" "B 3" "A 1"
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
LOGS="$HERE/../runs/logs"
mkdir -p "$LOGS"
for entry in "$@"; do
    set -- $entry
    log="$LOGS/train_${1}_seed${2}_$(date +%Y-%m-%d_%H-%M-%S).log"
    echo "$(date '+%F %T') start $1 seed $2 -> $log" >> "$LOGS/queue.log"
    .conda/bin/python "$HERE/launch.py" "$1" "$2" > "$log" 2>&1
    echo "$(date '+%F %T') end $1 seed $2 exit $?" >> "$LOGS/queue.log"
done
