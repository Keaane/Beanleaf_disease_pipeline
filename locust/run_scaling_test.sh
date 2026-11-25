#!/usr/bin/env bash
# Flood test with 1, 2 and 4 API containers behind nginx.
#   ./locust/run_scaling_test.sh            (defaults: 30 users, 60 s per run)
#   USERS=60 DURATION=90s ./locust/run_scaling_test.sh
set -euo pipefail
cd "$(dirname "$0")/.."

USERS="${USERS:-30}"
SPAWN="${SPAWN:-10}"
DURATION="${DURATION:-60s}"
HOST="${HOST:-http://localhost:8080}"
mkdir -p locust/results

for N in 1 2 4; do
  echo "=== $N container(s) ==="
  docker compose up -d --build --scale api="$N"
  # Restart nginx so it picks up the new set of containers, then wait for them.
  docker compose restart nginx >/dev/null
  until curl -sf "$HOST/health" >/dev/null; do sleep 2; done
  sleep 10  # let every replica load the model
  locust -f locust/locustfile.py --host "$HOST" --headless \
    -u "$USERS" -r "$SPAWN" -t "$DURATION" --csv "locust/results/api_$N" --only-summary
done

docker compose down
python locust/summarize.py locust/results | tee locust/results/summary.md
