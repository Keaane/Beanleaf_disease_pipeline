"""Turn Locust CSV results into a Markdown table: python locust/summarize.py locust/results"""

import csv
import glob
import os
import re
import sys

folder = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "results")
rows = []
for path in glob.glob(os.path.join(folder, "*_stats.csv")):
    m = re.search(r"api_(\d+)_stats\.csv$", path)
    if not m:
        continue
    with open(path) as f:
        stats = {r["Name"]: r for r in csv.DictReader(f)}
    agg, pred = stats["Aggregated"], stats.get("/predict", {})
    rows.append((
        int(m.group(1)),
        int(agg["Request Count"]),
        int(agg["Failure Count"]),
        float(agg["Requests/s"]),
        float(pred.get("Average Response Time", 0)),
        float(pred.get("50%", 0)),
        float(pred.get("95%", 0)),
        float(pred.get("99%", 0)),
        float(agg["50%"]),
    ))

print("| Containers | Requests | Failures | Throughput (req/s) | /predict avg (ms) | /predict median (ms) "
      "| /predict p95 (ms) | /predict p99 (ms) | All endpoints median (ms) |")
print("|---|---|---|---|---|---|---|---|---|")
for r in sorted(rows):
    fail_pct = 100 * r[2] / max(1, r[1])
    print(f"| {r[0]} | {r[1]:,} | {r[2]} ({fail_pct:.1f}%) | {r[3]:.1f} | {r[4]:,.0f} | {r[5]:,.0f} "
          f"| {r[6]:,.0f} | {r[7]:,.0f} | {r[8]:,.0f} |")
