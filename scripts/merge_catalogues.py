"""Join the batch tasks' catalogues (catalogue_task<k>.jsonl from run_batch.py) into one table.

    python scripts/merge_catalogues.py outputs/batch/ASTRID_047

Writes catalogue.csv in that folder: one row per galaxy, columns in the order they first appear
(sidx and status first), empty where a galaxy has no value (e.g. one that was skipped). A galaxy
listed more than once (a task that was run again) keeps its last line. Prints how many galaxies
are ok, skipped and failed.
"""
import csv
import json
import sys
from pathlib import Path

folder = Path(sys.argv[1])
rows = {}
for path in sorted(folder.glob("catalogue_task*.jsonl")):
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["sidx"]] = row
columns = []
for row in rows.values():
    columns += [c for c in row if c not in columns]
with open(folder / "catalogue.csv", "w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows.values())
status = [r["status"].split(":")[0] for r in rows.values()]
print(f"{folder / 'catalogue.csv'}: {len(rows)} galaxies; " +
      ", ".join(f"{status.count(k)} {k}" for k in ("ok", "skipped", "failed")))
