from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
PACKAGES_FILE = DATA_DIR / "packages_full.txt"
OUT_CSV = DATA_DIR / "depsdev_dependents_2026_05_05.csv"

SNAPSHOT_AT = "2026-05-05"
MAX_BYTES_BILLED = 500 * (1024 ** 3)


def main() -> int:
    try:
        from google.cloud import bigquery
    except ImportError:
        print("ERROR: google-cloud-bigquery missing; run `pip install google-cloud-bigquery`", file=sys.stderr)
        return 2

    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project:
        print("ERROR: GOOGLE_CLOUD_PROJECT not set; source ~/.zshenv first", file=sys.stderr)
        return 2


    packages: list[str] = []
    with PACKAGES_FILE.open() as f:
        header = next(f).rstrip("\n").split("\t")
        idx_name = header.index("name")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) > idx_name:
                packages.append(parts[idx_name])
    print(f"Loaded {len(packages)} packages", flush=True)


    quoted = ",".join('"' + p.replace('"', '\\"') + '"' for p in packages)


    query = f"""
SELECT
  Name AS name,
  COUNT(DISTINCT CONCAT(Dependent.System, "/", Dependent.Name)) AS dependents_count
FROM `bigquery-public-data.deps_dev_v1.Dependents`
WHERE SnapshotAt BETWEEN TIMESTAMP("{SNAPSHOT_AT} 00:00:00 UTC")
                     AND TIMESTAMP("{SNAPSHOT_AT} 23:59:59 UTC")
  AND System = "NPM"
  AND MinimumDepth = 1
  AND Name IN UNNEST([{quoted}])
GROUP BY Name
"""

    print(f"\nQuery text length: {len(query):,} chars (limit 1 MB)", flush=True)
    print(f"Max bytes billed cap: {MAX_BYTES_BILLED:,} bytes (500 GiB)", flush=True)
    print(f"Snapshot: {SNAPSHOT_AT}", flush=True)
    print(f"\nLaunching job...", flush=True)

    client = bigquery.Client(project=project)
    job_config = bigquery.QueryJobConfig(
        maximum_bytes_billed=MAX_BYTES_BILLED,
        use_query_cache=False,
    )
    t0 = time.time()
    query_job = client.query(query, job_config=job_config)
    print(f"Job ID: {query_job.job_id}", flush=True)

    rows = list(query_job.result())
    elapsed = time.time() - t0
    print(f"\n Completed in {elapsed:.1f}s", flush=True)
    print(f"  Bytes processed: {query_job.total_bytes_processed:,}", flush=True)
    print(f"  Bytes billed:    {query_job.total_bytes_billed:,}", flush=True)
    print(f"  Rows returned:   {len(rows)}", flush=True)


    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "dependents_count", "snapshot_at"])
        w.writeheader()
        for r in rows:
            w.writerow({"name": r["name"], "dependents_count": r["dependents_count"], "snapshot_at": SNAPSHOT_AT})
    print(f"\n Saved {OUT_CSV} ({len(rows)} rows)", flush=True)


    counts = sorted([int(r["dependents_count"]) for r in rows])
    if counts:
        n = len(counts)
        print(f"\nDependents distribution:", flush=True)
        print(f"  min={counts[0]}  med={counts[n//2]}  90%={counts[9*n//10]}  max={counts[-1]:,}", flush=True)
        not_found = len(packages) - len(rows)
        print(f"  packages not in deps.dev snapshot: {not_found}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
