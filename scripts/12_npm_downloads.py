"""
Task 3 Step 3.2 — npm Downloads API.

For each of 3000 packages, fetch:
  GET https://api.npmjs.org/downloads/range/2025-06-01:2026-05-10/{name}

Window: 2025-06-01 → 2026-05-10 (344 days = 49.14 weeks).
Cache raw responses to data/npm_downloads_raw/{safe_fname}.json (idempotent;
resumable across runs).

Output: data/npm_downloads_metrics.csv
  columns: package, total_downloads_window, weekly_downloads_avg,
           daily_downloads_avg, days_with_data, status

`status` ∈ {'ok', 'no_data' (empty downloads), 'not_found' (404),
            'rate_limited', 'error'}.

Rate budget: no formal hard limit per the npmjs.org API docs, but we sleep 0.1s
between requests for politeness; retry-with-backoff on 429/5xx.

Run: conda run --no-capture-output -n stage2c python scripts/12_npm_downloads.py
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from urllib.parse import quote

import requests

DATA_DIR = Path(__file__).parent.parent / "data"
PACKAGES_FILE = DATA_DIR / "packages_full.txt"
RAW_DIR = DATA_DIR / "npm_downloads_raw"
OUT_CSV = DATA_DIR / "npm_downloads_metrics.csv"

RAW_DIR.mkdir(parents=True, exist_ok=True)

API_BASE = "https://api.npmjs.org/downloads/range/2025-06-01:2026-05-10"
HEADERS = {"User-Agent": "stage2c-research/1.0 (Task 3 downloads)", "Accept": "application/json"}

SLEEP_BETWEEN = 0.1
RETRY_BACKOFFS = [3, 10, 30]

# Window: 2025-06-01 to 2026-05-10 inclusive
TOTAL_DAYS = 344
WEEKS = TOTAL_DAYS / 7.0


def safe_fname(pkg: str) -> str:
    return pkg.replace("/", "_").replace("@", "at_") + ".json"


def fetch_with_retry(pkg: str) -> tuple[dict | None, str]:
    """Returns (json_payload, status)."""
    url = f"{API_BASE}/{quote(pkg, safe='/@')}"
    last_err = ""
    for attempt, backoff in enumerate([0, *RETRY_BACKOFFS]):
        if backoff > 0:
            time.sleep(backoff)
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            if r.status_code == 200:
                return r.json(), "ok"
            if r.status_code == 404:
                return None, "not_found"
            if r.status_code == 429:
                last_err = "429 rate limited"
                continue
            if r.status_code >= 500:
                last_err = f"{r.status_code} server error"
                continue
            last_err = f"{r.status_code}: {r.text[:200]}"
            return None, "error"
        except Exception as e:
            last_err = f"{type(e).__name__}: {str(e)[:200]}"
            continue
    return None, "rate_limited" if "429" in last_err else "error"


def compute_metrics(payload: dict) -> tuple[int, float, float, int]:
    """Returns (total, weekly_avg, daily_avg, days_with_data)."""
    downloads = payload.get("downloads") or []
    total = 0
    days_with_data = 0
    for entry in downloads:
        n = int(entry.get("downloads", 0))
        total += n
        if n > 0:
            days_with_data += 1
    weekly = total / WEEKS if WEEKS > 0 else 0
    daily = total / max(len(downloads), 1)
    return total, weekly, daily, days_with_data


def main() -> int:
    # Load package list
    packages: list[str] = []
    with PACKAGES_FILE.open() as f:
        header = next(f).rstrip("\n").split("\t")
        idx_name = header.index("name")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) > idx_name:
                packages.append(parts[idx_name])
    print(f"Loaded {len(packages)} package names", flush=True)

    # Resume
    rows: list[dict] = []
    counts = {"ok": 0, "no_data": 0, "not_found": 0, "rate_limited": 0,
              "error": 0, "cached": 0}

    for i, pkg in enumerate(packages, start=1):
        cache_path = RAW_DIR / safe_fname(pkg)
        payload: dict | None = None
        status = ""
        if cache_path.exists():
            try:
                payload = json.loads(cache_path.read_text())
                status = payload.get("_status", "ok") if isinstance(payload, dict) else "ok"
                if status == "ok" and "downloads" not in payload:
                    status = "error"  # mid-write or stale cache
                counts["cached"] += 1
            except Exception:
                payload = None

        if payload is None:
            payload_data, fetch_status = fetch_with_retry(pkg)
            status = fetch_status
            if payload_data is not None:
                payload = payload_data
            else:
                payload = {"_status": status, "package": pkg}
            # Cache result (including non-ok) for resumability
            cache_path.write_text(json.dumps(payload))
            time.sleep(SLEEP_BETWEEN)

        # Compute metrics
        if status == "ok" and isinstance(payload, dict) and "downloads" in payload:
            total, weekly, daily, days = compute_metrics(payload)
            if total == 0 and days == 0:
                status = "no_data"
            rows.append({
                "package": pkg,
                "total_downloads_window": str(total),
                "weekly_downloads_avg": f"{weekly:.2f}",
                "daily_downloads_avg": f"{daily:.4f}",
                "days_with_data": str(days),
                "status": status,
            })
        else:
            rows.append({
                "package": pkg,
                "total_downloads_window": "",
                "weekly_downloads_avg": "",
                "daily_downloads_avg": "",
                "days_with_data": "0",
                "status": status,
            })
        counts[status] = counts.get(status, 0) + 1

        if i % 200 == 0 or i == len(packages):
            print(f"  [{i:>4}/{len(packages)}] {pkg[:40]:<40s}  status={status}", flush=True)

    # Save
    fields = ["package", "total_downloads_window", "weekly_downloads_avg",
              "daily_downloads_avg", "days_with_data", "status"]
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n✓ Saved {OUT_CSV} ({len(rows)} rows)", flush=True)

    print(f"\n=== Status counts ===", flush=True)
    for k in ("ok", "no_data", "not_found", "rate_limited", "error", "cached"):
        v = counts.get(k, 0)
        if v:
            print(f"  {k}: {v}", flush=True)

    # Summary
    ok_dls = [float(r["weekly_downloads_avg"]) for r in rows
              if r["status"] == "ok" and r["weekly_downloads_avg"]]
    if ok_dls:
        ok_dls.sort()
        n = len(ok_dls)
        print(f"\n=== weekly_downloads_avg distribution ({n} ok) ===", flush=True)
        print(f"  min={ok_dls[0]:>12,.0f}  10%={ok_dls[n//10]:>12,.0f}"
              f"  med={ok_dls[n//2]:>12,.0f}  90%={ok_dls[9*n//10]:>12,.0f}"
              f"  max={ok_dls[-1]:>12,.0f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
