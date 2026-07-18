from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
VERSIONS_CSV = DATA_DIR / "versions.csv"
MONTHLY_OUT = DATA_DIR / "rq1_monthly_attestation_rate.csv"
WEEKLY_OUT = DATA_DIR / "rq1_weekly_attestation_rate.csv"


def parse_dt(s: str) -> datetime | None:
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def main() -> int:
    monthly: dict[tuple[str, str], Counter] = defaultdict(Counter)
    weekly: dict[tuple[str, str], Counter] = defaultdict(Counter)
    stats = {"rows_read": 0, "parse_fail": 0}

    with VERSIONS_CSV.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            stats["rows_read"] += 1
            pub = parse_dt(r["published_at"])
            if pub is None:
                stats["parse_fail"] += 1
                continue
            month = f"{pub.year:04d}-{pub.month:02d}"

            week_start = (pub.date() - timedelta(days=pub.weekday())).isoformat()
            tier = r["tier"]
            has_att = r["has_attestation"].lower() == "true"
            for bucket, key in [(monthly, (month, tier)), (weekly, (week_start, tier))]:
                bucket[key]["total"] += 1
                if has_att:
                    bucket[key]["attested"] += 1

    print(f"Versions read: {stats['rows_read']:,}", flush=True)
    print(f"Parse failures: {stats['parse_fail']}", flush=True)

    def write(out_path: Path, bucket: dict, key_field: str):
        rows = []

        for (key, tier), c in sorted(bucket.items()):
            total = c["total"]
            attested = c["attested"]
            rate = attested / total if total else 0
            rows.append({key_field: key, "tier": tier, "total": total,
                         "attested": attested, "rate": f"{rate:.4f}"})

        combined: dict[str, Counter] = defaultdict(Counter)
        for (key, _), c in bucket.items():
            combined[key]["total"] += c["total"]
            combined[key]["attested"] += c["attested"]
        for key, c in sorted(combined.items()):
            total = c["total"]
            attested = c["attested"]
            rate = attested / total if total else 0
            rows.append({key_field: key, "tier": "all", "total": total,
                         "attested": attested, "rate": f"{rate:.4f}"})
        rows.sort(key=lambda r: (r[key_field], {"all": 0, "top": 1, "mid": 2, "tail": 3}[r["tier"]]))
        with out_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[key_field, "tier", "total", "attested", "rate"])
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f" Saved {out_path} ({len(rows)} rows)", flush=True)
        return rows

    monthly_rows = write(MONTHLY_OUT, monthly, "month")
    weekly_rows = write(WEEKLY_OUT, weekly, "week_start")


    print(f"\n=== Monthly overall (tier='all') ===", flush=True)
    print(f"  {'Month':<10s}  {'Total':>7s}  {'Att':>7s}  {'Rate':>6s}", flush=True)
    for r in monthly_rows:
        if r["tier"] == "all":
            print(f"  {r['month']:<10s}  {int(r['total']):>7,}  {int(r['attested']):>7,}  {float(r['rate'])*100:>5.1f}%", flush=True)


    print(f"\n=== Rate around Dec 9 deadline (per tier) ===", flush=True)
    print(f"  {'Month':<10s}  {'top':>6s}  {'mid':>6s}  {'tail':>6s}", flush=True)
    for month in ("2025-10", "2025-11", "2025-12", "2026-01", "2026-02", "2026-03"):
        cells = {}
        for r in monthly_rows:
            if r["month"] == month and r["tier"] in ("top", "mid", "tail"):
                cells[r["tier"]] = float(r["rate"]) * 100
        print(f"  {month:<10s}  " + "  ".join(f"{cells.get(t, 0):>5.1f}%" for t in ("top", "mid", "tail")), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
