from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
VERSIONS_CSV = DATA_DIR / "versions.csv"
DEPSDEV_CSV = DATA_DIR / "depsdev_dependents_2026_05_05.csv"
MONTHLY_OUT = DATA_DIR / "rq1_monthly_attestation_rate_depsdev.csv"
WEEKLY_OUT = DATA_DIR / "rq1_weekly_attestation_rate_depsdev.csv"


def dpc_tier(dpc: int) -> str:
    if dpc > 500:
        return "top"
    if dpc > 10:
        return "mid"
    return "tail"


def parse_dt(s: str) -> datetime | None:
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def main() -> int:

    pkg_tier: dict[str, str] = {}
    with DEPSDEV_CSV.open() as f:
        for r in csv.DictReader(f):
            try:
                pkg_tier[r["name"]] = dpc_tier(int(r["dependents_count"]))
            except ValueError:
                continue
    print(f"Loaded {len(pkg_tier)} packages with deps.dev tier", flush=True)


    monthly: dict[tuple[str, str], Counter] = defaultdict(Counter)
    weekly: dict[tuple[str, str], Counter] = defaultdict(Counter)
    stats = {"rows_read": 0, "missing_tier": 0, "parse_fail": 0}

    with VERSIONS_CSV.open() as f:
        for r in csv.DictReader(f):
            stats["rows_read"] += 1
            pub = parse_dt(r["published_at"])
            if pub is None:
                stats["parse_fail"] += 1
                continue
            month = f"{pub.year:04d}-{pub.month:02d}"
            week_start = (pub.date() - timedelta(days=pub.weekday())).isoformat()
            tier = pkg_tier.get(r["package"], "missing")
            if tier == "missing":
                stats["missing_tier"] += 1
            has_att = r["has_attestation"].lower() == "true"
            for bucket, key in [(monthly, (month, tier)), (weekly, (week_start, tier))]:
                bucket[key]["total"] += 1
                if has_att:
                    bucket[key]["attested"] += 1

    print(f"Versions read: {stats['rows_read']:,}", flush=True)
    print(f"  with missing depsdev tier: {stats['missing_tier']:,}", flush=True)

    def write(out_path: Path, bucket: dict, key_field: str):
        rows = []

        for (key, tier), c in sorted(bucket.items()):
            total = c["total"]
            attested = c["attested"]
            rate = attested / total if total else 0
            rows.append({key_field: key, "tier": tier, "total": total,
                         "attested": attested, "rate": f"{rate:.4f}"})

        combined_excl: dict[str, Counter] = defaultdict(Counter)

        combined_incl: dict[str, Counter] = defaultdict(Counter)
        for (key, t), c in bucket.items():
            combined_incl[key]["total"] += c["total"]
            combined_incl[key]["attested"] += c["attested"]
            if t != "missing":
                combined_excl[key]["total"] += c["total"]
                combined_excl[key]["attested"] += c["attested"]
        for key, c in sorted(combined_excl.items()):
            total = c["total"]
            attested = c["attested"]
            rate = attested / total if total else 0
            rows.append({key_field: key, "tier": "all_excl_missing", "total": total,
                         "attested": attested, "rate": f"{rate:.4f}"})
        for key, c in sorted(combined_incl.items()):
            total = c["total"]
            attested = c["attested"]
            rate = attested / total if total else 0
            rows.append({key_field: key, "tier": "all", "total": total,
                         "attested": attested, "rate": f"{rate:.4f}"})

        tier_order = {"all": 0, "all_excl_missing": 1, "top": 2, "mid": 3, "tail": 4, "missing": 5}
        rows.sort(key=lambda r: (r[key_field], tier_order.get(r["tier"], 99)))
        with out_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[key_field, "tier", "total", "attested", "rate"])
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f" Saved {out_path} ({len(rows)} rows)", flush=True)
        return rows

    monthly_rows = write(MONTHLY_OUT, monthly, "month")
    weekly_rows = write(WEEKLY_OUT, weekly, "week_start")

    print(f"\n=== Monthly rate per deps.dev-tier (excl. missing) ===", flush=True)
    print(f"  {'Month':<10s}  {'top':>6s}  {'mid':>6s}  {'tail':>6s}  {'missing':>8s}", flush=True)
    for month in ("2025-10", "2025-11", "2025-12", "2026-01", "2026-02", "2026-03"):
        cells = {}
        for r in monthly_rows:
            if r["month"] == month and r["tier"] in ("top", "mid", "tail", "missing"):
                cells[r["tier"]] = float(r["rate"]) * 100
        print(f"  {month:<10s}  " + "  ".join(
            f"{cells.get(t, 0):>5.1f}%" for t in ("top", "mid", "tail", "missing")
        ), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
