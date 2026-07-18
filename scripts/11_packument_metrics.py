from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
PACKAGES_FILE = DATA_DIR / "packages_full.txt"
RAW_DIR = DATA_DIR / "packuments_raw"
OUT_CSV = DATA_DIR / "packument_metrics.csv"

CUTOFF = datetime(2026, 5, 10, 23, 59, 59, tzinfo=timezone.utc)
WINDOW_START = datetime(2025, 5, 10, tzinfo=timezone.utc)


def safe_fname(pkg: str) -> str:
    return pkg.replace("/", "_").replace("@", "at_") + ".json"


def parse_dt(s: str) -> datetime | None:
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def main() -> int:

    packages: list[str] = []
    with PACKAGES_FILE.open() as f:
        header = next(f).rstrip("\n").split("\t")
        idx_name = header.index("name")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) > idx_name:
                packages.append(parts[idx_name])
    print(f"Loaded {len(packages)} package names", flush=True)

    rows: list[dict] = []
    stats = {
        "with_packument": 0, "missing_packument": 0,
        "404_marker": 0, "parse_error": 0,
        "missing_created": 0, "missing_maintainers": 0,
    }

    for pkg in packages:
        fpath = RAW_DIR / safe_fname(pkg)
        if not fpath.exists():
            stats["missing_packument"] += 1
            rows.append({
                "package": pkg, "package_age_days": "",
                "maintainers_count": "", "release_freq_12m": "",
                "is_scoped_package": "",
            })
            continue
        try:
            data = json.loads(fpath.read_text())
        except Exception:
            stats["parse_error"] += 1
            rows.append({
                "package": pkg, "package_age_days": "",
                "maintainers_count": "", "release_freq_12m": "",
                "is_scoped_package": "",
            })
            continue
        if data.get("_404"):
            stats["404_marker"] += 1
            rows.append({
                "package": pkg, "package_age_days": "",
                "maintainers_count": "", "release_freq_12m": "",
                "is_scoped_package": "",
            })
            continue
        stats["with_packument"] += 1


        times = data.get("time", {}) or {}
        created_str = times.get("created") or ""
        created_dt = parse_dt(created_str)
        if created_dt is None:
            stats["missing_created"] += 1
            age_days = ""
        else:
            age_days = max(0, (CUTOFF - created_dt).days)


        maintainers = data.get("maintainers") or []
        if not isinstance(maintainers, list):
            maintainers = []
        if not maintainers:
            stats["missing_maintainers"] += 1
        m_count = len(maintainers)


        release_count = 0
        for ver, ts in times.items():
            if ver in ("created", "modified"):
                continue
            t = parse_dt(str(ts))
            if t is None:
                continue
            if WINDOW_START <= t <= CUTOFF:
                release_count += 1


        has_org = "true" if pkg.startswith("@") else "false"

        rows.append({
            "package": pkg,
            "package_age_days": age_days,
            "maintainers_count": m_count,
            "release_freq_12m": release_count,
            "is_scoped_package": has_org,
        })


    fields = ["package", "package_age_days", "maintainers_count",
              "release_freq_12m", "is_scoped_package"]
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n Saved {OUT_CSV} ({len(rows)} rows)", flush=True)


    print(f"\n=== Extraction stats ===", flush=True)
    for k, v in stats.items():
        print(f"  {k}: {v}", flush=True)


    ages = [int(r["package_age_days"]) for r in rows if r["package_age_days"] != ""]
    maints = [int(r["maintainers_count"]) for r in rows if r["maintainers_count"] != ""]
    freqs = [int(r["release_freq_12m"]) for r in rows if r["release_freq_12m"] != ""]
    orgs = sum(1 for r in rows if r["is_scoped_package"] == "true")

    if ages:
        ages_sorted = sorted(ages)
        n = len(ages_sorted)
        print(f"\n=== Summary distributions (over {n} packages with packument) ===", flush=True)
        print(f"  package_age_days:  min={ages_sorted[0]:>5d}  med={ages_sorted[n//2]:>5d}"
              f"  max={ages_sorted[-1]:>5d}  mean={sum(ages_sorted)//n:>5d}", flush=True)
    if maints:
        maints_sorted = sorted(maints)
        n = len(maints_sorted)
        print(f"  maintainers_count: min={maints_sorted[0]:>3d}  med={maints_sorted[n//2]:>3d}"
              f"  max={maints_sorted[-1]:>3d}", flush=True)
    if freqs:
        freqs_sorted = sorted(freqs)
        n = len(freqs_sorted)
        print(f"  release_freq_12m:  min={freqs_sorted[0]:>3d}  med={freqs_sorted[n//2]:>3d}"
              f"  max={freqs_sorted[-1]:>3d}", flush=True)
    print(f"  is_scoped_package=true: {orgs}/{len(rows)} ({orgs/len(rows)*100:.1f}%)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
