"""
Task 1 Step 1.3 + 1.4 — Extract version-level data + stratified non-provenance sample.

Step 1.3:
- Reads each saved packument (`data/packuments_raw/*.json`)
- For every version with `2025-08-01 ≤ published_at ≤ 2026-05-10`, extracts:
    package, version, published_at, has_attestation, attestation_url,
    npm_user, npm_cli_version, tier, dependent_packages_count
- Writes `data/versions.csv`

Step 1.4:
- Filters versions where `has_attestation=False` AND `published_at > 2025-12-09`
  (the npm classic-token revocation deadline)
- Stratified sample of SAMPLE_TARGET (=500) versions:
    top 50% (=250), mid 30% (=150), tail 20% (=100); ≤ MAX_PER_PKG=3 versions per package
- Writes `data/non_provenance_sample.csv`

Run (default, reproduces the as-submitted data/versions.csv byte-for-byte):
  conda run --no-capture-output -n stage2c python scripts/03_extract_versions.py

Run (TSE R1 revision, extended pre-period per reviewer 1; extraction only, the
frozen 500-row sample is NOT redrawn):
  conda run --no-capture-output -n stage2c python scripts/03_extract_versions.py \
      --window-start 2024-01-01 --versions-out data/versions_extended.csv --skip-sample

Carried caveats:
- WINDOW_END is hard-pinned to the 2026-05-10 data cutoff and is NOT configurable;
  an assertion re-checks it on every run.
- --window-start only moves the LOWER bound. Moving it earlier re-reads the same
  already-collected packuments (data/packuments_raw/); it triggers no new
  collection. See reports/tse_r1_revision_plan.md constraint C-1.
- The non-provenance sample (Step 1.4) stays anchored to the 2025-12-09 deadline
  regardless of --window-start; use --skip-sample when extending so the frozen
  500-row sample and its kappa are not disturbed.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
PACKAGES_FILE = DATA_DIR / "packages_full.txt"
RAW_DIR = DATA_DIR / "packuments_raw"
VERSIONS_CSV = DATA_DIR / "versions.csv"
NON_PROV_CSV = DATA_DIR / "non_provenance_sample.csv"

# Spec: only count versions published 2025-08-01 → 2026-05-10.
# WINDOW_START is the protocol default and can be moved EARLIER via --window-start
# (TSE R1 reviewer-1 request). WINDOW_END is the 2026-05-10 data cutoff (铁律) and
# is not configurable.
DEFAULT_WINDOW_START = datetime(2025, 8, 1, tzinfo=timezone.utc)
WINDOW_END = datetime(2026, 5, 10, 23, 59, 59, tzinfo=timezone.utc)
DATA_CUTOFF = datetime(2026, 5, 10, 23, 59, 59, tzinfo=timezone.utc)
DEADLINE = datetime(2025, 12, 9, 23, 59, 59, 999999, tzinfo=timezone.utc)
# end-of-day UTC; spec "2025-12-09 < published_at" is date-level, so anything on 2025-12-09 (any UTC hour) is NOT post-deadline.
# Fixed 2026-05-16 after internal review round 2 (was 2025-12-09T00:00:00Z, which included Dec 9 evening records).
# npm classic-token revocation date = 2025-12-09; "post-deadline" means strictly 2025-12-10+ UTC.

SAMPLE_TARGET = 500
SAMPLE_FRAC = {"top": 0.50, "mid": 0.30, "tail": 0.20}  # 250 / 150 / 100
MAX_PER_PKG = 3
RNG = random.Random(42)


def safe_fname(pkg: str) -> str:
    return pkg.replace("/", "_").replace("@", "at_") + ".json"


def parse_dt(s: str) -> datetime | None:
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__ and __doc__.splitlines()[1] or "")
    ap.add_argument("--window-start", default="2025-08-01",
                    help="Lower bound (YYYY-MM-DD, UTC). Default 2025-08-01 = protocol spec.")
    ap.add_argument("--versions-out", default=str(VERSIONS_CSV))
    ap.add_argument("--sample-out", default=str(NON_PROV_CSV))
    ap.add_argument("--skip-sample", action="store_true",
                    help="Extraction only; do not redraw the frozen 500-row sample.")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    window_start = datetime.strptime(args.window_start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    versions_csv = Path(args.versions_out)
    non_prov_csv = Path(args.sample_out)
    assert WINDOW_END == DATA_CUTOFF, "WINDOW_END must remain pinned to the 2026-05-10 data cutoff"
    assert window_start < WINDOW_END, "--window-start must precede the data cutoff"
    if window_start != DEFAULT_WINDOW_START:
        if versions_csv.resolve() == VERSIONS_CSV.resolve():
            raise SystemExit("--window-start is non-default: pass --versions-out to a new file "
                             "(the protocol data/versions.csv is never overwritten)")
        if not args.skip_sample and non_prov_csv.resolve() == NON_PROV_CSV.resolve():
            raise SystemExit("--window-start is non-default: pass --skip-sample or --sample-out "
                             "(the frozen 500-row sample is never redrawn in place)")
        print(f"[non-default window] start={window_start.date()} (protocol default "
              f"{DEFAULT_WINDOW_START.date()}); end pinned at {WINDOW_END.date()}", flush=True)

    # Read package list with tier + dpc
    pkg_meta: dict[str, tuple[str, str]] = {}
    with PACKAGES_FILE.open() as f:
        header = next(f).rstrip("\n").split("\t")
        idx_name = header.index("name")
        idx_tier = header.index("dpc_tier")
        idx_dpc = header.index("dependent_packages_count")
        # NOTE: latest_release_at_collection_audit (renamed 2026-05-15 from latest_release_published_at)
        # is read elsewhere; not used in this script for version extraction.
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) > idx_dpc:
                pkg_meta[parts[idx_name]] = (parts[idx_tier], parts[idx_dpc])
    print(f"Loaded {len(pkg_meta)} packages with tier/dpc", flush=True)

    # Extract versions in window
    versions: list[dict] = []
    stats = {
        "packages_processed": 0,
        "404_marker_skipped": 0,
        "parse_error": 0,
        "versions_in_window": 0,
        "versions_with_attestation": 0,
        "non_prov_post_deadline": 0,
    }

    for pkg in pkg_meta:
        fpath = RAW_DIR / safe_fname(pkg)
        if not fpath.exists():
            continue
        try:
            data = json.loads(fpath.read_text())
        except Exception:
            stats["parse_error"] += 1
            continue
        if data.get("_404"):
            stats["404_marker_skipped"] += 1
            continue
        stats["packages_processed"] += 1

        times = data.get("time", {})
        versions_dict = data.get("versions", {})
        for ver, vdata in versions_dict.items():
            pub_str = times.get(ver, "")
            pub_time = parse_dt(pub_str)
            if pub_time is None:
                continue
            if not (window_start <= pub_time <= WINDOW_END):
                continue
            stats["versions_in_window"] += 1

            dist = vdata.get("dist") or {}
            attestations = dist.get("attestations") or {}
            att_url = attestations.get("url") if isinstance(attestations, dict) else None
            has_att = bool(att_url)
            if has_att:
                stats["versions_with_attestation"] += 1
            elif pub_time > DEADLINE:
                stats["non_prov_post_deadline"] += 1

            npm_user_obj = vdata.get("_npmUser") or {}
            npm_user_name = npm_user_obj.get("name", "") if isinstance(npm_user_obj, dict) else ""
            npm_cli = vdata.get("_npmVersion", "")

            tier, dpc = pkg_meta[pkg]
            versions.append({
                "package": pkg,
                "version": ver,
                "published_at": pub_str,
                "has_attestation": has_att,
                "attestation_url": att_url or "",
                "npm_user": npm_user_name,
                "npm_cli_version": npm_cli,
                "tier": tier,
                "dependent_packages_count": dpc,
            })

    print(f"\n=== Extraction stats ===", flush=True)
    for k, v in stats.items():
        print(f"  {k}: {v:,}", flush=True)

    # Save versions.csv
    fields = ["package", "version", "published_at", "has_attestation", "attestation_url",
              "npm_user", "npm_cli_version", "tier", "dependent_packages_count"]
    with versions_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for v in versions:
            w.writerow(v)
    print(f"\n✓ Saved {versions_csv} ({len(versions):,} rows)", flush=True)

    # 铁律 assertion: no extracted record may post-date the data cutoff
    latest = max((parse_dt(v["published_at"]) for v in versions
                  if parse_dt(v["published_at"])), default=None)
    assert latest is None or latest <= DATA_CUTOFF, \
        f"DATA CUTOFF VIOLATION: latest extracted record {latest} > {DATA_CUTOFF}"
    print(f"  cutoff assertion OK (latest record {latest})", flush=True)

    if args.skip_sample:
        print("\n[--skip-sample] Step 1.4 skipped; frozen sample left untouched.", flush=True)
        print("\n✅ Task 1 Step 1.3 PASS (extraction only)", flush=True)
        return 0

    # Step 1.4: filter non-provenance post-deadline
    np_pool = [v for v in versions
               if not v["has_attestation"]
               and (parse_dt(v["published_at"]) or DEADLINE) > DEADLINE]
    print(f"\n=== Step 1.4 stratified non-provenance sample ===", flush=True)
    print(f"Non-provenance pool (published > {DEADLINE.date()}): {len(np_pool):,}", flush=True)

    # By-tier breakdown
    by_tier_pool: dict[str, list[dict]] = {"top": [], "mid": [], "tail": []}
    for v in np_pool:
        by_tier_pool.setdefault(v["tier"], []).append(v)
    for t in ("top", "mid", "tail"):
        print(f"  pool {t}: {len(by_tier_pool[t]):,}", flush=True)

    if len(np_pool) < SAMPLE_TARGET:
        print(f"\n⚠ Pool ({len(np_pool)}) < target ({SAMPLE_TARGET}); sampling all available", flush=True)

    # Stratified pick: per tier, group by package, round-robin take ≤ MAX_PER_PKG
    sampled: list[dict] = []
    for tier in ("top", "mid", "tail"):
        target = int(round(SAMPLE_TARGET * SAMPLE_FRAC[tier]))
        pool = by_tier_pool[tier]
        by_pkg: dict[str, list[dict]] = {}
        for v in pool:
            by_pkg.setdefault(v["package"], []).append(v)
        # Shuffle within each package for randomness
        for vlist in by_pkg.values():
            RNG.shuffle(vlist)
        # Shuffle package order
        pkg_list = list(by_pkg.keys())
        RNG.shuffle(pkg_list)
        # Round-robin
        picked: list[dict] = []
        for round_idx in range(MAX_PER_PKG):
            for pkg in pkg_list:
                if len(picked) >= target:
                    break
                vlist = by_pkg[pkg]
                if round_idx < len(vlist):
                    picked.append(vlist[round_idx])
            if len(picked) >= target:
                break
        print(f"  sampled {tier}: {len(picked)}/{target}", flush=True)
        sampled.extend(picked)

    # Save non_provenance_sample.csv
    with non_prov_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for v in sampled:
            w.writerow(v)
    print(f"\n✓ Saved {non_prov_csv} ({len(sampled)} rows)", flush=True)

    # PASS / FAIL
    if len(sampled) < SAMPLE_TARGET * 0.9:
        print(f"\n⚠ Sample {len(sampled)} below 90% of {SAMPLE_TARGET} target", flush=True)
        return 1
    print(f"\n✅ Task 1 Step 1.3 + 1.4 PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
