from __future__ import annotations

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


WINDOW_START = datetime(2025, 8, 1, tzinfo=timezone.utc)
WINDOW_END = datetime(2026, 5, 10, 23, 59, 59, tzinfo=timezone.utc)
DEADLINE = datetime(2025, 12, 9, 23, 59, 59, 999999, tzinfo=timezone.utc)


SAMPLE_TARGET = 500
SAMPLE_FRAC = {"top": 0.50, "mid": 0.30, "tail": 0.20}
MAX_PER_PKG = 3
RNG = random.Random(42)


def safe_fname(pkg: str) -> str:
    return pkg.replace("/", "_").replace("@", "at_") + ".json"


def parse_dt(s: str) -> datetime | None:
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def main() -> int:

    pkg_meta: dict[str, tuple[str, str]] = {}
    with PACKAGES_FILE.open() as f:
        header = next(f).rstrip("\n").split("\t")
        idx_name = header.index("name")
        idx_tier = header.index("dpc_tier")
        idx_dpc = header.index("dependent_packages_count")


        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) > idx_dpc:
                pkg_meta[parts[idx_name]] = (parts[idx_tier], parts[idx_dpc])
    print(f"Loaded {len(pkg_meta)} packages with tier/dpc", flush=True)


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
            if not (WINDOW_START <= pub_time <= WINDOW_END):
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


    fields = ["package", "version", "published_at", "has_attestation", "attestation_url",
              "npm_user", "npm_cli_version", "tier", "dependent_packages_count"]
    with VERSIONS_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for v in versions:
            w.writerow(v)
    print(f"\n Saved {VERSIONS_CSV} ({len(versions):,} rows)", flush=True)


    np_pool = [v for v in versions
               if not v["has_attestation"]
               and (parse_dt(v["published_at"]) or DEADLINE) > DEADLINE]
    print(f"\n=== Step 1.4 stratified non-provenance sample ===", flush=True)
    print(f"Non-provenance pool (published > {DEADLINE.date()}): {len(np_pool):,}", flush=True)


    by_tier_pool: dict[str, list[dict]] = {"top": [], "mid": [], "tail": []}
    for v in np_pool:
        by_tier_pool.setdefault(v["tier"], []).append(v)
    for t in ("top", "mid", "tail"):
        print(f"  pool {t}: {len(by_tier_pool[t]):,}", flush=True)

    if len(np_pool) < SAMPLE_TARGET:
        print(f"\n Pool ({len(np_pool)}) < target ({SAMPLE_TARGET}); sampling all available", flush=True)


    sampled: list[dict] = []
    for tier in ("top", "mid", "tail"):
        target = int(round(SAMPLE_TARGET * SAMPLE_FRAC[tier]))
        pool = by_tier_pool[tier]
        by_pkg: dict[str, list[dict]] = {}
        for v in pool:
            by_pkg.setdefault(v["package"], []).append(v)

        for vlist in by_pkg.values():
            RNG.shuffle(vlist)

        pkg_list = list(by_pkg.keys())
        RNG.shuffle(pkg_list)

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


    with NON_PROV_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for v in sampled:
            w.writerow(v)
    print(f"\n Saved {NON_PROV_CSV} ({len(sampled)} rows)", flush=True)


    if len(sampled) < SAMPLE_TARGET * 0.9:
        print(f"\n Sample {len(sampled)} below 90% of {SAMPLE_TARGET} target", flush=True)
        return 1
    print(f"\n + 1.4 PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
