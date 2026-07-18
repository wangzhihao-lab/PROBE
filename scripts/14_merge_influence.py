from __future__ import annotations

import csv
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
PACKAGES_FILE = DATA_DIR / "packages_full.txt"
PACKUMENT_CSV = DATA_DIR / "packument_metrics.csv"
DOWNLOADS_CSV = DATA_DIR / "npm_downloads_metrics.csv"
GITHUB_CSV = DATA_DIR / "github_repo_meta.csv"
OWNER_TYPE_CSV = DATA_DIR / "github_owner_type.csv"
DEPSDEV_CSV = DATA_DIR / "depsdev_dependents_2026_05_05.csv"
OUT_CSV = DATA_DIR / "influence_metrics.csv"


def load_csv(path: Path, key: str = "package") -> dict[str, dict]:
    if not path.exists():
        print(f" missing: {path}", file=sys.stderr)
        return {}
    out: dict[str, dict] = {}
    with path.open() as f:
        for r in csv.DictReader(f):
            out[r[key]] = r
    return out


def main() -> int:
    pkg_tier: dict[str, str] = {}
    with PACKAGES_FILE.open() as f:
        header = next(f).rstrip("\n").split("\t")
        idx_name = header.index("name")
        idx_tier = header.index("dpc_tier")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) > idx_tier:
                pkg_tier[parts[idx_name]] = parts[idx_tier]
    print(f"Loaded {len(pkg_tier)} packages with tier", flush=True)

    packument = load_csv(PACKUMENT_CSV)
    downloads = load_csv(DOWNLOADS_CSV)
    github = load_csv(GITHUB_CSV)
    print(f"  packument rows: {len(packument)}", flush=True)
    print(f"  downloads rows: {len(downloads)}", flush=True)
    print(f"  github rows:    {len(github)}", flush=True)


    owner_type: dict[str, str] = {}
    if OWNER_TYPE_CSV.exists():
        with OWNER_TYPE_CSV.open() as f:
            for r in csv.DictReader(f):
                owner_type[r["owner"]] = r["type"]
        print(f"  owner_type rows: {len(owner_type)}", flush=True)


    depsdev: dict[str, str] = {}
    if DEPSDEV_CSV.exists():
        with DEPSDEV_CSV.open() as f:
            for r in csv.DictReader(f):
                depsdev[r["name"]] = r["dependents_count"]
        print(f"  depsdev rows: {len(depsdev)}", flush=True)

    fields = [
        "package", "tier",
        "weekly_downloads_avg", "total_downloads_window", "downloads_status",
        "depsdev_dependents_count",
        "github_stars", "github_forks",
        "package_age_days", "maintainers_count", "release_freq_12m",
        "is_scoped_package",
        "github_owner_type",
        "has_dependabot", "has_issue_template",
        "github_status",
    ]

    rows: list[dict] = []
    missing: dict[str, int] = {f: 0 for f in fields if f != "package"}

    for pkg, tier in sorted(pkg_tier.items()):
        p = packument.get(pkg, {})
        d = downloads.get(pkg, {})
        g = github.get(pkg, {})

        gh_owner = g.get("owner", "")
        r = {
            "package": pkg,
            "tier": tier,
            "weekly_downloads_avg": d.get("weekly_downloads_avg", ""),
            "total_downloads_window": d.get("total_downloads_window", ""),
            "downloads_status": d.get("status", ""),
            "depsdev_dependents_count": depsdev.get(pkg, "") if depsdev else "pending",
            "github_stars": g.get("stargazers_count", ""),
            "github_forks": g.get("forks_count", ""),
            "package_age_days": p.get("package_age_days", ""),
            "maintainers_count": p.get("maintainers_count", ""),
            "release_freq_12m": p.get("release_freq_12m", ""),
            "is_scoped_package": p.get("is_scoped_package", ""),
            "github_owner_type": owner_type.get(gh_owner, "") if gh_owner else "",
            "has_dependabot": g.get("has_dependabot", ""),
            "has_issue_template": g.get("has_issue_template", ""),
            "github_status": g.get("status", ""),
        }
        rows.append(r)
        for fk, v in r.items():
            if fk == "package":
                continue
            if v == "" or v is None:
                missing[fk] = missing.get(fk, 0) + 1

    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n Saved {OUT_CSV} ({len(rows)} rows × {len(fields)} cols)", flush=True)


    print(f"\n=== Missing-count per column ===", flush=True)
    for fk, c in missing.items():
        print(f"  {fk:32s}: {c:>4} missing", flush=True)


    from collections import Counter
    cross: dict[str, Counter] = {"top": Counter(), "mid": Counter(), "tail": Counter()}
    for r in rows:
        cross[r["tier"]][r["downloads_status"]] += 1
    print(f"\n=== Tier × downloads_status ===", flush=True)
    statuses = sorted({s for c in cross.values() for s in c})
    header_str = "  tier   " + "  ".join(f"{s:>14}" for s in statuses)
    print(header_str, flush=True)
    for t in ("top", "mid", "tail"):
        cells = [f"{cross[t].get(s, 0):>14}" for s in statuses]
        print(f"  {t:6s} " + "  ".join(cells), flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
