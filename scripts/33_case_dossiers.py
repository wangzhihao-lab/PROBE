"""
Task 6 Step 6.1 — Build per-case data dossiers.

For each of the 5 selected packages (axios, lodash, typescript, webpack,
@types/sinon as DefinitelyTyped representative), assemble:
  - Tier under ecosyste.ms vs deps.dev (with shift status)
  - Versions in 2025-08-01..2026-05-10 window with has_attestation
  - Cached workflow YAMLs in data/workflows_raw/
  - Classification + reason from data/classification.csv (only for the
    500-row non-prov sample — most cases won't be sampled)
  - influence_metrics.csv fields (downloads, stars, dependabot, etc.)
  - First-attested version time / event status (from Cox dataset prep)

Outputs:
  data/task_6_case_dossiers.json  (machine-readable)
  reports/task_6_case_dossiers.md (human-readable raw dossier)

Run: conda run --no-capture-output -n stage2c python scripts/33_case_dossiers.py
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
REPORTS_DIR = Path(__file__).parent.parent / "reports"
PACKAGES_FILE = DATA_DIR / "packages_full.txt"
VERSIONS_CSV = DATA_DIR / "versions.csv"
CLASS_CSV = DATA_DIR / "classification.csv"
INFLUENCE_CSV = DATA_DIR / "influence_metrics.csv"
DEPSDEV_CSV = DATA_DIR / "depsdev_dependents_2026_05_05.csv"
WORKFLOWS_DIR = DATA_DIR / "workflows_raw"
OUT_JSON = DATA_DIR / "task_6_case_dossiers.json"
OUT_MD = REPORTS_DIR / "task_6_case_dossiers.md"

CASES = ["axios", "lodash", "typescript", "webpack", "@types/sinon"]
DEADLINE = pd.Timestamp("2025-12-09", tz="UTC")


def dpc_tier(dpc):
    if pd.isna(dpc):
        return None
    d = int(dpc)
    return "top" if d > 500 else ("mid" if d > 10 else "tail")


def load_data():
    packages = pd.read_csv(PACKAGES_FILE, sep="\t")
    versions = pd.read_csv(VERSIONS_CSV, parse_dates=["published_at"])
    versions["published_at"] = pd.to_datetime(versions["published_at"], utc=True)
    cls = pd.read_csv(CLASS_CSV)
    inf = pd.read_csv(INFLUENCE_CSV)
    depsdev = pd.read_csv(DEPSDEV_CSV)
    return packages, versions, cls, inf, depsdev


def build_case(pkg, packages, versions, cls, inf, depsdev):
    out = {"package": pkg}

    # Sampling-frame meta
    p_row = packages[packages["name"] == pkg]
    if not p_row.empty:
        out["in_3000_sample"] = True
        out["eco_dpc"] = int(p_row["dependent_packages_count"].iloc[0])
        out["eco_tier"] = p_row["dpc_tier"].iloc[0]
        out["eco_rank_by_downloads"] = int(p_row["rank_by_downloads"].iloc[0])
        out["eco_downloads"] = int(p_row["downloads"].iloc[0])
        out["repo_url"] = p_row["repository_url"].iloc[0]
    else:
        out["in_3000_sample"] = False

    # deps.dev tier
    dd_row = depsdev[depsdev["name"] == pkg]
    if not dd_row.empty:
        out["dd_dependents"] = int(dd_row["dependents_count"].iloc[0])
        out["dd_tier"] = dpc_tier(out["dd_dependents"])
    else:
        out["dd_dependents"] = None
        out["dd_tier"] = "missing"

    # Tier-shift status
    out["tier_shift"] = "n/a"
    if out["in_3000_sample"] and out["dd_tier"] is not None:
        out["tier_shift"] = "same" if out["eco_tier"] == out["dd_tier"] else f"{out['eco_tier']}→{out['dd_tier']}"

    # Influence metrics
    inf_row = inf[inf["package"] == pkg]
    if not inf_row.empty:
        for fld in ["weekly_downloads_avg", "github_stars", "github_forks",
                    "package_age_days", "maintainers_count", "release_freq_12m",
                    "is_scoped_package", "github_owner_type",
                    "has_dependabot", "has_issue_template"]:
            v = inf_row[fld].iloc[0]
            out[fld] = None if pd.isna(v) else (str(v) if isinstance(v, (str, bool)) else float(v))

    # Versions in window
    pv = versions[versions["package"] == pkg].sort_values("published_at")
    pv_in_window = pv[(pv["published_at"] >= pd.Timestamp("2025-08-01", tz="UTC")) &
                       (pv["published_at"] <= pd.Timestamp("2026-05-10 23:59:59", tz="UTC"))]
    out["n_versions_in_window"] = int(len(pv_in_window))
    pv_in_window_attested = pv_in_window[pv_in_window["has_attestation"].astype(str).str.lower() == "true"]
    out["n_attested_versions_in_window"] = int(len(pv_in_window_attested))
    out["attestation_rate_in_window"] = (out["n_attested_versions_in_window"] / out["n_versions_in_window"]
                                          if out["n_versions_in_window"] else None)
    # First attested version
    first_att = pv[pv["has_attestation"].astype(str).str.lower() == "true"]
    out["first_attested_version"] = first_att.iloc[0]["version"] if not first_att.empty else None
    out["first_attested_at"] = (str(first_att.iloc[0]["published_at"])
                                if not first_att.empty else None)
    # Last version in window
    if not pv_in_window.empty:
        out["last_version_in_window"] = pv_in_window.iloc[-1]["version"]
        out["last_version_published_at"] = str(pv_in_window.iloc[-1]["published_at"])
        out["last_version_attested"] = bool(pv_in_window.iloc[-1]["has_attestation"])

    # Pre/post deadline split
    pre = pv_in_window[pv_in_window["published_at"] < DEADLINE]
    post = pv_in_window[pv_in_window["published_at"] >= DEADLINE]
    out["pre_deadline_n"] = int(len(pre))
    out["pre_deadline_attested"] = int((pre["has_attestation"].astype(str).str.lower() == "true").sum())
    out["post_deadline_n"] = int(len(post))
    out["post_deadline_attested"] = int((post["has_attestation"].astype(str).str.lower() == "true").sum())

    # Sample membership: is THIS package in the 500-row non-prov sample?
    cls_rows = cls[cls["package"] == pkg]
    out["in_500_classification_sample"] = bool(len(cls_rows) > 0)
    if not cls_rows.empty:
        out["classified_versions"] = []
        for _, row in cls_rows.iterrows():
            out["classified_versions"].append({
                "version": row["version"],
                "class": row["class"],
                "reason": row["reason"],
            })

    # Workflow files cached
    repo_url = out.get("repo_url") or ""
    if "github.com" in repo_url:
        tail = repo_url.split("github.com/")[-1]
        if tail.endswith(".git"):
            tail = tail[:-4]
        parts = tail.split("/")
        if len(parts) >= 2:
            owner, repo = parts[0], parts[1]
            out["github_owner"] = owner
            out["github_repo"] = repo
            wfs = sorted(p.name for p in WORKFLOWS_DIR.glob(f"{owner}__{repo}__*"))
            out["cached_workflows"] = wfs
            out["n_cached_workflows"] = len(wfs)

    return out


def main() -> int:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    packages, versions, cls, inf, depsdev = load_data()

    dossiers = [build_case(p, packages, versions, cls, inf, depsdev) for p in CASES]
    payload = {"task": "Task 6 — case dossiers", "cases": dossiers}
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=str))
    print(f"✓ Saved {OUT_JSON}", flush=True)

    # Markdown summary
    lines = ["# Task 6 case dossiers (raw)", "",
             "Auto-generated by `scripts/33_case_dossiers.py` for the 5 Task 6 cases:",
             "axios, lodash, typescript, webpack, @types/sinon."]

    for d in dossiers:
        lines += ["", f"## {d['package']}", ""]
        lines.append(f"- In 3000 sample: {d.get('in_3000_sample')}")
        if d.get('in_3000_sample'):
            lines.append(f"- ecosyste.ms tier: **{d['eco_tier']}** (DPC={d['eco_dpc']:,}; downloads rank={d['eco_rank_by_downloads']})")
        lines.append(f"- deps.dev tier (2026-05-05 snapshot): **{d['dd_tier']}** "
                     f"(direct dependents={d.get('dd_dependents', 'missing')})")
        lines.append(f"- Tier-shift status: {d.get('tier_shift')}")
        if "repo_url" in d:
            lines.append(f"- Repo: {d['repo_url']}")
        if "github_owner" in d:
            lines.append(f"- GitHub: {d['github_owner']}/{d['github_repo']}  "
                         f"(owner_type={d.get('github_owner_type', 'unknown')})")
        # Influence metrics
        if "weekly_downloads_avg" in d:
            lines.append(f"- Weekly downloads: {d['weekly_downloads_avg']:,.0f}")
        if "github_stars" in d and d['github_stars']:
            lines.append(f"- GitHub stars / forks: {int(d['github_stars']):,} / {int(d['github_forks']):,}")
        if "package_age_days" in d and d['package_age_days']:
            lines.append(f"- Package age: {int(d['package_age_days']):,} days "
                         f"(~{int(d['package_age_days'])/365.25:.1f} years)")
        if "maintainers_count" in d and d['maintainers_count'] is not None:
            lines.append(f"- Maintainers: {int(d['maintainers_count'])}")
        if "release_freq_12m" in d and d['release_freq_12m'] is not None:
            lines.append(f"- Release frequency 12mo: {int(d['release_freq_12m'])}")
        if "has_dependabot" in d:
            lines.append(f"- has_dependabot / has_issue_template / is_scoped: "
                         f"{d.get('has_dependabot')} / {d.get('has_issue_template')} / "
                         f"{d.get('is_scoped_package')}")

        lines.append("")
        lines.append("### Versions in window 2025-08-01 → 2026-05-10")
        lines.append(f"- Total versions in window: {d['n_versions_in_window']}")
        lines.append(f"- Attested in window: {d['n_attested_versions_in_window']}")
        if d.get('attestation_rate_in_window') is not None:
            lines.append(f"- Attestation rate in window: {d['attestation_rate_in_window']*100:.1f}%")
        lines.append(f"- Pre-deadline (08-01 → 12-08): {d['pre_deadline_n']} versions, "
                     f"{d['pre_deadline_attested']} attested")
        lines.append(f"- Post-deadline (12-09 → 05-10): {d['post_deadline_n']} versions, "
                     f"{d['post_deadline_attested']} attested")
        if d.get('first_attested_version'):
            lines.append(f"- First attested version (ever): {d['first_attested_version']} "
                         f"at {d['first_attested_at']}")
        else:
            lines.append(f"- First attested version (ever): NONE in our data")
        if "last_version_in_window" in d:
            lines.append(f"- Last version in window: {d['last_version_in_window']} "
                         f"(attested={d['last_version_attested']})")

        lines.append("")
        lines.append("### Classification (500-sample membership)")
        if d.get('in_500_classification_sample'):
            for cv in d['classified_versions']:
                lines.append(f"- {cv['version']} → class **{cv['class']}** — {cv['reason']}")
        else:
            lines.append("- NOT in the 500-row non-prov classification sample.")

        lines.append("")
        lines.append("### Cached GHA workflows")
        if "cached_workflows" in d and d["cached_workflows"]:
            for wf in d["cached_workflows"]:
                # Display short name (drop owner__repo__ prefix)
                short = wf.split("__", 2)[-1] if "__" in wf else wf
                lines.append(f"- `{short}`")
        else:
            lines.append("- (no cached workflows)")

    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"✓ Saved {OUT_MD}", flush=True)

    # Console summary
    print(f"\n=== Case summary ===", flush=True)
    for d in dossiers:
        in_sample = "✓" if d.get('in_3000_sample') else "✗"
        eco = d.get('eco_tier', '?')
        dd = d.get('dd_tier', '?')
        n_in = d.get('n_versions_in_window', 0)
        n_att = d.get('n_attested_versions_in_window', 0)
        print(f"  {d['package']:<20s} {in_sample} eco={eco:<5s} dd={dd:<8s} "
              f"versions={n_in:>3d} attested={n_att:>3d} "
              f"({(n_att/n_in*100 if n_in else 0):.0f}%)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
