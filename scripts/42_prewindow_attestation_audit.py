"""
Rebuttal B1 — Pre-window attestation audit (Cox left-truncation severity).

Reviewer concern: the Cox time origin is 2025-08-01 and the event is the first
*in-window* attested release. Because `dist.attestations` has existed on npm
since 2023 (provenance metadata from 2023; trusted-publisher mechanism in 2025,
see reports/task_7_plan.md), some packages were ALREADY attesting BEFORE the
window. The window-anchored Cox treats such "prevalent adopters" as if they
adopt inside the window -> left-truncation, which can bias the in-window hazard.

This script COUNTS; it does not re-fit. It answers, from the full version
history in data/packuments_raw/*.json:
  1. How many packages have >= 1 attested version published BEFORE 2025-08-01?
  2. How many of those sit in the Cox modeling subset (window-active +
     deps.dev-tiered + covariate-complete, mirroring scripts/29)?
  3. What is the earliest-attested-year distribution (2023 / 2024 / 2025-H1)?

The count decides the downstream rebuttal action (per user):
  exclude-prevalent sensitivity, prevalent-adopter indicator, or clarify the
  event as "first in-window attested release" only.

Run: conda run --no-capture-output -n stage2c python scripts/42_prewindow_attestation_audit.py
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent.parent / "data"
REPORTS = Path(__file__).parent.parent / "reports"
RAW = DATA / "packuments_raw"
VERSIONS_CSV = DATA / "versions.csv"
INFLUENCE_CSV = DATA / "influence_metrics.csv"
DEPSDEV_CSV = DATA / "depsdev_dependents_2026_05_05.csv"
OUT_CSV = DATA / "prewindow_attestation_audit.csv"
OUT_MD = REPORTS / "task_9_rebuttal_b1_prewindow_audit.md"

WINDOW_START = datetime(2025, 8, 1, tzinfo=timezone.utc)


def parse_dt(s: str):
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def dpc_tier(dpc: int) -> str:
    if dpc > 500:
        return "top"
    if dpc > 10:
        return "mid"
    return "tail"


def earliest_attested_per_package() -> dict[str, datetime]:
    """Scan FULL version history; return {package: earliest attested publish dt}."""
    earliest: dict[str, datetime] = {}
    n_scanned = n_404 = n_err = 0
    for fp in RAW.glob("*.json"):
        try:
            data = json.loads(fp.read_text())
        except Exception:
            n_err += 1
            continue
        if data.get("_404"):
            n_404 += 1
            continue
        n_scanned += 1
        name = data.get("name") or data.get("_id") or fp.stem
        times = data.get("time", {})
        versions = data.get("versions", {})
        att_times: list[datetime] = []
        for ver, vdata in versions.items():
            dist = (vdata or {}).get("dist") or {}
            att = dist.get("attestations") or {}
            url = att.get("url") if isinstance(att, dict) else None
            if url:
                pt = parse_dt(times.get(ver, ""))
                if pt:
                    att_times.append(pt)
        if att_times:
            earliest[name] = min(att_times)
    print(f"Scanned packuments: {n_scanned} (404 markers {n_404}, parse errors {n_err})", flush=True)
    print(f"Packages with >=1 attested version EVER: {len(earliest):,}", flush=True)
    return earliest


def cox_subset_packages() -> set[str]:
    """Reproduce the scripts/29 Cox modeling subset at package granularity:
    window-active (in versions.csv) + deps.dev tier in {top,mid,tail} +
    covariate-complete (same dropna columns as scripts/29)."""
    versions = pd.read_csv(VERSIONS_CSV)
    active = set(versions["package"].unique())

    inf = pd.read_csv(INFLUENCE_CSV)
    inf = inf[inf["package"].isin(active)].copy()

    depsdev = pd.read_csv(DEPSDEV_CSV)
    dd_map = dict(zip(depsdev["name"], depsdev["dependents_count"].astype(int)))
    inf["dd"] = inf["package"].map(dd_map)
    inf["tier"] = inf["dd"].apply(lambda x: dpc_tier(int(x)) if pd.notna(x) else "missing")
    inf = inf[inf["tier"].isin(["top", "mid", "tail"])].copy()

    # same covariates / dropna as scripts/29 build_survival_data
    inf["log_weekly_downloads"] = np.log1p(pd.to_numeric(inf["weekly_downloads_avg"], errors="coerce"))
    inf["log_age_days"] = np.log1p(pd.to_numeric(inf["package_age_days"], errors="coerce"))
    inf["log_maintainers"] = np.log1p(pd.to_numeric(inf["maintainers_count"], errors="coerce"))
    need = ["log_weekly_downloads", "log_age_days", "log_maintainers",
            "github_owner_type", "has_dependabot", "has_issue_template", "is_scoped_package"]
    inf = inf.dropna(subset=[c for c in need if c in inf.columns])
    return set(inf["package"].unique())


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)

    earliest = earliest_attested_per_package()
    cox_pkgs = cox_subset_packages()
    versions = pd.read_csv(VERSIONS_CSV)
    active_pkgs = set(versions["package"].unique())

    # window-internal first-attested (what Cox currently uses as the "event")
    vv = versions.copy()
    vv["att"] = vv["has_attestation"].astype(str).str.lower() == "true"
    in_window_first = (vv[vv["att"]].assign(pub=pd.to_datetime(vv.loc[vv["att"], "published_at"], utc=True))
                       .groupby("package")["pub"].min().to_dict())

    prewindow = {p: dt for p, dt in earliest.items() if dt < WINDOW_START}

    pw_active = prewindow.keys() & active_pkgs
    pw_cox = prewindow.keys() & cox_pkgs

    # year distribution of earliest-ever attestation among prevalent adopters
    yr = Counter(dt.year for dt in prewindow.values())

    # among the Cox subset, how many events (in-window first-attested) are actually prevalent?
    cox_events = {p for p in cox_pkgs if p in in_window_first}
    cox_event_prevalent = cox_events & prewindow.keys()

    # per-package audit rows (prevalent adopters in cox subset)
    rows = []
    for p in sorted(pw_cox):
        rows.append({
            "package": p,
            "earliest_attested_ever": earliest[p].date().isoformat(),
            "window_first_attested": (in_window_first[p].date().isoformat()
                                      if p in in_window_first else ""),
            "in_window_active": p in active_pkgs,
            "in_cox_subset": True,
            "is_cox_event": p in cox_events,
        })
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["package", "earliest_attested_ever",
                                          "window_first_attested", "in_window_active",
                                          "in_cox_subset", "is_cox_event"])
        w.writeheader()
        w.writerows(rows)

    # ---- console summary ----
    print("\n=== PRE-WINDOW ATTESTATION AUDIT ===", flush=True)
    print(f"packages ever attested (any time):           {len(earliest):,}", flush=True)
    print(f"  ... attested BEFORE 2025-08-01 (prevalent): {len(prewindow):,}", flush=True)
    print(f"  ... prevalent AND window-active:            {len(pw_active):,}", flush=True)
    print(f"  ... prevalent AND in Cox subset:            {len(pw_cox):,}", flush=True)
    print(f"Cox subset size (approx scripts/29):          {len(cox_pkgs):,}", flush=True)
    print(f"Cox 'events' (in-window first-attested):      {len(cox_events):,}", flush=True)
    print(f"  ... of which prevalent (left-truncated):    {len(cox_event_prevalent):,} "
          f"({100*len(cox_event_prevalent)/max(1,len(cox_events)):.1f}% of events)", flush=True)
    print(f"earliest-ever attestation year (prevalent):   {dict(sorted(yr.items()))}", flush=True)

    pct_events = 100 * len(cox_event_prevalent) / max(1, len(cox_events))
    if len(pw_cox) == 0:
        verdict = "NEGLIGIBLE — no prevalent adopters in the Cox subset; clarify event wording only."
    elif pct_events < 5:
        verdict = (f"MINOR — {len(cox_event_prevalent)} prevalent events "
                   f"({pct_events:.1f}% of events); clarify wording + report an exclude-prevalent "
                   f"sensitivity confirming HR stability.")
    else:
        verdict = (f"MATERIAL — {len(cox_event_prevalent)} prevalent events "
                   f"({pct_events:.1f}% of events); run an exclude-prevalent (incident-only) "
                   f"sensitivity AND/OR a prevalent-adopter indicator; report HR deltas.")
    print(f"\nVERDICT: {verdict}", flush=True)

    # ---- markdown ----
    md = [
        "# Rebuttal B1 — Pre-window attestation audit (Cox left-truncation)",
        "",
        f"**Date**: 2026-05-30  ",
        "**Purpose**: quantify how many packages were already attesting BEFORE the "
        "2025-08-01 Cox origin, to decide whether window-anchored survival suffers "
        "material left-truncation.  ",
        "**Method**: count only; full version history in `data/packuments_raw/`.",
        "",
        "## Counts",
        "",
        "| Quantity | n |",
        "|---|---:|",
        f"| Packages ever attested (any date) | {len(earliest):,} |",
        f"| ... attested **before 2025-08-01** (prevalent adopters) | {len(prewindow):,} |",
        f"| ... prevalent **and** window-active | {len(pw_active):,} |",
        f"| ... prevalent **and** in Cox subset | {len(pw_cox):,} |",
        f"| Cox subset size (approx. scripts/29) | {len(cox_pkgs):,} |",
        f"| Cox events (in-window first-attested) | {len(cox_events):,} |",
        f"| ... of which prevalent (left-truncated) | {len(cox_event_prevalent):,} "
        f"({pct_events:.1f}% of events) |",
        "",
        f"Earliest-ever attestation year among prevalent adopters: "
        f"`{dict(sorted(yr.items()))}`",
        "",
        "## Verdict",
        "",
        f"{verdict}",
        "",
        "## Downstream rebuttal options",
        "",
        "1. **Clarify wording** — define the event as *first in-window attested "
        "release* (interval-specific), not lifetime-first; state the prevalent count.",
        "2. **Exclude-prevalent sensitivity** — drop prevalent adopters (incident-only "
        "cohort) and report HR deltas vs the main Cox.",
        "3. **Prevalent-adopter indicator** — add a binary covariate flagging "
        "pre-window adopters and report whether the key HRs (log_age, tier_top, "
        "log_maintainers) move.",
        "",
        f"Per-package list of prevalent adopters in the Cox subset: `{OUT_CSV.name}` "
        f"({len(rows)} rows).",
    ]
    OUT_MD.write_text("\n".join(md) + "\n")
    print(f"\n✓ Saved {OUT_CSV} ({len(rows)} rows)", flush=True)
    print(f"✓ Saved {OUT_MD}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
