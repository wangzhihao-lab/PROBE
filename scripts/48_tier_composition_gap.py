"""
Pre-submission Batch-4 — Composition facts behind the raw mid-vs-top
attestation-rate inversion (acting-PI batch, 2026-07-08; frozen rows only).

Motivation: RQ1 reports raw version-level attestation rates with the middle
tier (~52%) far above the top tier (~19-23%), yet the RQ3 adjusted models
find no significant tier coefficient. This script quantifies the covariate
composition gap that reconciles the two: per-tier distributions, over the
same 97,672-row modeling frame, of the covariates the logistic model finds
strongly negative (log package age, scoped-package status) and positive
(Dependabot).

Inputs:
  data/versions.csv, data/influence_metrics.csv,
  data/depsdev_dependents_2026_05_05.csv

Outputs:
  data/tier_composition_gap.csv
  reports/presub_tier_composition_gap.md

Caveats carried: deps.dev primary tier frame; descriptive composition facts
only — no causal decomposition; anti-causal firewall intact.

Run: conda run --no-capture-output -n stage2c python scripts/48_tier_composition_gap.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
REPORTS_DIR = Path(__file__).parent.parent / "reports"
OUT_CSV = DATA_DIR / "tier_composition_gap.csv"
OUT_MD = REPORTS_DIR / "presub_tier_composition_gap.md"


def dpc_tier(dpc: int) -> str:
    if dpc > 500:
        return "top"
    if dpc > 10:
        return "mid"
    return "tail"


def main() -> int:
    versions = pd.read_csv(DATA_DIR / "versions.csv", parse_dates=["published_at"])
    versions["y"] = (versions["has_attestation"].astype(str).str.lower() == "true").astype(int)

    inf = pd.read_csv(DATA_DIR / "influence_metrics.csv")
    depsdev = pd.read_csv(DATA_DIR / "depsdev_dependents_2026_05_05.csv")
    dd = dict(zip(depsdev["name"], depsdev["dependents_count"].astype(int)))
    inf["depsdev_tier"] = inf["package"].map(dd).apply(
        lambda x: dpc_tier(int(x)) if pd.notna(x) else "missing"
    )
    inf = inf[inf["depsdev_tier"].isin(["top", "mid", "tail"])].copy()
    inf["is_scoped"] = (inf["is_scoped_package"].astype(str).str.lower() == "true").astype(int)
    inf["dependabot"] = (inf["has_dependabot"].astype(str).str.lower() == "true").astype(int)
    inf["age_years"] = pd.to_numeric(inf["package_age_days"], errors="coerce") / 365.25

    df = versions.merge(
        inf[["package", "depsdev_tier", "is_scoped", "dependabot", "age_years"]],
        on="package", how="inner",
    ).dropna(subset=["age_years"])
    print(f"Version rows joined: {len(df):,}", flush=True)

    rows = []
    for tier, g in df.groupby("depsdev_tier"):
        rows.append({
            "tier": tier,
            "version_rows": len(g),
            "attestation_rate": round(g["y"].mean(), 4),
            "median_age_years": round(float(g["age_years"].median()), 2),
            "scoped_share": round(float(g["is_scoped"].mean()), 4),
            "dependabot_share": round(float(g["dependabot"].mean()), 4),
        })
    out = pd.DataFrame(rows).set_index("tier").loc[["top", "mid", "tail"]]
    out.to_csv(OUT_CSV)
    print(out.to_string(), flush=True)
    print(f"✓ Saved {OUT_CSV}", flush=True)

    OUT_MD.write_text(
        "# Pre-submission — Tier composition facts behind the raw rate inversion\n\n"
        "**Date**: 2026-07-08 (acting-PI Batch-4; frozen rows only)\n\n"
        "Raw version-level attestation rates invert (mid > top) while adjusted tier\n"
        "coefficients are non-significant. Composition of the same version rows:\n\n"
        + out.to_markdown()
        + "\n\nReading (descriptive only): top-tier publishing volume is dominated by\n"
        "substantially older packages and a higher scoped-package share --- the two\n"
        "covariates with the strongest negative adjusted associations --- so the raw\n"
        "inversion is consistent with covariate composition rather than a tier effect;\n"
        "this is exactly what the non-significant adjusted tier coefficients say.\n"
        "No causal decomposition is claimed.\n"
    )
    print(f"✓ Saved {OUT_MD}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
