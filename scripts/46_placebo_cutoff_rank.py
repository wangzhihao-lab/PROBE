"""Pre-submission B2 — All-admissible-cutoff placebo rank for the top-tier ITS.

Batch-8 canonical convention (2026-07-11, Anomaly 9 repair): every candidate
cutoff is placed at the DEADLINE'S OWN WEEKDAY inside each week of the
41-week series (Dec-9 2025 is a Tuesday, so candidates are the Tuesdays
week_start + 1 day). Each candidate is fitted with the shared calendar-time,
cutoff-date-centered ITS of scripts/its_spec.py, which excludes the
candidate's own straddling week — so the actual Dec-9 partition and every
placebo partition are treated EXACTLY symmetrically (each loses one week,
each is centered at its own date). The earlier convention enumerated
cutoffs at week starts (Mondays), which dropped no week and put the 12-08
straddling week into pre — a different partition from the canonical Dec-9
fit; ranks from that convention were not the rank of the published beta2.

Rank statistics are reported for BOTH |beta2| (level change at the cutoff
date) and |beta3| (slope change), with randomization-inference-style
p-values = (# candidates with |coef| >= actual) / N.

Admissibility is pre-declared, not post-hoc: a candidate is admissible iff
its fit retains >= 8 clean pre weeks and >= 5 clean post weeks (the extremes
of the PI-approved B4b grid).

A built-in assertion re-derives the (Batch-8 regenerated) B4b top-tier grid
values with the shared helper and fails loudly on any divergence > 1e-6 pp.

Inputs:
  data/versions.csv                          (frozen; cutoff 2026-05-10)
  data/depsdev_dependents_2026_05_05.csv     (primary tier frame)
  data/baselines/task_7_b4_its_placebo_grid.csv  (consistency reference)

Outputs:
  data/placebo_cutoff_rank.csv
  reports/presub_placebo_rank.md

Carried-forward caveats: data cutoff 2026-05-10 (untouched); anti-causal
firewall applies — the rank statistic quantifies non-specificity of the
Dec-9 partition and cannot promote any estimate to causal.

Run: conda run --no-capture-output -n stage2c python scripts/46_placebo_cutoff_rank.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from datetime import date as _dt_date
_date_max = _dt_date.max
from typing import List, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox

import its_spec

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"

VERSIONS_CSV = DATA_DIR / "versions.csv"
DEPSDEV_CSV = DATA_DIR / "depsdev_dependents_2026_05_05.csv"
GRID_CSV = DATA_DIR / "baselines" / "task_7_b4_its_placebo_grid.csv"

OUT_CSV = DATA_DIR / "placebo_cutoff_rank.csv"
OUT_MD = REPORTS_DIR / "presub_placebo_rank.md"

# Constants identical to scripts/37_baseline_time_models.py.
DEADLINE = pd.Timestamp("2025-12-09", tz="UTC")
ORIGIN = pd.Timestamp("2025-08-01", tz="UTC")
CENSOR = pd.Timestamp("2026-05-10 23:59:59", tz="UTC")

MIN_PRE_WEEKS = 8   # matches B4b grid extreme (2025-09-21 cutoff)
MIN_POST_WEEKS = 5  # matches B4b grid extreme (2026-04-05 cutoff)

TIERS = ["top", "mid", "tail"]


def dd_tier(n) -> str:
    if pd.isna(n):
        return "missing"
    n = int(n)
    return "top" if n > 500 else "mid" if n > 10 else "tail"


def load_versions_with_tier() -> pd.DataFrame:
    """Identical loading path to scripts/37 (Task 7 B4)."""
    versions = pd.read_csv(VERSIONS_CSV)
    versions["published_at"] = pd.to_datetime(versions["published_at"], utc=True, errors="coerce")
    versions = versions.dropna(subset=["published_at"])
    versions = versions[(versions["published_at"] >= ORIGIN) & (versions["published_at"] <= CENSOR)]
    versions["has_attestation_int"] = (
        versions["has_attestation"].astype(str).str.lower() == "true"
    ).astype(int)
    depsdev = pd.read_csv(DEPSDEV_CSV)
    dd_map = dict(zip(depsdev["name"], depsdev["dependents_count"]))
    versions["dd_tier"] = versions["package"].map(dd_map).apply(dd_tier)
    return versions


def aggregate_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Weekly rate by tier; identical to scripts/37 aggregate_period('W')."""
    df = df.copy()
    df["period_start"] = (df["published_at"] -
                          pd.to_timedelta(df["published_at"].dt.weekday, unit="D")).dt.normalize()
    agg = (df.groupby(["dd_tier", "period_start"])
             .agg(n_versions=("has_attestation_int", "size"),
                  n_attested=("has_attestation_int", "sum"))
             .reset_index())
    agg["rate"] = agg["n_attested"] / agg["n_versions"]
    return agg.rename(columns={"dd_tier": "tier"})


def run_its(weekly: pd.DataFrame, tier: str, cutoff: pd.Timestamp) -> Optional[dict]:
    """Canonical Batch-8 fit (scripts/its_spec.py): calendar-time axis,
    cutoff-DATE-centered, the cutoff's straddling week excluded."""
    sub = weekly[weekly["tier"] == tier].copy()
    if len(sub) < 8:
        return None
    r = its_spec.fit_its(sub, cutoff, period_col="period_start",
                         granularity="W", min_pre=2, min_post=2)
    if r is None:
        return None
    return {
        "tier": tier,
        "n_periods": r["n_periods"],
        "n_pre_weeks": r["n_pre"],
        "n_post_weeks": r["n_post"],
        "beta2_level_change_pp": r["beta2_level_change_pp"],
        "ci95_beta2_low": r["ci95_beta2_low"],
        "ci95_beta2_high": r["ci95_beta2_high"],
        "pval_beta2": r["pval_beta2"],
        "beta3_slope_change": r["beta3_slope_change"],
        "ci95_beta3_low": r["ci95_beta3_low"],
        "ci95_beta3_high": r["ci95_beta3_high"],
        "pval_beta3": r["pval_beta3"],
        "DW": r["DW"],
        "LB_min_pval": r["LB_min_pval"],
    }


def check_grid_consistency(weekly: pd.DataFrame) -> None:
    """Re-derive the approved B4b top-tier grid; abort on divergence."""
    grid = pd.read_csv(GRID_CSV)
    top = grid[grid["tier"] == "top"]
    for _, ref in top.iterrows():
        cutoff = pd.Timestamp(ref["cutoff"], tz="UTC")
        r = run_its(weekly, "top", cutoff)
        if r is None or abs(r["beta2_level_change_pp"] - ref["beta2_level_change_pp"]) > 1e-6:
            raise AssertionError(
                f"B4b grid replication mismatch at {ref['cutoff']}: "
                f"got {r and r['beta2_level_change_pp']}, expected {ref['beta2_level_change_pp']}"
            )
    logger.info("B4b grid consistency check passed (%d top-tier cutoffs reproduce).", len(top))


def main() -> int:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    versions = load_versions_with_tier()
    weekly = aggregate_weekly(versions)
    check_grid_consistency(weekly)

    week_starts = sorted(weekly[weekly["tier"] == "top"]["period_start"].unique())
    n_weeks = len(week_starts)
    logger.info("Series: %d weeks, %s .. %s", n_weeks,
                pd.Timestamp(week_starts[0]).date(), pd.Timestamp(week_starts[-1]).date())

    # Batch-8 symmetric candidates: one cutoff per week, at the deadline's own
    # weekday inside that week (Dec-9 is week_start + 1 day). Each candidate's
    # fit drops its own straddling week (week k), so retained pre = k weeks and
    # retained post = n_weeks - k - 1 weeks; admissibility applies to RETAINED
    # clean weeks.
    weekday_offset = (DEADLINE.date()
                      - min(d for d in (pd.Timestamp(w).date() for w in week_starts)
                            if d <= DEADLINE.date() and (DEADLINE.date() - d).days < 7)).days
    admissible_idx = [k for k in range(n_weeks)
                      if k >= MIN_PRE_WEEKS and (n_weeks - k - 1) >= MIN_POST_WEEKS]

    actual_k = next(k for k in range(n_weeks)
                    if pd.Timestamp(week_starts[k]).date()
                    <= DEADLINE.date()
                    < pd.Timestamp(week_starts[k]).date() + pd.Timedelta(days=7))
    if actual_k not in admissible_idx:
        raise AssertionError("actual Dec-9 partition fell outside the admissible set")

    rows: List[dict] = []
    for k in admissible_idx:
        cutoff = (pd.Timestamp(week_starts[k]) + pd.Timedelta(days=weekday_offset))
        cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff
        assert (k != actual_k) or (cutoff.date() == DEADLINE.date())
        for tier in TIERS:
            r = run_its(weekly, tier, cutoff)
            if r is None:
                continue
            r["cutoff_date"] = str(cutoff.date())
            r["partition_k"] = k
            r["is_actual_dec9_partition"] = (k == actual_k)
            rows.append(r)

    df = pd.DataFrame(rows)
    fields = ["tier", "cutoff_date", "partition_k", "is_actual_dec9_partition",
              "n_periods", "n_pre_weeks", "n_post_weeks", "beta2_level_change_pp",
              "ci95_beta2_low", "ci95_beta2_high", "pval_beta2",
              "beta3_slope_change", "ci95_beta3_low", "ci95_beta3_high",
              "pval_beta3", "DW", "LB_min_pval"]
    df[fields].to_csv(OUT_CSV, index=False)
    logger.info("Saved %s", OUT_CSV)

    # Rank statistics (top tier primary) for BOTH beta2 and beta3.
    summary_lines: List[str] = []
    for tier in TIERS:
        sub = df[df["tier"] == tier].copy()
        actual = sub[sub["is_actual_dec9_partition"]].iloc[0]
        n_adm = len(sub)
        for coef, label in [("beta2_level_change_pp", "beta2"),
                            ("beta3_slope_change", "beta3")]:
            absv = sub[coef].abs()
            a = abs(actual[coef])
            n_geq = int((absv >= a).sum())  # includes actual
            n_strict = int((absv > a).sum())
            p_rand = n_geq / n_adm
            line = (f"{tier}: actual {label}={actual[coef]:+.2f}, "
                    f"rank {n_geq}/{n_adm} by |{label}| ({n_strict} strictly larger), "
                    f"randomization-style p = {n_geq}/{n_adm} = {p_rand:.3f}")
            logger.info(line)
            summary_lines.append(line)

    lines = [
        "# Pre-submission B2 — All-admissible-cutoff placebo rank (top-tier ITS)",
        "",
        "Batch-8 canonical convention: one candidate cutoff per week, at the",
        "deadline's own weekday (Tuesday = week_start + 1 day); every candidate",
        "fitted with the shared calendar-time cutoff-date-centered ITS",
        "(scripts/its_spec.py), each dropping its own straddling week — the",
        "actual Dec-9 partition and every placebo are treated symmetrically.",
        "Pre-declared admissibility: >= 8 retained pre weeks and >= 5 retained",
        "post weeks (the extremes of the approved Task 7 B4b grid). Ranks are",
        "reported for BOTH |beta2| (level change at the cutoff date) and",
        "|beta3| (slope change).",
        "",
        f"Admissible candidates: {len(admissible_idx)} of {n_weeks} weeks;",
        f"actual = {DEADLINE.date()} (straddling week "
        f"{pd.Timestamp(week_starts[actual_k]).date()} excluded from its fit).",
        "",
        "## Rank summary",
        "",
    ]
    lines += [f"- {s}" for s in summary_lines]
    lines += [
        "",
        "Interpretation boundary: the rank statistics measure how unusual the",
        "Dec-9 partition's level and slope-change coefficients are within the",
        "window's own placebo distribution. They quantify (non-)specificity;",
        "they do not identify a causal effect at any cutoff, including the",
        "actual deadline.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")
    logger.info("Saved %s", OUT_MD)
    return 0


if __name__ == "__main__":
    sys.exit(main())
