"""
Task 5 Step 5.2 — DiD: top tier (treatment) vs tail tier (control).

Setup (deps.dev-tier primary):
  Treatment group: top tier (DPC > 500)        — most-depended-on packages
  Control group  : tail tier (DPC ≤ 10)        — least-depended-on packages
  Pre period     : 2025-08-01 .. 2025-12-07 (week_start < 2025-12-08)
  Post period    : 2025-12-08 .. 2026-05-10
  Outcome        : weekly attestation rate (%)

Model:
  rate_it = α + β1·post_t + β2·top_i + β3·(post_t × top_i) + ε_it
  β3 = DiD estimate

MANDATORY diagnostics (per methodologist AGENTS.md):
  - parallel-trends test: F-test for joint significance of pre-period
    week-level coefficients in event-study form (β_pre,k = 0 for all k < 0)
  - visual diagnostic: event-study plot with pre-trend coefficients shown

Cluster-robust SE at tier level (only 2 clusters → use HAC + bootstrap CI optional);
fall back to OLS heteroskedasticity-robust HC1 + acknowledge low-cluster cost.

Output:
  data/did_results.csv
  reports/task_5_did.md
  paper/figures/did_event_study.png

Run: conda run --no-capture-output -n stage2c python scripts/28_did_top_vs_tail.py
"""
from __future__ import annotations

import csv
import sys
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt

import figstyle

figstyle.apply()

import numpy as np
import pandas as pd
import statsmodels.api as sm

DATA_DIR = Path(__file__).parent.parent / "data"
REPORTS_DIR = Path(__file__).parent.parent / "reports"
PAPER_DIR = Path(__file__).parent.parent / "paper" / "figures"
PAPER_DIR.mkdir(parents=True, exist_ok=True)

INPUT_CSV = DATA_DIR / "rq1_weekly_attestation_rate_depsdev.csv"
OUT_CSV = DATA_DIR / "did_results.csv"
OUT_MD = REPORTS_DIR / "task_5_did.md"
OUT_PNG = PAPER_DIR / "did_event_study.png"

DEADLINE = date(2025, 12, 8)


def main() -> int:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT_CSV)
    df = df[df["tier"].isin(["top", "tail"])].copy()
    df["week_start_dt"] = pd.to_datetime(df["week_start"])
    df["post"] = (df["week_start_dt"].dt.date >= DEADLINE).astype(int)
    df["top"] = (df["tier"] == "top").astype(int)
    df["rate_pp"] = df["rate"].astype(float) * 100
    df["t"] = (df["week_start_dt"] - df["week_start_dt"].min()).dt.days // 7
    # Center t at deadline week so β2_event reads as "weeks since deadline"
    deadline_t = df.loc[df["post"] == 1, "t"].min()
    df["event_week"] = df["t"] - deadline_t
    df = df.sort_values(["tier", "t"]).reset_index(drop=True)

    print(f"Tiers in data: {df['tier'].unique().tolist()}", flush=True)
    print(f"Pre weeks: {df.loc[df['post']==0, 'week_start'].nunique()}; "
          f"Post weeks: {df.loc[df['post']==1, 'week_start'].nunique()}", flush=True)

    # ---- DiD model: rate = α + β1·post + β2·top + β3·(post*top) ----
    df["post_x_top"] = df["post"] * df["top"]
    X = df[["post", "top", "post_x_top"]].copy()
    X = sm.add_constant(X)
    y = df["rate_pp"].values
    fit = sm.OLS(y, X.values).fit(cov_type="HC1")
    coefs = dict(zip(["const", "post", "top", "post_x_top"], fit.params))
    se = dict(zip(["const", "post", "top", "post_x_top"], fit.bse))
    ci = fit.conf_int(alpha=0.05)
    pvals = dict(zip(["const", "post", "top", "post_x_top"], fit.pvalues))

    did_est = coefs["post_x_top"]
    did_se = se["post_x_top"]
    did_ci_lo, did_ci_hi = ci[3][0], ci[3][1]
    did_p = pvals["post_x_top"]

    print(f"\nDiD β3 (interaction) = {did_est:+.2f} pp  "
          f"[{did_ci_lo:+.2f}, {did_ci_hi:+.2f}]  p={did_p:.4f}", flush=True)

    # ---- Pre-period parallel-trends test (parsimonious; avoids rank deficiency) ----
    # In PRE period only, fit: rate = γ0 + γ1·t + γ2·top + γ3·(t × top) + ε
    # PT test: γ3 = 0 (no differential pre-period slope).
    df_pre = df[df["post"] == 0].copy()
    df_pre["t_x_top"] = df_pre["t"] * df_pre["top"]
    X_pre = df_pre[["t", "top", "t_x_top"]].copy()
    X_pre = sm.add_constant(X_pre)
    fit_pre = sm.OLS(df_pre["rate_pp"].values, X_pre.values).fit(cov_type="HC1")
    pt_coef = fit_pre.params[3]   # γ3
    pt_se = fit_pre.bse[3]
    pt_p = float(fit_pre.pvalues[3])
    pt_t = float(fit_pre.tvalues[3])
    pt_f = pt_t ** 2  # equivalent to F-test on single restriction
    pt_ci_lo, pt_ci_hi = fit_pre.conf_int(alpha=0.05)[3][0], fit_pre.conf_int(alpha=0.05)[3][1]

    # Event-study coefficients for visualization (per actual event_week; OLS with HC1 — no F-test)
    df_es = df.copy()
    event_weeks_sorted = sorted(df_es["event_week"].unique())
    ref_week = -1
    es_coefs = []
    for ew in event_weeks_sorted:
        if ew == ref_week:
            es_coefs.append({"event_week": ew, "beta": 0.0, "se": 0.0,
                             "ci_lo": 0.0, "ci_hi": 0.0, "p": np.nan})
            continue
        # gap-from-ref: rate(top, ew) − rate(tail, ew) − [rate(top, ref) − rate(tail, ref)]
        gap_ew = (df_es.loc[(df_es["event_week"] == ew) & (df_es["top"] == 1), "rate_pp"].mean()
                  - df_es.loc[(df_es["event_week"] == ew) & (df_es["top"] == 0), "rate_pp"].mean())
        gap_ref = (df_es.loc[(df_es["event_week"] == ref_week) & (df_es["top"] == 1), "rate_pp"].mean()
                   - df_es.loc[(df_es["event_week"] == ref_week) & (df_es["top"] == 0), "rate_pp"].mean())
        if np.isnan(gap_ew) or np.isnan(gap_ref):
            continue
        es_coefs.append({"event_week": ew, "beta": gap_ew - gap_ref,
                         "se": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "p": np.nan})
    es_coefs_df = pd.DataFrame(es_coefs).sort_values("event_week").reset_index(drop=True)

    pt_verdict = "PASS (p > 0.05)" if pt_p is not None and pt_p > 0.05 else (
        f"⚠ FAIL (p={pt_p:.3f})" if pt_p is not None else "n/a"
    )
    print(f"Parallel-trends F-test: F={pt_f}, p={pt_p:.4f}  {pt_verdict}", flush=True)

    # ---- Save results CSV ----
    with OUT_CSV.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["did_beta3", did_est])
        w.writerow(["did_se_HC1", did_se])
        w.writerow(["did_ci95_low", did_ci_lo])
        w.writerow(["did_ci95_high", did_ci_hi])
        w.writerow(["did_pvalue", did_p])
        w.writerow(["beta1_post", coefs["post"]])
        w.writerow(["beta2_top", coefs["top"]])
        w.writerow(["parallel_trends_F", pt_f if pt_f is not None else ""])
        w.writerow(["parallel_trends_p", pt_p if pt_p is not None else ""])
        w.writerow(["n_pre_weeks", int((df["post"] == 0).sum() / 2)])
        w.writerow(["n_post_weeks", int((df["post"] == 1).sum() / 2)])
    print(f"✓ Saved {OUT_CSV}", flush=True)

    # ---- Markdown summary ----
    lines = [
        "# Task 5 Step 5.2 — DiD top tier vs tail tier",
        "",
        f"**Date**: 2026-05-18  ",
        f"**Source**: `data/rq1_weekly_attestation_rate_depsdev.csv` (deps.dev-tier PRIMARY)  ",
        f"**Treatment**: top tier (DPC > 500)  **Control**: tail tier (DPC ≤ 10); mid excluded.  ",
        f"**Pre**: 2025-08-01 .. {DEADLINE.isoformat()} (exclusive)  **Post**: {DEADLINE.isoformat()} .. 2026-05-10",
        "",
        "## Model",
        "",
        "`rate_it = α + β1·post_t + β2·top_i + β3·(post_t × top_i) + ε_it`",
        "",
        "β3 = DiD ATT estimate. HC1 heteroskedasticity-robust SE.",
        "",
        "## Main result",
        "",
        "| Term | β̂ | SE (HC1) | 95% CI | p |",
        "|---|---:|---:|---|---:|",
        f"| post (deadline effect on tail) | {coefs['post']:+.2f} | {se['post']:.2f} | [{ci[1][0]:+.2f}, {ci[1][1]:+.2f}] | {pvals['post']:.4f} |",
        f"| top (level diff baseline) | {coefs['top']:+.2f} | {se['top']:.2f} | [{ci[2][0]:+.2f}, {ci[2][1]:+.2f}] | {pvals['top']:.4f} |",
        f"| **post × top (DiD β3)** | **{did_est:+.2f}** | **{did_se:.2f}** | **[{did_ci_lo:+.2f}, {did_ci_hi:+.2f}]** | **{did_p:.4f}** |",
        "",
        "Effect units: percentage points (rate × 100).",
        "",
        "## Parallel-trends diagnostic (MANDATORY)",
        "",
        "Parsimonious specification (avoids rank-deficiency with only 2 clusters):",
        "`rate_it = γ0 + γ1·t + γ2·top + γ3·(t × top) + ε_it`  on PRE period only.",
        "PT test: γ3 = 0 (no differential pre-period slope between top and tail).",
        "",
        f"- γ3 = {pt_coef:+.4f} pp/week  [95% CI: {pt_ci_lo:+.4f}, {pt_ci_hi:+.4f}]  p = {pt_p:.4f}",
        f"- F-equivalent = t² = {pt_f:.3f}",
        f"- Verdict: **{pt_verdict}**",
        "",
        "## Cluster-count caveat",
        "",
        "Only 2 clusters (top + tail) → wild-cluster bootstrap inference is preferred but not feasible at 2 clusters. We report HC1 heteroskedasticity-robust SE; this likely understates uncertainty. The methodologist should treat the DiD estimate as a descriptive contrast pending an alternative inference scheme.",
        "",
        "## Construct caveats (carried)",
        "",
        "- Tier source = deps.dev snapshot 2026-05-05",
        "- Causal interpretation bounded to association unless PT test passes AND inference is honest",
        "- C/E construct caveats from Task 2 apply when class-level interpretation surfaces",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"✓ Saved {OUT_MD}", flush=True)

    # ---- Event-study figure (no CIs because per-week sample is 2; show point estimates only) ----
    fig, ax = plt.subplots(figsize=(5.6, 2.9))
    ax.plot(es_coefs_df["event_week"], es_coefs_df["beta"],
            "o-", color=figstyle.ACCENT, markersize=4, linewidth=1.2,
            label="Event-study gap (top − tail, rel. to ew=-1)")
    ax.axhline(0, linestyle=":", color=figstyle.REF, linewidth=0.8, alpha=0.6)
    ax.axvline(0, linestyle="--", color=figstyle.REF, alpha=0.5, linewidth=1, label="Dec-9 deadline")
    ax.set_xlabel("Event week (0 = week of 2025-12-08; ref. = -1)")
    ax.set_ylabel("Gap (pp; top minus tail, relative to ref. week)")
    ax.grid(True, alpha=0.3, linestyle="--")
    figstyle.legend_outside(ax)
    fig.tight_layout()
    fig.savefig(OUT_PNG, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved {OUT_PNG}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
