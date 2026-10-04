"""
Task 5 Step 5.1 — ITS segmented regression on weekly attestation rate.

Model (per tier) — canonical Batch-8 spec, see scripts/its_spec.py:
  rate_t = β0 + β1·(t−c) + β2·post_t + β3·(t−c)·post_t + ε_t
where:
  t       = TRUE CALENDAR week position (weeks since series origin; the
            excluded straddling week leaves a real gap — no re-indexing)
  c       = calendar position of the 2025-12-09 deadline on the same axis
  post_t  = 1 iff week_start ≥ 2025-12-09; the week straddling the deadline
            (2025-12-08) is excluded from the fit
  β1      = pre-deadline slope
  β2      = level change AT the deadline date (post minus pre fitted line at c)
  β3      = slope change after deadline
  β1+β3   = post-deadline slope

SE: Newey-West HAC with lag = ⌈n^(1/4)⌉.

MANDATORY diagnostics (per methodologist AGENTS.md):
  - Durbin-Watson statistic (autocorrelation)
  - Ljung-Box on residuals (white noise residuals at lags 4, 8, 12)
  - report effect sizes (β2, β3) with 95% CI, NOT only p-values

Tiers under analysis (deps.dev primary):
  all_excl_missing, top, mid, tail

Output:
  data/its_results.csv          (per tier: β0, β1, β2, β3, HAC-SE, 95% CI, DW, LB-p)
  reports/task_5_its.md          (Markdown summary)
  paper/figures/its_segmented_fit.png  (4-panel small-multiples: tier × fit)

Run: conda run --no-capture-output -n stage2c python scripts/27_its_segmented_regression.py
"""
from __future__ import annotations

import csv
import sys
from datetime import date
_date_max = date.max
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.lines import Line2D
from statsmodels.stats.diagnostic import acorr_ljungbox

import figstyle
import its_spec

figstyle.apply()

DATA_DIR = Path(__file__).parent.parent / "data"
REPORTS_DIR = Path(__file__).parent.parent / "reports"
PAPER_DIR = Path(__file__).parent.parent / "paper" / "figures"
PAPER_DIR.mkdir(parents=True, exist_ok=True)

INPUT_CSV = DATA_DIR / "rq1_weekly_attestation_rate_depsdev.csv"
OUT_CSV = DATA_DIR / "its_results.csv"
OUT_MD = REPORTS_DIR / "task_5_its.md"
OUT_PNG = PAPER_DIR / "its_segmented_fit.png"

DEADLINE = date(2025, 12, 9)  # actual cutoff; straddling week excluded from fits


def run_its(df: pd.DataFrame, tier: str) -> dict:
    """Canonical Batch-8 fit: calendar-time axis, cutoff-date-centered
    (scripts/its_spec.py). The Batch-7 interim spec re-indexed the axis after
    dropping the straddling week, compressing calendar time by one step at
    the seam (ANOMALY_REPORT.md, Anomaly 9)."""
    sub = df[df["tier"] == tier].copy()
    r = its_spec.fit_its(sub, DEADLINE, period_col="week_start",
                         granularity="W", min_pre=2, min_post=2)
    if r is None:
        return None
    lb = r["lb_pvals"]
    return {
        "tier": tier,
        "n_weeks": r["n_periods"],
        "hac_maxlags": r["hac_maxlags"],
        "β0_const": r["beta0"],
        "β1_pre_slope": r["beta1_pre_slope"],
        "β2_level_change": r["beta2_level_change_pp"],
        "β3_slope_change": r["beta3_slope_change"],
        "β1+β3_post_slope": r["post_slope"],
        "se_β2": r["se_beta2"],
        "se_β3": r["se_beta3"],
        "ci95_β2_low": r["ci95_beta2_low"], "ci95_β2_high": r["ci95_beta2_high"],
        "ci95_β3_low": r["ci95_beta3_low"], "ci95_β3_high": r["ci95_beta3_high"],
        "pval_β2": r["pval_beta2"],
        "pval_β3": r["pval_beta3"],
        "DW": r["DW"],
        "LB_p_lag1": lb[0] if len(lb) > 0 else None,
        "LB_p_lag2": lb[1] if len(lb) > 1 else None,
        "LB_p_lag3": lb[2] if len(lb) > 2 else None,
        "fit_obj": r["fit_obj"],
        "sub_df": r["sub_df"],
        "y": r["sub_df"]["rate"].astype(float).values * 100,
        "cutoff_c": r["cutoff_c"],
    }


def diagnose_str(r: dict) -> str:
    dw = r["DW"]
    dw_verdict = "OK (≈2)" if 1.5 <= dw <= 2.5 else f"⚠ {'pos.' if dw < 1.5 else 'neg.'} autocorrelation"
    lb_min = min(p for p in [r["LB_p_lag1"], r["LB_p_lag2"], r["LB_p_lag3"]] if p is not None)
    lb_verdict = "white-noise OK" if lb_min > 0.05 else f"⚠ residual autocorr (min p={lb_min:.3f})"
    return f"DW={dw:.2f} {dw_verdict}; LB min-p={lb_min:.3f} ({lb_verdict})"


def main() -> int:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} weekly rate rows", flush=True)

    tiers = ["all_excl_missing", "top", "mid", "tail"]
    results = [run_its(df, t) for t in tiers]

    # Write CSV
    fields = ["tier", "n_weeks", "hac_maxlags",
              "β0_const", "β1_pre_slope", "β2_level_change", "β3_slope_change",
              "β1+β3_post_slope", "se_β2", "se_β3",
              "ci95_β2_low", "ci95_β2_high", "ci95_β3_low", "ci95_β3_high",
              "pval_β2", "pval_β3",
              "DW", "LB_p_lag1", "LB_p_lag2", "LB_p_lag3"]
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({k: r[k] for k in fields})
    print(f"✓ Saved {OUT_CSV}", flush=True)

    # Markdown summary
    lines = [
        "# Task 5 Step 5.1 — ITS segmented regression",
        "",
        "**Date**: 2026-05-18  ",
        "**Source**: `data/rq1_weekly_attestation_rate_depsdev.csv` (deps.dev-tier PRIMARY)",
        "",
        "## Model",
        "",
        "`rate_t = β0 + β1·(t−c) + β2·post_t + β3·(t−c)·post_t + ε_t`  (canonical Batch-8 spec, scripts/its_spec.py)",
        "",
        "- `t` = calendar week position since 2025-08-01 (gaps preserved; no re-indexing)",
        f"- `c` = calendar position of the {DEADLINE.isoformat()} deadline; `post_t = 1` iff week_start ≥ deadline",
        "- the week straddling the deadline (2025-12-08) is excluded from the fit",
        "- `β2` = level change AT the deadline date; `β3` = slope change",
        "- HAC SE with `maxlags = ⌈n^(1/4)⌉`",
        "- Outcome `rate` × 100 = percentage points",
        "",
        "## Results table",
        "",
        "| Tier | n | β2 (level change, pp) | 95% CI | β3 (slope change, pp/week) | 95% CI | DW | LB min-p |",
        "|---|---:|---:|---|---:|---|---:|---:|",
    ]
    for r in results:
        b2 = r["β2_level_change"]
        b2_lo, b2_hi = r["ci95_β2_low"], r["ci95_β2_high"]
        b3 = r["β3_slope_change"]
        b3_lo, b3_hi = r["ci95_β3_low"], r["ci95_β3_high"]
        lb_min = min(p for p in [r["LB_p_lag1"], r["LB_p_lag2"], r["LB_p_lag3"]] if p is not None)
        sig_b2 = "***" if r["pval_β2"] < 0.01 else ("**" if r["pval_β2"] < 0.05 else ("*" if r["pval_β2"] < 0.10 else ""))
        sig_b3 = "***" if r["pval_β3"] < 0.01 else ("**" if r["pval_β3"] < 0.05 else ("*" if r["pval_β3"] < 0.10 else ""))
        lines.append(
            f"| {r['tier']} | {r['n_weeks']} | {b2:+.2f}{sig_b2} | [{b2_lo:+.2f}, {b2_hi:+.2f}] | "
            f"{b3:+.3f}{sig_b3} | [{b3_lo:+.3f}, {b3_hi:+.3f}] | {r['DW']:.2f} | {lb_min:.3f} |"
        )
    lines += ["",
              "Stars: `* p<0.10`, `** p<0.05`, `*** p<0.01`. Effect sizes are in percentage points (level) "
              "or pp/week (slope). DW close to 2 → no first-order autocorrelation. LB min-p > 0.05 → "
              "residuals consistent with white noise at lags 4/8/12.",
              "",
              "## Diagnostic verdicts per tier",
              ""]
    for r in results:
        lines.append(f"- **{r['tier']}**: {diagnose_str(r)}")

    lines += ["",
              "## Headline interpretation (deps.dev-tier PRIMARY)",
              "",
              "Per Task 4 Round 3 findings, mid-tier has the highest baseline attestation rate (~50%) while top-tier",
              "is lowest (~20%). The ITS estimates above test whether the Dec-9 deadline produced a sharp level shift",
              "(β2) and/or a slope change (β3) on top of these baselines.",
              "",
              "Causal interpretation is **bounded to association** unless diagnostics pass (DW≈2 + LB min-p>0.05).",
              "Where diagnostics fail, the segmented-regression coefficients reflect descriptive trend-change only.",
              "",
              "## Construct caveats (carried from Tasks 2/3/4)",
              "",
              "- Class C = possible external CI path (not verified)",
              "- Class E = no visible publish workflow in inspected sources (not asserted manual publish)",
              "- Tier source is deps.dev-snapshot 2026-05-05 (5-day delta from cutoff)",
              "- 62 packages excluded from deps.dev-tier (recent/scoped variants); see Task 3 strata-shift report",
              ]
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"✓ Saved {OUT_MD}", flush=True)

    # Figure: 4-panel small-multiples (styling via figstyle; titles live in caption)
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 4.6), sharex=True, sharey=True)
    for ax, r in zip(axes.flat, results):
        sub = r["sub_df"]
        y = r["y"]
        tier = r["tier"]
        # Per-tier hue (aggregate 'all_excl_missing' -> neutral secondary accent).
        tcolor = figstyle.TIER.get(tier, figstyle.ACCENT_2)
        ax.plot(sub["t"], y, "o", alpha=0.40, markersize=3.5, color=tcolor)
        # Fitted lines pre/post
        t_arr = sub["t"].values
        post = sub["post"].values
        # Use the model's own fitted values — no manual design reconstruction,
        # so the plot stays correct under the cutoff-centered parameterization.
        y_hat = np.asarray(r["fit_obj"].fittedvalues)
        # Pre line (dashed) / post line (solid), same tier hue.
        pre_mask = post == 0
        ax.plot(t_arr[pre_mask], y_hat[pre_mask], linestyle="--", linewidth=2, color=tcolor)
        post_mask = post == 1
        ax.plot(t_arr[post_mask], y_hat[post_mask], linestyle="-", linewidth=2, color=tcolor)
        # Deadline marker at the cutoff's true calendar position (not the
        # first post week), matching the date-centered estimand.
        ax.axvline(r["cutoff_c"], linestyle=":", color=figstyle.REF, alpha=0.85, linewidth=1.3)
        # Tier identifier: in-axes annotation replaces set_title (no in-figure titles).
        ax.text(0.03, 0.96, f"{tier}\n(n={r['n_weeks']}, DW={r['DW']:.2f})",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.28", facecolor="white",
                          edgecolor="none", alpha=0.72))
        ax.set_xlabel("Calendar weeks since 2025-08-01")
        ax.set_ylabel("Attestation rate (%)")

    # One shared, light legend (linestyle/marker key in neutral grey) below the
    # grid so it never occludes the fitted lines in any panel.
    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="none", color=figstyle.REF,
               alpha=0.55, markersize=6, label="observed weekly rate"),
        Line2D([0], [0], linestyle="--", color=figstyle.REF, linewidth=2,
               label="pre-deadline fit"),
        Line2D([0], [0], linestyle="-", color=figstyle.REF, linewidth=2,
               label="post-deadline fit"),
        Line2D([0], [0], linestyle=":", color=figstyle.REF, linewidth=1.4,
               label="Dec-9 deadline"),
    ]
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 0.0),
               ncol=4, frameon=True, framealpha=0.9, edgecolor="0.8")
    fig.savefig(OUT_PNG)
    plt.close(fig)
    print(f"✓ Saved {OUT_PNG}", flush=True)

    # Console headline
    print(f"\n=== ITS headline (β2 level change ± 95% CI) ===", flush=True)
    for r in results:
        print(f"  {r['tier']:<18s}: β2={r['β2_level_change']:+.2f} pp "
              f"[{r['ci95_β2_low']:+.2f}, {r['ci95_β2_high']:+.2f}]  "
              f"β3={r['β3_slope_change']:+.3f} pp/wk  {diagnose_str(r)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
