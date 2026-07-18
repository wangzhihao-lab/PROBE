from __future__ import annotations

import csv
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox

DATA_DIR = Path(__file__).parent.parent / "data"
REPORTS_DIR = Path(__file__).parent.parent / "reports"
INPUT_CSV = DATA_DIR / "rq1_weekly_attestation_rate_depsdev.csv"
OUT_CSV = DATA_DIR / "its_placebo_results.csv"
OUT_MD = REPORTS_DIR / "task_5_its_placebo.md"

CUTOFFS = {
    "actual_2025-12-08": date(2025, 12, 8),
    "placebo_2025-10-06": date(2025, 10, 6),
    "placebo_2026-02-23": date(2026, 2, 23),
}


def run_its_top(cutoff_date: date) -> dict:
    df = pd.read_csv(INPUT_CSV)
    sub = df[df["tier"] == "top"].copy()
    sub = sub.sort_values("week_start").reset_index(drop=True)
    sub["t"] = np.arange(len(sub))
    sub["post"] = (pd.to_datetime(sub["week_start"]).dt.date >= cutoff_date).astype(int)
    sub["t_post"] = sub["t"] * sub["post"]
    y = sub["rate"].astype(float).values * 100
    X = sub[["t", "post", "t_post"]].copy()
    X = sm.add_constant(X)
    n = len(sub)
    lag = max(1, int(np.ceil(n ** 0.25)))
    fit = sm.OLS(y, X.values).fit(cov_type="HAC", cov_kwds={"maxlags": lag})
    ci = fit.conf_int(alpha=0.05)
    resid = fit.resid
    dw = sm.stats.stattools.durbin_watson(resid)
    lb = acorr_ljungbox(resid, lags=[min(4, n // 3), min(8, n // 2), min(12, n - 4)], return_df=True)
    return {
        "β2_level_change": fit.params[2],
        "ci_lo": ci[2][0], "ci_hi": ci[2][1],
        "β2_p": fit.pvalues[2],
        "β3_slope_change": fit.params[3],
        "β3_p": fit.pvalues[3],
        "DW": dw,
        "LB_min_p": lb["lb_pvalue"].min(),
        "n_pre": int((sub["post"] == 0).sum()),
        "n_post": int((sub["post"] == 1).sum()),
    }


def main() -> int:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_rows = []
    for label, cutoff in CUTOFFS.items():
        r = run_its_top(cutoff)
        r["cutoff_label"] = label
        r["cutoff_date"] = cutoff.isoformat()
        out_rows.append(r)
        print(f"{label:<24s}  β2 = {r['β2_level_change']:+.2f} pp "
              f"[{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}]  "
              f"p={r['β2_p']:.4f}  DW={r['DW']:.2f}  LB={r['LB_min_p']:.3f}", flush=True)

    fields = ["cutoff_label", "cutoff_date", "n_pre", "n_post",
              "β2_level_change", "ci_lo", "ci_hi", "β2_p",
              "β3_slope_change", "β3_p", "DW", "LB_min_p"]
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in out_rows:
            w.writerow({k: r[k] for k in fields})
    print(f"\n Saved {OUT_CSV}", flush=True)


    lines = [
        "# — ITS placebo cutoff test (top-tier robustness)",
        "",
        "review : test whether the top-tier ITS level drop is specific",
        "to the Dec-9 deadline by re-running the segmented regression with placebo",
        "cutoffs (no real policy event at those weeks).",
        "",
        "## Method",
        "",
        "Identical model + HAC SE to `scripts/27`, on the top-tier weekly rate series",
        "only. Three cutoffs compared.",
        "",
        "## Results",
        "",
        "| Cutoff | β2 (level change, pp) | 95% CI | p | DW | LB min-p |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for r in out_rows:
        sig = "***" if r["β2_p"] < 0.01 else ("**" if r["β2_p"] < 0.05 else ("*" if r["β2_p"] < 0.10 else ""))
        lines.append(
            f"| `{r['cutoff_label']}` ({r['cutoff_date']}) | {r['β2_level_change']:+.2f}{sig} | "
            f"[{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}] | {r['β2_p']:.4f} | "
            f"{r['DW']:.2f} | {r['LB_min_p']:.3f} |"
        )


    actual_b2 = out_rows[0]["β2_level_change"]
    placebo_b2s = [out_rows[1]["β2_level_change"], out_rows[2]["β2_level_change"]]
    lines += ["",
              "## Interpretation",
              "",
              f"- Actual Dec-9 β2 = {actual_b2:+.2f} pp",
              f"- Placebo β2s: {placebo_b2s[0]:+.2f} pp (pre-deadline) / {placebo_b2s[1]:+.2f} pp (post-deadline)",
              "",
              "Verdict (qualitative):",
              ("- Placebo β2s are NOT statistically distinguishable from zero (CIs cross 0) "
               " actual Dec-9 drop appears SPECIFIC to the deadline window."
               if all(out_rows[i]["β2_p"] > 0.05 for i in (1, 2))
               else "- At least one placebo β2 is statistically significant  top-tier level drop is "
                    "NOT uniquely specific to Dec-9; alternative deadline-shock explanations remain."),
              "",
              "Caveats:",
              "- Placebo cutoffs are restricted to the same 41-week observation window. A truly",
              "  independent placebo would use an out-of-window series (e.g. 2024 historical data).",
              "- Even if the actual cutoff is specific, this does not establish causality; the",
              "  no-coincident-shock assumption (no other event near 2025-12-09 with similar",
              "  magnitude) remains untested.",
              ]
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f" Saved {OUT_MD}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
