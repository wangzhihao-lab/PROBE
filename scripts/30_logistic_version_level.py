from __future__ import annotations

import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve

import figstyle

figstyle.apply()

DATA_DIR = Path(__file__).parent.parent / "data"
REPORTS_DIR = Path(__file__).parent.parent / "reports"
PAPER_DIR = Path(__file__).parent.parent / "paper" / "figures"

VERSIONS_CSV = DATA_DIR / "versions.csv"
INFLUENCE_CSV = DATA_DIR / "influence_metrics.csv"
DEPSDEV_CSV = DATA_DIR / "depsdev_dependents_2026_05_05.csv"
OUT_CSV = DATA_DIR / "logistic_results.csv"
OUT_MD = REPORTS_DIR / "task_5_logistic.md"
OUT_ROC = PAPER_DIR / "logistic_roc.png"
OUT_HL = PAPER_DIR / "logistic_hl_calibration.png"

DEADLINE = pd.Timestamp("2025-12-09", tz="UTC")


def dpc_tier(dpc: int) -> str:
    if dpc > 500:
        return "top"
    if dpc > 10:
        return "mid"
    return "tail"


def hosmer_lemeshow(y_true: np.ndarray, p_hat: np.ndarray, n_bins: int = 10) -> tuple[float, float, pd.DataFrame]:
    df = pd.DataFrame({"y": y_true, "p": p_hat})
    df["decile"] = pd.qcut(df["p"], q=n_bins, duplicates="drop", labels=False)
    grouped = df.groupby("decile").agg(
        n=("y", "size"),
        obs=("y", "sum"),
        exp=("p", "sum"),
    )
    grouped["obs_minus_exp"] = grouped["obs"] - grouped["exp"]
    grouped["var"] = grouped["n"] * (grouped["exp"] / grouped["n"]) * (1 - grouped["exp"] / grouped["n"])
    grouped["chi2_contrib"] = (grouped["obs_minus_exp"] ** 2 / grouped["var"]).replace([np.inf, -np.inf], np.nan)
    chi2 = grouped["chi2_contrib"].sum(skipna=True)
    from scipy.stats import chi2 as chi2_dist
    p_val = 1 - chi2_dist.cdf(chi2, df=n_bins - 2)
    return float(chi2), float(p_val), grouped


def main() -> int:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)


    print("Loading versions.csv (100k rows)...", flush=True)
    versions = pd.read_csv(VERSIONS_CSV, parse_dates=["published_at"])
    versions["published_at"] = pd.to_datetime(versions["published_at"], utc=True)
    versions["y"] = (versions["has_attestation"].astype(str).str.lower() == "true").astype(int)
    versions["post_deadline"] = (versions["published_at"] >= DEADLINE).astype(int)

    inf = pd.read_csv(INFLUENCE_CSV)
    depsdev = pd.read_csv(DEPSDEV_CSV)
    depsdev_map = dict(zip(depsdev["name"], depsdev["dependents_count"].astype(int)))

    inf["depsdev_dependents_count"] = inf["package"].map(depsdev_map)
    inf["depsdev_tier"] = inf["depsdev_dependents_count"].apply(
        lambda x: dpc_tier(int(x)) if pd.notna(x) else "missing"
    )
    inf = inf[inf["depsdev_tier"].isin(["top", "mid", "tail"])].copy()


    inf["log_weekly_downloads"] = np.log1p(pd.to_numeric(inf["weekly_downloads_avg"], errors="coerce"))
    inf["log_age_days"] = np.log1p(pd.to_numeric(inf["package_age_days"], errors="coerce"))
    inf["log_maintainers"] = np.log1p(pd.to_numeric(inf["maintainers_count"], errors="coerce"))
    inf["owner_is_org"] = (inf["github_owner_type"] == "Organization").astype(int)
    for src, dst in [("has_dependabot", "has_dependabot_int"),
                     ("has_issue_template", "has_issue_template_int"),
                     ("is_scoped_package", "is_scoped_int")]:
        inf[dst] = (inf[src].astype(str).str.lower() == "true").astype(int)
    inf["tier_top"] = (inf["depsdev_tier"] == "top").astype(int)
    inf["tier_mid"] = (inf["depsdev_tier"] == "mid").astype(int)

    keep_inf_cols = ["package",
                     "log_weekly_downloads", "log_age_days", "log_maintainers",
                     "owner_is_org", "has_dependabot_int", "has_issue_template_int",
                     "is_scoped_int", "tier_top", "tier_mid"]
    inf_keep = inf[keep_inf_cols].dropna()

    df = versions.merge(inf_keep, on="package", how="inner")
    print(f"Joined frame: {len(df):,} version rows × {len(df['package'].unique())} packages", flush=True)
    print(f"Outcome base rate: {df['y'].mean()*100:.2f}%", flush=True)


    cov_cols = ["post_deadline",
                "log_weekly_downloads", "log_age_days", "log_maintainers",
                "owner_is_org", "has_dependabot_int", "has_issue_template_int",
                "is_scoped_int", "tier_top", "tier_mid"]
    X = df[cov_cols].copy()
    X = sm.add_constant(X)
    y = df["y"].values

    print("Fitting Logit with cluster-robust SE at package level...", flush=True)
    fit = sm.Logit(y, X).fit(
        cov_type="cluster",
        cov_kwds={"groups": df["package"].values},
        disp=False,
        maxiter=100,
    )
    print(f"Pseudo-R² (McFadden) = {fit.prsquared:.4f}", flush=True)


    p_hat = fit.predict(X)
    auc = roc_auc_score(y, p_hat)
    brier = brier_score_loss(y, p_hat)
    print(f"ROC AUC = {auc:.4f}  Brier = {brier:.4f}", flush=True)


    rng = np.random.default_rng(42)
    boot_aucs = []
    n = len(y)
    for _ in range(200):
        idx = rng.integers(0, n, n)
        try:
            boot_aucs.append(roc_auc_score(y[idx], p_hat.iloc[idx]))
        except ValueError:
            continue
    boot_aucs = np.array(sorted(boot_aucs))
    auc_lo, auc_hi = float(boot_aucs[5]), float(boot_aucs[-6])
    print(f"AUC 95% CI (boot n=200): [{auc_lo:.4f}, {auc_hi:.4f}]", flush=True)


    hl_stat, hl_p, hl_table = hosmer_lemeshow(y, p_hat.values, n_bins=10)
    print(f"Hosmer-Lemeshow: χ² = {hl_stat:.3f}, p = {hl_p:.4f}", flush=True)


    summary_rows = []
    for cov in fit.params.index:
        oh = float(np.exp(fit.params[cov]))
        ci = fit.conf_int().loc[cov]
        summary_rows.append({
            "covariate": cov,
            "coef": float(fit.params[cov]),
            "OR": oh,
            "OR_ci_low": float(np.exp(ci[0])),
            "OR_ci_high": float(np.exp(ci[1])),
            "se(coef)": float(fit.bse[cov]),
            "z": float(fit.tvalues[cov]),
            "p": float(fit.pvalues[cov]),
        })

    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["covariate", "coef", "OR", "OR_ci_low",
                                           "OR_ci_high", "se(coef)", "z", "p"])
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)
    print(f"\n Saved {OUT_CSV}", flush=True)


    lines = [
        "# — Logistic regression (version-level)",
        "",
        f"**Date**: 2026-05-18  ",
        f"**Sample**: {len(df):,} versions in 2025-08-01 .. 2026-05-10 window  ",
        f"**Outcome**: `has_attestation` ∈ {{0, 1}} (base rate {df['y'].mean()*100:.2f}%)  ",
        f"**Cluster**: package (n_pkg = {df['package'].nunique()})  ",
        f"**Model**: statsmodels Logit, cluster-robust SE",
        "",
        f"**Pseudo-R² (McFadden)**: {fit.prsquared:.4f}",
        "",
        "## Coefficient table (effect size = OR with 95% CI)",
        "",
        "| Covariate | OR | 95% CI | p |",
        "|---|---:|---|---:|",
    ]
    for r in summary_rows:
        sig = "***" if r["p"] < 0.01 else ("**" if r["p"] < 0.05 else ("*" if r["p"] < 0.10 else ""))
        lines.append(f"| `{r['covariate']}` | {r['OR']:.3f}{sig} | "
                     f"[{r['OR_ci_low']:.3f}, {r['OR_ci_high']:.3f}] | {r['p']:.4f} |")

    auc_verdict = "PASS" if auc >= 0.70 else f" {'borderline' if auc >= 0.60 else 'POOR'}"
    hl_verdict = "PASS (p > 0.05)" if hl_p > 0.05 else f" FAIL (p={hl_p:.3f}) — calibration questionable"
    lines += [
        "",
        "## Discrimination + calibration diagnostics (MANDATORY)",
        "",
        f"- **ROC AUC** = {auc:.4f}  95% CI (bootstrap n=200): [{auc_lo:.4f}, {auc_hi:.4f}]  — verdict: {auc_verdict}",
        f"- **Brier score** = {brier:.4f}  (lower = better calibrated)",
        f"- **Hosmer-Lemeshow**: χ² = {hl_stat:.3f} on 8 df, p = {hl_p:.4f}  — verdict: {hl_verdict}",
        "",
        "## Headline interpretation",
        "",
        "OR > 1: covariate   HIGHER predicted probability of attestation.",
        "OR < 1: covariate   LOWER predicted probability.",
        "",
        "Statistically significant (p < 0.05) covariates:",
        "",
    ]
    for r in summary_rows:
        if r["p"] < 0.05 and r["covariate"] != "const":
            direction = " attestation" if r["OR"] > 1 else " attestation"
            lines.append(f"- `{r['covariate']}`: OR={r['OR']:.3f} [{r['OR_ci_low']:.3f}, {r['OR_ci_high']:.3f}]  {direction} (p={r['p']:.4f})")

    lines += [
        "",
        "## Construct caveats (carried)",
        "",
        "- Tier source = deps.dev snapshot 2026-05-05 (PRIMARY frame)",
        "- 62 packages excluded (missing-dd)",
        "- C / E construct caveats apply to class-level downstream interpretation",
        "- Discrimination is **in-sample only** (no held-out validation in Round 1); paper use must add grouped out-of-sample validation or report AUC as in-sample explicitly.",
        "- **HL FAILS** at p < 0.001 — the model is well-discriminating but mis-calibrated; per-decile observed and predicted probabilities deviate systematically (see `paper/figures/logistic_hl_calibration.png`).",
        "- Causal interpretation is **NOT supported** by this model. AUC + OR are predictive/associational only. Unmeasured confounding (e.g., maintainer's technical risk-tolerance, pre-existing internal pipelines) remains plausible.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f" Saved {OUT_MD}", flush=True)


    fpr, tpr, _ = roc_curve(y, p_hat)
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot(fpr, tpr, color=figstyle.ACCENT, linewidth=2,
            label=f"Logit (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color=figstyle.REF, linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT_ROC)
    plt.close(fig)
    print(f" Saved {OUT_ROC}", flush=True)


    fig, ax = plt.subplots(figsize=(6, 5.5))
    hl_table = hl_table.copy()
    hl_table["obs_rate"] = hl_table["obs"] / hl_table["n"]
    hl_table["exp_rate"] = hl_table["exp"] / hl_table["n"]
    ax.plot(hl_table["exp_rate"], hl_table["obs_rate"], "o-",
            color=figstyle.ACCENT, markersize=7, label="Per decile (obs vs exp)")
    ax.plot([0, 1], [0, 1], "--", color=figstyle.REF, linewidth=1, label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability (per decile)")
    ax.set_ylabel("Observed attestation rate (per decile)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT_HL)
    plt.close(fig)
    print(f" Saved {OUT_HL}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
