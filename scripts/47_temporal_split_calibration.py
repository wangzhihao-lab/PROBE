from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2 as chi2_dist
from sklearn.metrics import brier_score_loss, roc_auc_score

import figstyle

figstyle.apply()

DATA_DIR = Path(__file__).parent.parent / "data"
REPORTS_DIR = Path(__file__).parent.parent / "reports"
PAPER_DIR = Path(__file__).parent.parent / "paper" / "figures"

VERSIONS_CSV = DATA_DIR / "versions.csv"
INFLUENCE_CSV = DATA_DIR / "influence_metrics.csv"
DEPSDEV_CSV = DATA_DIR / "depsdev_dependents_2026_05_05.csv"
OUT_CSV = DATA_DIR / "temporal_split_calibration.csv"
OUT_MD = REPORTS_DIR / "presub_temporal_split_calibration.md"
OUT_FIG = PAPER_DIR / "temporal_calibration.png"

DEADLINE = pd.Timestamp("2025-12-09", tz="UTC")


import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "rq3_primary", str(Path(__file__).parent / "52_rq3_sensitivity.py"))
_rq3 = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_rq3)


COV_COLS = [c for c in _rq3.LOGIT_COLS_PRIMARY if c != "post_deadline"]


def build_frame() -> pd.DataFrame:
    prewin = _rq3.prewindow_release_counts()
    df = _rq3.build_version_frame(prewin)

    assert len(df) == 97_672, f"joined frame {len(df):,} != frozen 97,672"
    assert df["package"].nunique() == 1_585, (
        f"package count {df['package'].nunique()} != frozen 1,585"
    )
    return df


def decile_ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    df = pd.DataFrame({"y": y, "p": p})
    df["bin"] = pd.qcut(df["p"], q=n_bins, duplicates="drop", labels=False)
    g = df.groupby("bin").agg(n=("y", "size"), obs=("y", "mean"), exp=("p", "mean"))
    return float((g["n"] / g["n"].sum() * (g["obs"] - g["exp"]).abs()).sum())


def hosmer_lemeshow(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> tuple[float, float]:
    df = pd.DataFrame({"y": y, "p": p})
    df["decile"] = pd.qcut(df["p"], q=n_bins, duplicates="drop", labels=False)
    g = df.groupby("decile").agg(n=("y", "size"), obs=("y", "sum"), exp=("p", "sum"))
    g["var"] = g["n"] * (g["exp"] / g["n"]) * (1 - g["exp"] / g["n"])
    chi2 = float(((g["obs"] - g["exp"]) ** 2 / g["var"]).replace([np.inf, -np.inf], np.nan).sum())
    return chi2, float(1 - chi2_dist.cdf(chi2, df=n_bins - 2))


def main() -> int:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    df = build_frame()
    pre = df[df["post_deadline"] == 0]
    post = df[df["post_deadline"] == 1]
    print(f"Pre-deadline train rows: {len(pre):,} ({pre['package'].nunique()} pkgs); "
          f"post-deadline test rows: {len(post):,} ({post['package'].nunique()} pkgs)", flush=True)
    print(f"Base rate pre = {pre['y'].mean()*100:.2f}%  post = {post['y'].mean()*100:.2f}%", flush=True)

    X_tr = sm.add_constant(pre[COV_COLS].copy())
    X_te = sm.add_constant(post[COV_COLS].copy(), has_constant="add")
    y_tr = pre["y"].values
    y_te = post["y"].values

    fit = sm.Logit(y_tr, X_tr).fit(
        cov_type="cluster",
        cov_kwds={"groups": pre["package"].values},
        disp=False,
        maxiter=100,
    )
    p_te = np.asarray(fit.predict(X_te))
    p_tr = np.asarray(fit.predict(X_tr))

    auc_te = roc_auc_score(y_te, p_te)
    auc_tr = roc_auc_score(y_tr, p_tr)
    brier_te = brier_score_loss(y_te, p_te)
    ece_te = decile_ece(y_te, p_te)
    citl = float(y_te.mean() - p_te.mean())
    hl_chi2, hl_p = hosmer_lemeshow(y_te, p_te)


    logit_p = np.log(np.clip(p_te, 1e-12, 1 - 1e-12) / (1 - np.clip(p_te, 1e-12, 1 - 1e-12)))
    recal = sm.Logit(y_te, sm.add_constant(logit_p)).fit(disp=False)
    cal_intercept = float(recal.params[0])
    cal_slope = float(recal.params[1])


    rng = np.random.default_rng(42)
    post_r = post.reset_index(drop=True)
    idx_by_pkg = post_r.groupby("package").indices
    pkgs = np.array(sorted(idx_by_pkg.keys()))
    boot = []
    for _ in range(200):
        sampled = rng.choice(pkgs, size=len(pkgs), replace=True)
        idx = np.concatenate([idx_by_pkg[p] for p in sampled])
        try:
            boot.append(roc_auc_score(y_te[idx], p_te[idx]))
        except ValueError:
            continue
    boot = np.array(sorted(boot))
    auc_lo, auc_hi = float(boot[5]), float(boot[-6])

    row_boot = []
    n = len(y_te)
    for _ in range(200):
        idx = rng.integers(0, n, n)
        try:
            row_boot.append(roc_auc_score(y_te[idx], p_te[idx]))
        except ValueError:
            continue
    row_boot = np.array(sorted(row_boot))
    row_lo, row_hi = float(row_boot[5]), float(row_boot[-6])

    print(f"Out-of-sample AUC = {auc_te:.4f} pkg-cluster CI [{auc_lo:.4f}, {auc_hi:.4f}] "
          f"(row-bootstrap contrast [{row_lo:.4f}, {row_hi:.4f}]; "
          f"in-sample train AUC = {auc_tr:.4f})", flush=True)
    print(f"Brier = {brier_te:.4f}  decile ECE = {ece_te:.4f}  CITL = {citl:+.4f}", flush=True)
    print(f"Recalibration slope = {cal_slope:.3f}  intercept = {cal_intercept:+.3f}", flush=True)
    print(f"HL(test) χ² = {hl_chi2:.1f}, p = {hl_p:.2e}", flush=True)

    rows = [
        ("train_rows_pre_deadline", len(pre)),
        ("test_rows_post_deadline", len(post)),
        ("base_rate_pre", round(float(pre["y"].mean()), 6)),
        ("base_rate_post", round(float(post["y"].mean()), 6)),
        ("auc_test", round(float(auc_te), 6)),
        ("auc_test_ci_low_pkgboot200", round(auc_lo, 6)),
        ("auc_test_ci_high_pkgboot200", round(auc_hi, 6)),
        ("auc_test_ci_low_rowboot200_contrast", round(row_lo, 6)),
        ("auc_test_ci_high_rowboot200_contrast", round(row_hi, 6)),
        ("auc_train", round(float(auc_tr), 6)),
        ("brier_test", round(float(brier_te), 6)),
        ("decile_ece_test", round(float(ece_te), 6)),
        ("calibration_in_the_large_test", round(citl, 6)),
        ("recalibration_slope_test", round(cal_slope, 6)),
        ("recalibration_intercept_test", round(cal_intercept, 6)),
        ("hosmer_lemeshow_chi2_test", round(hl_chi2, 4)),
        ("hosmer_lemeshow_p_test", hl_p),
    ]
    with OUT_CSV.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerows(rows)
    print(f" Saved {OUT_CSV}", flush=True)


    dfc = pd.DataFrame({"y": y_te, "p": p_te})
    dfc["bin"] = pd.qcut(dfc["p"], q=10, duplicates="drop", labels=False)
    g = dfc.groupby("bin").agg(obs=("y", "mean"), exp=("p", "mean"))
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot(g["exp"], g["obs"], "o-", color=figstyle.ACCENT, markersize=7,
            zorder=3, label="Post-deadline test deciles (obs vs pred)")
    ax.plot([0, 1], [0, 1], "--", color=figstyle.REF, linewidth=1,
            zorder=1, label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability (pre-deadline fit)")
    ax.set_ylabel("Observed attestation rate (post-deadline)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT_FIG)
    plt.close(fig)
    print(f" Saved {OUT_FIG}", flush=True)

    hl_verdict = "PASS (p > 0.05)" if hl_p > 0.05 else "FAIL — mis-calibrated out of sample too"
    OUT_MD.write_text("\n".join([
        "# Pre-submission — Temporal-split out-of-sample calibration check",
        "",
        "**Date**: 2026-07-08 (acting-PI pre-submission batch; frozen rows only)  ",
        "**Design**: fit scripts/30 specification minus `post_deadline` on pre-deadline",
        "versions; predict post-deadline versions. Split at 2025-12-09.",
        "",
        f"- Train (pre): {len(pre):,} rows, base rate {pre['y'].mean()*100:.2f}%",
        f"- Test (post): {len(post):,} rows, base rate {post['y'].mean()*100:.2f}%",
        f"- **Out-of-sample AUC = {auc_te:.4f}** (boot 95% CI [{auc_lo:.4f}, {auc_hi:.4f}]; "
        f"train AUC {auc_tr:.4f})",
        f"- Brier (test) = {brier_te:.4f}; **decile ECE = {ece_te:.4f}**; "
        f"calibration-in-the-large = {citl:+.4f}",
        f"- Recalibration slope = {cal_slope:.3f}, intercept = {cal_intercept:+.3f}",
        f"- Hosmer-Lemeshow (test) χ² = {hl_chi2:.1f}, p = {hl_p:.2e} — {hl_verdict}",
        "",
        "## Reading (descriptive-only)",
        "",
        "Discrimination transfers across the boundary; calibration status per the",
        "numbers above. Either way the verdict of Section 5.3.2 is unchanged:",
        "odds ratios remain predictive associations, never Dec-9 causal claims.",
        "The `post_deadline` covariate is necessarily excluded from the pre-period",
        "fit (constant within split); its OR is unaffected.",
    ]) + "\n")
    print(f" Saved {OUT_MD}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
