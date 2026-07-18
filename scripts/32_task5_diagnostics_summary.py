from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_JSON = DATA_DIR / "task_5_diagnostics_summary.json"


def _to_float(x):
    try:
        return float(x)
    except (ValueError, TypeError):
        return None


def its() -> dict:
    out = {"per_tier": []}
    with (DATA_DIR / "its_results.csv").open() as f:
        for r in csv.DictReader(f):
            out["per_tier"].append({
                "tier": r["tier"],
                "n_weeks": int(r["n_weeks"]),
                "β2_level_change_pp": _to_float(r["β2_level_change"]),
                "β2_ci95": [_to_float(r["ci95_β2_low"]), _to_float(r["ci95_β2_high"])],
                "β2_p": _to_float(r["pval_β2"]),
                "β3_slope_change_pp_per_week": _to_float(r["β3_slope_change"]),
                "β3_ci95": [_to_float(r["ci95_β3_low"]), _to_float(r["ci95_β3_high"])],
                "β3_p": _to_float(r["pval_β3"]),
                "durbin_watson": _to_float(r["DW"]),
                "ljung_box_min_p": min([p for p in [_to_float(r["LB_p_lag1"]),
                                                     _to_float(r["LB_p_lag2"]),
                                                     _to_float(r["LB_p_lag3"])]
                                        if p is not None]),
            })
    return out


def its_placebo() -> dict:
    out = {"top_tier_cutoffs": []}
    p = DATA_DIR / "its_placebo_results.csv"
    if not p.exists():
        return out
    with p.open() as f:
        for r in csv.DictReader(f):
            out["top_tier_cutoffs"].append({
                "cutoff_label": r["cutoff_label"],
                "cutoff_date": r["cutoff_date"],
                "n_pre": int(r["n_pre"]),
                "n_post": int(r["n_post"]),
                "β2_pp": _to_float(r["β2_level_change"]),
                "β2_ci95": [_to_float(r["ci_lo"]), _to_float(r["ci_hi"])],
                "β2_p": _to_float(r["β2_p"]),
                "durbin_watson": _to_float(r["DW"]),
                "ljung_box_min_p": _to_float(r["LB_min_p"]),
            })
    return out


def did() -> dict:
    from datetime import date
    import statsmodels.api as sm

    df = pd.read_csv(DATA_DIR / "rq1_weekly_attestation_rate_depsdev.csv")
    df = df[df["tier"].isin(["top", "tail"])].copy()
    df["week_start_dt"] = pd.to_datetime(df["week_start"])
    df["post"] = (df["week_start_dt"].dt.date >= date(2025, 12, 8)).astype(int)
    df["top"] = (df["tier"] == "top").astype(int)
    df["rate_pp"] = df["rate"].astype(float) * 100
    df["t"] = (df["week_start_dt"] - df["week_start_dt"].min()).dt.days // 7


    df["post_x_top"] = df["post"] * df["top"]
    X = sm.add_constant(df[["post", "top", "post_x_top"]])
    fit = sm.OLS(df["rate_pp"].values, X.values).fit(cov_type="HC1")
    β3 = float(fit.params[3])
    ci = fit.conf_int(alpha=0.05)
    main = {
        "β3_did_pp": β3,
        "β3_ci95": [float(ci[3][0]), float(ci[3][1])],
        "β3_p": float(fit.pvalues[3]),
        "se_HC1": float(fit.bse[3]),
        "n_pre_weeks": int((df["post"] == 0).sum() / 2),
        "n_post_weeks": int((df["post"] == 1).sum() / 2),
        "n_clusters": 2,
    }


    df_pre = df[df["post"] == 0].copy()
    df_pre["t_x_top"] = df_pre["t"] * df_pre["top"]
    X_pre = sm.add_constant(df_pre[["t", "top", "t_x_top"]])
    fit_pre = sm.OLS(df_pre["rate_pp"].values, X_pre.values).fit(cov_type="HC1")
    γ3 = float(fit_pre.params[3])
    ci_pre = fit_pre.conf_int(alpha=0.05)
    main["γ3_pre_trend_pp_per_week"] = γ3
    main["γ3_ci95"] = [float(ci_pre[3][0]), float(ci_pre[3][1])]
    main["γ3_p"] = float(fit_pre.pvalues[3])
    main["parallel_trends_verdict"] = "FAIL" if main["γ3_p"] < 0.05 else "PASS"
    return main


def cox() -> dict:

    out = {"per_covariate": []}
    failing = []
    with (DATA_DIR / "cox_results.csv").open() as f:
        for r in csv.DictReader(f):
            sf_p = _to_float(r["schoenfeld_p"])
            entry = {
                "covariate": r["covariate"],
                "HR": _to_float(r["HR"]),
                "HR_ci95": [_to_float(r["HR_ci_low"]), _to_float(r["HR_ci_high"])],
                "p": _to_float(r["p"]),
                "schoenfeld_p": sf_p,
                "ph_fail": (sf_p is not None and sf_p < 0.05),
            }
            out["per_covariate"].append(entry)
            if entry["ph_fail"]:
                failing.append(r["covariate"])

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "cox_mod", Path(__file__).parent / "29_cox_time_to_provenance.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    sd = m.build_survival_data()
    cox_cols = [
        "duration_days", "event",
        "log_weekly_downloads", "log_age_days", "log_maintainers",
        "owner_is_org", "has_dependabot_int", "has_issue_template_int",
        "is_scoped_int", "tier_top", "tier_mid"
    ]
    from lifelines import CoxPHFitter
    cph = CoxPHFitter()
    cph.fit(sd[cox_cols], duration_col="duration_days", event_col="event", robust=True)
    out["n_packages"] = int(len(sd))
    out["events"] = int(sd["event"].sum())
    out["censored"] = int((1 - sd["event"]).sum())
    out["concordance_index"] = float(cph.concordance_index_)
    out["covariates_failing_PH"] = failing
    out["count_failing_PH"] = len(failing)
    return out


def logistic() -> dict:

    from datetime import date
    import statsmodels.api as sm
    from sklearn.metrics import brier_score_loss, roc_auc_score
    from scipy.stats import chi2 as chi2_dist

    DEADLINE = pd.Timestamp("2025-12-09", tz="UTC")
    versions = pd.read_csv(DATA_DIR / "versions.csv", parse_dates=["published_at"])
    versions["published_at"] = pd.to_datetime(versions["published_at"], utc=True)
    versions["y"] = (versions["has_attestation"].astype(str).str.lower() == "true").astype(int)
    versions["post_deadline"] = (versions["published_at"] >= DEADLINE).astype(int)

    inf = pd.read_csv(DATA_DIR / "influence_metrics.csv")
    depsdev = pd.read_csv(DATA_DIR / "depsdev_dependents_2026_05_05.csv")
    depsdev_map = dict(zip(depsdev["name"], depsdev["dependents_count"].astype(int)))
    inf["depsdev_dependents_count"] = inf["package"].map(depsdev_map)
    def dpc_tier(dpc):
        if pd.isna(dpc):
            return "missing"
        d = int(dpc)
        return "top" if d > 500 else ("mid" if d > 10 else "tail")
    inf["depsdev_tier"] = inf["depsdev_dependents_count"].apply(dpc_tier)
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
    keep = ["package", "log_weekly_downloads", "log_age_days", "log_maintainers",
            "owner_is_org", "has_dependabot_int", "has_issue_template_int",
            "is_scoped_int", "tier_top", "tier_mid"]
    df = versions.merge(inf[keep].dropna(), on="package", how="inner")
    cov = ["post_deadline", "log_weekly_downloads", "log_age_days", "log_maintainers",
           "owner_is_org", "has_dependabot_int", "has_issue_template_int",
           "is_scoped_int", "tier_top", "tier_mid"]
    X = sm.add_constant(df[cov])
    fit = sm.Logit(df["y"].values, X).fit(
        cov_type="cluster", cov_kwds={"groups": df["package"].values},
        disp=False, maxiter=100,
    )
    p_hat = fit.predict(X)
    auc = float(roc_auc_score(df["y"].values, p_hat))
    brier = float(brier_score_loss(df["y"].values, p_hat))


    rng = np.random.default_rng(42)
    n = len(df)
    boot = []
    for _ in range(200):
        idx = rng.integers(0, n, n)
        try:
            boot.append(roc_auc_score(df["y"].values[idx], p_hat.iloc[idx]))
        except ValueError:
            continue
    boot = np.array(sorted(boot))
    auc_lo, auc_hi = float(boot[5]), float(boot[-6])


    tmp = pd.DataFrame({"y": df["y"].values, "p": p_hat.values})
    tmp["dec"] = pd.qcut(tmp["p"], q=10, duplicates="drop", labels=False)
    g = tmp.groupby("dec").agg(n=("y", "size"), obs=("y", "sum"), exp=("p", "sum"))
    g["var"] = g["n"] * (g["exp"] / g["n"]) * (1 - g["exp"] / g["n"])
    hl_chi2 = float(((g["obs"] - g["exp"]) ** 2 / g["var"]).replace([np.inf, -np.inf], np.nan).sum(skipna=True))
    hl_p = float(1 - chi2_dist.cdf(hl_chi2, df=len(g) - 2))

    return {
        "n_versions": int(len(df)),
        "n_packages": int(df["package"].nunique()),
        "base_rate": float(df["y"].mean()),
        "pseudo_R2_McFadden": float(fit.prsquared),
        "AUC": auc,
        "AUC_ci95_bootstrap_n200": [auc_lo, auc_hi],
        "Brier": brier,
        "HL_chi2": hl_chi2,
        "HL_p": hl_p,
        "HL_verdict": "PASS (p>0.05)" if hl_p > 0.05 else "FAIL (calibration mis-fit; common at large n)",
        "AUC_in_sample_only": True,
    }


def main() -> int:
    summary = {
        "task": "Task 5 (association analysis; demoted from causal per review)",
        "round": "1.5",
        "tier_frame": "deps.dev-tier PRIMARY (per )",
        "ITS": its(),
        "ITS_placebo": its_placebo(),
        "DiD": did(),
        "Cox": cox(),
        "Logistic": logistic(),
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, default=str))
    print(f" Saved {OUT_JSON}", flush=True)


    print(f"\n=== Re-derived for audit ===", flush=True)
    print(f"DiD β3: {summary['DiD']['β3_did_pp']:+.4f}  γ3: {summary['DiD']['γ3_pre_trend_pp_per_week']:+.4f}  PT verdict: {summary['DiD']['parallel_trends_verdict']}", flush=True)
    print(f"Cox n={summary['Cox']['n_packages']}, events={summary['Cox']['events']}, censored={summary['Cox']['censored']}, concordance={summary['Cox']['concordance_index']:.4f}", flush=True)
    print(f"Cox PH failing: {summary['Cox']['covariates_failing_PH']} ({summary['Cox']['count_failing_PH']} of {len(summary['Cox']['per_covariate'])})", flush=True)
    print(f"Logistic AUC={summary['Logistic']['AUC']:.4f} CI={summary['Logistic']['AUC_ci95_bootstrap_n200']}  HL p={summary['Logistic']['HL_p']:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
