"""Review-9 sensitivity batch: five checks answering the 2026-07-14 external
mock-review round (GPT round 9).

S1  Cox entry-tie convention: the primary delayed-entry model breaks
    entry==event integer-day ties by entering tied packages 0.5 day early
    (85 of 326 events are first-release adopters). Refit under 0.25-day and
    1.0-day offsets and with the tied packages excluded outright; report
    hazard-ratio drift, sign changes, and significance changes.
S2  ITS package-cluster bootstrap: the canonical top-tier cutoff-centered
    ITS uses HAC errors on ~40 weekly aggregates, which handles serial
    correlation but not the package-level uncertainty created by a few
    prolific packages dominating weekly composition. Resample packages with
    replacement (B=1000), rebuild the weekly top-tier series, refit the
    canonical spec (scripts/its_spec.py), and report the percentile interval
    for beta2 (and beta3).
S3  Calibration vs clustering: Hosmer-Lemeshow treats 97,672 version rows
    as independent. Hold the fitted primary logistic fixed, resample
    packages with replacement (B=1000), and recompute the in-sample decile
    ECE per resample: a package-cluster interval for the calibration-gap
    magnitude that does not lean on the row-independence chi-square.
S4  Per-class rater agreement: kappa is an aggregate. Reproduce the primary
    (kappa=0.8099) and post-adjudication (kappa=0.8192) pairings from the
    released label files, and tabulate the confusion matrix and per-class
    specific agreement (2*n_kk / (row_k + col_k)); flag classes too sparse
    to assess (D, F).
S5  Case-level predicted probabilities: mean/min/max predicted attestation
    probability for typescript's 242 in-window version rows under the
    primary logistic fit (observed attested = 0), formalizing the
    calibration-counterexample illustration in Section 6.3.

Inputs:
  data/versions.csv, data/influence_metrics.csv,
  data/depsdev_dependents_2026_05_05.csv, data/packuments_raw/*.json
  data/classification.csv, data/classification_v1_raw.csv,
  data/classification_v2_rules.csv, data/classification_v3_pi.csv,
  data/classification_rater2.csv
  (frame construction imported from scripts/52; weekly aggregation and the
  canonical ITS from scripts/46 + scripts/its_spec.py)

Outputs:
  data/review9_cox_tiebreak.csv
  data/review9_its_cluster_bootstrap.csv
  data/review9_ece_cluster_ci.csv
  data/review9_rater_perclass.csv
  data/review9_case_predicted.csv
  reports/presub_review9_sensitivities.md

Carried-forward caveats: data cutoff 2026-05-10 untouched; anti-causal
firewall applies — every quantity below is descriptive/diagnostic and none
identifies a December-9 effect.

Run: conda run --no-capture-output -n stage2c python scripts/53_review9_sensitivities.py

TSE R1 (S1 entry-tie sensitivity refit on the extended incident cohort; only
S1 runs under a non-default window, outputs are suffixed, and --out-suffix is
mandatory so the as-submitted files are never overwritten):
  conda run --no-capture-output -n stage2c python scripts/53_review9_sensitivities.py \
      --origin 2024-01-01 --versions-csv versions_extended.csv --out-suffix _tse_r1_long
"""
from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
SCRIPTS = ROOT / "scripts"
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
sys.path.insert(0, str(SCRIPTS))

SEED = 42
B = 1000


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


mod52 = _load("mod52", SCRIPTS / "52_rq3_sensitivity.py")

# TSE R1: optional extended-window run (S1 only) through scripts/52's configure().
import argparse as _argparse  # noqa: E402
_ap = _argparse.ArgumentParser(add_help=False)
_ap.add_argument("--origin", default="2025-08-01")
_ap.add_argument("--versions-csv", default="versions.csv")
_ap.add_argument("--out-suffix", default="")
_ARGS, _ = _ap.parse_known_args()
_NON_DEFAULT = (_ARGS.origin, _ARGS.versions_csv) != ("2025-08-01", "versions.csv")
if _NON_DEFAULT and not _ARGS.out_suffix:
    raise SystemExit("--out-suffix is required whenever --origin or --versions-csv is "
                     "non-default (protects the as-submitted artefacts)")
_SUFFIX = _ARGS.out_suffix
if _NON_DEFAULT:
    mod52.configure(_ARGS.origin, _ARGS.versions_csv, _ARGS.out_suffix)
mod46 = _load("mod46", SCRIPTS / "46_placebo_cutoff_rank.py")
import its_spec  # noqa: E402  (shared canonical ITS spec)

from lifelines import CoxPHFitter  # noqa: E402


# ---------------------------------------------------------------- S1: Cox ties
def s1_cox_tiebreak() -> pd.DataFrame:
    prevalent = mod52.prevalent_packages()
    prewin = mod52.prewindow_release_counts()
    sd = mod52.build_survival(prevalent, prewin)
    sd_inc = sd[sd["prevalent"] == 0].copy()
    # build_survival returns entry_days already clipped to duration - 0.5 for
    # entry==event ties; raw integer-day entries are exactly recoverable
    # (ties are the only fractional entries, and their raw value == duration).
    dur = sd_inc["duration_days"].astype(float)
    clipped = sd_inc["entry_days"].astype(float)
    tie_mask = (clipped % 1.0) == 0.5
    entry_raw = pd.Series(np.where(tie_mask, dur, clipped), index=sd_inc.index)
    logger.info("S1: incident n=%d events=%d entry==event ties=%d",
                len(sd_inc), int(sd_inc["event"].sum()), int(tie_mask.sum()))
    # Tie composition is computed, never hard-coded, so the report text follows
    # whichever window is being refit.
    s1_cox_tiebreak.tie_info = {
        "n_incident": int(len(sd_inc)), "events": int(sd_inc["event"].sum()),
        "n_tied": int(tie_mask.sum()),
        "tied_events": int(sd_inc.loc[tie_mask, "event"].sum()),
        "tied_censored": int(tie_mask.sum() - sd_inc.loc[tie_mask, "event"].sum()),
    }

    def fit_variant(label: str, entry: pd.Series, frame: pd.DataFrame) -> pd.DataFrame:
        f = frame[["duration_days", "event"] + mod52.COX_COLS_PRIMARY].copy()
        f["entry_days"] = entry.loc[f.index].values
        cph = CoxPHFitter()
        cph.fit(f, duration_col="duration_days", event_col="event",
                entry_col="entry_days", robust=True)
        out = cph.summary.reset_index()[["covariate", "exp(coef)", "p"]]
        out.columns = ["covariate", "HR", "p"]
        out.insert(0, "variant", label)
        return out

    rows = []
    for eps in (0.5, 0.25, 1.0):
        entry = np.minimum(entry_raw, sd_inc["duration_days"] - eps).clip(lower=0.0)
        rows.append(fit_variant(f"eps_{eps}", entry, sd_inc))
    keep = sd_inc[~tie_mask]
    entry_keep = np.minimum(
        entry_raw.loc[keep.index], keep["duration_days"] - 0.5
    ).clip(lower=0.0)
    rows.append(fit_variant("drop_tied", entry_keep, keep))
    tab = pd.concat(rows, ignore_index=True)

    base = tab[tab["variant"] == "eps_0.5"].set_index("covariate")
    drift = []
    for variant in tab["variant"].unique():
        cur = tab[tab["variant"] == variant].set_index("covariate")
        dlog = np.abs(np.log(cur["HR"]) - np.log(base["HR"]))
        sign_flip = ((cur["HR"] > 1) != (base["HR"] > 1)).sum()
        sig_flip = ((cur["p"] < 0.05) != (base["p"] < 0.05)).sum()
        drift.append({"variant": variant, "max_abs_dlogHR": float(dlog.max()),
                      "worst_covariate": str(dlog.idxmax()),
                      "n_sign_changes": int(sign_flip),
                      "n_sig_changes_at_.05": int(sig_flip)})
        logger.info("S1 %-10s max|dlogHR|=%.4f (%s) sign_changes=%d sig_changes=%d",
                    variant, dlog.max(), dlog.idxmax(), sign_flip, sig_flip)
    tab.to_csv(DATA / f"review9_cox_tiebreak{_SUFFIX}.csv", index=False)
    return pd.DataFrame(drift)


# ------------------------------------------------- S2: ITS package bootstrap
def s2_its_cluster_bootstrap() -> dict:
    versions = mod46.load_versions_with_tier()
    top = versions[versions["dd_tier"] == "top"].copy()
    top["period_start"] = (
        top["published_at"]
        - pd.to_timedelta(top["published_at"].dt.weekday, unit="D")
    ).dt.normalize()
    pkgs = sorted(top["package"].unique())
    weeks = sorted(top["period_start"].unique())
    p_idx = {p: i for i, p in enumerate(pkgs)}
    w_idx = {w: i for i, w in enumerate(weeks)}
    N = np.zeros((len(pkgs), len(weeks)))
    A = np.zeros((len(pkgs), len(weeks)))
    g = top.groupby(["package", "period_start"])["has_attestation_int"].agg(["size", "sum"])
    for (p, w), r in g.iterrows():
        N[p_idx[p], w_idx[w]] = r["size"]
        A[p_idx[p], w_idx[w]] = r["sum"]

    def fit_from_counts(counts: np.ndarray):
        n_w = counts @ N
        a_w = counts @ A
        keep = n_w > 0
        sub = pd.DataFrame({"period_start": pd.Series(weeks)[keep],
                            "rate": a_w[keep] / n_w[keep]})
        return its_spec.fit_its(sub, mod52.DEADLINE, period_col="period_start",
                                granularity="W", min_pre=2, min_post=2)

    ident = fit_from_counts(np.ones(len(pkgs)))
    logger.info("S2: identity-weight check beta2=%.4f (canonical -4.42)",
                ident["beta2_level_change_pp"])
    assert abs(ident["beta2_level_change_pp"] - (-4.42)) < 0.05, "canonical mismatch"

    rng = np.random.default_rng(SEED)
    b2s, b3s = [], []
    for _ in range(B):
        draw = rng.integers(0, len(pkgs), len(pkgs))
        counts = np.bincount(draw, minlength=len(pkgs)).astype(float)
        r = fit_from_counts(counts)
        if r is not None:
            b2s.append(r["beta2_level_change_pp"])
            b3s.append(r["beta3_slope_change"])
    b2s, b3s = np.array(b2s), np.array(b3s)
    out = {"n_boot": len(b2s), "n_pkgs": len(pkgs),
           "beta2_point": float(ident["beta2_level_change_pp"]),
           "beta2_ci_low": float(np.percentile(b2s, 2.5)),
           "beta2_ci_high": float(np.percentile(b2s, 97.5)),
           "beta3_point": float(ident["beta3_slope_change"]),
           "beta3_ci_low": float(np.percentile(b3s, 2.5)),
           "beta3_ci_high": float(np.percentile(b3s, 97.5))}
    pd.DataFrame([out]).to_csv(DATA / "review9_its_cluster_bootstrap.csv", index=False)
    logger.info("S2: beta2 cluster CI [%.2f, %.2f]; beta3 [%.3f, %.3f] (B=%d)",
                out["beta2_ci_low"], out["beta2_ci_high"],
                out["beta3_ci_low"], out["beta3_ci_high"], len(b2s))
    return out


# ------------------------------------------- S3 + S5: ECE cluster CI + cases
def s3_s5_logistic() -> tuple[dict, pd.DataFrame]:
    prewin = mod52.prewindow_release_counts()
    df = mod52.build_version_frame(prewin)
    fit, X = mod52.fit_logit(df, mod52.LOGIT_COLS_PRIMARY)
    p_hat = np.asarray(fit.predict(X))
    y = df["y"].values
    ece_point = mod52.decile_ece(y, p_hat)

    pkg_rows: dict[str, np.ndarray] = {
        p: idx.values for p, idx in df.groupby("package").groups.items()
    }
    pkg_list = list(pkg_rows)
    rng = np.random.default_rng(SEED)
    eces = []
    for _ in range(B):
        draw = rng.integers(0, len(pkg_list), len(pkg_list))
        idx = np.concatenate([pkg_rows[pkg_list[i]] for i in draw])
        eces.append(mod52.decile_ece(y[idx], p_hat[idx]))
    eces = np.array(eces)
    s3 = {"ece_point": float(ece_point),
          "ece_ci_low": float(np.percentile(eces, 2.5)),
          "ece_ci_high": float(np.percentile(eces, 97.5)),
          "n_boot": B, "n_rows": len(df), "n_pkgs": len(pkg_list)}
    pd.DataFrame([s3]).to_csv(DATA / "review9_ece_cluster_ci.csv", index=False)
    logger.info("S3: in-sample decile ECE %.4f, package-cluster CI [%.4f, %.4f]",
                ece_point, s3["ece_ci_low"], s3["ece_ci_high"])

    cases = []
    for pkg in ("typescript", "axios", "lodash"):
        m = df["package"] == pkg
        if m.any():
            cases.append({"package": pkg, "n_rows": int(m.sum()),
                          "observed_attested": int(y[m.values].sum()),
                          "mean_p_hat": float(p_hat[m.values].mean()),
                          "min_p_hat": float(p_hat[m.values].min()),
                          "max_p_hat": float(p_hat[m.values].max())})
    case_tab = pd.DataFrame(cases)
    case_tab.to_csv(DATA / "review9_case_predicted.csv", index=False)
    logger.info("S5: %s", case_tab.to_dict("records"))
    return s3, case_tab


# ------------------------------------- S6: parallel-trends test under HAC SEs
def s6_pt_hac() -> dict:
    """The DiD pre-trend test (scripts/28) uses HC1 errors, which handle
    heteroskedasticity but not the weekly series' serial correlation. Refit
    the same pre-period differential-trend coefficient under (a) the stacked
    interaction spec with Newey-West HAC errors and (b) the cleanest
    formulation: the top-minus-tail difference series regressed on time with
    Newey-West HAC errors (for balanced weeks the slope equals gamma3)."""
    import statsmodels.api as sm
    df = pd.read_csv(DATA / "rq1_weekly_attestation_rate_depsdev.csv")
    df = df[df["tier"].isin(["top", "tail"])].copy()
    df["week_start_dt"] = pd.to_datetime(df["week_start"])
    # scripts/28 convention: the deadline WEEK (Monday 2025-12-08, containing
    # December 9) is post — matches the event-study x=0 = week of 2025-12-08.
    df["post"] = (df["week_start_dt"] >= pd.Timestamp("2025-12-08")).astype(int)
    df["top"] = (df["tier"] == "top").astype(int)
    df["rate_pp"] = df["rate"].astype(float) * 100
    df["t"] = (df["week_start_dt"] - df["week_start_dt"].min()).dt.days // 7
    pre = df[df["post"] == 0].sort_values(["tier", "t"]).copy()
    pre["t_x_top"] = pre["t"] * pre["top"]

    X = sm.add_constant(pre[["t", "top", "t_x_top"]].values)
    y = pre["rate_pp"].values
    f_hc1 = sm.OLS(y, X).fit(cov_type="HC1")
    lag = max(1, int(np.ceil(len(pre) ** 0.25)))
    f_hac = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": lag})

    piv = pre.pivot(index="t", columns="tier", values="rate_pp").dropna()
    dser = (piv["top"] - piv["tail"]).values
    td = piv.index.values.astype(float)
    Xd = sm.add_constant(td)
    lag_d = max(1, int(np.ceil(len(dser) ** 0.25)))
    f_diff = sm.OLS(dser, Xd).fit(cov_type="HAC", cov_kwds={"maxlags": lag_d})

    out = {"gamma3_hc1": float(f_hc1.params[3]), "p_hc1": float(f_hc1.pvalues[3]),
           "gamma3_hac_stacked": float(f_hac.params[3]),
           "p_hac_stacked": float(f_hac.pvalues[3]), "hac_lag": lag,
           "gamma3_diffseries": float(f_diff.params[1]),
           "p_diffseries_hac": float(f_diff.pvalues[1]), "diff_lag": lag_d,
           "n_pre_weeks": int(len(piv))}
    assert abs(out["gamma3_hc1"] - (-1.6106)) < 0.005, "PT replication mismatch"
    pd.DataFrame([out]).to_csv(DATA / "review9_pt_hac.csv", index=False)
    logger.info("S6: gamma3 HC1 %.4f (p=%.4f) | HAC-stacked p=%.4f | "
                "diff-series NW p=%.4f (lag %d, %d pre weeks)",
                out["gamma3_hc1"], out["p_hc1"], out["p_hac_stacked"],
                out["p_diffseries_hac"], lag_d, out["n_pre_weeks"])
    return out


# ---------------------------------------------------- S4: per-class agreement
def s4_perclass() -> pd.DataFrame:
    r2 = pd.read_csv(DATA / "classification_rater2.csv")[["package", "version", "class"]]
    r2 = r2.rename(columns={"class": "class_r2"})
    out_rows = []
    for fname in ("classification_v1_raw.csv", "classification_v2_rules.csv",
                  "classification_v3_pi.csv", "classification.csv"):
        r1 = pd.read_csv(DATA / fname)[["package", "version", "class"]]
        r1 = r1.rename(columns={"class": "class_r1"})
        m = r2.merge(r1, on=["package", "version"], how="inner")
        if len(m) < 90:
            continue
        classes = sorted(set(m["class_r1"]) | set(m["class_r2"]))
        conf = pd.crosstab(m["class_r1"], m["class_r2"]).reindex(
            index=classes, columns=classes, fill_value=0)
        po = np.trace(conf.values) / len(m)
        marg1 = conf.sum(axis=1) / len(m)
        marg2 = conf.sum(axis=0) / len(m)
        pe = float((marg1 * marg2).sum())
        kappa = (po - pe) / (1 - pe)
        logger.info("S4: %-32s n=%d kappa=%.4f", fname, len(m), kappa)
        for k in classes:
            nkk = conf.loc[k, k] if k in conf.index and k in conf.columns else 0
            rk = conf.loc[k].sum() if k in conf.index else 0
            ck = conf[k].sum() if k in conf.columns else 0
            sa = (2 * nkk / (rk + ck)) if (rk + ck) > 0 else np.nan
            out_rows.append({"pairing": fname, "kappa": round(float(kappa), 4),
                             "n_overlap": len(m), "class": k,
                             "n_r1": int(rk), "n_r2": int(ck),
                             "n_agree": int(nkk),
                             "specific_agreement": round(float(sa), 4)})
    tab = pd.DataFrame(out_rows)
    tab.to_csv(DATA / "review9_rater_perclass.csv", index=False)
    return tab


def main() -> int:
    # --s4-only: recompute the per-class rater-agreement block (S4) alone.
    # S4 reads only the frozen classification tables shipped with the public
    # package; the full run additionally needs the access-gated raw slice
    # (data/packuments_raw/) for the Cox blocks.
    if "--s4-only" in sys.argv[1:]:
        perclass = s4_perclass()
        print(perclass.to_markdown(index=False))
        return 0
    drift = s1_cox_tiebreak()
    ti = s1_cox_tiebreak.tie_info
    tie_text = (f"drop_tied excludes the entry==event packages ({ti['tied_events']} events + "
                f"{ti['tied_censored']} censored; {ti['n_tied']} of {ti['n_incident']} "
                f"incident packages).")
    if _NON_DEFAULT:
        lines = [f"# Review-9 S1 refit on the extended incident cohort (window origin "
                 f"{_ARGS.origin}; version table `{_ARGS.versions_csv}`)", "",
                 "## S1 Cox entry-tie convention (extended series)",
                 "Primary = eps_0.5 (paper). Variants: 0.25 d, 1.0 d offsets; " + tie_text,
                 "", drift.to_markdown(index=False), "",
                 "Only S1 is refit under a non-default window; S2-S6 remain protocol-window "
                 "quantities in `reports/presub_review9_sensitivities.md`.",
                 "Anti-causal firewall: all quantities descriptive/diagnostic; none identifies "
                 "a December-9 effect. Data cutoff 2026-05-10."]
        out = REPORTS / f"presub_review9_sensitivities{_SUFFIX}.md"
        out.write_text("\n".join(lines))
        logger.info("wrote %s", out)
        return 0
    s2 = s2_its_cluster_bootstrap()
    s3, cases = s3_s5_logistic()
    perclass = s4_perclass()
    s6 = s6_pt_hac()

    lines = ["# Review-9 sensitivity batch (2026-07-14)", ""]
    lines += ["## S1 Cox entry-tie convention",
              "Primary = eps_0.5 (paper). Variants: 0.25 d, 1.0 d offsets; " + tie_text, "",
              drift.to_markdown(index=False), ""]
    lines += ["## S2 ITS package-cluster bootstrap (top tier, canonical spec)",
              f"beta2 point {s2['beta2_point']:.2f} pp; percentile CI "
              f"[{s2['beta2_ci_low']:.2f}, {s2['beta2_ci_high']:.2f}] "
              f"(B={s2['n_boot']}, {s2['n_pkgs']} packages). "
              f"beta3 point {s2['beta3_point']:.2f}; CI "
              f"[{s2['beta3_ci_low']:.2f}, {s2['beta3_ci_high']:.2f}].", ""]
    lines += ["## S3 In-sample decile ECE under package resampling (fixed fit)",
              f"ECE {s3['ece_point']:.4f}; package-cluster percentile CI "
              f"[{s3['ece_ci_low']:.4f}, {s3['ece_ci_high']:.4f}] (B={s3['n_boot']}).", ""]
    lines += ["## S4 Per-class rater agreement", "",
              perclass.to_markdown(index=False), ""]
    lines += ["## S5 Case-level predicted probabilities (primary logistic)", "",
              cases.to_markdown(index=False), ""]
    lines += ["## S6 Parallel-trends test under HAC errors",
              f"gamma3 HC1 {s6['gamma3_hc1']:.4f} (p={s6['p_hc1']:.4f}); "
              f"stacked-interaction HAC p={s6['p_hac_stacked']:.4f}; "
              f"top-minus-tail difference-series Newey-West "
              f"p={s6['p_diffseries_hac']:.4f} "
              f"(lag {s6['diff_lag']}, {s6['n_pre_weeks']} pre weeks).", ""]
    lines += ["Anti-causal firewall: all quantities descriptive/diagnostic; "
              "none identifies a December-9 effect. Data cutoff 2026-05-10."]
    (REPORTS / "presub_review9_sensitivities.md").write_text("\n".join(lines))
    logger.info("wrote reports/presub_review9_sensitivities.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
