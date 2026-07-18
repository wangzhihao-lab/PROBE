"""
Pre-submission RQ3 batch — PRIMARY Cox + PRIMARY logistic (Batch-8 spec)
plus the sensitivity ladder connecting them to the earlier specifications.

This script does NOT modify scripts/29, 30 or 44. Loading and covariate
construction are copied from those scripts, then extended per the Batch-8
review repairs (Anomaly 10 in ANOMALY_REPORT.md):

Part 1 — Cox (paper's primary time-to-first-attested-release table):
  PRIMARY spec = incident-only cohort (packages with >=1 attested version
  before 2025-08-01 excluded), DELAYED ENTRY at each package's first
  in-window release (a package is not at risk of a first in-window attested
  release before it releases in-window; entry clipped to duration - 0.5 day
  for the packages whose entry ties its endpoint at integer-day resolution,
  85 of them events and one a final-day censored row), 11 covariates =
  scripts/29's 9 + log_prewin_release_freq (log1p of the release count in
  the twelve months BEFORE the window opens -- a clean baseline-opportunity
  attribute that, unlike a within-window release count, cannot carry
  post-event information) + github_unobserved (missingness indicator for
  github_status != 'ok' packages, whose GitHub covariates are otherwise
  zero-imputed). Robust SE. Schoenfeld PH is read from an AUXILIARY
  entry-at-window-start fit of the same 11 covariates, because lifelines
  does not implement residuals for left-truncated (delayed-entry) fits.
  SENSITIVITY = the same 11 covariates on the same cohort but with the
  Batch-7 risk set (entry at window start), reported for the audit trail.
  Also writes the incident-cohort left-truncated KM figure
  (paper/figures/km_incident_by_tier.png).

Part 2 — version-level logistic:
  PRIMARY spec = scripts/30's 10 covariates + log_prewin_release_freq +
  github_unobserved, cluster-robust SE by package. The zero-imputation
  spec without the indicator (scripts/30 reproduction) and the
  complete-case fit are SENSITIVITIES.
  (c) package-disjoint holdout: GroupShuffleSplit (test 0.3, seed 42) on the
      primary spec, plus REPEATED grouped splits (20 seeds) -> AUC mean and
      2.5/97.5 percentiles.
  (d) package-cluster bootstrap AUC CI for the primary in-sample model
      (B=200, seed 42, refit-free); naive row bootstrap kept for contrast.

Inputs:
  data/versions.csv                       (100,168 in-window version rows)
  data/influence_metrics.csv              (package-level covariates + github_status)
  data/depsdev_dependents_2026_05_05.csv  (primary tier frame)
  data/packuments_raw/*.json              (full history; prevalent-adopter flag)

Outputs:
  data/cox_incident_primary.csv           (11 covariates: HR, CI, p; Schoenfeld p from auxiliary entry-at-start fit)
  data/logistic_sensitivity.csv           (tidy long table: model, name, kind, ...)
  reports/presub_rq3_sensitivity.md

Carried-forward caveats: deps.dev primary tier frame; 62 missing-dd packages
excluded; anti-causal firewall (all estimates are descriptive/predictive
associations, never Dec-9 causal claims); data cutoff 2026-05-10 untouched.

Run: conda run --no-capture-output -n stage2c python scripts/52_rq3_sensitivity.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

DATA = Path(__file__).parent.parent / "data"
REPORTS = Path(__file__).parent.parent / "reports"
RAW = DATA / "packuments_raw"
VERSIONS_CSV = DATA / "versions.csv"
INFLUENCE_CSV = DATA / "influence_metrics.csv"
DEPSDEV_CSV = DATA / "depsdev_dependents_2026_05_05.csv"
OUT_COX_CSV = DATA / "cox_incident_primary.csv"
OUT_COX_SENS_CSV = DATA / "cox_incident_entry_at_start_sensitivity.csv"
OUT_COX_ISSUETPL_2X2_CSV = DATA / "cox_issue_template_2x2.csv"
OUT_LOG_CSV = DATA / "logistic_sensitivity.csv"
OUT_MD = REPORTS / "presub_rq3_sensitivity.md"

ORIGIN = pd.Timestamp("2025-08-01", tz="UTC")
CENSOR = pd.Timestamp("2026-05-10 23:59:59", tz="UTC")
WINDOW_START = datetime(2025, 8, 1, tzinfo=timezone.utc)
# Pre-window opportunity: releases in the 12 months BEFORE the window. This is a
# clean baseline release-opportunity covariate; influence_metrics.release_freq_12m
# spans the outcome window (2025-05 .. 2026-05) and therefore leaks post-event
# information for early adopters (Batch-9 fix, ANOMALY 12).
PREWIN_START = datetime(2024, 8, 1, tzinfo=timezone.utc)
DEADLINE = pd.Timestamp("2025-12-09", tz="UTC")

COX_COLS = ["log_weekly_downloads", "log_age_days", "log_maintainers",
            "owner_is_org", "has_dependabot_int", "has_issue_template_int",
            "is_scoped_int", "tier_top", "tier_mid"]
# Batch-9 primary covariate set: + pre-window release opportunity + missingness.
COX_COLS_PRIMARY = COX_COLS + ["log_prewin_release_freq", "github_unobserved"]

LOGIT_COLS = ["post_deadline",
              "log_weekly_downloads", "log_age_days", "log_maintainers",
              "owner_is_org", "has_dependabot_int", "has_issue_template_int",
              "is_scoped_int", "tier_top", "tier_mid"]
LOGIT_COLS_PRIMARY = LOGIT_COLS + ["log_prewin_release_freq", "github_unobserved"]

KEY_ORS = ["has_dependabot_int", "has_issue_template_int", "is_scoped_int",
           "tier_top", "log_age_days", "post_deadline", "log_prewin_release_freq",
           "owner_is_org", "log_maintainers"]


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


# ---------------------------------------------------------------- Part 1: Cox
def prevalent_packages() -> set[str]:
    """Packages with >=1 attested version published before 2025-08-01 (full history).

    Copied verbatim from scripts/44_cox_incident_only_sensitivity.py.
    """
    prevalent: set[str] = set()
    for fp in RAW.glob("*.json"):
        try:
            data = json.loads(fp.read_text())
        except Exception:
            continue
        if data.get("_404"):
            continue
        name = data.get("name") or data.get("_id") or fp.stem
        times = data.get("time", {})
        for ver, vdata in (data.get("versions", {}) or {}).items():
            dist = (vdata or {}).get("dist") or {}
            att = dist.get("attestations") or {}
            if isinstance(att, dict) and att.get("url"):
                pt = parse_dt(times.get(ver, ""))
                if pt and pt < WINDOW_START:
                    prevalent.add(name)
                    break
    return prevalent


def prewindow_release_counts() -> dict[str, int]:
    """Per package, number of versions published in the 12 months BEFORE the
    analysis window ([2024-08-01, 2025-08-01)). A clean baseline
    release-opportunity covariate: it uses only pre-window information and
    cannot leak post-event releases (unlike influence_metrics.release_freq_12m,
    which spans the outcome window). Packages absent from the raw cache, or
    with no pre-window release, get 0."""
    counts: dict[str, int] = {}
    for fp in RAW.glob("*.json"):
        try:
            data = json.loads(fp.read_text())
        except Exception:
            continue
        if data.get("_404"):
            continue
        name = data.get("name") or data.get("_id") or fp.stem
        times = data.get("time", {})
        n = 0
        for ver, ts in times.items():
            if ver in ("created", "modified"):
                continue
            pt = parse_dt(ts or "")
            if pt and PREWIN_START <= pt < WINDOW_START:
                n += 1
        counts[name] = n
    return counts


def build_survival(prevalent: set[str], prewin: dict[str, int]) -> pd.DataFrame:
    """Mirror scripts/29 build_survival_data via scripts/44 (keeps package + prevalent),
    extended with delayed-entry time (first in-window release) and the Batch-9
    primary covariates (log_prewin_release_freq, github_unobserved)."""
    v = pd.read_csv(VERSIONS_CSV, parse_dates=["published_at"])
    v["published_at"] = pd.to_datetime(v["published_at"], utc=True)
    v = v[(v["published_at"] >= ORIGIN) & (v["published_at"] <= CENSOR)].copy()
    att = v[v["has_attestation"].astype(str).str.lower() == "true"]
    first_att = att.groupby("package")["published_at"].min().reset_index()
    first_att.columns = ["package", "first_attested_at"]
    first_rel = v.groupby("package")["published_at"].min().reset_index()
    first_rel.columns = ["package", "first_release_at"]
    sd = pd.DataFrame({"package": v["package"].drop_duplicates().reset_index(drop=True)})
    sd = sd.merge(first_att, on="package", how="left")
    sd = sd.merge(first_rel, on="package", how="left")
    sd["event"] = sd["first_attested_at"].notna().astype(int)
    sd.loc[sd["event"] == 0, "first_attested_at"] = CENSOR
    sd["duration_days"] = (sd["first_attested_at"] - ORIGIN).dt.days.clip(lower=1)
    # Delayed entry: at risk of a first in-window attested release only once
    # the package has released in-window. Entry == event ties are clipped to
    # duration - 0.5 (the package enters the risk set at its event instant).
    sd["entry_days"] = (sd["first_release_at"] - ORIGIN).dt.days.clip(lower=0).astype(float)
    sd["entry_days"] = np.minimum(sd["entry_days"], sd["duration_days"] - 0.5)

    inf = pd.read_csv(INFLUENCE_CSV)
    sd = sd.merge(inf, on="package", how="left")
    depsdev = pd.read_csv(DEPSDEV_CSV)
    dd = dict(zip(depsdev["name"], depsdev["dependents_count"].astype(int)))
    sd["dd"] = sd["package"].map(dd)
    sd["tier"] = sd["dd"].apply(lambda x: dpc_tier(int(x)) if pd.notna(x) else "missing")
    sd = sd[sd["tier"].isin(["top", "mid", "tail"])].copy()

    sd["log_weekly_downloads"] = np.log1p(pd.to_numeric(sd["weekly_downloads_avg"], errors="coerce"))
    sd["log_age_days"] = np.log1p(pd.to_numeric(sd["package_age_days"], errors="coerce"))
    sd["log_maintainers"] = np.log1p(pd.to_numeric(sd["maintainers_count"], errors="coerce"))
    sd["log_prewin_release_freq"] = np.log1p(sd["package"].map(prewin).fillna(0).astype(float))
    sd["owner_is_org"] = (sd["github_owner_type"] == "Organization").astype(int)
    sd["github_unobserved"] = (sd["github_status"].astype(str) != "ok").astype(int)
    for s, d in [("has_dependabot", "has_dependabot_int"),
                 ("has_issue_template", "has_issue_template_int"),
                 ("is_scoped_package", "is_scoped_int")]:
        sd[d] = (sd[s].astype(str).str.lower() == "true").astype(int)
    sd["tier_top"] = (sd["tier"] == "top").astype(int)
    sd["tier_mid"] = (sd["tier"] == "mid").astype(int)
    sd["prevalent"] = sd["package"].isin(prevalent).astype(int)

    cols = ["package", "duration_days", "entry_days", "event",
            "prevalent"] + COX_COLS_PRIMARY
    sd = sd[cols].dropna()
    return sd.astype({"duration_days": int, "event": int})


def _cox_table(cph, fit_frame, ph_cph=None, ph_frame=None) -> pd.DataFrame:
    """Coefficient table for `cph`. Schoenfeld PH p-values are computed on
    (ph_cph, ph_frame) when given — lifelines cannot compute Schoenfeld
    residuals under left truncation (delayed entry), so for the primary fit
    the PH diagnostic runs on the same-covariate entry-at-window-start fit
    and is disclosed as such."""
    summary = cph.summary
    if ph_cph is None:
        ph_cph, ph_frame = cph, fit_frame
    ph_test = proportional_hazard_test(ph_cph, ph_frame, time_transform="rank")
    ph_df = ph_test.summary
    rows = []
    for cov in summary.index:
        rows.append({
            "covariate": cov,
            "HR": summary.loc[cov, "exp(coef)"],
            "HR_ci_low": summary.loc[cov, "exp(coef) lower 95%"],
            "HR_ci_high": summary.loc[cov, "exp(coef) upper 95%"],
            "coef": summary.loc[cov, "coef"],
            "se(coef)": summary.loc[cov, "se(coef)"],
            "p": summary.loc[cov, "p"],
            "schoenfeld_p": float(ph_df.loc[cov, "p"]) if cov in ph_df.index else None,
        })
    return pd.DataFrame(rows)


def _km_incident_figure(sd_inc: pd.DataFrame) -> None:
    """Left-truncated KM by tier on the incident cohort (Batch-8 primary view)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from lifelines import KaplanMeierFitter
    sys.path.insert(0, str(Path(__file__).parent))
    import figstyle
    figstyle.apply()
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    tiers = [("top", sd_inc["tier_top"] == 1),
             ("mid", sd_inc["tier_mid"] == 1),
             ("tail", (sd_inc["tier_top"] == 0) & (sd_inc["tier_mid"] == 0))]
    for name, mask in tiers:
        sub = sd_inc[mask]
        kmf = KaplanMeierFitter()
        kmf.fit(sub["duration_days"], event_observed=sub["event"],
                entry=sub["entry_days"], label=name)
        kmf.plot_survival_function(ax=ax, ci_show=True, ci_alpha=0.15,
                                   color=figstyle.TIER[name], linewidth=2)
    ax.set_xlabel("Days since 2025-08-01")
    ax.set_ylabel("Share with no in-window attested release yet")
    ax.set_ylim(0, 1.02)
    ax.legend(title="deps.dev tier", frameon=True, framealpha=0.9,
              edgecolor="0.8", loc="lower left")
    fig.tight_layout()
    out = Path(__file__).parent.parent / "paper" / "figures" / "km_incident_by_tier.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"[Part 1] Saved {out}", flush=True)


def part1_cox_incident() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """PRIMARY: incident cohort, delayed entry at first in-window release,
    11 covariates (9 + log_prewin_release_freq + github_unobserved), robust
    SE, Schoenfeld on an auxiliary entry-at-window-start fit. SENSITIVITY:
    the same 11 covariates on the same cohort with the Batch-7 risk set
    (entry at window start)."""
    prevalent = prevalent_packages()
    prewin = prewindow_release_counts()
    sd = build_survival(prevalent, prewin)
    sd_inc = sd[sd["prevalent"] == 0].copy()
    n_tie = int((sd_inc["entry_days"] >= sd_inc["duration_days"] - 0.5).sum())
    meta = {"n_full": len(sd), "events_full": int(sd["event"].sum()),
            "n_prevalent": int(sd["prevalent"].sum()),
            "n": len(sd_inc), "events": int(sd_inc["event"].sum()),
            "n_entry_tie": n_tie,
            "entry_median_days": float(sd_inc["entry_days"].median())}
    print(f"[Part 1] Full cohort: {meta['n_full']} pkgs, {meta['events_full']} events, "
          f"prevalent={meta['n_prevalent']}", flush=True)
    print(f"[Part 1] Incident cohort: {meta['n']} pkgs, {meta['events']} events; "
          f"median entry {meta['entry_median_days']:.1f} d; "
          f"entry==event ties {n_tie}", flush=True)

    # PRIMARY: delayed entry + 11 covariates. lifelines cannot compute
    # Schoenfeld residuals under left truncation, so the PH diagnostic is
    # computed on an auxiliary entry-at-window-start fit of the SAME 11
    # covariates (disclosed in the paper).
    frame_p = sd_inc[["duration_days", "entry_days", "event"] + COX_COLS_PRIMARY]
    cph_p = CoxPHFitter()
    cph_p.fit(frame_p, duration_col="duration_days", event_col="event",
              entry_col="entry_days", robust=True)
    meta["c_index"] = float(cph_p.concordance_index_)
    frame_aux = sd_inc[["duration_days", "event"] + COX_COLS_PRIMARY]
    cph_aux = CoxPHFitter()
    cph_aux.fit(frame_aux, duration_col="duration_days", event_col="event", robust=True)
    tab_p = _cox_table(cph_p, frame_p, ph_cph=cph_aux, ph_frame=frame_aux)
    tab_p.to_csv(OUT_COX_CSV, index=False)

    # SENSITIVITY: entry-at-window-start with the SAME 11 covariates as the
    # primary (the like-for-like risk-set comparison the paper cites). This is
    # what OUT_COX_SENS_CSV holds.
    frame_s = sd_inc[["duration_days", "event"] + COX_COLS_PRIMARY]
    cph_s = CoxPHFitter()
    cph_s.fit(frame_s, duration_col="duration_days", event_col="event", robust=True)
    tab_s = _cox_table(cph_s, frame_s)
    tab_s.to_csv(OUT_COX_SENS_CSV, index=False)

    # 2x2 interaction check for the issue-template coefficient across
    # {entry-at-start, delayed entry} x {9 core, 11 primary} covariate sets,
    # so the paper's "the two choices interact" claim is reproducible.
    grid_rows = []
    for cov_lbl, cols in [("9core", COX_COLS), ("11primary", COX_COLS_PRIMARY)]:
        f_es = CoxPHFitter(); f_es.fit(
            sd_inc[["duration_days", "event"] + cols],
            duration_col="duration_days", event_col="event", robust=True)
        f_de = CoxPHFitter(); f_de.fit(
            sd_inc[["duration_days", "entry_days", "event"] + cols],
            duration_col="duration_days", event_col="event",
            entry_col="entry_days", robust=True)
        for risk_lbl, f in [("entry_at_start", f_es), ("delayed_entry", f_de)]:
            r = f.summary.loc["has_issue_template_int"]
            grid_rows.append({"covariate_set": cov_lbl, "risk_set": risk_lbl,
                              "issue_template_HR": float(r["exp(coef)"]),
                              "p": float(r["p"])})
    pd.DataFrame(grid_rows).to_csv(OUT_COX_ISSUETPL_2X2_CSV, index=False)
    print("[Part 1] issue-template 2x2 (risk-set x covariate-set):", flush=True)
    for gr in grid_rows:
        print(f"    {gr['covariate_set']:10s} {gr['risk_set']:15s} "
              f"HR={gr['issue_template_HR']:.3f} p={gr['p']:.4f}", flush=True)

    print(f"[Part 1] PRIMARY (delayed entry, 11 cov). Concordance = {meta['c_index']:.4f}",
          flush=True)
    show = tab_p.copy()
    for c in ["HR", "HR_ci_low", "HR_ci_high"]:
        show[c] = show[c].round(3)
    show["p"] = show["p"].round(4)
    show["schoenfeld_p"] = show["schoenfeld_p"].round(4)
    print(show[["covariate", "HR", "HR_ci_low", "HR_ci_high", "p", "schoenfeld_p"]]
          .to_string(index=False), flush=True)
    print("[Part 1] SENSITIVITY (entry at window start, 11 cov):", flush=True)
    show_s = tab_s.copy()
    for c in ["HR", "HR_ci_low", "HR_ci_high"]:
        show_s[c] = show_s[c].round(3)
    show_s["p"] = show_s["p"].round(4)
    print(show_s[["covariate", "HR", "p"]].to_string(index=False), flush=True)
    print(f"[Part 1] Saved {OUT_COX_CSV} and {OUT_COX_SENS_CSV}", flush=True)

    _km_incident_figure(sd_inc)
    return tab_p, tab_s, meta


# ----------------------------------------------------------- Part 2: logistic
def build_version_frame(prewin: dict[str, int]) -> pd.DataFrame:
    """Version-level analysis frame, copied from scripts/30 main() and extended
    with the clean pre-window opportunity covariate."""
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
    inf["log_prewin_release_freq"] = np.log1p(inf["package"].map(prewin).fillna(0).astype(float))
    inf["owner_is_org"] = (inf["github_owner_type"] == "Organization").astype(int)
    inf["github_unobserved"] = (inf["github_status"].astype(str) != "ok").astype(int)
    for src, dst in [("has_dependabot", "has_dependabot_int"),
                     ("has_issue_template", "has_issue_template_int"),
                     ("is_scoped_package", "is_scoped_int")]:
        inf[dst] = (inf[src].astype(str).str.lower() == "true").astype(int)
    inf["tier_top"] = (inf["depsdev_tier"] == "top").astype(int)
    inf["tier_mid"] = (inf["depsdev_tier"] == "mid").astype(int)

    keep_inf_cols = ["package",
                     "log_weekly_downloads", "log_age_days", "log_maintainers",
                     "log_prewin_release_freq", "github_unobserved",
                     "owner_is_org", "has_dependabot_int", "has_issue_template_int",
                     "is_scoped_int", "tier_top", "tier_mid"]
    inf_keep = inf[keep_inf_cols].dropna()
    df = versions.merge(inf_keep, on="package", how="inner").reset_index(drop=True)
    return df


def fit_logit(df: pd.DataFrame, cols: list[str]):
    """Main-spec fit: statsmodels Logit, cluster-robust SE by package (scripts/30)."""
    X = sm.add_constant(df[cols].copy())
    fit = sm.Logit(df["y"].values, X).fit(
        cov_type="cluster",
        cov_kwds={"groups": df["package"].values},
        disp=False,
        maxiter=100,
    )
    return fit, X


def or_rows(fit, model: str) -> list[dict]:
    rows = []
    ci = fit.conf_int()
    for cov in fit.params.index:
        rows.append({"model": model, "name": cov, "kind": "OR",
                     "value": float(np.exp(fit.params[cov])),
                     "ci_low": float(np.exp(ci.loc[cov][0])),
                     "ci_high": float(np.exp(ci.loc[cov][1])),
                     "p": float(fit.pvalues[cov])})
    return rows


def decile_ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Equal-frequency-bin expected calibration error (weighted); per scripts/47."""
    df = pd.DataFrame({"y": y, "p": p})
    df["bin"] = pd.qcut(df["p"], q=n_bins, duplicates="drop", labels=False)
    g = df.groupby("bin").agg(n=("y", "size"), obs=("y", "mean"), exp=("p", "mean"))
    return float((g["n"] / g["n"].sum() * (g["obs"] - g["exp"]).abs()).sum())


def boot_ci(vals: list[float]) -> tuple[float, float]:
    """95% CI via the scripts/30 convention: sorted values, index 5 / -6 of 200."""
    v = np.array(sorted(vals))
    return float(v[5]), float(v[-6])


def cluster_bootstrap_auc(df: pd.DataFrame, y: np.ndarray, p_hat: np.ndarray,
                          n_boot: int = 200, seed: int = 42) -> list[float]:
    """Resample PACKAGES with replacement; score the SAME fitted model (refit-free)."""
    idx_by_pkg = df.groupby("package").indices
    pkgs = np.array(sorted(idx_by_pkg.keys()))
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_boot):
        sampled = rng.choice(pkgs, size=len(pkgs), replace=True)
        idx = np.concatenate([idx_by_pkg[p] for p in sampled])
        try:
            aucs.append(roc_auc_score(y[idx], p_hat[idx]))
        except ValueError:
            continue
    return aucs


def hosmer_lemeshow(y: np.ndarray, p: np.ndarray, n_bins: int = 10):
    """HL statistic on equal-frequency deciles (copied from scripts/30)."""
    from scipy.stats import chi2 as chi2_dist
    d = pd.DataFrame({"y": y, "p": p})
    d["bin"] = pd.qcut(d["p"], q=n_bins, duplicates="drop", labels=False)
    g = d.groupby("bin").agg(n=("y", "size"), obs=("y", "sum"), exp=("p", "sum"))
    g["var"] = g["n"] * (g["exp"] / g["n"]) * (1 - g["exp"] / g["n"])
    chi2 = float((((g["obs"] - g["exp"]) ** 2 / g["var"])
                  .replace([np.inf, -np.inf], np.nan).sum(skipna=True)))
    df_hl = len(g) - 2
    return chi2, float(1 - chi2_dist.cdf(chi2, df=df_hl)), df_hl, g


def _primary_logistic_figures(y: np.ndarray, p: np.ndarray) -> None:
    """Regenerate paper/figures ROC + HL calibration for the PRIMARY model,
    so the appendix figures match the 12-covariate primary, not scripts/30."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    sys.path.insert(0, str(Path(__file__).parent))
    import figstyle
    figstyle.apply()
    figdir = Path(__file__).parent.parent / "paper" / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    auc = roc_auc_score(y, p)
    fpr, tpr, _ = roc_curve(y, p)
    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    ax.plot(fpr, tpr, color=figstyle.ACCENT, linewidth=2,
            label=f"Logit (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color=figstyle.REF, linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right", frameon=True, framealpha=0.9, edgecolor="0.8")
    fig.tight_layout()
    fig.savefig(figdir / "logistic_roc.png")
    plt.close(fig)

    _, _, _, g = hosmer_lemeshow(y, p, n_bins=10)
    g = g.copy()
    g["obs_rate"] = g["obs"] / g["n"]
    g["exp_rate"] = g["exp"] / g["n"]
    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    ax.plot([0, 1], [0, 1], "--", color=figstyle.REF, linewidth=1,
            label="perfect calibration")
    ax.plot(g["exp_rate"], g["obs_rate"], "o-", color=figstyle.ACCENT,
            linewidth=1.5, markersize=6, label="per-decile")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed attestation rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", frameon=True, framealpha=0.9, edgecolor="0.8")
    fig.tight_layout()
    fig.savefig(figdir / "logistic_hl_calibration.png")
    plt.close(fig)
    print(f"[Part 2] Regenerated logistic_roc.png + logistic_hl_calibration.png "
          f"for the PRIMARY model", flush=True)


def part2_logistic() -> tuple[pd.DataFrame, dict]:
    prewin = prewindow_release_counts()
    df = build_version_frame(prewin)
    n_bad_rows = int(df["github_unobserved"].sum())
    sub_bad = df[df["github_unobserved"] == 1]
    n_bad_pkgs = sub_bad["package"].nunique()
    meta = {"n_rows": len(df), "n_pkgs": df["package"].nunique(),
            "base_rate": float(df["y"].mean()),
            "n_bad_pkgs_frame": n_bad_pkgs,
            "bad_row_share": n_bad_rows / len(df), "n_bad_rows": n_bad_rows,
            "bad_y1_rows": int(sub_bad["y"].sum()),
            "bad_pkgs_any_y1": int(sub_bad.groupby("package")["y"].max().sum())
            if n_bad_pkgs else 0,
            "obs_y_rate": float(df.loc[df["github_unobserved"] == 0, "y"].mean())}
    print(f"\n[Part 2] Frame: {meta['n_rows']:,} rows x {meta['n_pkgs']} pkgs "
          f"(base rate {meta['base_rate']*100:.2f}%)", flush=True)
    print(f"[Part 2] github_unobserved: {n_bad_pkgs} pkgs -> {n_bad_rows:,} rows "
          f"({meta['bad_row_share']*100:.1f}%)", flush=True)

    rows: list[dict] = []

    def metric(model: str, name: str, value: float,
               lo: float | None = None, hi: float | None = None) -> None:
        rows.append({"model": model, "name": name, "kind": "metric", "value": value,
                     "ci_low": lo, "ci_high": hi, "p": None})

    y = df["y"].to_numpy()

    # ---- PRIMARY: missingness-explicit + pre-window release opportunity ----
    fit_p, X_p = fit_logit(df, LOGIT_COLS_PRIMARY)
    p_p = np.asarray(fit_p.predict(X_p))
    auc_p = roc_auc_score(y, p_p)
    brier_p = brier_score_loss(y, p_p)
    hl_chi2, hl_p, hl_df, _ = hosmer_lemeshow(y, p_p, n_bins=10)
    ece_p = decile_ece(y, p_p)
    meta["auc_primary"] = auc_p
    meta["brier_primary"] = brier_p
    meta["hl_chi2_primary"] = hl_chi2
    meta["hl_p_primary"] = hl_p
    meta["hl_df_primary"] = hl_df
    meta["ece_primary"] = ece_p
    rows += or_rows(fit_p, "primary")
    metric("primary", "auc_in_sample", auc_p)
    metric("primary", "brier", brier_p)
    metric("primary", "hl_chi2", hl_chi2)
    metric("primary", "hl_p", hl_p)
    metric("primary", "ece_in_sample", ece_p)
    metric("primary", "n_rows", len(df))
    metric("primary", "n_pkgs", df["package"].nunique())
    _primary_logistic_figures(y, p_p)
    print(f"[Part 2] PRIMARY calibration: Brier={brier_p:.4f}, "
          f"HL chi2={hl_chi2:.2f} (df={hl_df}, p={hl_p:.2e}), "
          f"in-sample decile ECE={ece_p:.4f}", flush=True)
    ci_p = fit_p.conf_int()
    meta["gh_unobs_or"] = float(np.exp(fit_p.params["github_unobserved"]))
    meta["gh_unobs_ci"] = (float(np.exp(ci_p.loc["github_unobserved"][0])),
                           float(np.exp(ci_p.loc["github_unobserved"][1])))
    meta["gh_unobs_p"] = float(fit_p.pvalues["github_unobserved"])
    print(f"[Part 2] PRIMARY (12 covariates): AUC = {auc_p:.4f}; "
          f"github_unobserved OR = {meta['gh_unobs_or']:.3f} "
          f"[{meta['gh_unobs_ci'][0]:.3f}, {meta['gh_unobs_ci'][1]:.3f}]", flush=True)

    # ---- SENSITIVITY (i): zero-imputation spec (scripts/30 reproduction) ----
    fit_z, X_z = fit_logit(df, LOGIT_COLS)
    p_z = np.asarray(fit_z.predict(X_z))
    auc_z = roc_auc_score(y, p_z)
    meta["auc_zerofill"] = auc_z
    rows += or_rows(fit_z, "zerofill_sensitivity")
    metric("zerofill_sensitivity", "auc_in_sample", auc_z)
    print(f"[Part 2i] zero-imputation sensitivity (scripts/30 spec): AUC = {auc_z:.4f}",
          flush=True)

    # ---- SENSITIVITY (ii): primary minus release frequency ----
    fit_a, _ = fit_logit(df, LOGIT_COLS + ["github_unobserved"])
    rows += or_rows(fit_a, "no_release_freq_sensitivity")

    # ---- SENSITIVITY (iii): complete-case (GitHub-observed rows only) ----
    df_cc = df[df["github_unobserved"] == 0].reset_index(drop=True)
    cc_cols = [c for c in LOGIT_COLS_PRIMARY if c != "github_unobserved"]
    fit_b, X_b = fit_logit(df_cc, cc_cols)
    p_b = np.asarray(fit_b.predict(X_b))
    auc_b = roc_auc_score(df_cc["y"].to_numpy(), p_b)
    meta["auc_cc"] = auc_b
    meta["n_cc"] = len(df_cc)
    rows += or_rows(fit_b, "complete_case")
    metric("complete_case", "auc_in_sample", auc_b)
    metric("complete_case", "n_rows", len(df_cc))
    metric("complete_case", "n_pkgs", df_cc["package"].nunique())
    print(f"[Part 2iii] complete-case: {len(df_cc):,} rows, AUC = {auc_b:.4f}", flush=True)

    # ---- key-OR comparison print (primary vs zerofill vs complete-case) ----
    print(f"[Part 2] Key ORs (primary vs zerofill vs complete-case):", flush=True)
    for cov in KEY_ORS:
        vals = []
        for f in (fit_p, fit_z, fit_b):
            vals.append(f"{float(np.exp(f.params[cov])):7.3f}"
                        if cov in f.params.index else "      —")
        print(f"    {cov:24s} primary={vals[0]}  zerofill={vals[1]}  cc={vals[2]}",
              flush=True)

    # ---- (c) package-disjoint holdout on the PRIMARY spec ----
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    tr_idx, te_idx = next(gss.split(df, groups=df["package"]))
    df_tr = df.iloc[tr_idx].reset_index(drop=True)
    df_te = df.iloc[te_idx].reset_index(drop=True)
    fit_c, X_tr = fit_logit(df_tr, LOGIT_COLS_PRIMARY)
    p_tr = np.asarray(fit_c.predict(X_tr))
    X_te = sm.add_constant(df_te[LOGIT_COLS_PRIMARY].copy())
    p_te = np.asarray(fit_c.predict(X_te))
    y_te = df_te["y"].to_numpy()
    auc_tr = roc_auc_score(df_tr["y"].to_numpy(), p_tr)
    auc_te = roc_auc_score(y_te, p_te)
    ece_te = decile_ece(y_te, p_te)
    meta.update({"auc_holdout_train": auc_tr, "auc_holdout_test": auc_te,
                 "ece_holdout_test": ece_te,
                 "n_train": len(df_tr), "n_test": len(df_te),
                 "pkgs_train": df_tr["package"].nunique(),
                 "pkgs_test": df_te["package"].nunique()})
    metric("grouped_holdout", "auc_train", auc_tr)
    metric("grouped_holdout", "auc_test", auc_te)
    metric("grouped_holdout", "ece_test_decile10", ece_te)
    print(f"[Part 2c] grouped holdout (primary spec): "
          f"train AUC = {auc_tr:.4f}, TEST AUC = {auc_te:.4f}, "
          f"test decile ECE = {ece_te:.4f}", flush=True)

    # ---- (c-bis) REPEATED grouped splits: AUC mean + percentile band ----
    rep = GroupShuffleSplit(n_splits=20, test_size=0.3, random_state=7)
    rep_aucs: list[float] = []
    for tr_i, te_i in rep.split(df, groups=df["package"]):
        d_tr, d_te = df.iloc[tr_i], df.iloc[te_i]
        try:
            f_r, _ = fit_logit(d_tr.reset_index(drop=True), LOGIT_COLS_PRIMARY)
            X_r = sm.add_constant(d_te[LOGIT_COLS_PRIMARY].copy())
            rep_aucs.append(roc_auc_score(d_te["y"].to_numpy(),
                                          np.asarray(f_r.predict(X_r))))
        except Exception:
            continue
    rep_arr = np.sort(np.array(rep_aucs))
    meta["rep_holdout_mean"] = float(rep_arr.mean())
    meta["rep_holdout_lo"] = float(np.percentile(rep_arr, 2.5))
    meta["rep_holdout_hi"] = float(np.percentile(rep_arr, 97.5))
    meta["rep_holdout_n"] = len(rep_arr)
    metric("grouped_holdout", "auc_test_repeated_mean", meta["rep_holdout_mean"],
           meta["rep_holdout_lo"], meta["rep_holdout_hi"])
    print(f"[Part 2c-bis] repeated grouped splits (n={len(rep_arr)}): "
          f"test AUC mean = {meta['rep_holdout_mean']:.4f} "
          f"[{meta['rep_holdout_lo']:.4f}, {meta['rep_holdout_hi']:.4f}]", flush=True)

    # ---- (d) package-cluster bootstrap AUC CI (primary model, refit-free) ----
    cl_aucs = cluster_bootstrap_auc(df, y, p_p, n_boot=200, seed=42)
    cl_lo, cl_hi = boot_ci(cl_aucs)
    meta["cluster_boot_ci"] = (cl_lo, cl_hi)
    rng = np.random.default_rng(42)
    row_aucs = []
    n = len(y)
    for _ in range(200):
        idx = rng.integers(0, n, n)
        try:
            row_aucs.append(roc_auc_score(y[idx], p_p[idx]))
        except ValueError:
            continue
    row_lo, row_hi = boot_ci(row_aucs)
    meta["row_boot_ci"] = (row_lo, row_hi)
    metric("primary", "auc_cluster_boot_ci", auc_p, cl_lo, cl_hi)
    metric("primary", "auc_row_boot_ci", auc_p, row_lo, row_hi)
    print(f"[Part 2d] package-cluster bootstrap AUC 95% CI (B=200, seed 42): "
          f"[{cl_lo:.4f}, {cl_hi:.4f}]", flush=True)
    print(f"[Part 2d] naive row bootstrap AUC 95% CI: [{row_lo:.4f}, {row_hi:.4f}]",
          flush=True)

    tab = pd.DataFrame(rows)
    tab.to_csv(OUT_LOG_CSV, index=False)
    print(f"[Part 2] Saved {OUT_LOG_CSV}", flush=True)
    meta["fits"] = {"primary": fit_p, "zerofill": fit_z, "cc": fit_b}
    return tab, meta


# ------------------------------------------------------------------ reporting
def _star(p: float) -> str:
    return "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else ""))


def write_report(cox_tab: pd.DataFrame, cox_sens: pd.DataFrame,
                 cox_meta: dict, log_meta: dict) -> None:
    f_p, f_z, f_b = (log_meta["fits"][k] for k in ("primary", "zerofill", "cc"))
    sens_ix = cox_sens.set_index("covariate")
    md = [
        "# Pre-submission — RQ3 batch (Batch-8 primary Cox + primary logistic)",
        "",
        "**Date**: 2026-07-11 (Batch-8; frozen rows only)  ",
        "**Script**: `scripts/52_rq3_sensitivity.py`",
        "",
        "## Part 1 — Cox PRIMARY: incident cohort, delayed entry, 11 covariates",
        "",
        f"Incident cohort (never attested before 2025-08-01): "
        f"n = {cox_meta['n']}, events = {cox_meta['events']} "
        f"(full cohort n = {cox_meta['n_full']}, events = {cox_meta['events_full']}, "
        f"prevalent excluded = {cox_meta['n_prevalent']}). "
        f"Delayed entry at first in-window release: median entry "
        f"{cox_meta['entry_median_days']:.1f} days after window start; "
        f"{cox_meta['n_entry_tie']} packages whose entry ties its endpoint "
        f"(85 events + 1 final-day censored) enter at duration − 0.5 d. "
        f"Concordance = {cox_meta['c_index']:.3f}. Robust SE; Schoenfeld "
        "(auxiliary entry-at-start fit) = rank time-transform.",
        "",
        "| Covariate | HR | 95% CI | p | Schoenfeld p (aux.) | entry-at-start HR (11-cov) |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for _, r in cox_tab.iterrows():
        sig = _star(r["p"])
        ph_flag = " (fails PH)" if r["schoenfeld_p"] < 0.05 else ""
        s_hr = (f"{sens_ix.loc[r['covariate'], 'HR']:.3f}"
                f"{_star(float(sens_ix.loc[r['covariate'], 'p']))}"
                if r["covariate"] in sens_ix.index else "—")
        md.append(f"| `{r['covariate']}` | {r['HR']:.3f}{sig} | "
                  f"[{r['HR_ci_low']:.3f}, {r['HR_ci_high']:.3f}] | {r['p']:.4f} | "
                  f"{r['schoenfeld_p']:.3f}{ph_flag} | {s_hr} |")
    n_fail = int((cox_tab["schoenfeld_p"] < 0.05).sum())
    md += ["",
           f"PH compliance: {n_fail} of {len(cox_tab)} covariates fail Schoenfeld "
           "(p < 0.05) in the auxiliary entry-at-window-start fit (lifelines has no "
           "left-truncated residuals, so the delayed-entry primary is not directly tested).",
           "",
           "## Part 2 — version-level logistic (PRIMARY = missingness-explicit "
           "+ release frequency)",
           "",
           f"Frame: {log_meta['n_rows']:,} rows x {log_meta['n_pkgs']} packages; "
           f"primary in-sample AUC = {log_meta['auc_primary']:.4f} "
           f"(zero-imputation sensitivity {log_meta['auc_zerofill']:.4f}; "
           f"complete-case {log_meta['auc_cc']:.4f} on {log_meta['n_cc']:,} rows).",
           "",
           f"`github_unobserved` OR = {log_meta['gh_unobs_or']:.4f} "
           f"[{log_meta['gh_unobs_ci'][0]:.4f}, {log_meta['gh_unobs_ci'][1]:.4f}], "
           f"p = {log_meta['gh_unobs_p']:.3g}. Near-separating: "
           f"{log_meta['bad_y1_rows']} of {log_meta['n_bad_rows']:,} "
           f"unobserved-GitHub rows attested "
           f"({log_meta['bad_y1_rows']/max(1,log_meta['n_bad_rows'])*100:.2f}% vs "
           f"{log_meta['obs_y_rate']*100:.2f}% among observed) — read qualitatively.",
           "",
           "### Key ORs across specifications (*** p<0.01, ** p<0.05, * p<0.10)",
           "",
           "| Covariate | PRIMARY | zero-imputation sens. | complete-case |",
           "|---|---:|---:|---:|",
           ]
    for cov in KEY_ORS:
        cells = []
        for f in (f_p, f_z, f_b):
            cells.append(f"{float(np.exp(f.params[cov])):.3f}{_star(f.pvalues[cov])}"
                         if cov in f.params.index else "—")
        md.append(f"| `{cov}` | {cells[0]} | {cells[1]} | {cells[2]} |")
    md += [
        "",
        "### Package-disjoint holdout (primary spec)",
        "",
        f"- Single split (test 0.3, seed 42): test AUC = "
        f"{log_meta['auc_holdout_test']:.4f} "
        f"(train {log_meta['auc_holdout_train']:.4f}); "
        f"test decile ECE = {log_meta['ece_holdout_test']:.4f}",
        f"- Repeated grouped splits (n={log_meta['rep_holdout_n']}): "
        f"test AUC mean = {log_meta['rep_holdout_mean']:.4f} "
        f"[{log_meta['rep_holdout_lo']:.4f}, {log_meta['rep_holdout_hi']:.4f}] "
        "(2.5/97.5 percentiles)",
        "",
        "### Package-cluster bootstrap AUC CI (primary, refit-free, B=200, seed 42)",
        "",
        f"- Cluster (package) bootstrap 95% CI: "
        f"[{log_meta['cluster_boot_ci'][0]:.4f}, {log_meta['cluster_boot_ci'][1]:.4f}]",
        f"- Naive row bootstrap 95% CI (contrast only): "
        f"[{log_meta['row_boot_ci'][0]:.4f}, {log_meta['row_boot_ci'][1]:.4f}] "
        "— within-package dependence makes the row CI misleadingly narrow.",
        "",
        "## Reading (firewall-compliant)",
        "",
        "All quantities are descriptive/predictive associations under the",
        "anti-causal firewall; nothing here identifies a Dec-9 causal effect.",
        "Delayed entry + release frequency model the RELEASE-OPPORTUNITY process;",
        "governance coefficients should be read net of it. Deps.dev primary",
        "tier frame; 62 missing-dd packages excluded; data cutoff 2026-05-10.",
    ]
    OUT_MD.write_text("\n".join(md) + "\n")
    print(f"\n✓ Saved {OUT_MD}", flush=True)


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    cox_tab, cox_sens, cox_meta = part1_cox_incident()
    log_tab, log_meta = part2_logistic()
    write_report(cox_tab, cox_sens, cox_meta, log_meta)
    print(f"✓ Saved {OUT_COX_CSV}", flush=True)
    print(f"✓ Saved {OUT_LOG_CSV}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
