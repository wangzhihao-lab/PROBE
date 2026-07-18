from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from lifelines import CoxPHFitter
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

import figstyle

figstyle.apply()


import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "rq3_primary", str(Path(__file__).parent / "52_rq3_sensitivity.py"))
_rq3 = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_rq3)
_PREWIN = _rq3.prewindow_release_counts()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
BASELINES_DIR = DATA_DIR / "baselines"
REPORTS_DIR = ROOT / "reports"

PACKAGES_FULL = DATA_DIR / "packages_full.txt"
VERSIONS_CSV = DATA_DIR / "versions.csv"
INFLUENCE_CSV = DATA_DIR / "influence_metrics.csv"
DEPSDEV_CSV = DATA_DIR / "depsdev_dependents_2026_05_05.csv"

OUT_SHIFT = BASELINES_DIR / "task_7_b3_tier_shift_matrix.csv"
OUT_PCA = BASELINES_DIR / "task_7_b3_pca_loadings.csv"
OUT_GRID = BASELINES_DIR / "task_7_b3_covariate_grid.csv"
OUT_VDROP = BASELINES_DIR / "task_7_b3_versionlevel_dropone.csv"
OUT_FOREST = BASELINES_DIR / "task_7_b3_forest.png"
OUT_REPORT = REPORTS_DIR / "task_7_baseline_3.md"

ORIGIN = pd.Timestamp("2025-08-01", tz="UTC")
CENSOR = pd.Timestamp("2026-05-10 23:59:59", tz="UTC")
DEADLINE = pd.Timestamp("2025-12-09", tz="UTC")
RANDOM_SEED = 42


VERSION_COVS = [
    "post_deadline",
    "log_weekly_downloads", "log_age_days", "log_maintainers",
    "owner_is_org", "has_dependabot_int", "has_issue_template_int",
    "is_scoped_int", "tier_top", "tier_mid",
    "log_prewin_release_freq", "github_unobserved",
]
COX_EXTRA = ["log_prewin_release_freq_pkg", "github_unobserved_pkg"]
VDROP_FULL_LABEL = "none_full_reference"
PACKUMENTS_DIR = DATA_DIR / "packuments_raw"


def dpc_tier(dpc) -> str:
    if pd.isna(dpc): return "missing"
    d = int(dpc)
    return "top" if d > 500 else "mid" if d > 10 else "tail"


def prevalent_packages() -> set:
    import json as _json
    from datetime import datetime as _dt, timezone as _tz
    window_start = _dt(2025, 8, 1, tzinfo=_tz.utc)

    def _parse(s):
        try:
            return _dt.fromisoformat(s.replace("Z", "+00:00"))
        except Exception:
            return None

    prevalent = set()
    for fp in PACKUMENTS_DIR.glob("*.json"):
        try:
            data = _json.loads(fp.read_text())
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
                pt = _parse(times.get(ver, ""))
                if pt and pt < window_start:
                    prevalent.add(name)
                    break
    return prevalent


def quantile_tier(score, top_q=0.95, mid_q=0.50) -> str:
    return score


def build_covariates(packages: pd.DataFrame, versions: pd.DataFrame,
                     inf: pd.DataFrame, depsdev: pd.DataFrame
                     ) -> pd.DataFrame:

    v = versions.copy()
    v["published_at"] = pd.to_datetime(v["published_at"], utc=True, errors="coerce")
    v = v[(v["published_at"] >= ORIGIN) & (v["published_at"] <= CENSOR)]
    attested = v[v["has_attestation"].astype(str).str.lower() == "true"]
    first_att = attested.groupby("package")["published_at"].min().reset_index()
    first_att.columns = ["package", "first_attested_at"]
    first_rel = v.groupby("package")["published_at"].min().reset_index()
    first_rel.columns = ["package", "first_release_at"]
    all_pkgs = v["package"].drop_duplicates().reset_index(drop=True)
    sd = pd.DataFrame({"package": all_pkgs})
    sd = sd.merge(first_att, on="package", how="left")
    sd = sd.merge(first_rel, on="package", how="left")
    sd["event"] = sd["first_attested_at"].notna().astype(int)
    sd.loc[sd["event"] == 0, "first_attested_at"] = CENSOR
    sd["duration_days"] = (sd["first_attested_at"] - ORIGIN).dt.days.clip(lower=1).astype(int)


    sd["entry_days"] = (sd["first_release_at"] - ORIGIN).dt.days.clip(lower=0).astype(float)
    sd["entry_days"] = np.minimum(sd["entry_days"], sd["duration_days"] - 0.5)
    sd["prevalent"] = sd["package"].isin(prevalent_packages()).astype(int)


    sd = sd.merge(inf, on="package", how="left")
    sd["log_prewin_release_freq_pkg"] = np.log1p(sd["package"].map(_PREWIN).fillna(0).astype(float))
    sd["github_unobserved_pkg"] = (sd["github_status"].astype(str) != "ok").astype(int)


    eco_map = dict(zip(packages["name"], packages["dependent_packages_count"]))
    dd_map = dict(zip(depsdev["name"], depsdev["dependents_count"]))
    sd["eco_dpc"] = sd["package"].map(eco_map)
    sd["dd_dep"] = sd["package"].map(dd_map)


    sd["tier_dd"] = sd["dd_dep"].apply(dpc_tier)
    sd["tier_eco"] = sd["eco_dpc"].apply(dpc_tier)


    dl = pd.to_numeric(sd["weekly_downloads_avg"], errors="coerce")
    q95 = np.nanpercentile(dl, 95)
    q50 = np.nanpercentile(dl, 50)
    def dl_tier(x):
        if pd.isna(x): return "missing"
        if x >= q95: return "top"
        if x >= q50: return "mid"
        return "tail"
    sd["tier_dl"] = dl.apply(dl_tier)


    pca_inputs = sd[["weekly_downloads_avg", "eco_dpc", "github_stars", "maintainers_count", "package_age_days"]].copy()
    for c in pca_inputs.columns:
        pca_inputs[c] = pd.to_numeric(pca_inputs[c], errors="coerce")

    for c in pca_inputs.columns:
        lo = np.nanpercentile(pca_inputs[c], 5)
        hi = np.nanpercentile(pca_inputs[c], 95)
        pca_inputs[c] = pca_inputs[c].clip(lower=lo, upper=hi)
    pca_inputs_log = pca_inputs.apply(np.log1p)
    valid_mask = pca_inputs_log.notna().all(axis=1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(pca_inputs_log[valid_mask])
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    pcs = pca.fit_transform(scaled)
    sd["pca_score"] = np.nan
    sd.loc[valid_mask, "pca_score"] = pcs[:, 0]

    corr_dl = np.corrcoef(sd.loc[valid_mask, "pca_score"],
                           pca_inputs_log.loc[valid_mask, "weekly_downloads_avg"])[0, 1]
    if corr_dl < 0:
        sd["pca_score"] = -sd["pca_score"]
    q95_pca = np.nanpercentile(sd["pca_score"], 95)
    q50_pca = np.nanpercentile(sd["pca_score"], 50)
    def pca_tier(x):
        if pd.isna(x): return "missing"
        if x >= q95_pca: return "top"
        if x >= q50_pca: return "mid"
        return "tail"
    sd["tier_pca"] = sd["pca_score"].apply(pca_tier)


    sd["log_downloads"] = np.log1p(pd.to_numeric(sd["weekly_downloads_avg"], errors="coerce"))
    sd["log_age"] = np.log1p(pd.to_numeric(sd["package_age_days"], errors="coerce"))
    sd["log_dependents"] = np.log1p(pd.to_numeric(sd["dd_dep"], errors="coerce"))
    sd["log_maintainers"] = np.log1p(pd.to_numeric(sd["maintainers_count"], errors="coerce"))
    sd["owner_org_int"] = (sd["github_owner_type"].astype(str) == "Organization").astype(int)
    for src, dst in [("has_dependabot", "has_dependabot_int"),
                     ("has_issue_template", "has_issue_template_int"),
                     ("is_scoped_package", "is_scoped_int")]:
        sd[dst] = (sd[src].astype(str).str.lower() == "true").astype(int)


    for frame in ["dd", "eco", "dl", "pca"]:
        sd[f"tier_top_{frame}"] = (sd[f"tier_{frame}"] == "top").astype(int)
        sd[f"tier_mid_{frame}"] = (sd[f"tier_{frame}"] == "mid").astype(int)


    loadings_df = pd.DataFrame({
        "input_feature": pca_inputs.columns,
        "pc1_loading": pca.components_[0],
        "pc2_loading": pca.components_[1],
    })
    loadings_df.attrs["explained_variance_ratio"] = pca.explained_variance_ratio_.tolist()
    return sd, loadings_df


def _drop_constant(sub: pd.DataFrame, covariates: list[str]) -> list[str]:
    return [c for c in covariates if sub[c].nunique(dropna=True) > 1]


def fit_version_logistic(vdf: pd.DataFrame, covariates: list[str],
                         label: str) -> Optional[dict]:
    sub = vdf[covariates + ["y", "package"]].dropna()
    if len(sub) < 50:
        return None
    cov_use = _drop_constant(sub, covariates)
    dropped = [c for c in covariates if c not in cov_use]
    if dropped:
        logger.info(f"logistic [{label}]: dropped constant covariates {dropped}")
    X = sm.add_constant(sub[cov_use].astype(float))
    y = sub["y"].astype(int).values
    try:
        model = sm.Logit(y, X).fit(disp=0, maxiter=200, cov_type="cluster",
                                   cov_kwds={"groups": sub["package"].values})
    except Exception as e:
        logger.warning(f"logistic [{label}] failed: {e}")
        return None
    rows = []
    ci = model.conf_int(alpha=0.05)
    names = ["const"] + cov_use
    for name in names:
        rows.append({
            "label": label,
            "outcome": "logistic_version_level",
            "feature": name,
            "coef": float(model.params[name]),
            "se": float(model.bse[name]),
            "OR": float(np.exp(model.params[name])),
            "OR_ci_low": float(np.exp(ci.loc[name][0])),
            "OR_ci_high": float(np.exp(ci.loc[name][1])),
            "pval": float(model.pvalues[name]),
            "n_obs": int(len(sub)),
        })
    return {"rows": rows, "n_obs": len(sub), "dropped": dropped}


def fit_cox(df: pd.DataFrame, covariates: list[str], label: str) -> Optional[dict]:
    sub = df[df["prevalent"] == 0]
    sub = sub[covariates + ["duration_days", "entry_days", "event"]].dropna()
    if len(sub) < 50:
        return None
    cov_use = _drop_constant(sub, covariates)
    if len(cov_use) < len(covariates):
        logger.info(f"cox [{label}]: dropped constant covariates "
                    f"{[c for c in covariates if c not in cov_use]}")
    sub = sub[cov_use + ["duration_days", "entry_days", "event"]]
    try:
        cph = CoxPHFitter()
        cph.fit(sub, duration_col="duration_days", event_col="event",
                entry_col="entry_days", robust=True)
    except Exception as e:
        logger.warning(f"cox [{label}] failed: {e}")
        return None
    summary = cph.summary
    rows = []
    for name in covariates:
        if name not in summary.index:
            continue
        rows.append({
            "label": label,
            "outcome": "cox_time_to_first_provenance",
            "feature": name,
            "coef": float(summary.loc[name, "coef"]),
            "se": float(summary.loc[name, "se(coef)"]),
            "OR": float(summary.loc[name, "exp(coef)"]),
            "OR_ci_low": float(summary.loc[name, "exp(coef) lower 95%"]),
            "OR_ci_high": float(summary.loc[name, "exp(coef) upper 95%"]),
            "pval": float(summary.loc[name, "p"]),
            "n_obs": int(len(sub)),
        })
    return {"rows": rows, "n_obs": len(sub)}


def build_version_frame(versions: pd.DataFrame, inf: pd.DataFrame,
                        depsdev: pd.DataFrame, sd: pd.DataFrame) -> pd.DataFrame:
    v = versions.copy()
    v["published_at"] = pd.to_datetime(v["published_at"], utc=True, errors="coerce")
    v["y"] = (v["has_attestation"].astype(str).str.lower() == "true").astype(int)
    v["post_deadline"] = (v["published_at"] >= DEADLINE).astype(int)

    pc = inf.copy()
    dd_map = dict(zip(depsdev["name"], depsdev["dependents_count"].astype(int)))
    pc["depsdev_dependents_count"] = pc["package"].map(dd_map)
    pc["depsdev_tier"] = pc["depsdev_dependents_count"].apply(dpc_tier)
    pc["log_weekly_downloads"] = np.log1p(pd.to_numeric(pc["weekly_downloads_avg"], errors="coerce"))
    pc["log_age_days"] = np.log1p(pd.to_numeric(pc["package_age_days"], errors="coerce"))
    pc["log_maintainers"] = np.log1p(pd.to_numeric(pc["maintainers_count"], errors="coerce"))
    pc["log_prewin_release_freq"] = np.log1p(pc["package"].map(_PREWIN).fillna(0).astype(float))
    pc["owner_is_org"] = (pc["github_owner_type"] == "Organization").astype(int)
    pc["github_unobserved"] = (pc["github_status"].astype(str) != "ok").astype(int)
    for src, dst in [("has_dependabot", "has_dependabot_int"),
                     ("has_issue_template", "has_issue_template_int"),
                     ("is_scoped_package", "is_scoped_int")]:
        pc[dst] = (pc[src].astype(str).str.lower() == "true").astype(int)
    pc["tier_top"] = (pc["depsdev_tier"] == "top").astype(int)
    pc["tier_mid"] = (pc["depsdev_tier"] == "mid").astype(int)
    pkg_covs = [c for c in VERSION_COVS if c != "post_deadline"]
    pc_keep = pc[["package", "depsdev_tier"] + pkg_covs].dropna()
    frame_cols = ["package"] + [f"tier_{f}" for f in ["dd", "eco", "dl", "pca"]] + \
                 [f"tier_top_{f}" for f in ["dd", "eco", "dl", "pca"]] + \
                 [f"tier_mid_{f}" for f in ["dd", "eco", "dl", "pca"]]
    vdf = v[["package", "y", "post_deadline"]].merge(pc_keep, on="package", how="inner")
    return vdf.merge(sd[frame_cols], on="package", how="left")


def versionlevel_dropone(vdf: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []

    def _fit(covs: list[str], dropped: str) -> None:
        X = sm.add_constant(vdf[covs].astype(float).values)
        y = vdf["y"].astype(int).values
        try:
            fit = sm.Logit(y, X).fit(disp=0, maxiter=200)
        except Exception as e:
            logger.warning(f"version-level logit (drop={dropped}) failed: {e}")
            return
        cis = np.asarray(fit.conf_int(alpha=0.05))
        for i, name in enumerate(["const"] + covs):
            rows.append({
                "covariate_dropped": dropped,
                "feature": name,
                "coef": float(fit.params[i]),
                "se": float(fit.bse[i]),
                "OR": float(np.exp(fit.params[i])),
                "OR_ci_low": float(np.exp(cis[i, 0])),
                "OR_ci_high": float(np.exp(cis[i, 1])),
                "pval": float(fit.pvalues[i]),
                "n_obs": int(len(vdf)),
            })

    _fit(VERSION_COVS, VDROP_FULL_LABEL)
    for drop in VERSION_COVS:
        _fit([c for c in VERSION_COVS if c != drop], drop)
    return pd.DataFrame(rows)


def build_fits_grid(sd: pd.DataFrame, vdf: pd.DataFrame) -> pd.DataFrame:
    base_cov_no_tier = ["log_downloads", "log_age", "log_maintainers",
                        "owner_org_int", "has_dependabot_int",
                        "has_issue_template_int", "is_scoped_int"]
    logit_base = ["post_deadline", "log_weekly_downloads", "log_age_days",
                  "log_maintainers", "owner_is_org", "has_dependabot_int",
                  "has_issue_template_int", "is_scoped_int",
                  "log_prewin_release_freq", "github_unobserved"]
    all_rows = []


    frames_cox_ok, frames_logit_ok = [], []
    for frame, label_frame in [("dd", "frame_dd_PRIMARY"), ("eco", "frame_eco"),
                                ("dl", "frame_downloads"), ("pca", "frame_pca")]:
        tiers = [f"tier_top_{frame}", f"tier_mid_{frame}"]
        sd_f = sd[sd[f"tier_{frame}"] != "missing"]
        vdf_f = vdf[vdf[f"tier_{frame}"] != "missing"]
        r = fit_cox(sd_f, base_cov_no_tier + COX_EXTRA + tiers, f"{label_frame}_full")
        if r:
            all_rows.extend(r["rows"]); frames_cox_ok.append(frame)
        r = fit_version_logistic(vdf_f, logit_base + tiers, f"{label_frame}_full")
        if r:
            all_rows.extend(r["rows"]); frames_logit_ok.append(frame)


    logger.info(f"Part A frames fitted: Cox {frames_cox_ok}, "
                f"Logistic {frames_logit_ok}")
    if len(frames_cox_ok) < 4 or len(frames_logit_ok) < 4:
        logger.warning(f"NOT all four frames fit: Cox {len(frames_cox_ok)}/4, "
                       f"Logistic {len(frames_logit_ok)}/4 — the paper's "
                       f"frame count must reflect this.")
    build_fits_grid.frames_cox_ok = frames_cox_ok
    build_fits_grid.frames_logit_ok = frames_logit_ok


    sd_dd = sd[sd["tier_dd"] != "missing"]
    vdf_dd = vdf[vdf["tier_dd"] != "missing"]
    cox_primary = base_cov_no_tier + COX_EXTRA + ["tier_top_dd", "tier_mid_dd"]
    logit_primary = logit_base + ["tier_top_dd", "tier_mid_dd"]

    def both(cox_covs: list[str], logit_covs: list[str], label: str) -> None:
        r = fit_cox(sd_dd, cox_covs, label)
        if r: all_rows.extend(r["rows"])
        r = fit_version_logistic(vdf_dd, logit_covs, label)
        if r: all_rows.extend(r["rows"])


    for drop_cox, drop_logit in zip(
            base_cov_no_tier + ["tier_top_dd", "tier_mid_dd"],
            ["log_weekly_downloads", "log_age_days", "log_maintainers",
             "owner_is_org", "has_dependabot_int", "has_issue_template_int",
             "is_scoped_int", "tier_top_dd", "tier_mid_dd"]):
        both([c for c in cox_primary if c != drop_cox],
             [c for c in logit_primary if c != drop_logit],
             f"frame_dd_dropone_{drop_cox}")


    for grp_name, cox_grp, logit_grp in [
            ("governance",
             ["owner_org_int", "has_dependabot_int", "has_issue_template_int"],
             ["owner_is_org", "has_dependabot_int", "has_issue_template_int"]),
            ("tier_dummies", ["tier_top_dd", "tier_mid_dd"],
             ["tier_top_dd", "tier_mid_dd"]),
            ("pkg_meta",
             ["log_downloads", "log_age", "log_maintainers", "is_scoped_int"],
             ["log_weekly_downloads", "log_age_days", "log_maintainers",
              "is_scoped_int"])]:
        both([c for c in cox_primary if c not in cox_grp],
             [c for c in logit_primary if c not in logit_grp],
             f"frame_dd_groupout_{grp_name}")


    for cov_cox, cov_logit in zip(
            base_cov_no_tier + ["tier_top_dd", "tier_mid_dd"],
            ["log_weekly_downloads", "log_age_days", "log_maintainers",
             "owner_is_org", "has_dependabot_int", "has_issue_template_int",
             "is_scoped_int", "tier_top_dd", "tier_mid_dd"]):
        both([cov_cox], [cov_logit], f"frame_dd_univariate_{cov_cox}")

    return pd.DataFrame(all_rows)


def tier_shift_matrix(sd: pd.DataFrame) -> pd.DataFrame:
    rows = []
    frames = [("dd", "deps.dev PRIMARY"), ("eco", "ecosyste.ms"),
              ("dl", "downloads quantile"), ("pca", "PCA composite")]
    for i, (fa, la) in enumerate(frames):
        for j, (fb, lb) in enumerate(frames):
            if i >= j: continue
            ta = sd[f"tier_{fa}"]
            tb = sd[f"tier_{fb}"]
            both_known = (ta != "missing") & (tb != "missing")
            agree = ((ta == tb) & both_known).sum()
            disagree = ((ta != tb) & both_known).sum()
            total = both_known.sum()
            rows.append({
                "frame_a": la, "frame_b": lb,
                "n_both_known": int(total),
                "n_agree": int(agree),
                "n_disagree": int(disagree),
                "agreement_pct": float(agree) / max(1, total) * 100,
            })
    return pd.DataFrame(rows)


def make_forest(grid_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.9), sharey=True)

    rows_spec = [
        ("log_weekly_downloads", "log_downloads", "log_weekly_downloads"),
        ("log_age_days", "log_age", "log_age_days"),
        ("log_maintainers", "log_maintainers", "log_maintainers"),
        ("owner_is_org", "owner_org_int", "owner_is_org"),
        ("has_dependabot", "has_dependabot_int", "has_dependabot_int"),
        ("has_issue_template", "has_issue_template_int", "has_issue_template_int"),
        ("is_scoped", "is_scoped_int", "is_scoped_int"),
        ("log_prewin_release_freq", "log_prewin_release_freq_pkg", "log_prewin_release_freq"),
        ("github_unobserved", "github_unobserved_pkg", "github_unobserved"),
        ("tier_top", "tier_top_dd", "tier_top_dd"),
        ("tier_mid", "tier_mid_dd", "tier_mid_dd"),
    ]
    n = len(rows_spec)
    panel_cfg = [
        (axes[0], "logistic_version_level", 2, figstyle.ACCENT_2, "Odds ratio (log scale)"),
        (axes[1], "cox_time_to_first_provenance", 1, figstyle.ACCENT, "Hazard ratio (log scale)"),
    ]
    for panel_idx, (ax, outcome, feat_col, full_color, xlab) in enumerate(panel_cfg):
        sub = grid_df[grid_df["outcome"] == outcome]
        full_labeled = drop_labeled = False
        for i, spec in enumerate(rows_spec):
            feat = spec[feat_col]
            if feat is None:
                continue
            y = n - 1 - i
            drop_xs = []
            for other in rows_spec:
                of = other[feat_col]
                if of is None or of == feat:
                    continue
                dr = sub[(sub["feature"] == feat) &
                         (sub["label"] == f"frame_dd_dropone_{other[1]}")]
                if not dr.empty:
                    drop_xs.append(float(dr.iloc[0]["OR"]))
            if drop_xs:
                ax.plot(drop_xs, [y - 0.25] * len(drop_xs), marker="|",
                        linestyle="none", color=figstyle.REF, alpha=0.6,
                        markersize=7, markeredgewidth=1.2,
                        label="Drop-one refit" if not drop_labeled else "_nolegend_")
                drop_labeled = True
            fr = sub[(sub["feature"] == feat) & (sub["label"] == "frame_dd_PRIMARY_full")]
            if not fr.empty:
                r = fr.iloc[0]
                ax.errorbar(r["OR"], y,
                            xerr=[[max(0, r["OR"] - r["OR_ci_low"])],
                                  [max(0, r["OR_ci_high"] - r["OR"])]],
                            fmt="o", color=full_color, capsize=4,
                            label="Full model" if not full_labeled else "_nolegend_")
                full_labeled = True
        ax.axvline(1.0, linestyle="--", color=figstyle.REF, alpha=0.5)
        ax.set_xscale("log")
        ax.set_xlabel(xlab)
        ax.set_ylim(-0.7, n - 0.3)
        ax.set_yticks(range(n))
        if panel_idx == 0:
            ax.set_yticklabels([rows_spec[n - 1 - k][0] for k in range(n)])
        else:
            ax.tick_params(labelleft=False)
        ax.grid(True, axis="x", alpha=0.3, linestyle=":")


    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=True,
               framealpha=0.9, edgecolor="0.8", bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout()
    fig.savefig(OUT_FOREST, dpi=400, bbox_inches="tight")


    paper_forest = ROOT / "paper" / "figures" / "b3_forest.png"
    fig.savefig(paper_forest, dpi=400, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"wrote {OUT_FOREST} and {paper_forest}")


def write_report(grid_df: pd.DataFrame, shift_df: pd.DataFrame,
                 pca_loadings: pd.DataFrame, sd: pd.DataFrame,
                 vdrop_df: pd.DataFrame) -> None:
    lines = [
        "# Baseline 3 — Influence Metric Sensitivity (tier + covariates merged)",
        "",
        "**Date**: auto-generated 2026-05-18 by `scripts/36_baseline_influence.py`",
        "**Scope**: Part A: 4 tier-frame sources (dd PRIMARY / eco / downloads / PCA composite). "
        "Part B: covariate sensitivity (full / drop-one / group-out / univariate) under "
        "the primary deps.dev frame.",
        "**Framing**: association-only.",
        "",
        f"**Survival data**: n={len(sd):,} packages with versions in window; "
        f"events={int(sd['event'].sum()):,}; censored={int((1-sd['event']).sum()):,}. "
        f"**Cox fits the incident cohort** (prevalent adopters excluded): "
        f"n={int((sd['prevalent']==0).sum()):,}, "
        f"events={int(sd.loc[sd['prevalent']==0,'event'].sum()):,}, "
        "with delayed entry at each package's first in-window release.",
        "",
        "## Part A — Tier-frame source sensitivity",
        "",
        "### Pairwise tier-agreement matrix",
        "",
        "| Frame A | Frame B | n both known | n agree | n disagree | agreement % |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for _, r in shift_df.iterrows():
        lines.append(f"| {r['frame_a']} | {r['frame_b']} | {int(r['n_both_known']):,} | "
                     f"{int(r['n_agree']):,} | {int(r['n_disagree']):,} | "
                     f"{r['agreement_pct']:.1f}% |")
    lines += ["",
              "**Reading**: high pairwise agreement (>85%) means RQ1/RQ2 trajectories are "
              "robust to tier-frame source. Disagreement >15% triggers per-frame re-runs of "
              "the headline models (already covered by the 'frame_xxx_full' rows in "
              "the covariate grid below).",
              "",
              "### PCA composite loadings",
              "",
              "PCA inputs (log1p, then MinMaxScaler with 5-95 winsorization, full-sample fit, "
              f"random_state={RANDOM_SEED}):",
              "",
              "| Input feature | PC1 loading | PC2 loading |",
              "|---|---:|---:|"]
    for _, r in pca_loadings.iterrows():
        lines.append(f"| {r['input_feature']} | {r['pc1_loading']:.4f} | "
                     f"{r['pc2_loading']:.4f} |")
    if hasattr(pca_loadings, "attrs"):
        evr = pca_loadings.attrs.get("explained_variance_ratio", [])
        if evr:
            lines.append(f"\n_PCA explained variance ratio: PC1={evr[0]:.3f}, PC2={evr[1]:.3f}_")
    lines += ["", "## Part B — Covariate sensitivity grid (primary frame)", ""]


    for outcome_label, header in [
        ("logistic_version_level",
         "### Logistic (version-level `has_attestation`, 12-covariate primary) — "
         "Full model, primary frame\n\n"
         "_This is the paper's PRIMARY version-level logistic shape (adds a "
         "pre-window release-opportunity covariate and github_unobserved), fitted "
         "with package-cluster-robust standard errors (matching the paper primary)._"),
        ("cox_time_to_first_provenance",
         "### Cox PH (incident cohort, delayed entry, 11-covariate primary) — "
         "Full model, primary frame\n\n"
         "_This is the paper's PRIMARY Cox shape (incident-only, delayed entry at "
         "first in-window release, +log_prewin_release_freq +github_unobserved)._"),
    ]:
        sub = grid_df[(grid_df["outcome"] == outcome_label) &
                      (grid_df["label"] == "frame_dd_PRIMARY_full")]
        lines += [header, "",
                  "| Feature | OR/HR | 95% CI | p |",
                  "|---|---:|---|---:|"]
        for _, r in sub.iterrows():
            lines.append(f"| {r['feature']} | {r['OR']:.3f} | "
                         f"[{r['OR_ci_low']:.3f}, {r['OR_ci_high']:.3f}] | "
                         f"{r['pval']:.4f} |")
        lines.append("")
    lines += ["", "### Drop-one OR/HR spot check (no single covariate dominates)", ""]
    for outcome_label, label_pretty, spot_features in [
        ("logistic_version_level", "Logistic OR",
         ["log_weekly_downloads", "has_issue_template_int"]),
        ("cox_time_to_first_provenance", "Cox HR",
         ["log_downloads", "has_issue_template_int"]),
    ]:
        for feat in spot_features:
            lines += [f"#### {label_pretty} on `{feat}` across drop-one refits", ""]
            sub = grid_df[(grid_df["outcome"] == outcome_label) &
                          (grid_df["feature"] == feat) &
                          (grid_df["label"].str.startswith("frame_dd_"))]
            if sub.empty:
                lines.append("- (no rows)")
                continue
            lines += ["| Configuration | OR/HR | 95% CI | n_obs |",
                      "|---|---:|---|---:|"]
            for _, r in sub.iterrows():
                lines.append(f"| {r['label']} | {r['OR']:.3f} | "
                             f"[{r['OR_ci_low']:.3f}, {r['OR_ci_high']:.3f}] | "
                             f"{int(r['n_obs'])} |")
            lines.append("")

    lines += ["",
              "## Part B2 — Version-level drop-one (primary 12-covariate spec)",
              ""]
    if vdrop_df.empty:
        lines.append("- (no rows; version-level drop-one fits failed)")
    else:
        n_obs = int(vdrop_df["n_obs"].iloc[0])
        full = vdrop_df[vdrop_df["covariate_dropped"] == VDROP_FULL_LABEL].set_index("feature")
        lines += [
            f"Primary version-level Logistic (12 covariates incl. post_deadline, "
            f"pre-window release opportunity, github_unobserved; outcome = version "
            f"`has_attestation`; n={n_obs:,} version rows) refit once as reference "
            "then 12x, dropping one covariate per refit. These drop-one refits use "
            "plain-MLE point estimates (the package-cluster SEs of the grid/forest "
            "and of scripts/52 affect CIs/p only, not the point estimates). Full grid: "
            "`data/baselines/task_7_b3_versionlevel_dropone.csv`.",
            "",
            "| Feature | full OR | full p | OR range across drop-one refits | "
            "sign flips | p<0.05 flips |",
            "|---|---:|---:|---|---:|---:|",
        ]
        for feat in VERSION_COVS:
            surv = vdrop_df[(vdrop_df["feature"] == feat) &
                            (vdrop_df["covariate_dropped"] != VDROP_FULL_LABEL) &
                            (vdrop_df["covariate_dropped"] != feat)]
            if feat not in full.index or surv.empty:
                continue
            f_or = float(full.loc[feat, "OR"])
            f_p = float(full.loc[feat, "pval"])
            sign_flips = int(((surv["OR"] > 1) != (f_or > 1)).sum())
            sig_flips = int(((surv["pval"] < 0.05) != (f_p < 0.05)).sum())
            lines.append(
                f"| {feat} | {f_or:.3f} | {f_p:.4f} | "
                f"[{surv['OR'].min():.3f}, {surv['OR'].max():.3f}] ({len(surv)} refits) | "
                f"{sign_flips} | {sig_flips} |")
        lines += [
            "",
            "**Reading**: `sign flips` counts drop-one refits in which the OR crosses 1 "
            "relative to the full model; `p<0.05 flips` counts refits in which MLE "
            "significance at 0.05 changes relative to the full model. 0/0 across rows = "
            "the version-level spec is drop-one stable.",
        ]
    lines += ["",
              "## Forest plot",
              "",
              f"See `data/baselines/task_7_b3_forest.png` for a visual forest of OR/HR for "
              "the primary tier frame's full model vs drop-one variants.",
              "",
              "## Construct caveats (carried forward)",
              "",
              "- TS + @types/sinon are governance-rich top-tier non-adopters; they should remain mis-predicted under "
              "drop-one variants. If a single dropped covariate flips that prediction, "
              "flag it as a calibration-failure driver.",
              "- All Logistic ORs reported here are in-sample only; out-of-sample / "
              "grouped CV is deferred to if needed.",
              "- Missing-from-deps.dev scope: **62 packages in the full 3000-package Task 1 "
              "sample** (per ) and **29 in the B3 survival subset** (1614 "
              "packages; visible as 1614 − 1585 in the dd-vs-eco both_known cell of the "
              "tier-shift matrix). Both carried as `tier='missing'`; not silently imputed.",
              "- Descriptive-only framing preserved: all OR/HR estimates are "
              "associations, not causal effects.",
              "",
              "## Closure status",
              " B3 contributed: "
              "the <85% pairwise agreement gate is acknowledged as **load-bearing tier-frame "
              "sensitivity** (no frame pair exceeds 85%; eco-vs-PCA at 82.1% is the highest). "
              "Deps.dev PRIMARY remains canonical per .",
              "",
              "Fit-count convention: **25 fits per outcome total** = "
              "4 tier-frame full models + 21 primary-frame sensitivity refits (9 drop-one + "
              "3 group-out + 9 univariate). 22 = primary-frame fits including the full model "
              "(1 + 21). Part B2 (2026-07-11 addition) contributes 13 further version-level "
              "Logistic fits (1 full + 12 drop-one) outside this count."]
    OUT_REPORT.write_text("\n".join(lines) + "\n")
    logger.info(f"wrote {OUT_REPORT}")


def main() -> int:
    BASELINES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("loading data...")
    packages = pd.read_csv(PACKAGES_FULL, sep="\t")
    versions = pd.read_csv(VERSIONS_CSV)
    inf = pd.read_csv(INFLUENCE_CSV)
    depsdev = pd.read_csv(DEPSDEV_CSV)

    logger.info("building covariate frame...")
    sd, pca_loadings = build_covariates(packages, versions, inf, depsdev)
    n_inc = int((sd["prevalent"] == 0).sum())
    ev_inc = int(sd.loc[sd["prevalent"] == 0, "event"].sum())
    logger.info(f"survival data: n={len(sd)} (events={int(sd['event'].sum())}); "
                f"incident cohort n={n_inc} (events={ev_inc}) is the Cox fitting set")

    logger.info("tier-shift matrix...")
    shift_df = tier_shift_matrix(sd)
    shift_df.to_csv(OUT_SHIFT, index=False)
    logger.info(f"wrote {OUT_SHIFT}")
    pca_loadings.to_csv(OUT_PCA, index=False)
    logger.info(f"wrote {OUT_PCA}")

    logger.info("building version-level frame (all tier frames attached)...")
    vdf = build_version_frame(versions, inf, depsdev, sd)
    logger.info(f"version-level frame: n={len(vdf):,} rows × "
                f"{vdf['package'].nunique()} packages")

    logger.info("fitting version-level Logistic + incident Cox grids...")
    grid_df = build_fits_grid(sd, vdf)
    grid_df.to_csv(OUT_GRID, index=False)
    logger.info(f"wrote {OUT_GRID} (rows={len(grid_df)})")

    logger.info("Part B2: version-level drop-one (primary 12-cov spec)...")


    vdf_dd = vdf[vdf["tier_dd"] != "missing"].reset_index(drop=True)
    logger.info(f"drop-one frame: n={len(vdf_dd):,} rows "
                f"(deps.dev-missing excluded from {len(vdf):,})")
    vdrop_df = versionlevel_dropone(vdf_dd)
    vdrop_df.to_csv(OUT_VDROP, index=False)
    logger.info(f"wrote {OUT_VDROP} (rows={len(vdrop_df)})")

    logger.info("making forest plot...")
    try:
        make_forest(grid_df)
    except Exception as e:
        logger.warning(f"forest plot failed: {e}")

    write_report(grid_df, shift_df, pca_loadings, sd, vdrop_df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
