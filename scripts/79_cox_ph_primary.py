"""
TSE revision -- proportional-hazards check computed directly on the PRIMARY
delayed-entry Cox model (extended series, incident cohort, 11 covariates).

Schoenfeld residuals are defined for counting-process (entry, stop] data; the
earlier check ran on an entry-at-start auxiliary fit only because lifelines
raises NotImplementedError ("Residuals for entries not implemented") for
entry_col fits. This script removes that workaround:

  1. rebuilds the primary analysis frame with scripts/52's own cohort builders
     (--origin 2024-01-01 --versions-csv versions_extended.csv), so the cohort,
     entry clipping and covariates are identical to Table IV;
  2. runs scripts/79_cox_zph.R (R survival::cox.zph, score test on the
     left-truncated risk sets) on that frame;
  3. cross-checks R against lifelines: same coefficients, and an independent
     Python implementation of counting-process Schoenfeld residuals (Efron ties)
     that reproduces lifelines' residuals on the untruncated fit and satisfies
     the score equations at the delayed-entry MLE;
  4. writes the per-covariate p-value figure for the supplement and a report.

Inputs:  as scripts/52 (data/versions_extended.csv, data/influence_metrics.csv,
         data/depsdev_dependents_2026_05_05.csv, data/packuments_raw/*.json); without the
         access-gated packument slice, steps 2-4 run from the frozen
         data/cox_primary_frame_tse_r1_long.csv
Outputs: data/cox_primary_frame_tse_r1_long.csv
         data/cox_zph_primary_tse_r1_long.csv, data/cox_primary_R_coef_tse_r1_long.csv
         paper_tse/figures/cox_schoenfeld_primary_extended.pdf (reports/ without the paper tree)
         reports/tse_r2_cox_ph_primary.md
Carried-forward caveats: deps.dev primary tier frame; hazard ratios are
time-averaged descriptive associations; anti-causal firewall; data cutoff
2026-05-10 untouched.

Run: conda run --no-capture-output -n stage2c python scripts/79_cox_ph_primary.py
     (step 2 calls `conda run -n rsurv Rscript scripts/79_cox_zph.R`; override
     with the RSCRIPT_CMD environment variable)
"""
from __future__ import annotations

import importlib.util
import os
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from scipy import stats

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
FRAME_CSV = DATA / "cox_primary_frame_tse_r1_long.csv"
ZPH_CSV = DATA / "cox_zph_primary_tse_r1_long.csv"
RCOEF_CSV = DATA / "cox_primary_R_coef_tse_r1_long.csv"
TABLE_CSV = DATA / "cox_incident_primary_tse_r1_long.csv"
FIG_PDF = (ROOT / "paper_tse" / "figures" if (ROOT / "paper_tse").is_dir() else ROOT / "reports") \
    / "cox_schoenfeld_primary_extended.pdf"
RAW = DATA / "packuments_raw"
OUT_MD = ROOT / "reports" / "tse_r2_cox_ph_primary.md"
RSCRIPT_CMD = os.environ.get("RSCRIPT_CMD", "conda run -n rsurv Rscript")
TRANSFORM = "km"      # survival::cox.zph default; rank and identity are reported alongside
ALPHA = 0.05
FLAG = "#332288"      # first colour of the default figure palette
NEUTRAL = "#9A9A9A"

LABELS = {"log_weekly_downloads": "log weekly downloads", "log_age_days": "log package age",
          "log_maintainers": "log maintainers", "owner_is_org": "organization owner",
          "has_dependabot_int": "Dependabot", "has_issue_template_int": "issue template",
          "is_scoped_int": "scoped package", "tier_top": "top tier", "tier_mid": "middle tier",
          "log_prewin_release_freq": "pre-window release freq.",
          "github_unobserved": "GitHub unobserved"}


def load_rq3_module():
    """Import scripts/52 (file name starts with a digit) and point it at the extended series."""
    spec = importlib.util.spec_from_file_location("rq3", ROOT / "scripts" / "52_rq3_sensitivity.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.configure("2024-01-01", "versions_extended.csv", "_tse_r1_long")
    return mod


def build_frame(rq3) -> pd.DataFrame:
    """Incident cohort exactly as scripts/52 part1_cox_incident builds it."""
    sd = rq3.build_survival(rq3.prevalent_packages(), rq3.prewindow_release_counts())
    inc = sd[sd["prevalent"] == 0]
    return inc[["package", "entry_days", "duration_days", "event"] + rq3.COX_COLS_PRIMARY].reset_index(drop=True)


def schoenfeld_cp(x: np.ndarray, t: np.ndarray, e: np.ndarray, entry: np.ndarray,
                  beta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Efron-tie Schoenfeld residuals with left-truncated risk sets.

    Returns (residuals, event times, original row positions), one row per event in time order.
    """
    order = np.argsort(t, kind="mergesort")
    x, t, e, entry = x[order], t[order], e[order], entry[order]
    phi = np.exp(x @ beta)
    res, times, rows = [], [], []
    for ti in np.unique(t[e == 1]):
        risk = (entry < ti) & (t >= ti)
        dead = (t == ti) & (e == 1)
        m = int(dead.sum())
        s0, s1 = phi[risk].sum(), (phi[risk, None] * x[risk]).sum(0)
        d0, d1 = phi[dead].sum(), (phi[dead, None] * x[dead]).sum(0)
        xbar = sum((s1 - k / m * d1) / (s0 - k / m * d0) for k in range(m)) / m
        for i in np.where(dead)[0]:
            res.append(x[i] - xbar)
            times.append(ti)
            rows.append(order[i])
    return np.array(res), np.array(times), np.array(rows)


def python_cross_check(frame: pd.DataFrame, cols: list[str]) -> dict:
    """Independent residual implementation: validation against lifelines and the score equations."""
    out = {}
    fp = frame[["duration_days", "entry_days", "event"] + cols]
    cph = CoxPHFitter().fit(fp, duration_col="duration_days", event_col="event",
                            entry_col="entry_days", robust=True)
    fa = frame[["duration_days", "event"] + cols]
    aux = CoxPHFitter().fit(fa, duration_col="duration_days", event_col="event", robust=True)
    x = frame[cols].to_numpy(float)
    t = frame["duration_days"].to_numpy(float)
    e = frame["event"].to_numpy(int)
    entry = frame["entry_days"].to_numpy(float)
    # (a) untruncated fit: residuals must equal lifelines' own
    lf = aux.compute_residuals(fa, kind="schoenfeld")
    r_aux, _, rows = schoenfeld_cp(x, t, e, np.full(len(t), -np.inf), aux.params_[cols].to_numpy())
    mine = pd.DataFrame(r_aux, columns=cols, index=frame.index[rows]).loc[lf.index]
    out["max_abs_diff_vs_lifelines"] = float(np.abs(mine.to_numpy() - lf[cols].to_numpy()).max())
    # (b) score equations at the delayed-entry MLE vanish only with truncated risk sets
    beta = cph.params_[cols].to_numpy()
    r_tr, tt, _ = schoenfeld_cp(x, t, e, entry, beta)
    r_naive, _, _ = schoenfeld_cp(x, t, e, np.full(len(t), -np.inf), beta)
    out["score_sum_truncated"] = float(np.abs(r_tr.sum(0)).max())
    out["score_sum_entry_ignored"] = float(np.abs(r_naive.sum(0)).max())
    # (c) Grambsch-Therneau approximate test, rank transform (average ranks), model-based variance
    v = cph.variance_matrix_.loc[cols, cols].to_numpy()
    d = int(e.sum())
    scaled = d * r_tr @ v
    g = stats.rankdata(tt)
    gc = g - g.mean()
    stat = (gc[:, None] * scaled).sum(0) ** 2 / (d * np.diag(v) * (gc ** 2).sum())
    out["approx_rank_p"] = pd.Series(stats.chi2.sf(stat, 1), index=cols)
    out["lifelines_coef"] = cph.params_[cols]
    out["lifelines_hr"] = np.exp(cph.params_[cols])
    return out


def figure(zph: pd.DataFrame, cols: list[str]) -> None:
    """Per-covariate cox.zph p-values (log scale), flagged covariates filled."""
    os.environ.setdefault("FIGSTYLE_VENUE", "ieee")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(ROOT / "scripts"))
    import figstyle
    figstyle.apply()
    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
                         "mathtext.fontset": "stix", "pdf.fonttype": 42})
    p = zph.set_index("term").loc[cols, "p"].sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    y = np.arange(len(p))
    flagged = (p < ALPHA).to_numpy()
    ax.hlines(y, p.min() / 2, p.to_numpy(), color=NEUTRAL, linewidth=0.8, zorder=1)
    ax.scatter(p[~flagged], y[~flagged], s=22, facecolors="white", edgecolors=NEUTRAL,
               linewidths=1.0, zorder=2, label=r"$p \geq 0.05$")
    ax.scatter(p[flagged], y[flagged], s=24, color=FLAG, marker="o", zorder=3,
               label=r"$p < 0.05$ (flagged)")
    ax.axvline(ALPHA, color=figstyle.REF, linestyle="--", linewidth=0.8, zorder=0)
    ax.set_xscale("log")
    ax.set_xlim(p.min() / 2, 1.3)
    ax.set_yticks(y, [LABELS[c] for c in p.index])
    ax.set_xlabel(r"cox.zph $p$-value (Kaplan-Meier time transform)")
    ax.legend(loc="upper right", frameon=True, framealpha=0.9, edgecolor="0.8", fontsize=7)
    fig.tight_layout()
    FIG_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PDF)
    plt.close(fig)
    print(f"Saved {FIG_PDF}", flush=True)


def main() -> int:
    rq3 = load_rq3_module()
    cols = rq3.COX_COLS_PRIMARY
    if RAW.is_dir() and any(RAW.glob("*.json")):
        frame = build_frame(rq3)
        frame.to_csv(FRAME_CSV, index=False)
        print(f"Frame: {len(frame)} packages, {int(frame['event'].sum())} events -> {FRAME_CSV}", flush=True)
    else:
        # The prevalent-adopter flag needs the access-gated packument history; without it,
        # rebuilding would silently keep prevalent adopters, so use the frozen frame instead.
        frame = pd.read_csv(FRAME_CSV)
        print(f"Raw packument slice not present: using the frozen frame {FRAME_CSV} "
              f"({len(frame)} packages, {int(frame['event'].sum())} events)", flush=True)

    r = subprocess.run(shlex.split(RSCRIPT_CMD) + [str(ROOT / "scripts" / "79_cox_zph.R")],
                       capture_output=True, text=True)
    print(r.stdout, flush=True)
    if r.returncode:
        print(r.stderr, flush=True)
        return 1
    zph = pd.read_csv(ZPH_CSV)
    rcoef = pd.read_csv(RCOEF_CSV).set_index("term")

    chk = python_cross_check(frame, cols)
    table = pd.read_csv(TABLE_CSV).set_index("covariate")
    coef_diff = float((rcoef.loc[cols, "coef"] - chk["lifelines_coef"]).abs().max())
    hr_diff = float((chk["lifelines_hr"] - table.loc[cols, "HR"]).abs().max())
    print(f"max |coef_R - coef_lifelines| = {coef_diff:.2e}; max |HR - Table IV HR| = {hr_diff:.2e}", flush=True)
    print(f"residual check vs lifelines (untruncated): {chk['max_abs_diff_vs_lifelines']:.2e}; "
          f"score sums: truncated {chk['score_sum_truncated']:.2e}, entry ignored "
          f"{chk['score_sum_entry_ignored']:.1f}", flush=True)

    main_tab = zph[zph["transform"] == TRANSFORM].set_index("term")
    wide = pd.DataFrame({
        "HR": table.loc[cols, "HR"].round(3),
        "zph_km_p": zph[zph["transform"] == "km"].set_index("term").loc[cols, "p"],
        "zph_rank_p": zph[zph["transform"] == "rank"].set_index("term").loc[cols, "p"],
        "zph_identity_p": zph[zph["transform"] == "identity"].set_index("term").loc[cols, "p"],
        "py_approx_rank_p": chk["approx_rank_p"],
        "aux_entry_at_start_p": table.loc[cols, "schoenfeld_p"],
    })
    print(wide.round(4).to_string(), flush=True)
    figure(zph[zph["transform"] == TRANSFORM], cols)

    flagged = {tr: sorted(zph[(zph["transform"] == tr) & (zph["term"] != "GLOBAL") & (zph["p"] < ALPHA)]["term"])
               for tr in ["km", "rank", "identity"]}
    md = [
        "# Proportional-hazards check on the primary delayed-entry Cox model",
        "",
        f"Frame: {len(frame)} packages, {int(frame['event'].sum())} events (extended series, incident "
        "cohort, delayed entry, 11 covariates; built with scripts/52's cohort functions).",
        "",
        f"Test: R survival::cox.zph score test on counting-process (entry, stop] data; reported "
        f"transform: {TRANSFORM}. Global p ({TRANSFORM}) = {main_tab.loc['GLOBAL', 'p']:.4g}.",
        "",
        "Cross-checks: max |coef_R - coef_lifelines| = "
        f"{coef_diff:.2e}; max |HR - Table IV| = {hr_diff:.2e}; Python residuals vs lifelines on the "
        f"untruncated fit {chk['max_abs_diff_vs_lifelines']:.2e}; score sums at the delayed-entry MLE "
        f"{chk['score_sum_truncated']:.2e} with truncated risk sets vs "
        f"{chk['score_sum_entry_ignored']:.1f} with entry ignored.",
        "",
        "| Covariate | HR | cox.zph p (km) | (rank) | (identity) | Python approx. rank p | aux. entry-at-start p |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in cols:
        w = wide.loc[c]
        md.append(f"| {c} | {w['HR']:.3f} | {w['zph_km_p']:.3f} | {w['zph_rank_p']:.3f} | "
                  f"{w['zph_identity_p']:.3f} | {w['py_approx_rank_p']:.3f} | {w['aux_entry_at_start_p']:.3f} |")
    md += ["", "Flagged at p < 0.05: " + "; ".join(f"{k}: {', '.join(v) or 'none'}" for k, v in flagged.items()), ""]
    OUT_MD.write_text("\n".join(md) + "\n")
    print(f"Saved {OUT_MD}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
