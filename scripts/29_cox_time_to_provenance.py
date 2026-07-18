from __future__ import annotations

import csv
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import proportional_hazard_test

import figstyle

figstyle.apply()

DATA_DIR = Path(__file__).parent.parent / "data"
REPORTS_DIR = Path(__file__).parent.parent / "reports"
PAPER_DIR = Path(__file__).parent.parent / "paper" / "figures"

VERSIONS_CSV = DATA_DIR / "versions.csv"
INFLUENCE_CSV = DATA_DIR / "influence_metrics.csv"
DEPSDEV_CSV = DATA_DIR / "depsdev_dependents_2026_05_05.csv"
OUT_CSV = DATA_DIR / "cox_results.csv"
OUT_MD = REPORTS_DIR / "task_5_cox.md"
OUT_KM_PNG = PAPER_DIR / "cox_km_by_tier.png"
OUT_DIAG_PNG = PAPER_DIR / "cox_schoenfeld_diagnostic.png"

ORIGIN = pd.Timestamp("2025-08-01", tz="UTC")
CENSOR = pd.Timestamp("2026-05-10 23:59:59", tz="UTC")
MAX_DURATION = (CENSOR - ORIGIN).days


def dpc_tier(dpc: int) -> str:
    if dpc > 500:
        return "top"
    if dpc > 10:
        return "mid"
    return "tail"


def build_survival_data() -> pd.DataFrame:

    versions = pd.read_csv(VERSIONS_CSV, parse_dates=["published_at"])
    versions["published_at"] = pd.to_datetime(versions["published_at"], utc=True)
    versions = versions[(versions["published_at"] >= ORIGIN) &
                        (versions["published_at"] <= CENSOR)].copy()


    attested = versions[versions["has_attestation"].astype(str).str.lower() == "true"]
    first_att = attested.groupby("package")["published_at"].min().reset_index()
    first_att.columns = ["package", "first_attested_at"]


    all_pkgs = versions["package"].drop_duplicates().reset_index(drop=True)
    sd = pd.DataFrame({"package": all_pkgs})
    sd = sd.merge(first_att, on="package", how="left")
    sd["event"] = sd["first_attested_at"].notna().astype(int)
    sd.loc[sd["event"] == 0, "first_attested_at"] = CENSOR
    sd["duration_days"] = (sd["first_attested_at"] - ORIGIN).dt.days.clip(lower=1)


    inf = pd.read_csv(INFLUENCE_CSV)
    sd = sd.merge(inf, on="package", how="left")

    depsdev = pd.read_csv(DEPSDEV_CSV)
    depsdev_map = dict(zip(depsdev["name"], depsdev["dependents_count"].astype(int)))
    sd["depsdev_dependents_count"] = sd["package"].map(depsdev_map)
    sd["depsdev_tier"] = sd["depsdev_dependents_count"].apply(
        lambda x: dpc_tier(int(x)) if pd.notna(x) else "missing"
    )


    sd = sd[sd["depsdev_tier"].isin(["top", "mid", "tail"])].copy()


    sd["log_dependents"] = np.log1p(sd["depsdev_dependents_count"].astype(float))
    sd["log_weekly_downloads"] = np.log1p(pd.to_numeric(sd["weekly_downloads_avg"], errors="coerce"))
    sd["log_age_days"] = np.log1p(pd.to_numeric(sd["package_age_days"], errors="coerce"))
    sd["log_maintainers"] = np.log1p(pd.to_numeric(sd["maintainers_count"], errors="coerce"))
    sd["owner_is_org"] = (sd["github_owner_type"] == "Organization").astype(int)


    for src, dst in [("has_dependabot", "has_dependabot_int"),
                     ("has_issue_template", "has_issue_template_int"),
                     ("is_scoped_package", "is_scoped_int")]:
        sd[dst] = (sd[src].astype(str).str.lower() == "true").astype(int)
    sd["tier_top"] = (sd["depsdev_tier"] == "top").astype(int)
    sd["tier_mid"] = (sd["depsdev_tier"] == "mid").astype(int)


    keep_cols = [
        "duration_days", "event",
        "log_dependents", "log_weekly_downloads", "log_age_days", "log_maintainers",
        "owner_is_org", "has_dependabot_int", "has_issue_template_int",
        "is_scoped_int", "tier_top", "tier_mid", "depsdev_tier"
    ]
    sd = sd[keep_cols].dropna()
    sd = sd.astype({"duration_days": int, "event": int})
    return sd


def _style_schoenfeld(fig: "plt.Figure") -> None:
    from matplotlib.collections import LineCollection, PathCollection

    for ax in fig.axes:
        for coll in ax.collections:
            if isinstance(coll, PathCollection):
                coll.set_facecolor(figstyle.ACCENT)
                coll.set_edgecolor(figstyle.ACCENT)
            elif isinstance(coll, LineCollection):
                coll.set_color(figstyle.REF)


def main() -> int:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    sd = build_survival_data()
    print(f"Survival dataset: {len(sd)} packages "
          f"(events={sd['event'].sum()}, censored={(1-sd['event']).sum()})", flush=True)


    fig, ax = plt.subplots(figsize=(8, 5))
    for t in ("top", "mid", "tail"):
        sub = sd[sd["depsdev_tier"] == t]
        if len(sub) == 0:
            continue
        km = KaplanMeierFitter()
        km.fit(sub["duration_days"], sub["event"], label=f"{t} (n={len(sub)}, "
               f"events={sub['event'].sum()})")
        km.plot_survival_function(ax=ax, color=figstyle.TIER[t], ci_alpha=0.15)
    deadline_day = (pd.Timestamp("2025-12-09", tz="UTC") - ORIGIN).days
    ax.axvline(deadline_day, linestyle="--", color=figstyle.REF, alpha=0.7, linewidth=1.2)
    ax.text(deadline_day + 2, 0.97, "Dec-9", color=figstyle.REF, fontsize=8,
            va="top", transform=ax.get_xaxis_transform())
    ax.set_xlabel("Days since 2025-08-01")
    ax.set_ylabel("S(t) — share with no observed in-window attested release yet")
    figstyle.legend_outside(ax)
    fig.tight_layout()
    fig.savefig(OUT_KM_PNG, bbox_inches="tight")
    plt.close(fig)
    print(f" Saved {OUT_KM_PNG}", flush=True)


    cox_cols = [
        "duration_days", "event",
        "log_weekly_downloads", "log_age_days", "log_maintainers",
        "owner_is_org", "has_dependabot_int", "has_issue_template_int",
        "is_scoped_int", "tier_top", "tier_mid"
    ]
    cph = CoxPHFitter()
    cph.fit(sd[cox_cols], duration_col="duration_days", event_col="event",
            robust=True)
    summary = cph.summary
    print(f"\nCox PH fitted. Concordance = {cph.concordance_index_:.4f}", flush=True)
    print(summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].round(3), flush=True)


    try:
        ph_test = proportional_hazard_test(cph, sd[cox_cols],
                                            time_transform="rank")
        ph_df = ph_test.summary
        print(f"\nSchoenfeld PH test (rank transform):", flush=True)
        print(ph_df[["test_statistic", "p"]].round(4), flush=True)
        global_p = ph_df["p"].min()
    except Exception as e:
        ph_df = pd.DataFrame()
        global_p = None
        print(f" Schoenfeld test failed: {e}", flush=True)


    try:
        d = ph_df.reindex(cox_cols).dropna(subset=["p"]).sort_values("p")
        covs = list(d.index)
        pvals = [float(v) for v in d["p"].values]
        ypos = list(range(len(covs) - 1, -1, -1))
        colors = [figstyle.ACCENT if p < 0.05 else figstyle.REF for p in pvals]
        n_fail = sum(p < 0.05 for p in pvals)

        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        ax.hlines(ypos, 0, pvals, color=colors, linewidth=2.0, alpha=0.75, zorder=2)
        ax.scatter(pvals, ypos, c=colors, s=62, zorder=3,
                   edgecolor="white", linewidth=0.6)
        ax.axvline(0.05, color=figstyle.REF, linestyle="--", linewidth=1.0, zorder=1)
        ax.text(0.05, len(covs) - 0.35, r"  $\alpha = 0.05$", color=figstyle.REF,
                fontsize=8.5, va="top", ha="left")
        for p, yi in zip(pvals, ypos):
            ax.annotate(f"{p:.3f}", (p, yi), xytext=(6, 0),
                        textcoords="offset points", va="center",
                        fontsize=8, color="0.25")
        ax.set_yticks(ypos)
        ax.set_yticklabels(covs, fontsize=8.5)
        ax.set_xlabel(r"Schoenfeld test $p$-value (rank time-transform)")
        ax.set_xlim(-0.03, 1.04)
        ax.set_ylim(-0.6, len(covs) - 0.2)
        ax.grid(axis="y", visible=False)
        from matplotlib.lines import Line2D
        handles = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=figstyle.ACCENT,
                   markersize=8, label=f"fails PH ($p<0.05$): {n_fail} of {len(covs)}"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=figstyle.REF,
                   markersize=8, label="passes PH"),
        ]
        figstyle.legend_outside(ax, handles=handles)
        fig.savefig(OUT_DIAG_PNG)
        plt.close("all")
        print(f" Saved {OUT_DIAG_PNG} (lollipop; {n_fail} of {len(covs)} fail PH)",
              flush=True)
    except Exception as e:
        print(f" Schoenfeld plot failed: {e}", flush=True)


    out_rows = []
    for cov in summary.index:
        out_rows.append({
            "covariate": cov,
            "HR": summary.loc[cov, "exp(coef)"],
            "HR_ci_low": summary.loc[cov, "exp(coef) lower 95%"],
            "HR_ci_high": summary.loc[cov, "exp(coef) upper 95%"],
            "coef": summary.loc[cov, "coef"],
            "se(coef)": summary.loc[cov, "se(coef)"],
            "p": summary.loc[cov, "p"],
            "schoenfeld_p": float(ph_df.loc[cov, "p"]) if cov in ph_df.index else None,
        })
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["covariate", "HR", "HR_ci_low", "HR_ci_high",
                                          "coef", "se(coef)", "p", "schoenfeld_p"])
        w.writeheader()
        for r in out_rows:
            w.writerow(r)
    print(f"\n Saved {OUT_CSV}", flush=True)


    lines = [
        "# — Cox PH on time-to-first-provenance",
        "",
        f"**Date**: 2026-05-18  ",
        f"**Sample**: {len(sd)} packages (events={sd['event'].sum()}, censored={(1-sd['event']).sum()})  ",
        f"**Window**: {ORIGIN.date()}  {CENSOR.date()} (max duration {MAX_DURATION} days)  ",
        f"**Model**: Cox PH with robust SE, lifelines CoxPHFitter  ",
        f"**Concordance index**: {cph.concordance_index_:.4f}",
        "",
        "## Covariate effects (HR with 95% CI; effect size = HR)",
        "",
        "| Covariate | HR | 95% CI | p | Schoenfeld p |",
        "|---|---:|---|---:|---:|",
    ]
    for r in out_rows:
        sf_p = f"{r['schoenfeld_p']:.3f}" if r['schoenfeld_p'] is not None else "—"
        sig = "***" if r["p"] < 0.01 else ("**" if r["p"] < 0.05 else ("*" if r["p"] < 0.10 else ""))
        ph_flag = " " if r['schoenfeld_p'] is not None and r['schoenfeld_p'] < 0.05 else ""
        lines.append(
            f"| `{r['covariate']}` | {r['HR']:.3f}{sig} | "
            f"[{r['HR_ci_low']:.3f}, {r['HR_ci_high']:.3f}] | {r['p']:.4f} | "
            f"{sf_p}{ph_flag} |"
        )

    ph_verdict = "PASS (all covariates p > 0.05)" if global_p is not None and global_p > 0.05 else (
        f" FAIL (min covariate p = {global_p:.3f})  PH assumption violated for at least one covariate"
    ) if global_p is not None else "n/a"

    lines += ["",
              "## Schoenfeld residuals PH test (MANDATORY diagnostic)",
              "",
              f"- Global verdict: **{ph_verdict}**",
              "- Per-covariate Schoenfeld p-values in last column of table above (`` if p<0.05)",
              "",
              "## Construct caveats (carried)",
              "",
              "- Tier source = deps.dev snapshot 2026-05-05 (PRIMARY frame)",
              "- 62 packages excluded (missing-dd) — see strata-shift report",
              "- Class C / E construct caveats apply for downstream interpretation",
              "- Causal interpretation bounded to association unless PH assumption holds AND no unmeasured confounding",
              "",
              "## Headline interpretation",
              "",
              "HR > 1  covariate  associates with FASTER first-provenance adoption.",
              "HR < 1  slower / lower hazard.",
              "Significant HRs:",
              ""]
    for r in out_rows:
        if r["p"] < 0.05:
            direction = "FASTER" if r["HR"] > 1 else "SLOWER"
            lines.append(f"- `{r['covariate']}`: HR={r['HR']:.3f} [{r['HR_ci_low']:.3f}, {r['HR_ci_high']:.3f}] "
                         f" {direction} (p={r['p']:.4f})")

    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f" Saved {OUT_MD}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
