"""
Rebuttal B1 — Cox left-truncation sensitivity (incident-only + prevalent indicator).

B1 audit (scripts/42) found that 336 of 661 Cox events (50.8%) are PREVALENT
adopters: packages already carrying dist.attestations before the 2025-08-01
window origin. The main Cox (scripts/29) anchors duration at 2025-08-01 and
treats these as in-window incident adoptions -> left-truncation.

This script does NOT modify scripts/29. It re-fits three Cox models on the same
covariate set and compares hazard ratios:

  M-main      : full cohort (reproduces scripts/29)
  M-incident  : exclude prevalent adopters (incident-only cohort)
  M-indicator : full cohort + a binary `prevalent` covariate

If the key HRs (log_age_days, tier_top, log_maintainers) are stable across
M-main and M-incident, the main Cox description survives and the sensitivity
佐证s the rebuttal. If they move materially, the main interpretation must be
re-scoped. Either way this is survival-method rigor, not a causal claim.

Run: conda run --no-capture-output -n stage2c python scripts/44_cox_incident_only_sensitivity.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

DATA = Path(__file__).parent.parent / "data"
REPORTS = Path(__file__).parent.parent / "reports"
RAW = DATA / "packuments_raw"
VERSIONS_CSV = DATA / "versions.csv"
INFLUENCE_CSV = DATA / "influence_metrics.csv"
DEPSDEV_CSV = DATA / "depsdev_dependents_2026_05_05.csv"
OUT_MD = REPORTS / "task_9_rebuttal_b1_cox_sensitivity.md"
OUT_CSV = DATA / "cox_incident_sensitivity.csv"

ORIGIN = pd.Timestamp("2025-08-01", tz="UTC")
CENSOR = pd.Timestamp("2026-05-10 23:59:59", tz="UTC")
WINDOW_START = datetime(2025, 8, 1, tzinfo=timezone.utc)

COX_COLS = ["log_weekly_downloads", "log_age_days", "log_maintainers",
            "owner_is_org", "has_dependabot_int", "has_issue_template_int",
            "is_scoped_int", "tier_top", "tier_mid"]


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


def prevalent_packages() -> set[str]:
    """Packages with >=1 attested version published before 2025-08-01 (full history)."""
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


def build_survival(prevalent: set[str]) -> pd.DataFrame:
    """Mirror scripts/29 build_survival_data, but keep `package` and a `prevalent` flag."""
    v = pd.read_csv(VERSIONS_CSV, parse_dates=["published_at"])
    v["published_at"] = pd.to_datetime(v["published_at"], utc=True)
    v = v[(v["published_at"] >= ORIGIN) & (v["published_at"] <= CENSOR)].copy()
    att = v[v["has_attestation"].astype(str).str.lower() == "true"]
    first_att = att.groupby("package")["published_at"].min().reset_index()
    first_att.columns = ["package", "first_attested_at"]
    sd = pd.DataFrame({"package": v["package"].drop_duplicates().reset_index(drop=True)})
    sd = sd.merge(first_att, on="package", how="left")
    sd["event"] = sd["first_attested_at"].notna().astype(int)
    sd.loc[sd["event"] == 0, "first_attested_at"] = CENSOR
    sd["duration_days"] = (sd["first_attested_at"] - ORIGIN).dt.days.clip(lower=1)

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
    sd["owner_is_org"] = (sd["github_owner_type"] == "Organization").astype(int)
    for s, d in [("has_dependabot", "has_dependabot_int"),
                 ("has_issue_template", "has_issue_template_int"),
                 ("is_scoped_package", "is_scoped_int")]:
        sd[d] = (sd[s].astype(str).str.lower() == "true").astype(int)
    sd["tier_top"] = (sd["tier"] == "top").astype(int)
    sd["tier_mid"] = (sd["tier"] == "mid").astype(int)
    sd["prevalent"] = sd["package"].isin(prevalent).astype(int)

    cols = ["package", "duration_days", "event", "prevalent"] + COX_COLS
    sd = sd[cols].dropna()
    return sd.astype({"duration_days": int, "event": int})


def fit_cox(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    cph = CoxPHFitter()
    cph.fit(df[["duration_days", "event"] + cols], duration_col="duration_days",
            event_col="event", robust=True)
    s = cph.summary
    out = s[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].copy()
    out.columns = ["HR", "lo", "hi", "p"]
    out.attrs["c_index"] = cph.concordance_index_
    out.attrs["n"] = len(df)
    out.attrs["events"] = int(df["event"].sum())
    return out


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    prevalent = prevalent_packages()
    sd = build_survival(prevalent)
    sd_inc = sd[sd["prevalent"] == 0].copy()

    print(f"Full cohort:     {len(sd)} pkgs, {int(sd['event'].sum())} events, "
          f"prevalent={int(sd['prevalent'].sum())}", flush=True)
    print(f"Incident-only:   {len(sd_inc)} pkgs, {int(sd_inc['event'].sum())} events", flush=True)

    m_main = fit_cox(sd, COX_COLS)
    m_inc = fit_cox(sd_inc, COX_COLS)
    m_ind = fit_cox(sd, COX_COLS + ["prevalent"])

    # ---- comparison table ----
    rows = []
    for cov in COX_COLS:
        hr_main = m_main.loc[cov, "HR"]
        hr_inc = m_inc.loc[cov, "HR"]
        delta = (hr_inc - hr_main) / hr_main * 100
        rows.append({
            "covariate": cov,
            "HR_main": round(hr_main, 3),
            "p_main": round(m_main.loc[cov, "p"], 4),
            "HR_incident": round(hr_inc, 3),
            "p_incident": round(m_inc.loc[cov, "p"], 4),
            "HR_pct_change": round(delta, 1),
            "sign_flip": (hr_main - 1) * (hr_inc - 1) < 0,
        })
    comp = pd.DataFrame(rows)
    comp.to_csv(OUT_CSV, index=False)
    print("\n=== HR: main vs incident-only ===", flush=True)
    print(comp.to_string(index=False), flush=True)

    prev_hr = m_ind.loc["prevalent", "HR"]
    prev_p = m_ind.loc["prevalent", "p"]
    print(f"\nPrevalent-indicator HR = {prev_hr:.3f} (p={prev_p:.4g})", flush=True)

    key = ["log_age_days", "tier_top", "log_maintainers"]
    max_shift = comp[comp["covariate"].isin(key)]["HR_pct_change"].abs().max()
    any_flip = comp["sign_flip"].any()
    if any_flip:
        verdict = "RE-SCOPE — at least one HR sign flips under incident-only; main Cox interpretation must be revised."
    elif max_shift < 15:
        verdict = (f"STABLE — key HRs shift < 15% (max {max_shift:.1f}%) and no sign flip; "
                   f"main Cox description survives; report incident-only as a sensitivity.")
    else:
        verdict = (f"SENSITIVE — key HRs shift up to {max_shift:.1f}% (no sign flip); "
                   f"keep main Cox but foreground the incident-only sensitivity and the prevalent share.")
    print(f"\nVERDICT: {verdict}", flush=True)

    # ---- markdown ----
    md = [
        "# Rebuttal B1 — Cox left-truncation sensitivity",
        "",
        "**Date**: 2026-05-30  ",
        f"**Prevalent adopters (attested before 2025-08-01)**: {int(sd['prevalent'].sum())} "
        f"of {int(sd['event'].sum())} events ({100*sd['prevalent'].sum()/sd['event'].sum():.1f}%)  ",
        "**Models** (covariates identical to scripts/29; robust SE):",
        "",
        f"- M-main (full cohort): n={m_main.attrs['n']}, events={m_main.attrs['events']}, "
        f"C-index={m_main.attrs['c_index']:.3f}",
        f"- M-incident (exclude prevalent): n={m_inc.attrs['n']}, events={m_inc.attrs['events']}, "
        f"C-index={m_inc.attrs['c_index']:.3f}",
        f"- M-indicator (full + `prevalent` dummy): prevalent HR={prev_hr:.3f} (p={prev_p:.4g})",
        "",
        "## Hazard ratios: main vs incident-only",
        "",
        "| Covariate | HR (main) | p | HR (incident) | p | Δ% | sign flip |",
        "|---|---:|---:|---:|---:|---:|:--:|",
    ]
    for r in rows:
        md.append(f"| `{r['covariate']}` | {r['HR_main']} | {r['p_main']} | "
                  f"{r['HR_incident']} | {r['p_incident']} | {r['HR_pct_change']:+} | "
                  f"{'⚠ YES' if r['sign_flip'] else 'no'} |")
    md += [
        "",
        f"**Prevalent-adopter indicator** (M-indicator): HR = {prev_hr:.3f}, p = {prev_p:.4g} "
        "(direction of the pre-window-adoption signal, holding covariates fixed).",
        "",
        "## Verdict",
        "",
        verdict,
        "",
        "## Rebuttal wording (draft)",
        "",
        "> The Cox event is the first *in-window* attested release (origin 2025-08-01), "
        "an interval-specific quantity; `dist.attestations` predates the window (npm "
        "provenance since 2023), so a portion of the cohort were prevalent adopters. We "
        "added (i) an incident-only sensitivity excluding pre-window adopters and (ii) a "
        "prevalent-adopter indicator. The PH-passing hazard-ratio directions "
        "(`log_age_days`, `tier_top`, `log_maintainers`) are "
        f"{'preserved' if not any_flip else 'revised'} under both, and the estimates remain "
        "time-averaged descriptive associations (not causal).",
    ]
    OUT_MD.write_text("\n".join(md) + "\n")
    print(f"\n✓ Saved {OUT_CSV}", flush=True)
    print(f"✓ Saved {OUT_MD}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
