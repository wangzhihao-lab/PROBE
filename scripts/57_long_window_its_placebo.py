"""
TSE R1 revision, T1.4 + T1.5 — Long-window ITS and widened placebo scan.

Reviewer 1 asked for a longer pre-period to "increase the statistical power" and
"widen the placebo grid". This script refits the canonical ITS on the extended
2024-01-01 -> 2026-05-10 panel and reruns the all-admissible-cutoff placebo scan
over the widened candidate set.

Nothing about the specification changes. scripts/its_spec.py is imported and
used verbatim: calendar-time axis, cutoff-DATE-centered, each candidate's own
straddling week excluded, Newey-West HAC. Candidates sit at the deadline's own
weekday (Tuesday) inside each week, and admissibility (>= MIN_PRE_WEEKS clean
pre weeks and >= MIN_POST_WEEKS clean post weeks) is the SAME pre-declared rule
as scripts/46 -- it is not retuned for the longer series, which would be a
specification search. Two stricter balance restrictions travel as sensitivity.

Regression guard: the script first refits the AS-SUBMITTED 41-week window with
this same code and asserts it reproduces the published top-tier beta2 (-4.42 pp)
and placebo rank (17/28). If that fails, nothing downstream is trustworthy.

Inputs:
  data/versions_extended.csv               (extended panel)
  data/versions.csv                        (as-submitted panel, regression guard)
  data/depsdev_dependents_2026_05_05.csv
  scripts/its_spec.py                      (imported, NOT modified)

Outputs:
  data/tse_r1_long_its.csv                 (per tier, extended vs as-submitted)
  data/tse_r1_long_placebo_scan.csv        (every candidate cutoff, all tiers)
  data/tse_r1_long_estimand_its.csv        (top tier under all three estimands)
  reports/tse_r1_long_its_placebo.md

Carried caveats:
  - Anti-causal firewall: beta2/beta3 are descriptive trajectory coefficients.
    A widened placebo grid can only ever WEAKEN or fail to weaken the
    non-identification verdict; it cannot license a causal reading.
  - The rank is a specificity heuristic, not a randomization test: candidate
    cutoffs are neither randomly assigned nor exchangeable.
  - Tier is collection-time (deps.dev 2026-05-05) applied retroactively across
    28 months; survivorship stretches with the window.

Run: conda run --no-capture-output -n stage2c python scripts/57_long_window_its_placebo.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import its_spec  # noqa: E402

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"

EXTENDED_CSV = DATA_DIR / "versions_extended.csv"
BASE_CSV = DATA_DIR / "versions.csv"
DEPSDEV_CSV = DATA_DIR / "depsdev_dependents_2026_05_05.csv"

OUT_ITS = DATA_DIR / "tse_r1_long_its.csv"
OUT_SCAN = DATA_DIR / "tse_r1_long_placebo_scan.csv"
OUT_ESTIMAND = DATA_DIR / "tse_r1_long_estimand_its.csv"
REPORT_MD = REPORTS_DIR / "tse_r1_long_its_placebo.md"

DEADLINE = pd.Timestamp("2025-12-09", tz="UTC")
CENSOR = pd.Timestamp("2026-05-10 23:59:59", tz="UTC")
BASE_ORIGIN = pd.Timestamp("2025-08-01", tz="UTC")
LONG_ORIGIN = pd.Timestamp("2024-01-01", tz="UTC")

MIN_PRE_WEEKS = 8    # pre-declared in scripts/46; NOT retuned for the long series
MIN_POST_WEEKS = 5
BALANCE_SENSITIVITIES = [(26, 13), (52, 13)]  # (min_pre, min_post) robustness

TIERS = ["top", "mid", "tail"]
PUBLISHED_BASE_BETA2_TOP = -4.42   # as-submitted paper value, section 5.1
PUBLISHED_BASE_RANK_TOP = 17       # as-submitted paper value, 17 of 28


def dd_tier(n) -> str:
    if pd.isna(n):
        return "missing"
    n = int(n)
    return "top" if n > 500 else "mid" if n > 10 else "tail"


def load_weekly(path: Path, origin: pd.Timestamp) -> pd.DataFrame:
    v = pd.read_csv(path)
    v["published_at"] = pd.to_datetime(v["published_at"], utc=True, errors="coerce")
    v = v.dropna(subset=["published_at"])
    v = v[(v["published_at"] >= origin) & (v["published_at"] <= CENSOR)]
    v["has_attestation_int"] = (
        v["has_attestation"].astype(str).str.lower() == "true").astype(int)
    depsdev = pd.read_csv(DEPSDEV_CSV)
    v["dd_tier"] = v["package"].map(dict(zip(depsdev["name"],
                                             depsdev["dependents_count"]))).apply(dd_tier)
    v["period_start"] = (v["published_at"]
                         - pd.to_timedelta(v["published_at"].dt.weekday, unit="D")).dt.normalize()
    agg = (v.groupby(["dd_tier", "period_start"])
             .agg(n_versions=("has_attestation_int", "size"),
                  n_attested=("has_attestation_int", "sum"))
             .reset_index())
    agg["rate"] = agg["n_attested"] / agg["n_versions"]
    return agg.rename(columns={"dd_tier": "tier"})


def load_rows(path: Path, origin: pd.Timestamp) -> pd.DataFrame:
    """Version rows with tier and week, before any aggregation."""
    v = pd.read_csv(path)
    v["published_at"] = pd.to_datetime(v["published_at"], utc=True, errors="coerce")
    v = v.dropna(subset=["published_at"])
    v = v[(v["published_at"] >= origin) & (v["published_at"] <= CENSOR)]
    v["has_attestation_int"] = (
        v["has_attestation"].astype(str).str.lower() == "true").astype(int)
    depsdev = pd.read_csv(DEPSDEV_CSV)
    v["dd_tier"] = v["package"].map(dict(zip(depsdev["name"],
                                             depsdev["dependents_count"]))).apply(dd_tier)
    v["period_start"] = (v["published_at"]
                         - pd.to_timedelta(v["published_at"].dt.weekday, unit="D")).dt.normalize()
    return v


def estimand_panel(rows: pd.DataFrame, estimand: str, drop_top5: bool = False) -> pd.DataFrame:
    """Weekly tier panel under one of the three named estimands (scripts/51 defs)."""
    d = rows.copy()
    if drop_top5:
        top5 = d[d["dd_tier"] == "top"].groupby("package").size().nlargest(5).index
        d = d[~d["package"].isin(top5)]
    per_pkg = (d.groupby(["dd_tier", "period_start", "package"])
                 .agg(n=("has_attestation_int", "size"), k=("has_attestation_int", "sum"))
                 .reset_index())
    per_pkg["r"] = per_pkg["k"] / per_pkg["n"]
    if estimand == "version_weighted":
        g = (d.groupby(["dd_tier", "period_start"])
               .agg(n=("has_attestation_int", "size"), k=("has_attestation_int", "sum"))
               .reset_index())
        g["rate"] = g["k"] / g["n"]
    elif estimand == "equal_package":
        g = per_pkg.groupby(["dd_tier", "period_start"]).agg(rate=("r", "mean")).reset_index()
    elif estimand == "any_attested_package":
        per_pkg["any"] = (per_pkg["k"] > 0).astype(int)
        g = per_pkg.groupby(["dd_tier", "period_start"]).agg(rate=("any", "mean")).reset_index()
    else:
        raise ValueError(estimand)
    return g.rename(columns={"dd_tier": "tier"})


def fit(weekly: pd.DataFrame, tier: str, cutoff: pd.Timestamp) -> Optional[dict]:
    sub = weekly[weekly["tier"] == tier]
    if len(sub) < 8:
        return None
    return its_spec.fit_its(sub, cutoff, period_col="period_start",
                            granularity="W", min_pre=2, min_post=2)


def candidates(weekly: pd.DataFrame, min_pre: int, min_post: int) -> list[pd.Timestamp]:
    """One cutoff per week at the deadline's weekday; pre-declared admissibility."""
    week_starts = sorted(weekly["period_start"].unique())
    n = len(week_starts)
    offset = (DEADLINE.date()
              - (DEADLINE - pd.Timedelta(days=DEADLINE.weekday())).date()).days
    return [pd.Timestamp(week_starts[k]) + pd.Timedelta(days=offset)
            for k in range(n) if k >= min_pre and (n - k - 1) >= min_post]


def scan(weekly: pd.DataFrame, cands: list[pd.Timestamp], label: str) -> pd.DataFrame:
    rows: list[dict] = []
    for tier in TIERS:
        for c in cands:
            r = fit(weekly, tier, c)
            if r is None:
                continue
            rows.append({
                "panel": label, "tier": tier, "cutoff": c.date().isoformat(),
                "is_actual": c.date() == DEADLINE.date(),
                "n_pre": r["n_pre"], "n_post": r["n_post"],
                "beta2": r["beta2_level_change_pp"],
                "ci_lo": r["ci95_beta2_low"], "ci_hi": r["ci95_beta2_high"],
                "p_beta2": r["pval_beta2"],
                "beta3": r["beta3_slope_change"], "p_beta3": r["pval_beta3"],
                "DW": r["DW"], "LB_min_p": r["LB_min_pval"],
            })
    return pd.DataFrame(rows)


def rank_of_actual(scan_df: pd.DataFrame, tier: str, col: str) -> tuple[int, int, float]:
    """Rank of |actual| among |candidates|, descending. Returns (rank, N,
    rank_fraction = rank / N). The fraction is an empirical specificity
    heuristic over non-exchangeable candidate cutoffs, NOT a p-value, and is
    never labelled `p` in any output."""
    sub = scan_df[scan_df["tier"] == tier]
    actual = sub[sub["is_actual"]]
    if actual.empty:
        return (-1, len(sub), float("nan"))
    a = abs(float(actual.iloc[0][col]))
    mags = sub[col].abs().values
    strictly_larger = int((mags > a).sum())
    n = len(mags)
    return (strictly_larger + 1, n, (strictly_larger + 1) / n)


def main() -> int:
    # ---------- regression guard on the as-submitted panel -------------------
    base_weekly = load_weekly(BASE_CSV, BASE_ORIGIN)
    base_top = fit(base_weekly, "top", DEADLINE)
    assert base_top is not None, "as-submitted top-tier fit failed"
    b2 = base_top["beta2_level_change_pp"]
    assert abs(b2 - PUBLISHED_BASE_BETA2_TOP) < 0.01, \
        f"REGRESSION: as-submitted top beta2 {b2:.4f} != published {PUBLISHED_BASE_BETA2_TOP}"
    base_scan = scan(base_weekly, candidates(base_weekly, MIN_PRE_WEEKS, MIN_POST_WEEKS),
                     "as_submitted")
    rk, nk, pk = rank_of_actual(base_scan, "top", "beta2")
    assert (rk, nk) == (PUBLISHED_BASE_RANK_TOP, 28), \
        f"REGRESSION: as-submitted rank {rk}/{nk} != published {PUBLISHED_BASE_RANK_TOP}/28"
    print(f"✓ regression guard: as-submitted top beta2 = {b2:.2f} pp, "
          f"placebo rank {rk}/{nk} (rank/N={pk:.3f}) — matches the published values", flush=True)

    # ---------- extended panel ----------------------------------------------
    long_weekly = load_weekly(EXTENDED_CSV, LONG_ORIGIN)
    n_weeks = long_weekly[long_weekly['tier'] == 'top']['period_start'].nunique()
    print(f"extended panel: {n_weeks} weeks/tier (as-submitted 41)", flush=True)

    long_cands = candidates(long_weekly, MIN_PRE_WEEKS, MIN_POST_WEEKS)
    long_scan = scan(long_weekly, long_cands, "extended")
    print(f"admissible candidates: {len(long_cands)} (as-submitted 28)", flush=True)

    its_rows: list[dict] = []
    for panel, wk in (("as_submitted", base_weekly), ("extended", long_weekly)):
        for tier in TIERS:
            r = fit(wk, tier, DEADLINE)
            if r is None:
                continue
            its_rows.append({
                "panel": panel, "tier": tier,
                "n_pre_weeks": r["n_pre"], "n_post_weeks": r["n_post"],
                "beta2_level_change_pp": round(r["beta2_level_change_pp"], 4),
                "ci95_beta2_low": round(r["ci95_beta2_low"], 4),
                "ci95_beta2_high": round(r["ci95_beta2_high"], 4),
                "ci_halfwidth_pp": round((r["ci95_beta2_high"] - r["ci95_beta2_low"]) / 2, 4),
                "pval_beta2": round(r["pval_beta2"], 4),
                "beta3_slope_change": round(r["beta3_slope_change"], 4),
                "ci95_beta3_low": round(r["ci95_beta3_low"], 4),
                "ci95_beta3_high": round(r["ci95_beta3_high"], 4),
                "pval_beta3": round(r["pval_beta3"], 4),
                "DW": round(r["DW"], 3),
                "LB_min_pval": round(r["LB_min_pval"], 4) if r["LB_min_pval"] else None,
            })
    its_df = pd.DataFrame(its_rows)
    its_df.to_csv(OUT_ITS, index=False)
    pd.concat([base_scan, long_scan]).to_csv(OUT_SCAN, index=False)
    print(f"✓ {OUT_ITS.name}  ✓ {OUT_SCAN.name}", flush=True)

    # ---------- balance sensitivities ---------------------------------------
    sens: list[dict] = []
    for mp, mq in BALANCE_SENSITIVITIES:
        cs = candidates(long_weekly, mp, mq)
        sc = scan(long_weekly, cs, f"extended_bal{mp}_{mq}")
        for tier in TIERS:
            r2, n2, p2 = rank_of_actual(sc, tier, "beta2")
            r3, _, p3 = rank_of_actual(sc, tier, "beta3")
            sens.append({"min_pre": mp, "min_post": mq, "tier": tier, "n_cand": n2,
                         "rank_b2": r2, "rank_fraction_b2": round(p2, 3),
                         "rank_b3": r3, "rank_fraction_b3": round(p3, 3)})

    # ---------- estimand sensitivity of the long-window coefficient -----------
    # Estimand sensitivity, not a decomposition: how much does the large
    # extended-window level change depend on how packages are weighted?
    # Equal-package and any-attested-package weighting re-weight the active
    # set (they do not partition the coefficient; the exact additive
    # decomposition is scripts/45's contrast), each with its own placebo scan.
    rows = load_rows(EXTENDED_CSV, LONG_ORIGIN)
    est_rows: list[dict] = []
    for label, est, drop in (("version_weighted", "version_weighted", False),
                             ("equal_package", "equal_package", False),
                             ("any_attested_package", "any_attested_package", False),
                             ("version_weighted_drop_top5", "version_weighted", True)):
        g = estimand_panel(rows, est, drop)
        r = fit(g, "top", DEADLINE)
        sc = scan(g, candidates(g, MIN_PRE_WEEKS, MIN_POST_WEEKS), label)
        rk2, n2, p2 = rank_of_actual(sc, "top", "beta2")
        est_rows.append({
            "estimand": label,
            "beta2_level_change_pp": round(r["beta2_level_change_pp"], 4),
            "ci95_beta2_low": round(r["ci95_beta2_low"], 4),
            "ci95_beta2_high": round(r["ci95_beta2_high"], 4),
            "pval_beta2": round(r["pval_beta2"], 4),
            "placebo_rank": rk2, "n_candidates": n2, "rank_fraction": round(p2, 4),
        })
    est_df = pd.DataFrame(est_rows)
    est_df.to_csv(OUT_ESTIMAND, index=False)
    print(f"✓ {OUT_ESTIMAND.name}", flush=True)

    # ---------- report -------------------------------------------------------
    L: list[str] = []
    L.append("# T1.4 + T1.5 — 长窗 ITS 与加宽 placebo 扫描（TSE R1 / R1-1）\n")
    L.append("**Anti-causal firewall**: β₂/β₃ 为描述性轨迹系数。加宽 placebo 网格"
             "只可能削弱或不改变非识别裁决，不可能反向许可因果读法。\n")
    L.append(f"**回归保护**：本脚本先用同一套代码重拟初投的 41 周面板，"
             f"复现 top β₂ = {b2:.2f} pp 与 placebo 排名 {rk}/{nk}（论文值 "
             f"{PUBLISHED_BASE_BETA2_TOP} / {PUBLISHED_BASE_RANK_TOP}/28）——断言通过。\n")
    L.append("## 1. ITS 系数：初投窗 vs 扩展窗\n")
    L.append("| 面板 | tier | pre 周 | post 周 | β₂ (pp) | 95% CI | CI 半宽 | p | β₃ | p(β₃) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for _, r in its_df.iterrows():
        L.append(f"| {r['panel']} | {r['tier']} | {r['n_pre_weeks']} | {r['n_post_weeks']} | "
                 f"{r['beta2_level_change_pp']:+.2f} | "
                 f"[{r['ci95_beta2_low']:+.2f}, {r['ci95_beta2_high']:+.2f}] | "
                 f"{r['ci_halfwidth_pp']:.2f} | {r['pval_beta2']:.3f} | "
                 f"{r['beta3_slope_change']:+.3f} | {r['pval_beta3']:.3f} |")
    L.append("\n## 2. placebo 排名（|β₂| 与 |β₃| 降序，1 = 最大）\n")
    L.append("| 面板 | tier | 候选数 | β₂ 排名 | 排名/N | β₃ 排名 | 排名/N |")
    L.append("|---|---|---|---|---|---|---|")
    for panel, sc in (("as_submitted", base_scan), ("extended", long_scan)):
        for tier in TIERS:
            r2, n2, p2 = rank_of_actual(sc, tier, "beta2")
            r3, _, p3 = rank_of_actual(sc, tier, "beta3")
            L.append(f"| {panel} | {tier} | {n2} | {r2} | {p2:.3f} | {r3} | {p3:.3f} |")
    L.append("\n## 3. 候选集平衡性敏感性（扩展窗）\n")
    L.append("长序列下沿用初投的 8/5 门槛会纳入极不平衡的切点（如 8 前 / 110 后）。"
             "主结果仍用**预先声明的同一规则**以避免规格搜索；下表为更严平衡限制下的复核。\n")
    L.append("| min_pre | min_post | tier | 候选数 | β₂ 排名 | 排名/N | β₃ 排名 | 排名/N |")
    L.append("|---|---|---|---|---|---|---|---|")
    for s in sens:
        L.append(f"| {s['min_pre']} | {s['min_post']} | {s['tier']} | {s['n_cand']} | "
                 f"{s['rank_b2']} | {s['rank_fraction_b2']} | {s['rank_b3']} | {s['rank_fraction_b3']} |")
    L.append("\n## 4. 估计量敏感性：长窗系数对加权方式的敏感程度（top tier，扩展窗）\n")
    L.append("估计量敏感性（非精确分解）。version-weighted 的大系数若在 equal-package 下坍缩，"
             "说明它对发布量加权高度敏感；范畴性的成分陈述只保留给 8 周对比的精确加法分解（scripts/45）。\n")
    L.append("| 估计量 | β₂ (pp) | 95% CI | p | placebo 排名 | 排名/N |")
    L.append("|---|---|---|---|---|---|")
    for _, r in est_df.iterrows():
        L.append(f"| {r['estimand']} | {r['beta2_level_change_pp']:+.2f} | "
                 f"[{r['ci95_beta2_low']:+.2f}, {r['ci95_beta2_high']:+.2f}] | "
                 f"{r['pval_beta2']:.3f} | {r['placebo_rank']}/{r['n_candidates']} | "
                 f"{r['rank_fraction']:.3f} |")
    L.append("")
    L.append("\n## 5. 扩展窗中 |β₂| 最大的 10 个候选切点（top tier）\n")
    top_scan = long_scan[long_scan["tier"] == "top"].copy()
    top_scan["abs_b2"] = top_scan["beta2"].abs()
    L.append("| 切点 | β₂ (pp) | 95% CI | p | pre/post 周 | 是否实际 deadline |")
    L.append("|---|---|---|---|---|---|")
    for _, r in top_scan.nlargest(10, "abs_b2").iterrows():
        L.append(f"| {r['cutoff']} | {r['beta2']:+.2f} | [{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}] | "
                 f"{r['p_beta2']:.3f} | {r['n_pre']}/{r['n_post']} | "
                 f"{'**是**' if r['is_actual'] else ''} |")
    REPORT_MD.write_text("\n".join(L) + "\n")
    print(f"✓ {REPORT_MD.name}", flush=True)
    print("\n✅ T1.4 + T1.5 complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
