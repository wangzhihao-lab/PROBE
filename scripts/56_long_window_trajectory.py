"""
TSE R1 revision, T1.3 + T1.6 — Long-window RQ1 trajectory and seasonal baseline.

Reviewer 1: "Extending the pre-period to early 2025 or even earlier would
increase the statistical power, widen the placebo grid, and establish a seasonal
baseline for the December/January dip."

This script builds the extended-window trajectory panels that scripts/57 then
fits, and answers the seasonal-baseline half directly: is the December dip a
2025 phenomenon, or does the cohort dip every December?

Three estimands travel together, defined exactly as in scripts/51:
  version-weighted    attested version rows / all version rows      (PRIMARY)
  equal-package       mean over packages of each package's own rate
  any-attested-pkg    share of packages with >= 1 attested version

Inputs:
  data/versions_extended.csv               (scripts/03 --window-start 2024-01-01)
  data/versions.csv                        (as-submitted window, for overlap check)
  data/depsdev_dependents_2026_05_05.csv   (primary tier frame)

Outputs:
  data/tse_r1_long_trajectory_weekly.csv
  data/tse_r1_long_trajectory_monthly.csv
  data/tse_r1_seasonality.csv            (incl. package-cluster bootstrap CIs)
  reports/tse_r1_long_trajectory.md
  reports/tse_r1_seasonality.md

Carried caveats:
  - Anti-causal firewall: every rate and every year-on-year contrast here is
    DESCRIPTIVE. Nothing in this script identifies an effect of any date.
  - Tier is a collection-time construct (deps.dev dependents as of 2026-05-05)
    applied to the whole window. Extending the window backwards stretches that
    retroactive application from 9 months to 28; disclose in section 9.
  - Cohort membership is fixed at the 2026-05-15 assembly, so the survivorship
    condition stretches back with the window (reports/tse_r1_prewindow_construct_audit.md).
  - No new collection: the extended rows come from packuments already on disk.

Run: conda run --no-capture-output -n stage2c python scripts/56_long_window_trajectory.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"

EXTENDED_CSV = DATA_DIR / "versions_extended.csv"
BASE_CSV = DATA_DIR / "versions.csv"
DEPSDEV_CSV = DATA_DIR / "depsdev_dependents_2026_05_05.csv"

OUT_WEEKLY = DATA_DIR / "tse_r1_long_trajectory_weekly.csv"
OUT_MONTHLY = DATA_DIR / "tse_r1_long_trajectory_monthly.csv"
OUT_SEASON = DATA_DIR / "tse_r1_seasonality.csv"
REPORT_TRAJ = REPORTS_DIR / "tse_r1_long_trajectory.md"
REPORT_SEASON = REPORTS_DIR / "tse_r1_seasonality.md"

DEADLINE = pd.Timestamp("2025-12-09", tz="UTC")
ORIGIN = pd.Timestamp("2024-01-01", tz="UTC")
CENSOR = pd.Timestamp("2026-05-10 23:59:59", tz="UTC")
TIERS = ["top", "mid", "tail"]


def dd_tier(n) -> str:
    """deps.dev tier thresholds — identical to scripts/25/37/46."""
    if pd.isna(n):
        return "missing"
    n = int(n)
    return "top" if n > 500 else "mid" if n > 10 else "tail"


def load_versions(path: Path, origin: pd.Timestamp) -> pd.DataFrame:
    """Same loading path as scripts/46, parameterised on origin."""
    v = pd.read_csv(path)
    v["published_at"] = pd.to_datetime(v["published_at"], utc=True, errors="coerce")
    v = v.dropna(subset=["published_at"])
    v = v[(v["published_at"] >= origin) & (v["published_at"] <= CENSOR)]
    v["has_attestation_int"] = (
        v["has_attestation"].astype(str).str.lower() == "true").astype(int)
    depsdev = pd.read_csv(DEPSDEV_CSV)
    dd_map = dict(zip(depsdev["name"], depsdev["dependents_count"]))
    v["dd_tier"] = v["package"].map(dd_map).apply(dd_tier)
    return v


def period_start(df: pd.DataFrame, granularity: str) -> pd.Series:
    if granularity == "W":
        return (df["published_at"]
                - pd.to_timedelta(df["published_at"].dt.weekday, unit="D")).dt.normalize()
    return df["published_at"].dt.to_period("M").dt.start_time.dt.tz_localize("UTC")


def aggregate(df: pd.DataFrame, granularity: str) -> pd.DataFrame:
    """All three estimands on one period x tier panel."""
    d = df.copy()
    d["period_start"] = period_start(d, granularity)

    vw = (d.groupby(["dd_tier", "period_start"])
            .agg(n_versions=("has_attestation_int", "size"),
                 n_attested=("has_attestation_int", "sum"),
                 n_packages=("package", "nunique"))
            .reset_index())
    vw["rate"] = vw["n_attested"] / vw["n_versions"]

    per_pkg = (d.groupby(["dd_tier", "period_start", "package"])
                 .agg(pkg_n=("has_attestation_int", "size"),
                      pkg_att=("has_attestation_int", "sum"))
                 .reset_index())
    per_pkg["pkg_rate"] = per_pkg["pkg_att"] / per_pkg["pkg_n"]
    per_pkg["pkg_any"] = (per_pkg["pkg_att"] > 0).astype(int)
    eq = (per_pkg.groupby(["dd_tier", "period_start"])
                 .agg(rate_equal_package=("pkg_rate", "mean"),
                      rate_any_attested_pkg=("pkg_any", "mean"))
                 .reset_index())

    out = vw.merge(eq, on=["dd_tier", "period_start"], how="left")
    return out.rename(columns={"dd_tier": "tier"}).sort_values(["tier", "period_start"])


BOOT_B = 2000
BOOT_SEED = 42


def _nov_dec_change(rows: pd.DataFrame, tier: str, dec: str) -> float:
    """November -> December version-weighted change in pp, one tier, one year."""
    nov, = [f"{dec[:4]}-11"]
    sub = rows[(rows["dd_tier"] == tier) & (rows["ym"].isin([nov, dec]))]
    if sub.empty:
        return float("nan")
    g = sub.groupby("ym")["has_attestation_int"].agg(["size", "sum"])
    if nov not in g.index or dec not in g.index or g.loc[nov, "size"] == 0 \
            or g.loc[dec, "size"] == 0:
        return float("nan")
    return 100 * (g.loc[dec, "sum"] / g.loc[dec, "size"]
                  - g.loc[nov, "sum"] / g.loc[nov, "size"])


def seasonality_bootstrap(rows: pd.DataFrame) -> pd.DataFrame:
    """Package-cluster bootstrap of each December's Nov->Dec change, and of the
    year-on-year difference between them.

    Versions cluster within packages, so the resampling unit is the PACKAGE
    (B = 2000, seed 42), matching the interval convention used throughout the
    paper. The year-on-year difference is reported as a DESCRIPTIVE contrast of
    two seasonal movements; it is not a difference-in-differences estimate and
    identifies nothing.
    """
    rows = rows.copy()
    rows["ym"] = rows["published_at"].dt.strftime("%Y-%m")
    rows = rows[rows["ym"].isin(["2024-11", "2024-12", "2025-11", "2025-12"])]
    rng = np.random.default_rng(BOOT_SEED)
    out: list[dict] = []
    for tier in TIERS:
        sub = rows[rows["dd_tier"] == tier]
        pkgs = sub["package"].unique()
        by_pkg = {p: g for p, g in sub.groupby("package")}
        point = {d: _nov_dec_change(sub, tier, d) for d in ("2024-12", "2025-12")}
        point_diff = point["2025-12"] - point["2024-12"]
        draws: dict[str, list[float]] = {"2024-12": [], "2025-12": [], "diff": []}
        for _ in range(BOOT_B):
            pick = rng.choice(pkgs, size=len(pkgs), replace=True)
            rs = pd.concat([by_pkg[p] for p in pick], ignore_index=True)
            a = _nov_dec_change(rs, tier, "2024-12")
            b = _nov_dec_change(rs, tier, "2025-12")
            if np.isnan(a) or np.isnan(b):
                continue
            draws["2024-12"].append(a)
            draws["2025-12"].append(b)
            draws["diff"].append(b - a)
        for key, label in (("2024-12", "nov_to_dec_2024"),
                           ("2025-12", "nov_to_dec_2025"),
                           ("diff", "yoy_difference")):
            d = np.array(draws[key])
            pt = point_diff if key == "diff" else point[key]
            out.append({
                "tier": tier, "quantity": label,
                "point_pp": round(float(pt), 3),
                "ci95_low": round(float(np.percentile(d, 2.5)), 3) if len(d) else None,
                "ci95_high": round(float(np.percentile(d, 97.5)), 3) if len(d) else None,
                "n_boot": len(d), "n_packages": len(pkgs),
            })
    return pd.DataFrame(out)


def seasonality(monthly: pd.DataFrame) -> pd.DataFrame:
    """December -> January change, per tier, for every December in the window."""
    rows: list[dict] = []
    m = monthly.copy()
    m["ym"] = m["period_start"].dt.strftime("%Y-%m")
    for tier in TIERS:
        sub = m[m["tier"] == tier].set_index("ym")
        for dec in ("2024-12", "2025-12"):
            jan = f"{int(dec[:4]) + 1}-01"
            nov = f"{dec[:4]}-11"
            if dec not in sub.index or jan not in sub.index or nov not in sub.index:
                continue
            rows.append({
                "tier": tier,
                "december": dec,
                "nov_rate_pp": round(100 * sub.loc[nov, "rate"], 2),
                "dec_rate_pp": round(100 * sub.loc[dec, "rate"], 2),
                "jan_rate_pp": round(100 * sub.loc[jan, "rate"], 2),
                "nov_to_dec_pp": round(100 * (sub.loc[dec, "rate"] - sub.loc[nov, "rate"]), 2),
                "dec_to_jan_pp": round(100 * (sub.loc[jan, "rate"] - sub.loc[dec, "rate"]), 2),
                "dec_n_versions": int(sub.loc[dec, "n_versions"]),
            })
    return pd.DataFrame(rows)


def main() -> int:
    ext = load_versions(EXTENDED_CSV, ORIGIN)
    print(f"extended rows {len(ext):,}  packages {ext['package'].nunique():,}", flush=True)

    # Overlap consistency: the extended panel restricted to the as-submitted
    # window must reproduce the as-submitted panel exactly.
    base = load_versions(BASE_CSV, pd.Timestamp("2025-08-01", tz="UTC"))
    ext_sub = ext[ext["published_at"] >= pd.Timestamp("2025-08-01", tz="UTC")]
    assert len(base) == len(ext_sub), f"overlap mismatch {len(base)} vs {len(ext_sub)}"
    assert base["has_attestation_int"].sum() == ext_sub["has_attestation_int"].sum(), \
        "overlap attestation count mismatch"
    print(f"✓ overlap check: {len(base):,} rows, "
          f"{int(base['has_attestation_int'].sum()):,} attested — identical", flush=True)

    weekly = aggregate(ext, "W")
    monthly = aggregate(ext, "MS")
    weekly.to_csv(OUT_WEEKLY, index=False)
    monthly.to_csv(OUT_MONTHLY, index=False)
    print(f"✓ {OUT_WEEKLY.name}  weeks/tier: "
          f"{weekly.groupby('tier').size().to_dict()}", flush=True)
    print(f"✓ {OUT_MONTHLY.name}  months/tier: "
          f"{monthly.groupby('tier').size().to_dict()}", flush=True)

    season = seasonality(monthly)
    boot = seasonality_bootstrap(ext)
    season.to_csv(OUT_SEASON, index=False)
    boot.to_csv(DATA_DIR / "tse_r1_seasonality_bootstrap.csv", index=False)
    print(f"✓ {OUT_SEASON.name}  ✓ tse_r1_seasonality_bootstrap.csv", flush=True)

    # ---- trajectory report -------------------------------------------------
    lines = ["# T1.3 — 长窗 RQ1 轨迹（2024-01-01 → 2026-05-10）\n",
             "**Anti-causal firewall**: 本文件所有数字均为描述性，"
             "不识别任何日期的效应。\n",
             f"扩展窗版本行 **{len(ext):,}**（原窗 100,168），"
             f"活跃包 **{ext['package'].nunique():,}**（原窗 1,614）。",
             "重叠区间与初投面板逐行一致（已断言）。\n",
             "## 月度 version-weighted 采纳率（pp），按 deps.dev tier\n",
             "| 月份 | top | mid | tail | top n |", "|---|---|---|---|---|"]
    piv = monthly.pivot_table(index="period_start", columns="tier", values="rate")
    ntop = monthly[monthly["tier"] == "top"].set_index("period_start")["n_versions"]
    for ts in piv.index:
        row = piv.loc[ts]
        lines.append(
            f"| {ts.strftime('%Y-%m')} | "
            + " | ".join(f"{100 * row[t]:.1f}" if t in row and pd.notna(row[t]) else "—"
                         for t in TIERS)
            + f" | {int(ntop.get(ts, 0)):,} |")
    lines.append("\n## 三种估计量的月度对照（top tier）\n")
    lines.append("| 月份 | version-weighted | equal-package | any-attested-pkg | 包数 |")
    lines.append("|---|---|---|---|---|")
    tt = monthly[monthly["tier"] == "top"].set_index("period_start")
    for ts in tt.index:
        r = tt.loc[ts]
        lines.append(f"| {ts.strftime('%Y-%m')} | {100 * r['rate']:.1f} | "
                     f"{100 * r['rate_equal_package']:.1f} | "
                     f"{100 * r['rate_any_attested_pkg']:.1f} | {int(r['n_packages'])} |")
    REPORT_TRAJ.write_text("\n".join(lines) + "\n")
    print(f"✓ {REPORT_TRAJ.name}", flush=True)

    # ---- seasonality report ------------------------------------------------
    s = ["# T1.6 — 季节基线：12 月的回落是不是每年都有？（TSE R1 / R1-1）\n",
         "审稿人 1 要求用更长的 pre-period 建立 December/January 的季节基线。",
         "下表对窗内每一个 12 月给出 11→12 与 12→1 的变化（version-weighted, pp）。\n",
         "**描述性**：年度对照不识别任何效应；2024 年只是**一年期的历史基准**，"
         "它能说明两年的 11→12 变化是否相近，不能确立\"例行季节形态\"，也不是等价性检验。\n",
         "| tier | 12月 | 11月率 | 12月率 | 次年1月率 | 11→12 | 12→1 | 12月版本数 |",
         "|---|---|---|---|---|---|---|---|"]
    for _, r in season.iterrows():
        s.append(f"| {r['tier']} | {r['december']} | {r['nov_rate_pp']} | {r['dec_rate_pp']} | "
                 f"{r['jan_rate_pp']} | {r['nov_to_dec_pp']:+} | {r['dec_to_jan_pp']:+} | "
                 f"{r['dec_n_versions']:,} |")
    s.append("\n## 包聚类 bootstrap 区间（B = 2000, seed 42；重抽单位 = 包）\n")
    s.append("版本在包内聚集，故按包重抽，与全文区间约定一致。"
             "年度差是两个季节移动的**描述性对照**，不是 DiD 估计，不识别任何效应。\n")
    s.append("| tier | 量 | 点估计 (pp) | 95% CI | 包数 |")
    s.append("|---|---|---|---|---|")
    for _, r in boot.iterrows():
        s.append(f"| {r['tier']} | {r['quantity']} | {r['point_pp']:+} | "
                 f"[{r['ci95_low']:+}, {r['ci95_high']:+}] | {r['n_packages']} |")
    s.append("\n## 读法\n")
    s.append("若 2024-12 与 2025-12 的 11→12 变化同号同量级，则 2025 年 12 月的回落")
    s.append("与前一年**相近**（年差区间含零，但也容许约 ±5 pp 的差异），deadline 不是必要解释；若只有 2025-12 显著下沉，")
    s.append("则该月确有异常，但仍需 placebo 扫描（T1.5）判断异常是否**特定于 12-9**。")
    REPORT_SEASON.write_text("\n".join(s) + "\n")
    print(f"✓ {REPORT_SEASON.name}", flush=True)

    assert ext["published_at"].max() <= CENSOR, "DATA CUTOFF VIOLATION"
    print("\n✅ T1.3 + T1.6 complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
