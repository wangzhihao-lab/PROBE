"""
TSE R1 revision, T1.7 (redesigned) — RQ2 evidence-state composition across the
boundary, with no new collection.

The original plan was to draw a pre-deadline classification sample and compare
labels. That design is empty: the A--F label is a function of collection-time
repository state, so it is PACKAGE-CONSTANT (verified here by assertion; the
500-row sample has 97 multi-version packages and zero label disagreements).
Re-labelling earlier versions of the same packages would reproduce the same
labels by construction. Labelling NEW packages would require fetching workflows
in 2026-09, four months past the data cutoff -- forbidden by constraint C-1.

The package-constancy is itself the instrument. If no package can change its
evidence state, then any shift in the evidence-state mix of the non-provenance
pool across the boundary is attributable to one thing only: which packages are
in the pool and how many rows each contributes. This is the composition finding
of RQ1, restated in the RQ2 domain, and it answers reviewer 2's point that A--F
is a descriptive rather than a diagnostic instrument -- we grant that, and show
what the descriptive instrument can still isolate.

Design:
  universe   packages that (a) carry an A--F label and (b) have non-attested
             versions on BOTH sides of 2025-12-09
  control    package-level (equal-weight) A--F distribution, pre vs post.
             Identical by construction; reported to make the invariance visible.
  finding    row-weighted A--F distribution, pre vs post. Any movement is
             100% publishing-volume reallocation.

Inputs:
  data/classification.csv              (500 labelled rows, final A--F rule)
  data/non_provenance_sample.csv       (eco sampling stratum per sampled row)
  data/versions_extended.csv           (extended non-provenance pool)
  data/depsdev_dependents_2026_05_05.csv

Outputs:
  data/tse_r1_rq2_cross_boundary.csv
  reports/tse_r1_rq2_cross_boundary.md

Carried caveats:
  - A--F describes the COLLECTION-TIME evidence state (mid-May 2026) of a
    package, not the publish-time workflow state of any version. Applying those
    labels to versions published as early as 2024 widens that gap; the labels
    are still not claims about what any publish actually did.
  - The universe is the 500-row sample's packages, not the full pool, so shares
    are design-consistent for that sample, not ecosystem prevalence.
  - No new collection: labels and workflow evidence are reused as-is.

Run: conda run --no-capture-output -n stage2c python scripts/59_rq2_cross_boundary.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
REPORTS = ROOT / "reports"

CLASSIFICATION_CSV = DATA / "classification.csv"
SAMPLE_CSV = DATA / "non_provenance_sample.csv"
EXTENDED_CSV = DATA / "versions_extended.csv"
DEPSDEV_CSV = DATA / "depsdev_dependents_2026_05_05.csv"

OUT_CSV = DATA / "tse_r1_rq2_cross_boundary.csv"
OUT_MD = REPORTS / "tse_r1_rq2_cross_boundary.md"

DEADLINE = pd.Timestamp("2025-12-09 23:59:59.999999", tz="UTC")
CENSOR = pd.Timestamp("2026-05-10 23:59:59", tz="UTC")
CLASSES = ["A", "B", "C", "D", "E"]
BOOT_B = 2000
BOOT_SEED = 42


def dd_tier(n) -> str:
    if pd.isna(n):
        return "missing"
    n = int(n)
    return "top" if n > 500 else "mid" if n > 10 else "tail"


def load_labels() -> pd.DataFrame:
    """package -> A--F label, with the package-constancy assertion."""
    c = pd.read_csv(CLASSIFICATION_CSV)
    per_pkg = c.groupby("package")["class"].nunique()
    non_constant = per_pkg[per_pkg > 1]
    assert len(non_constant) == 0, \
        f"A--F is not package-constant for {len(non_constant)} packages: {list(non_constant.index)[:5]}"
    multi = c.groupby("package").size()
    print(f"labels: {len(c)} rows, {c['package'].nunique()} packages, "
          f"{int((multi > 1).sum())} multi-version, 0 label disagreements ✓", flush=True)
    lab = c.groupby("package")["class"].first().rename("class").reset_index()
    eco = pd.read_csv(SAMPLE_CSV)[["package", "tier"]].drop_duplicates("package")
    return lab.merge(eco.rename(columns={"tier": "eco_tier"}), on="package", how="left")


def load_pool(labels: pd.DataFrame) -> pd.DataFrame:
    """Non-attested rows of the extended window, restricted to labelled packages."""
    v = pd.read_csv(EXTENDED_CSV)
    v["published_at"] = pd.to_datetime(v["published_at"], utc=True, errors="coerce")
    v = v.dropna(subset=["published_at"])
    v = v[v["published_at"] <= CENSOR]
    v = v[v["has_attestation"].astype(str).str.lower() != "true"]
    v = v.merge(labels, on="package", how="inner")
    v["side"] = np.where(v["published_at"] > DEADLINE, "post", "pre")
    depsdev = pd.read_csv(DEPSDEV_CSV)
    v["dd_tier"] = v["package"].map(dict(zip(depsdev["name"],
                                             depsdev["dependents_count"]))).apply(dd_tier)
    return v


def shares(df: pd.DataFrame, weight: str) -> pd.Series:
    """Class shares in percentage points, row-weighted or package-weighted."""
    if weight == "row":
        counts = df.groupby("class").size()
    else:
        counts = df.groupby("class")["package"].nunique()
    counts = counts.reindex(CLASSES, fill_value=0)
    total = counts.sum()
    return 100 * counts / total if total else counts.astype(float)


def bootstrap_shift(pool: pd.DataFrame, pkgs: np.ndarray) -> dict[str, tuple[float, float]]:
    """Package-cluster bootstrap of the post-minus-pre row-weighted share shift."""
    rng = np.random.default_rng(BOOT_SEED)
    by_pkg = {p: g for p, g in pool.groupby("package")}
    draws: dict[str, list[float]] = {c: [] for c in CLASSES}
    for _ in range(BOOT_B):
        pick = rng.choice(pkgs, size=len(pkgs), replace=True)
        rs = pd.concat([by_pkg[p] for p in pick], ignore_index=True)
        pre_s = shares(rs[rs["side"] == "pre"], "row")
        post_s = shares(rs[rs["side"] == "post"], "row")
        for c in CLASSES:
            draws[c].append(float(post_s.get(c, 0.0) - pre_s.get(c, 0.0)))
    return {c: (float(np.percentile(draws[c], 2.5)),
                float(np.percentile(draws[c], 97.5))) for c in CLASSES}


def main() -> int:
    labels = load_labels()
    pool = load_pool(labels)

    both = (pool.groupby("package")["side"].nunique() == 2)
    universe_af = both[both].index.to_numpy()
    common_af = pool[pool["package"].isin(universe_af)].copy()
    # The shares below condition on A--E (F = no repository URL is excluded from
    # every A--E denominator in the paper), so the analysis universe, the driver
    # table and the bootstrap all use the SAME A--E population; F is reported
    # separately so no denominator mixes the two.
    common = common_af[common_af["class"].isin(CLASSES)].copy()
    universe = common["package"].drop_duplicates().to_numpy()
    f_rows = common_af[~common_af["class"].isin(CLASSES)]
    n_f_pkgs = int(f_rows["package"].nunique())
    print(f"universe (A--F): {len(universe_af)} packages with non-attested rows on both "
          f"sides ({len(common_af):,} rows); A--E analysis universe: {len(universe)} "
          f"packages ({len(common):,} rows: {int((common['side'] == 'pre').sum()):,} pre / "
          f"{int((common['side'] == 'post').sum()):,} post); F set aside: {n_f_pkgs} "
          f"packages, {len(f_rows):,} rows", flush=True)

    pre, post = common[common["side"] == "pre"], common[common["side"] == "post"]
    pkg_pre, pkg_post = shares(pre, "pkg"), shares(post, "pkg")
    row_pre, row_post = shares(pre, "row"), shares(post, "row")
    ci = bootstrap_shift(common, universe)

    # invariance control: package-level shares must be identical on the common set
    max_pkg_gap = float((pkg_post - pkg_pre).abs().max())
    assert max_pkg_gap < 1e-9, \
        f"package-level shares should be identical by construction; max gap {max_pkg_gap}"
    print(f"✓ invariance control: package-level A--E shares identical pre/post "
          f"(max gap {max_pkg_gap:.2e})", flush=True)

    rows = []
    for c in CLASSES:
        rows.append({
            "scope": "all_tiers", "class": c,
            "pkg_share_pre_pp": round(float(pkg_pre[c]), 2),
            "pkg_share_post_pp": round(float(pkg_post[c]), 2),
            "row_share_pre_pp": round(float(row_pre[c]), 2),
            "row_share_post_pp": round(float(row_post[c]), 2),
            "row_shift_pp": round(float(row_post[c] - row_pre[c]), 2),
            "shift_ci_low": round(ci[c][0], 2), "shift_ci_high": round(ci[c][1], 2),
        })
    for tier in ("top", "mid", "tail"):
        sub = common[common["dd_tier"] == tier]
        if sub.empty:
            continue
        rp, rq = shares(sub[sub["side"] == "pre"], "row"), shares(sub[sub["side"] == "post"], "row")
        pp, pq = shares(sub[sub["side"] == "pre"], "pkg"), shares(sub[sub["side"] == "post"], "pkg")
        for c in CLASSES:
            rows.append({
                "scope": f"dd_{tier}", "class": c,
                "pkg_share_pre_pp": round(float(pp[c]), 2),
                "pkg_share_post_pp": round(float(pq[c]), 2),
                "row_share_pre_pp": round(float(rp[c]), 2),
                "row_share_post_pp": round(float(rq[c]), 2),
                "row_shift_pp": round(float(rq[c] - rp[c]), 2),
                "shift_ci_low": None, "shift_ci_high": None,
            })
    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"✓ {OUT_CSV.name}", flush=True)

    # which packages drive the row-weighted shift
    vol = (common.groupby(["package", "class", "side"]).size().unstack("side", fill_value=0)
           .reset_index())
    vol["delta_rows"] = vol.get("post", 0) - vol.get("pre", 0)
    top_drivers = vol.reindex(vol["delta_rows"].abs().sort_values(ascending=False).index).head(10)

    L = ["# T1.7 — 非-provenance 池的证据态构成跨边界变化（TSE R1 / R2-1d）\n",
         "**设计要点**：A–F 标签是 2026 年 5 月采集时刻仓库状态的函数，按构造**每包一个标签**"
         f"（{len(labels)} 个已标注包，多版本包中标签不一致者 0 个，脚本内断言）。",
         "这是**测量构造**，不是经验发现：分类器把同一份 2026 年 5 月仓库状态套用到该包的每一次发布，",
         "因此它观测不到任何历史上的状态变化。在标签按包固定、且包集合固定的条件下，",
         "包级份额前后相等是代数恒等式，行加权份额的变化则只能经由各包贡献行数（发布量权重）产生。",
         "下面的分解是**固定标签下的精确重加权分解**，不是关于包行为是否改变的检验。\n",
         f"**分析域**：{len(universe_af)} 个在 2025-12-09 两侧都有非-attested 版本的已标注包（A–F，"
         f"{len(common_af):,} 行）；其中 A–E 分析域 {len(universe)} 个包、{len(common):,} 行"
         f"（前 {len(pre):,} / 后 {len(post):,}）；F（无仓库 URL）单列：{n_f_pkgs} 个包、"
         f"{len(f_rows):,} 行，不进入任何 A–E 分母。\n",
         "## 1. 不变性对照（包级等权）——固定标签 + 固定包集合下的恒等式\n",
         "| 类 | 前（包级 %） | 后（包级 %） | 差 |", "|---|---|---|---|"]
    for c in CLASSES:
        L.append(f"| {c} | {pkg_pre[c]:.2f} | {pkg_post[c]:.2f} | "
                 f"{pkg_post[c] - pkg_pre[c]:+.2f} |")
    L += ["", f"最大偏差 {max_pkg_gap:.1e}——与恒等式一致（这只核对了实现，不是经验结论）。\n",
          "## 2. 行加权分布（发布量加权）——这里才有变化\n",
          "包聚类 bootstrap，B = 2000，seed 42，重抽单位 = 包。\n",
          "| 类 | 前（行 %） | 后（行 %） | 变化 (pp) | 95% CI |", "|---|---|---|---|---|"]
    for c in CLASSES:
        lo, hi = ci[c]
        star = " **" if (lo > 0 or hi < 0) else ""
        L.append(f"| {c} | {row_pre[c]:.2f} | {row_post[c]:.2f} | "
                 f"{row_post[c] - row_pre[c]:+.2f}{star} | [{lo:+.2f}, {hi:+.2f}] |")
    L += ["", "（`**` = 区间排除零）\n",
          "## 3. 分 deps.dev tier 的行加权变化\n",
          "| tier | 类 | 前 | 后 | 变化 (pp) |", "|---|---|---|---|---|"]
    for _, r in out[out["scope"] != "all_tiers"].iterrows():
        L.append(f"| {r['scope'].replace('dd_', '')} | {r['class']} | "
                 f"{r['row_share_pre_pp']} | {r['row_share_post_pp']} | {r['row_shift_pp']:+} |")
    L += ["", "## 4. 驱动行加权移动的包（A–E 域内，|Δ行数| 前 10）\n",
          "| 包 | 类 | 前行数 | 后行数 | Δ |", "|---|---|---|---|---|"]
    for _, r in top_drivers.iterrows():
        L.append(f"| `{r['package']}` | {r['class']} | {int(r.get('pre', 0)):,} | "
                 f"{int(r.get('post', 0)):,} | {int(r['delta_rows']):+,} |")
    L += ["", "## 5. 可写进论文的结论（措辞已按方法学评审收窄）\n",
          "在标签按构造固定、且限定于两侧共有的 A–E 包集合的条件下，非-provenance 池的",
          "行加权证据态份额跨边界的移动**由发布量权重的重新分配生成**（精确重加权分解）。",
          "这与 RQ1 的 composition 结果（version-weighted β₂ = −20.13 pp 在 equal-package 下坍缩为 −4.07 pp）",
          "方向一致，但它**不是**\"没有任何包改变了证据态\"的经验证据：分类器只见 2026 年 5 月一份状态，",
          "历史上的状态变化对它不可见。A–F 仍是**描述性证据态工具**（R2-1d 所言不虚）。\n",
          "**构念警示**：标签描述 2026 年 5 月的仓库状态。把它应用到最早 2024 年的发布，",
          "发布与观测之间的间隔比初投稿更长；标签仍然不是关于任何一次发布实际走了什么路径的断言。"]
    OUT_MD.write_text("\n".join(L) + "\n")
    print(f"✓ {OUT_MD.name}", flush=True)
    print("\n✅ T1.7 complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
