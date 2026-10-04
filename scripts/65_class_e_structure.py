"""
TSE R1 revision, T2.7 — Structuring the Class E pool (reviewer 2, R2-1a).

Reviewer 2: Class E is a large "other" category whose internal heterogeneity is
under-analysed. The paper already splits E deterministically by packument
monorepo markers and release-automation hints (scripts/49: E-dir / E-auto /
E-plain). This script adds the orthogonal axis the frozen record supports --
what the repository's own automation surface looked like at collection time --
and crosses the two, turning "other" into a 2x3 structured space:

  activity axis (from data/repo_status.csv, frozen):
    E-active-CI     cached GitHub Actions workflows exist; none is a publish
                    workflow (and no external-CI config was recorded, else C)
    E-no-workflows  no workflow cached at all: no automation visible anywhere
  marker axis (from data/class_e_subtypes.csv, scripts/49, untouched):
    E-dir / E-auto / E-plain

Every signal is presence-only and collection-time; no sub-state upgrades to a
verified publish path. scripts/49 and its outputs are NOT modified.

Inputs:  data/classification.csv, data/class_e_subtypes.csv, data/repo_status.csv
Outputs: data/tse_r1_class_e_structure.csv, reports/tse_r1_class_e_structure.md

Run: conda run --no-capture-output -n stage2c python scripts/65_class_e_structure.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
DATA, REPORTS = ROOT / "data", ROOT / "reports"
OUT_CSV = DATA / "tse_r1_class_e_structure.csv"
OUT_MD = REPORTS / "tse_r1_class_e_structure.md"


def dd_tier(n) -> str:
    if pd.isna(n):
        return "missing"
    n = int(n)
    return "top" if n > 500 else "mid" if n > 10 else "tail"


def main() -> int:
    c = pd.read_csv(DATA / "classification.csv")
    e = c[c["class"] == "E"].copy()
    sub = pd.read_csv(DATA / "class_e_subtypes.csv")[["package", "version", "subtype"]]
    e = e.merge(sub, on=["package", "version"], how="left")
    assert e["subtype"].notna().all(), "every E row must carry a scripts/49 subtype"
    rs = pd.read_csv(DATA / "repo_status.csv")
    e = e.merge(rs[["package", "version", "workflow_count", "repo_status"]],
                on=["package", "version"], how="left")
    dm = pd.read_csv(DATA / "depsdev_dependents_2026_05_05.csv")
    e["dd_tier"] = e["package"].map(dict(zip(dm["name"], dm["dependents_count"]))).apply(dd_tier)
    e["activity"] = (e["workflow_count"].fillna(0) > 0).map(
        {True: "E-active-CI", False: "E-no-workflows"})
    e.to_csv(OUT_CSV, index=False)
    n = len(e)
    print(f"E rows: {n}  packages: {e['package'].nunique()}", flush=True)

    xt = e.groupby(["subtype", "activity"]).size().unstack(fill_value=0)
    xt["rows"] = xt.sum(axis=1)
    top = e[e["dd_tier"] == "top"]
    xt_top = top.groupby(["subtype", "activity"]).size().unstack(fill_value=0)

    L = ["# T2.7 — Class E 的结构化（R2-1a）\n",
         f"E 池 {n} 行 / {e['package'].nunique()} 包。两条确定性轴交叉：",
         "packument 标记轴（scripts/49：E-dir / E-auto / E-plain，未改动）×",
         "自动化活动轴（repo_status：有无缓存 workflow）。全部 presence-only。\n",
         "## 1. 全池交叉表（行数）\n",
         "| 标记 \\\\ 活动 | E-active-CI | E-no-workflows | 合计 | 份额 |",
         "|---|---|---|---|---|"]
    for idx, r in xt.iterrows():
        L.append(f"| {idx} | {r.get('E-active-CI', 0)} | {r.get('E-no-workflows', 0)} | "
                 f"{r['rows']} | {100 * r['rows'] / n:.1f}% |")
    L += ["", "## 2. top tier 的同一张表\n",
          "| 标记 \\\\ 活动 | E-active-CI | E-no-workflows |", "|---|---|---|"]
    for idx, r in xt_top.iterrows():
        L.append(f"| {idx} | {r.get('E-active-CI', 0)} | {r.get('E-no-workflows', 0)} |")
    act = e.groupby("activity").size()
    L += ["", "## 3. 六个子态的解读（写进 §5.2 / supplement 的底稿）\n",
          f"- **E-dir × active-CI**：monorepo 标记 + 仓库有 CI 但无发布 workflow——",
          "  发布自动化在包根之外（@types/sinon 型）最一致的形态。",
          "- **E-dir × no-workflows**：monorepo 标记但整个仓库无任何缓存 workflow——",
          "  自动化可能整体位于别处（外部服务或私有基础设施）。",
          "- **E-plain × active-CI**：有 CI 活动（测试/lint）但发布不可见——发布路径独立于",
          "  可见 CI；\"手动发布\"与\"外部工具\"都相容，不可区分。",
          "- **E-plain × no-workflows**：完全无可见自动化——最接近\"手动发布\"的画像，",
          "  但仍只是 absence-of-evidence。",
          f"- 活动轴合计：E-active-CI {act.get('E-active-CI', 0)} 行 / "
          f"E-no-workflows {act.get('E-no-workflows', 0)} 行。\n",
          "## 4. 结论句\n",
          "E 不再是无结构的\"其他\"：两条冻结轴把它切成六个可命名子态，每个子态对应",
          "一组明确的（但仓库证据无法裁决的）候选发布机制。诊断性上限来自数据构念本身，",
          "而非分析缺位——这正是回复 R2-1a 的立足点。"]
    OUT_MD.write_text("\n".join(L) + "\n")
    print(f"✓ {OUT_CSV.name}  ✓ {OUT_MD.name}\n✅ T2.7 complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
