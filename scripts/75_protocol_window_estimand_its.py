"""
TSE R1 independent-audit follow-up — Estimand companions on the protocol window.

Why: Reviewer 1 (item R1-5) asked for the cutoff-centered ITS and its placebo
rank under an equal-package estimand (or excluding the most prolific
packages) beside the version-weighted primary. scripts/57 supplied those
companions on the extended series only; the paper keeps the protocol window
as its frozen anchor, so the same companions are needed on that window.
Nothing here changes the primary estimand or any published number: the
version-weighted row is a regression guard that must reproduce the published
anchor (-4.42 pp, placebo rank 17 of 28).

Design: scripts/57's own panel builder, fitter, candidate rule (eight pre-,
five post-weeks; same weekday as the deadline) and rank function are imported
unchanged and applied to the protocol-window rows (origin 2025-08-01, censor
2026-05-10). Three estimands: version-weighted (guard), equal-package, and
version-weighted with the five packages most prolific in the protocol window
dropped. The placebo scan uses the same candidate set in every panel.

Inputs:
  data/versions.csv, data/depsdev_dependents_2026_05_05.csv
  scripts/57_long_window_its_placebo.py (imported; not modified)
Outputs:
  data/tse_r1_protocol_estimand_its.csv
  reports/tse_r1_protocol_estimand_its.md

Run: conda run --no-capture-output -n stage2c python scripts/75_protocol_window_estimand_its.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
DATA, REPORTS, SCRIPTS = ROOT / "data", ROOT / "reports", ROOT / "scripts"
OUT_CSV = DATA / "tse_r1_protocol_estimand_its.csv"
OUT_MD = REPORTS / "tse_r1_protocol_estimand_its.md"

_spec = importlib.util.spec_from_file_location("s57", SCRIPTS / "57_long_window_its_placebo.py")
s57 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s57)  # type: ignore[union-attr]

PANELS = (("version_weighted", "version_weighted", False),
          ("equal_package", "equal_package", False),
          ("version_weighted_drop_top5", "version_weighted", True))


def main() -> int:
    rows = s57.load_rows(s57.BASE_CSV, s57.BASE_ORIGIN)
    out = []
    for label, est, drop in PANELS:
        weekly = s57.estimand_panel(rows, est, drop)
        r = s57.fit(weekly, "top", s57.DEADLINE)
        assert r is not None, label
        cands = s57.candidates(weekly, s57.MIN_PRE_WEEKS, s57.MIN_POST_WEEKS)
        sc = s57.scan(weekly, cands, f"protocol_{label}")
        rank, n, frac = s57.rank_of_actual(sc, "top", "beta2")
        out.append({"window": "protocol", "estimand": label, "n_pre_weeks": r["n_pre"], "n_post_weeks": r["n_post"],
                    "beta2_level_change_pp": round(r["beta2_level_change_pp"], 4),
                    "ci95_beta2_low": round(r["ci95_beta2_low"], 4), "ci95_beta2_high": round(r["ci95_beta2_high"], 4),
                    "pval_beta2": round(r["pval_beta2"], 4), "DW": round(r["DW"], 3),
                    "LB_min_pval": round(r["LB_min_pval"], 4) if r["LB_min_pval"] is not None else None,
                    "placebo_rank": rank, "n_candidates": n, "rank_fraction": round(frac, 3)})
        print(f"{label:28s} beta2={r['beta2_level_change_pp']:+.2f} [{r['ci95_beta2_low']:+.2f}, {r['ci95_beta2_high']:+.2f}] "
              f"rank {rank}/{n} DW={r['DW']:.2f} LB={r['LB_min_pval']:.3f}", flush=True)
    df = pd.DataFrame(out)
    guard = df[df["estimand"] == "version_weighted"].iloc[0]
    assert abs(guard["beta2_level_change_pp"] - s57.PUBLISHED_BASE_BETA2_TOP) < 0.01, "REGRESSION: anchor beta2"
    assert int(guard["placebo_rank"]) == s57.PUBLISHED_BASE_RANK_TOP and int(guard["n_candidates"]) == 28, "REGRESSION: anchor rank"
    df.to_csv(OUT_CSV, index=False)
    print(f"✓ regression guard: anchor reproduced ({guard['beta2_level_change_pp']:+.2f} pp, rank {int(guard['placebo_rank'])}/28)")
    L = ["# R1-5 同窗配套：protocol window 的估计量伴随 ITS 与 placebo 排名（TSE R1 独立复核后补）\n",
         "scripts/57 的面板构造、拟合、候选规则（≥8 pre / ≥5 post 周、与 deadline 同星期几）与排名函数原样导入，",
         "施加于 protocol window（2025-08-01 起，2026-05-10 截止）的版本行；version-weighted 行为回归保护（须复现 −4.42 pp、17/28）。\n",
         "| 估计量 | β₂ (pp) | 95% HAC CI | p | DW | LB min p | placebo 排名 | 排名/N |", "|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        L.append(f"| {r['estimand']} | {r['beta2_level_change_pp']:+.2f} | [{r['ci95_beta2_low']:+.2f}, {r['ci95_beta2_high']:+.2f}] | "
                 f"{r['pval_beta2']:.3f} | {r['DW']:.2f} | {r['LB_min_pval']:.3f} | {int(r['placebo_rank'])}/{int(r['n_candidates'])} | {r['rank_fraction']:.3f} |")
    L += ["", "## 读法\n",
          "- 三个区间都跨零；没有任何估计量在锚点窗口给出可与 placebo 区分的边界变化。",
          "- equal-package 在锚点窗口**没有**把系数缩向零（−4.56 对 −4.42），其 placebo 排名（5/28）比版本加权（17/28）更靠前；",
          "  扩展窗上的\"坍缩\"（−20.1 → −4.1）是长窗现象，正文措辞须限定到扩展窗。",
          "- 去掉 protocol window 内最多产的五个包后系数为 −7.33、区间更宽（[−21.4, +6.7]），排名 14/28。",
          "- 与 8 周对比（equal-package +1.49 pp）不是同一对象：那是两期均值差，这是 ITS 的截距跳变。"]
    OUT_MD.write_text("\n".join(L) + "\n")
    print(f"✓ {OUT_CSV.name}  ✓ {OUT_MD.name}\n✅ protocol-window estimand companions complete", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
