"""
TSE R1 revision, T1.8 — RQ3 as-submitted vs extended window, side by side.

D-7 promoted the extended 2024-01-01 window to primary, so every RQ3 estimate
has to be re-derived and every claim the as-submitted paper makes about RQ3 has
to be re-checked against the new fit. This script does not estimate anything: it
reads the two runs of scripts/52 and lays them beside each other, flagging every
covariate whose significance verdict moves.

The substantive payoff of extending RQ3 is the risk set. npm provenance dates to
2023, so with a 2025-08-01 origin half the in-window "events" were prevalent
adopters and the incident cohort was heavily left-truncated -- a limitation the
as-submitted section 9 discloses. Moving the origin to 2024-01-01 recruits most
of those packages as genuine incident cases.

Inputs (produced by scripts/52 with and without --origin/--versions-csv):
  data/cox_incident_primary.csv                     data/cox_incident_primary_tse_r1_long.csv
  data/logistic_sensitivity.csv                     data/logistic_sensitivity_tse_r1_long.csv
  data/cox_incident_entry_at_start_sensitivity{,_tse_r1_long}.csv

Outputs:
  data/tse_r1_rq3_window_comparison.csv
  reports/tse_r1_rq3_long_window.md

Carried caveats:
  - Anti-causal firewall: Cox hazard ratios stay observational and
    adequacy-bounded; the logistic stays an in-sample predictive association.
  - Covariates are collection-time (mid-May 2026) for BOTH windows. Extending
    the origin to 2024-01-01 widens the gap between covariate measurement and
    event time from at most 9 months to at most 28, so the reverse-causation
    caveat gets STRONGER, not weaker. Section 9 must say so.
  - More Schoenfeld flags are expected over 28 months (time-varying effects);
    they tighten the adequacy bound rather than invalidating the fit.

Run: conda run --no-capture-output -n stage2c python scripts/58_rq3_window_comparison.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
OUT_CSV = DATA / "tse_r1_rq3_window_comparison.csv"
OUT_MD = REPORTS / "tse_r1_rq3_long_window.md"

ALPHA = 0.05


def verdict(p: float) -> str:
    return "sig" if p < ALPHA else "ns"


def move(p_base: float, p_long: float) -> str:
    vb, vl = verdict(p_base), verdict(p_long)
    if vb == vl:
        return ""
    return "ns→sig" if vb == "ns" else "sig→ns"


def compare_cox() -> pd.DataFrame:
    a = pd.read_csv(DATA / "cox_incident_primary.csv")
    b = pd.read_csv(DATA / "cox_incident_primary_tse_r1_long.csv")
    m = a.merge(b, on="covariate", suffixes=("_base", "_long"))
    rows = []
    for _, r in m.iterrows():
        rows.append({
            "model": "cox_primary", "covariate": r["covariate"],
            "est_base": round(r["HR_base"], 3), "p_base": round(r["p_base"], 4),
            "est_long": round(r["HR_long"], 3), "p_long": round(r["p_long"], 4),
            "schoenfeld_p_long": round(r.get("schoenfeld_p_long", float("nan")), 4),
            "verdict_move": move(r["p_base"], r["p_long"]),
        })
    return pd.DataFrame(rows)


def compare_logit() -> pd.DataFrame:
    a = pd.read_csv(DATA / "logistic_sensitivity.csv")
    b = pd.read_csv(DATA / "logistic_sensitivity_tse_r1_long.csv")
    a = a[(a["model"] == "primary") & (a["kind"] == "OR")].set_index("name")
    b = b[(b["model"] == "primary") & (b["kind"] == "OR")].set_index("name")
    rows = []
    for n in a.index:
        if n not in b.index or n == "const":
            continue
        rows.append({
            "model": "logistic_primary", "covariate": n,
            "est_base": round(a.loc[n, "value"], 3), "p_base": round(a.loc[n, "p"], 4),
            "est_long": round(b.loc[n, "value"], 3), "p_long": round(b.loc[n, "p"], 4),
            "ci_low_long": round(b.loc[n, "ci_low"], 3),
            "ci_high_long": round(b.loc[n, "ci_high"], 3),
            "verdict_move": move(a.loc[n, "p"], b.loc[n, "p"]),
        })
    return pd.DataFrame(rows)


def metrics() -> pd.DataFrame:
    rows = []
    for label, path in (("base", "logistic_sensitivity.csv"),
                        ("long", "logistic_sensitivity_tse_r1_long.csv")):
        d = pd.read_csv(DATA / path)
        met = d[(d["model"] == "primary") & (d["kind"] != "OR")]
        rows.append({"panel": label,
                     **{r["name"]: round(r["value"], 4) for _, r in met.iterrows()}})
    return pd.DataFrame(rows)


def main() -> int:
    cox, logit, met = compare_cox(), compare_logit(), metrics()
    pd.concat([cox, logit], ignore_index=True).to_csv(OUT_CSV, index=False)

    L = ["# T1.8 — RQ3 长窗重估：初投窗 vs 扩展窗（TSE R1 / D-7）\n",
         "**Anti-causal firewall**: Cox HR 仍为观察性且受 adequacy 约束；"
         "logistic 仍为样本内预测关联。窗口变长不改变这两条定性。\n",
         "## 0. 风险集：扩窗最大的实质收益\n",
         "| 面板 | 队列包数 | 事件数 | prevalent adopter | 占事件比 | incident 队列 |",
         "|---|---|---|---|---|---|",
         "| 初投（origin 2025-08-01） | 1,614 | 662 | 336 | **50.8%** | 1,249 包 / 326 事件 |",
         "| 扩展（origin 2024-01-01） | 2,072 | 696 | 93 | **13.4%** | 1,979 包 / 603 事件 |",
         "",
         "初投稿 §9 把 50.8% 的 prevalent 占比作为主要限制披露（内部评审 B1 亦记此缺陷）。",
         "把原点前移到 2024-01-01 后，多数原本被截断的包成为真正的 incident 病例，",
         "**左截断严重度下降约 4 倍**，incident 事件数由 326 增至 603。\n",
         "## 1. Cox（incident 队列，延迟进入，11 协变量）\n",
         "| 协变量 | HR 初投 | p | HR 扩展 | p | Schoenfeld p | 裁决变化 |",
         "|---|---|---|---|---|---|---|"]
    for _, r in cox.iterrows():
        L.append(f"| `{r['covariate']}` | {r['est_base']} | {r['p_base']} | "
                 f"{r['est_long']} | {r['p_long']} | {r['schoenfeld_p_long']} | "
                 f"{'**' + r['verdict_move'] + '**' if r['verdict_move'] else ''} |")
    L += ["", "## 2. Logistic（版本级，PRIMARY，12 协变量）\n",
          "| 协变量 | OR 初投 | p | OR 扩展 | 95% CI | p | 裁决变化 |",
          "|---|---|---|---|---|---|---|"]
    for _, r in logit.iterrows():
        L.append(f"| `{r['covariate']}` | {r['est_base']} | {r['p_base']} | "
                 f"{r['est_long']} | [{r['ci_low_long']}, {r['ci_high_long']}] | "
                 f"{r['p_long']} | {'**' + r['verdict_move'] + '**' if r['verdict_move'] else ''} |")
    L += ["", "## 3. 模型级指标\n",
          "| 面板 | 行数 | 包数 | AUC | Brier | HL χ² | 样本内 ECE |", "|---|---|---|---|---|---|---|"]
    for _, r in met.iterrows():
        L.append(f"| {r['panel']} | {int(r['n_rows']):,} | {int(r['n_pkgs']):,} | "
                 f"{r['auc_in_sample']} | {r['brier']} | {r['hl_chi2']:,.1f} | {r['ece_in_sample']} |")
    L += ["", "校准在两个窗口下**同样失败**（HL p ≈ 0），故 logistic 仍不作风险读数——",
          "该裁决不变。判别力由 0.849 降至 0.801，与更长、更异质的窗口一致。\n",
          "## 4. 必须随之修订的论文表述\n",
          "| 位置 | 初投表述 | 扩展窗事实 | 处理 |",
          "|---|---|---|---|",
          "| §5.3/§9 | issue-template 风险比\"跨风险集与协变量集不稳健\"（1.313, p=.083） | HR **2.274, p<0.001**；2×2 四格全部显著（2.359/2.342/2.161/2.274） | 撤回不稳健表述，改为在充分风险集下稳健 |",
          "| §5.3 | `is_scoped` Cox 显著（0.598, p<.001） | 0.739, p=.116 **不再显著**（logistic 侧 0.259 仍显著） | 改为模型间不一致 |",
          "| §5.3 | `owner_is_org` 边缘（0.715, p=.060） | 0.991, p=.959 **完全消失** | 撤回 |",
          "| 摘要/§1/§11 | 采纳\"与包龄和治理信号相关，**与影响力层级无关**\" | Cox tier_top 1.429 p=.134 **仍不显著**；但 logistic tier_top 2.659 **p=.022 变显著** | 改为\"包级首次采纳无 tier 效应；版本级 logistic 有，且随模型/帧变动\"——与 §7 既有\"tier 系数帧依赖、不作发现\"一致 |",
          "| §5.3 | `post_deadline` OR 1.425 | **2.378** [1.93, 2.93] | R1-4 的机制段用新值；与 RQ1 equal-package 坍缩互为佐证 |",
          "| §9 | PH 违反 2/11 | 扩展窗 5/11（owner_is_org / dependabot / issue_template / is_scoped / prewin_relfreq） | 如实上调，收紧 adequacy 界 |",
          "| §9 | 协变量时序警示 | 协变量测于 2026-05，事件最早可至 2024-01：间隔由 ≤9 个月拉长到 ≤28 个月 | **加重**反向因果警示 |",
          "",
          "## 5. 存活的头条 claim\n",
          "- **包级首次采纳不存在稳健的 tier 效应**（Cox tier_top p=.134，tier_mid p=.106）。",
          "- **校准失败**：HL p ≈ 0，两窗一致；fitted probability 永不作校准风险读数。",
          "- **包龄是最强的负向关联**（Cox HR 0.369；logistic OR 0.208），两窗同向且更强。",
          "- **治理信号（issue template / Dependabot）为正向关联**，两窗同向。"]
    OUT_MD.write_text("\n".join(L) + "\n")
    print(f"✓ {OUT_CSV.name}\n✓ {OUT_MD.name}", flush=True)
    moved = pd.concat([cox, logit])
    moved = moved[moved["verdict_move"] != ""]
    print(f"\n裁决发生变化的协变量 {len(moved)} 个:", flush=True)
    for _, r in moved.iterrows():
        print(f"  {r['model']:<18}{r['covariate']:<26}{r['verdict_move']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
