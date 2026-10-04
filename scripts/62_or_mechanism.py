"""
TSE R1 revision, T2.4 — Nested-model sensitivity of the post-deadline odds ratio
(why the adjusted coefficient exceeds 1 while the top-tier rate falls; reviewer 1, item R1-4).
The stepwise path is a specification sensitivity, not a decomposition: odds ratios are
non-collapsible, so movement across nested models does not by itself establish composition.

Reviewer 1: Table III's post_deadline OR (1.425 as submitted; 2.378 on the
extended window) "could be seen as suggesting that the versions were more likely
to be attested after the deadline. This is at variance with the declining
top-tier trend." The two are not at variance: the aggregate rate is dominated by
publishing-volume composition, while the within-model coefficient conditions on
the covariates that carry that composition. This script reports two
sensitivities, neither of which is a decomposition:

  (a) a stepwise coefficient path -- the post_deadline OR refit under
      progressively richer covariate sets, cluster-robust by package, showing
      which adjustment moves it and by how much;
  (b) tier-stratified unadjusted ORs -- the Simpson structure directly: the
      pooled unadjusted OR against the per-tier unadjusted ORs.

Everything reuses scripts/52's own frame builder and fitter (imported, not
re-implemented), so the full-model column reproduces the paper's Table III
coefficient exactly; that reproduction is asserted.

Inputs:  scripts/52 (frame + fitter), data/versions{,_extended}.csv and friends
Outputs: data/tse_r1_or_mechanism.csv, reports/tse_r1_or_mechanism.md

Carried caveats:
  - In-sample predictive association; calibration fails (HL p ~ 0), so no OR is
    a calibrated risk. Anti-causal firewall: post_deadline is a period contrast,
    not an effect of the deadline -- the ITS placebo verdict forecloses that.

Run: conda run --no-capture-output -n stage2c python scripts/62_or_mechanism.py
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
DATA, REPORTS = ROOT / "data", ROOT / "reports"
OUT_CSV = DATA / "tse_r1_or_mechanism.csv"
OUT_MD = REPORTS / "tse_r1_or_mechanism.md"

_spec = importlib.util.spec_from_file_location("s52", Path(__file__).parent / "52_rq3_sensitivity.py")
s52 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s52)

STEPS: list[tuple[str, list[str]]] = [
    ("1 unadjusted", ["post_deadline"]),
    ("2 + tier", ["post_deadline", "tier_top", "tier_mid"]),
    ("3 + age", ["post_deadline", "tier_top", "tier_mid", "log_age_days"]),
    ("4 + release opportunity", ["post_deadline", "tier_top", "tier_mid", "log_age_days",
                                 "log_prewin_release_freq"]),
    ("5 + scale", ["post_deadline", "tier_top", "tier_mid", "log_age_days",
                   "log_prewin_release_freq", "log_weekly_downloads", "log_maintainers"]),
    ("6 + governance", ["post_deadline", "tier_top", "tier_mid", "log_age_days",
                        "log_prewin_release_freq", "log_weekly_downloads", "log_maintainers",
                        "owner_is_org", "has_dependabot_int", "has_issue_template_int",
                        "is_scoped_int"]),
    ("7 full primary", s52.LOGIT_COLS_PRIMARY),
]


def run_window(label: str) -> tuple[list[dict], list[dict]]:
    prewin = s52.prewindow_release_counts()
    df = s52.build_version_frame(prewin)
    path_rows: list[dict] = []
    for step, cols in STEPS:
        fit, _ = s52.fit_logit(df, cols)
        ci = fit.conf_int()
        path_rows.append({
            "window": label, "step": step, "n_covariates": len(cols),
            "or_post_deadline": float(np.exp(fit.params["post_deadline"])),
            "ci_low": float(np.exp(ci.loc["post_deadline"][0])),
            "ci_high": float(np.exp(ci.loc["post_deadline"][1])),
            "p": float(fit.pvalues["post_deadline"]),
        })
    strat_rows: list[dict] = []
    for scope, sub in [("pooled", df), ("top", df[df["tier_top"] == 1]),
                       ("mid", df[df["tier_mid"] == 1]),
                       ("tail", df[(df["tier_top"] == 0) & (df["tier_mid"] == 0)])]:
        pre = sub[sub["post_deadline"] == 0]["y"]
        post = sub[sub["post_deadline"] == 1]["y"]
        if len(pre) == 0 or len(post) == 0:
            continue
        fit, _ = s52.fit_logit(sub, ["post_deadline"])
        ci = fit.conf_int()
        strat_rows.append({
            "window": label, "scope": scope,
            "rate_pre_pct": round(100 * pre.mean(), 2),
            "rate_post_pct": round(100 * post.mean(), 2),
            "or_unadjusted": float(np.exp(fit.params["post_deadline"])),
            "ci_low": float(np.exp(ci.loc["post_deadline"][0])),
            "ci_high": float(np.exp(ci.loc["post_deadline"][1])),
            "n_pre": int(len(pre)), "n_post": int(len(post)),
        })
    return path_rows, strat_rows


def main() -> int:
    # as-submitted window (defaults)
    base_path, base_strat = run_window("as_submitted")
    # extended window (D-7 primary)
    s52.configure("2024-01-01", "versions_extended.csv", "_tse_r1_long")
    long_path, long_strat = run_window("extended")

    # assert the full-model column reproduces the published Table III values
    pub = pd.read_csv(DATA / "logistic_sensitivity.csv")
    pub_or = float(pub[(pub["model"] == "primary") & (pub["kind"] == "OR")
                       & (pub["name"] == "post_deadline")]["value"].iloc[0])
    got = base_path[-1]["or_post_deadline"]
    assert abs(got - pub_or) < 1e-6, f"full-model OR {got} != published {pub_or}"
    pub_l = pd.read_csv(DATA / "logistic_sensitivity_tse_r1_long.csv")
    pub_or_l = float(pub_l[(pub_l["model"] == "primary") & (pub_l["kind"] == "OR")
                           & (pub_l["name"] == "post_deadline")]["value"].iloc[0])
    got_l = long_path[-1]["or_post_deadline"]
    assert abs(got_l - pub_or_l) < 1e-6, f"extended full-model OR {got_l} != {pub_or_l}"
    print(f"✓ full-model reproduction: as-submitted {got:.3f} (published {pub_or:.3f}); "
          f"extended {got_l:.3f} (published {pub_or_l:.3f})", flush=True)

    path = pd.DataFrame(base_path + long_path)
    strat = pd.DataFrame(base_strat + long_strat)
    pd.concat([path.assign(table="stepwise"), strat.assign(table="stratified")],
              ignore_index=True).to_csv(OUT_CSV, index=False)

    L = ["# T2.4 — post-deadline OR 的嵌套模型规格敏感性（R1-4）\n",
         "**问题**（R1 原文大意）：post_deadline OR > 1 似乎说明 deadline 后版本更可能被",
         "attested，与 top-tier 汇总率下滑矛盾。**答案**：两个数测的不是同一个量：汇总率是",
         "版本加权的期别率，模型内系数是给定协变量剖面下的期别对比；本路径只作规格敏感性，不作机制或构成陈述。\n",
         "## 1. 逐步加入协变量的系数轨迹（cluster-robust by package）\n",
         "| 窗口 | 步骤 | 协变量数 | OR(post_deadline) | 95% CI | p |",
         "|---|---|---|---|---|---|"]
    for _, r in path.iterrows():
        L.append(f"| {r['window']} | {r['step']} | {r['n_covariates']} | {r['or_post_deadline']:.3f} | "
                 f"[{r['ci_low']:.2f}, {r['ci_high']:.2f}] | {r['p']:.4f} |")
    L += ["", "## 2. 分 tier 的未调整期别对比（Simpson 结构）\n",
          "| 窗口 | 范围 | 率(pre) | 率(post) | 未调整 OR | 95% CI | n(pre)/n(post) |",
          "|---|---|---|---|---|---|---|"]
    for _, r in strat.iterrows():
        L.append(f"| {r['window']} | {r['scope']} | {r['rate_pre_pct']}% | {r['rate_post_pct']}% | "
                 f"{r['or_unadjusted']:.3f} | [{r['ci_low']:.2f}, {r['ci_high']:.2f}] | "
                 f"{r['n_pre']:,}/{r['n_post']:,} |")
    L += ["", "## 3. 可写进论文的句子（措辞已按方法学评审收窄）\n",
          "post_deadline 是一个期别对比而非任何日期的效应（ITS placebo 裁决在先）。",
          "分 tier 看：top 的未调整 OR < 1 而 mid/tail > 1，pooled 率在协议窗持平、在扩展窗上升，",
          "只有 top 层的总体率下降——这是分层间的效应异质性 / Simpson 型聚合。",
          "逐步加入协变量后系数总体上移但**不单调**（协议窗 1.098→1.097、扩展窗 1.680→1.534 在加入包龄时回落），",
          "且 OR 不可坍缩（non-collapsibility）：嵌套模型间的系数变化本身不能证明混杂或构成机制，",
          "其幅度还取决于协变量的加入顺序与相关性。因此这条路径只作**嵌套模型规格敏感性**报告，",
          "不作构成分解或机制陈述；与 RQ1 侧 equal-package 坍缩（−20.1 → −4.1，二次趋势下 +0.7）",
          "方向一致。logistic 校准失败（HL p≈0），任何 OR 均不作校准风险读数。\n",
          "## 4. logistic vs Cox 的用途区分（R1-4 后半问）\n",
          "logistic 回答\"**这个版本**是否带 attestation\"（版本级、判别力 AUC 0.80–0.85、",
          "校准失败故仅作判别与关联）；Cox 回答\"**这个包**何时首次出现 attested 发布\"",
          "（包级事件史、incident 队列 + 延迟进入处理左截断、受 PH 与 adequacy 约束）。",
          "两者共享九协变量核心但估计不同构念上的量，不互为替代。"]
    OUT_MD.write_text("\n".join(L) + "\n")
    print(path.to_string(index=False), flush=True)
    print(f"✓ {OUT_CSV.name}  ✓ {OUT_MD.name}\n✅ T2.4 complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
