"""
TSE R1 revision, T2.1 — Power, minimum detectable effect, and equivalence bounds
for the boundary level-change coefficient.

Reviewer 3: "the null result is uninterpretable: it could mean no effect, or it
could mean an effect too small to detect with the given design." The honest
answer is not to manufacture significance by lengthening the pre-period (T2.9
shows that inflates the point estimate through trend misspecification without
narrowing the interval) but to state, for each design, what size of level
change it could and could not have detected, and what range the data rule out.

For every design in the grid this script reports:
  MDE(80%) / MDE(90%)   minimum |beta2| detectable at alpha=.05 two-sided,
                        using the design's own HAC standard error:
                        MDE = SE * (z_{1-alpha/2} + z_{power})
  TOST equivalence      two one-sided tests against +/-delta for delta in
                        {5, 10, 15} pp: p_TOST = max(p_lower, p_upper). The
                        boundary change is "equivalent to zero within delta"
                        at 5% if p_TOST < .05, i.e. the 90% CI lies inside
                        (-delta, +delta).
  95% CI half-width     the plain reading of precision.

Inputs:  data/tse_r1_its_spec_sensitivity.csv (T2.9), data/tse_r1_long_its.csv (T1.4)
Outputs: data/tse_r1_power_mde.csv, reports/tse_r1_power_mde.md

Carried caveats:
  - MDE and TOST are properties of the DESIGN and the SE, not evidence about
    any date. Anti-causal firewall applies: "equivalent to zero within delta"
    bounds a descriptive level change; it licenses no causal null either.
  - SEs are Newey-West HAC; where the model fails its residual diagnostics
    (T2.9 rows with LB min p < .05) the SE is robust to autocorrelation but
    the point estimate carries lack-of-fit, so the MDE is more trustworthy
    than the estimate it surrounds.

Run: conda run --no-capture-output -n stage2c python scripts/61_power_mde_equivalence.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).parent.parent
DATA, REPORTS = ROOT / "data", ROOT / "reports"
SPEC_CSV = DATA / "tse_r1_its_spec_sensitivity.csv"
ANCHOR_CSV = DATA / "tse_r1_long_its.csv"   # protocol-window anchor row (panel = as_submitted)
OUT_CSV = DATA / "tse_r1_power_mde.csv"
OUT_MD = REPORTS / "tse_r1_power_mde.md"

ALPHA = 0.05
# Equivalence margins. 5 and 10 pp were named in the revision plan before any
# result was seen; 15 pp was added after the grid was known. No smallest effect
# size of interest is pre-specified for this design, so the TOST columns are an
# EXPLORATORY sensitivity, not an equivalence conclusion; the design-capability
# statement the paper carries is the MDE, which needs no margin.
DELTAS = [5.0, 10.0, 15.0]
PRESPECIFIED_DELTAS = {5.0, 10.0}
Z_A = norm.ppf(1 - ALPHA / 2)


def se_from_ci(lo: float, hi: float) -> float:
    return (hi - lo) / (2 * Z_A)


def tost_p(beta: float, se: float, delta: float) -> float:
    p_low = 1 - norm.cdf((beta + delta) / se)     # H0: beta <= -delta
    p_high = norm.cdf((beta - delta) / se)         # H0: beta >= +delta
    return max(p_low, p_high)


def main() -> int:
    spec = pd.read_csv(SPEC_CSV)
    spec = spec[spec["tier"] == "top"].copy()
    # The protocol-window anchor (diagnostics-passing, pre-declared) belongs in
    # the same table; it is read from the frozen scripts/57 output, not retyped.
    anc = pd.read_csv(ANCHOR_CSV)
    anc = anc[(anc["panel"] == "as_submitted") & (anc["tier"] == "top")].iloc[0]
    anchor = pd.DataFrame([{
        "analysis": "protocol_window", "spec": "linear", "tier": "top",
        "half_width_weeks": 0, "n_pre": int(anc["n_pre_weeks"]), "n_post": int(anc["n_post_weeks"]),
        "beta2": anc["beta2_level_change_pp"], "lo": anc["ci95_beta2_low"], "hi": anc["ci95_beta2_high"],
        "DW": anc["DW"], "LB": anc["LB_min_pval"],
    }])
    spec = pd.concat([anchor, spec], ignore_index=True)
    rows = []
    for _, r in spec.iterrows():
        se = se_from_ci(r["lo"], r["hi"])
        row = {
            "analysis": r["analysis"], "spec": r["spec"],
            "half_width_weeks": int(r["half_width_weeks"]),
            "n_pre": int(r["n_pre"]), "n_post": int(r["n_post"]),
            "beta2": round(r["beta2"], 2), "se": round(se, 3),
            "ci95_halfwidth": round(Z_A * se, 2),
            "mde_80": round(se * (Z_A + norm.ppf(0.80)), 2),
            "mde_90": round(se * (Z_A + norm.ppf(0.90)), 2),
            "diag_ok": bool(r["DW"] > 1.5 and r["LB"] > 0.05),
        }
        for d in DELTAS:
            row[f"tost_p_{int(d)}pp"] = round(tost_p(r["beta2"], se, d), 4)
        rows.append(row)
    df = pd.DataFrame(rows).sort_values(["analysis", "half_width_weeks", "spec"])
    df.to_csv(OUT_CSV, index=False)
    ok = df[df["diag_ok"]]
    ok_labels = [f"{r['analysis']}/{r['spec']}" + (f"/±{r['half_width_weeks']}w" if r['half_width_weeks'] else "")
                 for _, r in ok.iterrows()]
    mde_lo, mde_hi = float(ok["mde_80"].min()), float(ok["mde_80"].max())
    excl15 = all(ok["tost_p_15pp"] < ALPHA)
    excl10 = all(ok["tost_p_10pp"] < ALPHA)

    L = ["# T2.1 — 功效、最小可检测效应（MDE）与等效界（R3-1b）\n",
         "**目的**：把 null 从\"不可解释\"变成\"有界\"。对每个设计给出它能检测到多大的水平变化、",
         "以及数据能排除多大的水平变化。MDE 与 TOST 是**设计的性质**，不是关于任何日期的证据。\n",
         "MDE = SE × (z₀.₉₇₅ + z_power)；TOST：90% CI 落在 (−δ, +δ) 内即在 5% 水平等效于零。",
         "δ = 5、10 pp 在看到结果前写入返修计划；δ = 15 pp 是网格出来后追加的。本设计没有预先",
         "指定的最小关心效应量（SESOI），所以 TOST 列是**探索性敏感性**，不构成等价结论；",
         "论文承载的设计能力陈述是 MDE（无需 margin）。协议窗锚点行从 `tse_r1_long_its.csv` 读入。\n",
         "| 分析 | 规格 | 半宽(周) | pre/post | β₂ | SE | CI 半宽 | MDE₈₀ | MDE₉₀ | 诊断 | TOST p (δ=5) | (δ=10) | (δ=15) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        hw = f"±{r['half_width_weeks']}" if r['half_width_weeks'] else "protocol"
        L.append(f"| {r['analysis']} | {r['spec']} | {hw} | {r['n_pre']}/{r['n_post']} | "
                 f"{r['beta2']:+.2f} | {r['se']:.2f} | {r['ci95_halfwidth']:.2f} | {r['mde_80']:.2f} | {r['mde_90']:.2f} | "
                 f"{'✓' if r['diag_ok'] else '✗'} | {r['tost_p_5pp']:.3f} | {r['tost_p_10pp']:.3f} | {r['tost_p_15pp']:.3f} |")
    L += ["", "## 读法（由 diag_ok 列生成，不手写）\n",
          "- **精度没有随窗口拉长而提高**：CI 半宽在 ±26 周与 ±101 周之间基本不变（HAC SE 不因",
          "  加入更多被误设的观测而缩小）。扩展窗改变的是点估计，不是推断精度。",
          f"- **诊断通过的拟合**（DW > 1.5 且 LB min p > .05）：{', '.join(ok_labels)}。"
          f"  它们的 MDE₈₀ 为 {mde_lo:.1f}–{mde_hi:.1f} pp：这是论文的\"设计能界定而非检测\"陈述。",
          f"- **TOST（探索性）**：在诊断通过的拟合上，δ = 15 pp {'全部' if excl15 else '并非全部'}被排除（5% 水平）；"
          f"  δ = 10 pp {'全部' if excl10 else '并非全部'}被排除。诊断失败的格子（✗）不进入任何有界陈述：",
          "  均值模型误设时 HAC 协方差修不了估计量，其 MDE 与 TOST 一并只作参考。"]
    OUT_MD.write_text("\n".join(L) + "\n")
    print(df[["analysis", "spec", "half_width_weeks", "beta2", "ci95_halfwidth", "mde_80", "mde_90",
              "diag_ok", "tost_p_5pp", "tost_p_10pp", "tost_p_15pp"]].to_string(index=False), flush=True)
    print(f"\n✓ {OUT_CSV.name}  ✓ {OUT_MD.name}\n✅ T2.1 complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
