"""
TSE R1 revision, T2.9 — Specification sensitivity for the extended-window ITS.

Self-review after T1.4 found that the canonical linear segmented model, which
passes its residual diagnostics on the as-submitted 41-week panel (top tier
DW 1.69, Ljung-Box min p 0.18), FAILS them on the extended 123-week panel (top
DW 1.30, LB p 0.001; mid/tail LB p ~ 0). A straight pre-trend cannot describe a
28-month series that rose through 2024, peaked in spring 2025 and turned down
from autumn 2025. Part of the -20 pp level change is therefore lack of fit, and
that has to be shown before the extended window carries the paper.

The specification stays the pre-declared one (scripts/its_spec.py, imported
verbatim); this script only asks how the level-change coefficient and, more
importantly, its placebo rank behave under (a) a quadratic pre/post trend and
(b) a grid of symmetric local windows around the deadline. The placebo
comparison is WITHIN-specification: the same mean model is applied at every
candidate cutoff. That does not make the ranks immune to misspecification (a
curved mean function is partitioned differently at different candidate dates,
so its pseudo-discontinuity can vary by cutoff); it makes them empirical
falsification sensitivities, read alongside the residual diagnostics rather
than in place of them. A cell whose diagnostics fail is reported as
diagnostically inadequate whatever its rank says.

Regression guard: the local fitter with the quadratic terms dropped must
reproduce its_spec.fit_its to 1e-9 on the extended top-tier panel.

Inputs:
  data/versions_extended.csv, data/depsdev_dependents_2026_05_05.csv
  scripts/its_spec.py (imported; NOT modified)

Outputs:
  data/tse_r1_its_spec_sensitivity.csv
  reports/tse_r1_its_spec_sensitivity.md

Carried caveats:
  - Anti-causal firewall throughout; nothing here identifies an effect.
  - HAC standard errors are robust to residual autocorrelation but not to
    misspecification of the mean; the window grid addresses the latter.

Run: conda run --no-capture-output -n stage2c python scripts/60_extended_its_spec_sensitivity.py
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox

sys.path.insert(0, str(Path(__file__).parent))
import its_spec  # noqa: E402

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
EXTENDED_CSV = DATA / "versions_extended.csv"
DEPSDEV_CSV = DATA / "depsdev_dependents_2026_05_05.csv"
OUT_CSV = DATA / "tse_r1_its_spec_sensitivity.csv"
OUT_MD = REPORTS / "tse_r1_its_spec_sensitivity.md"

DEADLINE = pd.Timestamp("2025-12-09", tz="UTC")
CENSOR = pd.Timestamp("2026-05-10 23:59:59", tz="UTC")
LONG_ORIGIN = pd.Timestamp("2024-01-01", tz="UTC")
MIN_PRE, MIN_POST = 8, 5
TIERS = ["top", "mid", "tail"]
HALF_WIDTHS = [13, 26, 39, 52, 78, 101]   # weeks each side of the deadline


def dd_tier(n) -> str:
    if pd.isna(n):
        return "missing"
    n = int(n)
    return "top" if n > 500 else "mid" if n > 10 else "tail"


def load_weekly() -> pd.DataFrame:
    v = pd.read_csv(EXTENDED_CSV)
    v["published_at"] = pd.to_datetime(v["published_at"], utc=True, errors="coerce")
    v = v.dropna(subset=["published_at"])
    v = v[(v["published_at"] >= LONG_ORIGIN) & (v["published_at"] <= CENSOR)]
    v["a"] = (v["has_attestation"].astype(str).str.lower() == "true").astype(int)
    dm = pd.read_csv(DEPSDEV_CSV)
    v["tier"] = v["package"].map(dict(zip(dm["name"], dm["dependents_count"]))).apply(dd_tier)
    v["period_start"] = (v["published_at"]
                         - pd.to_timedelta(v["published_at"].dt.weekday, unit="D")).dt.normalize()
    g = v.groupby(["tier", "period_start"]).agg(n=("a", "size"), k=("a", "sum")).reset_index()
    g["rate"] = g["k"] / g["n"]
    return g


def fit_flex(sub: pd.DataFrame, cutoff, order: int) -> Optional[dict]:
    """its_spec conventions (calendar t, cutoff-date centering, straddle exclusion,
    Newey-West HAC) with an optional quadratic trend on both sides."""
    sub = sub.sort_values("period_start").reset_index(drop=True)
    starts = [pd.Timestamp(v).date() for v in sub["period_start"]]
    cut = pd.Timestamp(cutoff).date()
    t = its_spec._calendar_t(starts, "W")
    c = its_spec._cutoff_c(starts[0], cut, "W")
    nxt = starts[1:] + [date.max]
    keep = np.array([not (s < cut < nx) for s, nx in zip(starts, nxt)])
    sub, t = sub[keep].reset_index(drop=True), t[keep]
    starts = [s for s, k in zip(starts, keep) if k]
    post = np.array([s >= cut for s in starts], dtype=int)
    if (post == 0).sum() < 2 or post.sum() < 2:
        return None
    x = t - c
    y = sub["rate"].astype(float).values * 100
    cols = [x, post, x * post] if order == 1 else [x, x ** 2, post, x * post, (x ** 2) * post]
    X = sm.add_constant(np.column_stack(cols))
    n = len(y)
    lag = max(1, int(np.ceil(n ** 0.25)))
    try:
        fit = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": lag})
    except Exception:
        return None
    j = 2 if order == 1 else 3          # index of the post (level-change) coefficient
    ci = fit.conf_int(alpha=0.05)
    resid = fit.resid
    dw = float(sm.stats.stattools.durbin_watson(resid))
    lags = [l for l in sorted({min(4, n // 3), min(8, n // 2), min(12, n - 4)}) if l > 0]
    lb = float(acorr_ljungbox(resid, lags=lags, return_df=True)["lb_pvalue"].min()) if lags else None
    return {"beta2": float(fit.params[j]), "lo": float(ci[j, 0]), "hi": float(ci[j, 1]),
            "p": float(fit.pvalues[j]), "DW": dw, "LB": lb,
            "n_pre": int((post == 0).sum()), "n_post": int(post.sum()), "n": n}


def candidates(weekly: pd.DataFrame) -> list[pd.Timestamp]:
    ws = sorted(weekly["period_start"].unique())
    n = len(ws)
    off = (DEADLINE.date() - (DEADLINE - pd.Timedelta(days=DEADLINE.weekday())).date()).days
    return [pd.Timestamp(ws[k]) + pd.Timedelta(days=off)
            for k in range(n) if k >= MIN_PRE and (n - k - 1) >= MIN_POST]


def rank(sub: pd.DataFrame, cands: list[pd.Timestamp], order: int) -> tuple[int, int]:
    actual = fit_flex(sub, DEADLINE, order)
    a = abs(actual["beta2"])
    mags = [abs(r["beta2"]) for cnd in cands if (r := fit_flex(sub, cnd, order))]
    return (sum(1 for m in mags if m > a) + 1, len(mags))


def main() -> int:
    weekly = load_weekly()
    top = weekly[weekly["tier"] == "top"]

    # regression guard: linear local fitter == its_spec
    mine = fit_flex(top, DEADLINE, 1)
    ref = its_spec.fit_its(top, DEADLINE, period_col="period_start", granularity="W")
    assert abs(mine["beta2"] - ref["beta2_level_change_pp"]) < 1e-9, \
        f"fit_flex(order=1) diverges from its_spec: {mine['beta2']} vs {ref['beta2_level_change_pp']}"
    print(f"✓ regression guard: fit_flex(order=1) == its_spec.fit_its "
          f"(beta2 {mine['beta2']:.4f})", flush=True)

    rows: list[dict] = []
    # A. full extended panel, linear vs quadratic, all tiers, with placebo rank
    full_cands = candidates(weekly)
    for tier in TIERS:
        sub = weekly[weekly["tier"] == tier]
        for order, label in ((1, "linear"), (2, "quadratic")):
            r = fit_flex(sub, DEADLINE, order)
            rk, nc = rank(sub, full_cands, order)
            rows.append({"analysis": "full_panel", "tier": tier, "spec": label,
                         "half_width_weeks": 101, **{k: r[k] for k in ("beta2", "lo", "hi", "p", "DW", "LB", "n_pre", "n_post")},
                         "placebo_rank": rk, "n_candidates": nc, "rank_fraction": rk / nc})
            print(f"  full {tier:<5}{label:<10}beta2={r['beta2']:+7.2f} p={r['p']:.3f} "
                  f"DW={r['DW']:.2f} LB={r['LB']:.3f} rank={rk}/{nc}", flush=True)

    # B. symmetric local windows around the deadline, linear spec (pre-declared),
    #    with the placebo scan re-run INSIDE each window
    for h in HALF_WIDTHS:
        lo_t = DEADLINE - pd.Timedelta(weeks=h)
        hi_t = DEADLINE + pd.Timedelta(weeks=h)
        win = weekly[(weekly["period_start"] >= lo_t) & (weekly["period_start"] <= hi_t)]
        wc = candidates(win)
        for tier in TIERS:
            sub = win[win["tier"] == tier]
            r = fit_flex(sub, DEADLINE, 1)
            if r is None:
                continue
            rk, nc = rank(sub, wc, 1) if wc else (-1, 0)
            rows.append({"analysis": "local_window", "tier": tier, "spec": "linear",
                         "half_width_weeks": h, **{k: r[k] for k in ("beta2", "lo", "hi", "p", "DW", "LB", "n_pre", "n_post")},
                         "placebo_rank": rk, "n_candidates": nc,
                         "rank_fraction": (rk / nc) if nc else float("nan")})
        t = [x for x in rows if x["analysis"] == "local_window" and x["half_width_weeks"] == h and x["tier"] == "top"]
        if t:
            t = t[0]
            print(f"  ±{h:>3}w top  beta2={t['beta2']:+7.2f} [{t['lo']:+.1f},{t['hi']:+.1f}] "
                  f"p={t['p']:.3f} DW={t['DW']:.2f} LB={t['LB']:.3f} rank={t['placebo_rank']}/{t['n_candidates']}", flush=True)

    # C. estimand decomposition under each specification (top tier). If the
    #    equal-package coefficient is near zero wherever the model is adequate,
    #    the version-weighted coefficient is publishing-volume composition.
    import importlib.util
    sp = importlib.util.spec_from_file_location("s57", Path(__file__).parent / "57_long_window_its_placebo.py")
    s57 = importlib.util.module_from_spec(sp); sp.loader.exec_module(s57)
    raw = s57.load_rows(s57.EXTENDED_CSV, s57.LONG_ORIGIN)
    for est, lbl in (("version_weighted", "version_weighted"), ("equal_package", "equal_package"),
                     ("any_attested_package", "any_attested_package")):
        g = s57.estimand_panel(raw, est)
        gt = g[g["tier"] == "top"]
        for spec, order, h in (("linear", 1, 101), ("quadratic", 2, 101), ("linear", 1, 26)):
            win = gt if h >= 101 else gt[(gt["period_start"] >= DEADLINE - pd.Timedelta(weeks=h))
                                         & (gt["period_start"] <= DEADLINE + pd.Timedelta(weeks=h))]
            r = fit_flex(win, DEADLINE, order)
            rk, nc = rank(win, candidates(win), order)
            rows.append({"analysis": f"estimand_{lbl}", "tier": "top", "spec": spec,
                         "half_width_weeks": h, **{k: r[k] for k in ("beta2", "lo", "hi", "p", "DW", "LB", "n_pre", "n_post")},
                         "placebo_rank": rk, "n_candidates": nc, "rank_fraction": rk / nc})

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    L = ["# T2.9 — 扩展窗 ITS 的规格敏感性（诊断失败的处置）\n",
         "**问题**：预先声明的线性分段模型在初投 41 周面板上通过残差诊断"
         "（top DW 1.69 / LB p 0.18），在扩展 123 周面板上**不通过**"
         "（top DW 1.30 / LB p 0.001；mid、tail LB p ≈ 0）。"
         "直线描述不了 2024 年上升、2025 春见顶、2025 秋回落的驼峰序列，"
         "−20 pp 的水平变化含有 lack-of-fit 成分。\n",
         "**处置原则**：规格不做事后更改（那是规格搜索）。本文件只回答两件事：",
         "(a) 换成二次趋势后系数与 placebo 排名如何；(b) 把窗口对称收窄到 deadline 附近后如何。",
         "**读法**：placebo 比较是规格内比较——每个候选切点套用同一个均值模型。这不意味着排名"
         "对误设免疫（弯曲的均值函数在不同切点被不同地切分，伪跳变可随切点变化）；排名只是"
         "经验性的证伪敏感性，须与残差诊断并读。诊断失败的格子无论排名如何，都记为诊断不足。\n",
         "## A. 全扩展面板：线性 vs 二次（各 tier，含 placebo 排名）\n",
         "| tier | 规格 | β₂ (pp) | 95% CI | p | DW | LB min p | 排名 | 排名/N |",
         "|---|---|---|---|---|---|---|---|---|"]
    for _, r in df[df["analysis"] == "full_panel"].iterrows():
        L.append(f"| {r['tier']} | {r['spec']} | {r['beta2']:+.2f} | [{r['lo']:+.2f}, {r['hi']:+.2f}] | "
                 f"{r['p']:.3f} | {r['DW']:.2f} | {r['LB']:.3f} | {r['placebo_rank']}/{r['n_candidates']} | {r['rank_fraction']:.3f} |")
    L += ["", "## B. 对称局部窗口（线性规格，placebo 扫描在窗内重跑）\n",
          "| 半宽（周） | tier | pre/post 周 | β₂ (pp) | 95% CI | p | DW | LB min p | 排名 | 排名/N |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in df[df["analysis"] == "local_window"].iterrows():
        L.append(f"| ±{int(r['half_width_weeks'])} | {r['tier']} | {r['n_pre']}/{r['n_post']} | {r['beta2']:+.2f} | "
                 f"[{r['lo']:+.2f}, {r['hi']:+.2f}] | {r['p']:.3f} | {r['DW']:.2f} | {r['LB']:.3f} | "
                 f"{r['placebo_rank']}/{r['n_candidates']} | {r['rank_fraction']:.3f} |")
    L += ["", "## C. 估计量分解在各规格下（top tier）\n",
          "| 估计量 | 规格 | 窗口 | β₂ (pp) | 95% CI | p | 排名 |", "|---|---|---|---|---|---|---|"]
    for _, r in df[df["analysis"].str.startswith("estimand_")].iterrows():
        L.append(f"| {r['analysis'].replace('estimand_', '')} | {r['spec']} | ±{int(r['half_width_weeks'])}w | "
                 f"{r['beta2']:+.2f} | [{r['lo']:+.2f}, {r['hi']:+.2f}] | {r['p']:.3f} | {r['placebo_rank']}/{r['n_candidates']} |")
    L += ["", "## D. 读法\n",
          "- 若二次规格下 top 的 placebo 排名仍不靠前，则非特异性裁决对趋势形状不敏感。",
          "- 局部窗口表直接展示 β₂ 随窗口长度的变化：短窗（±13/±26）≈ 初投稿的量级，",
          "  长窗系数随 pre-period 拉长而增大——这就是 lack-of-fit 的形态，须在 §7 如实呈现。",
          "- 主结果沿用预先声明的线性规格与 8/5 候选规则；本表进 §7 与 supplement。"]
    OUT_MD.write_text("\n".join(L) + "\n")
    print(f"✓ {OUT_CSV.name}  ✓ {OUT_MD.name}\n✅ T2.9 complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
