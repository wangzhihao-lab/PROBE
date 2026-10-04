"""
Task 4 Round 3 Step 3 — Re-generate RQ1 + RQ2 figures under deps.dev-tier.

Outputs:
  paper/figures/rq1_timeline_depsdev.png + .pdf
  paper/figures/rq2_classification_depsdev.png + .pdf
  paper/figures/tier_comparison_eco_vs_depsdev.png + .pdf  (NEW comparison panel)

Run: conda run --no-capture-output -n stage2c python scripts/26_reweight_charts_depsdev.py
"""
from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import figstyle

figstyle.apply()  # shared publication style (rcParams) before any plotting

DATA_DIR = Path(__file__).parent.parent / "data"
PAPER_DIR = Path(__file__).parent.parent / "paper" / "figures"
PAPER_DIR.mkdir(parents=True, exist_ok=True)


def _save(fig, stem: str) -> None:
    fig.savefig(str(PAPER_DIR / f"{stem}.png"))
    fig.savefig(str(PAPER_DIR / f"{stem}.pdf"))
    plt.close(fig)
    print(f"  {stem}.{{png,pdf}}", flush=True)


def main() -> int:
    # ---- RQ1 timeline under deps.dev-tier ----
    series: dict[str, list[tuple[str, float]]] = defaultdict(list)
    with (DATA_DIR / "rq1_monthly_attestation_rate_depsdev.csv").open() as f:
        for r in csv.DictReader(f):
            series[r["tier"]].append((r["month"], float(r["rate"]) * 100))

    months = sorted({m for items in series.values() for m, _ in items})
    series_order = ["all_excl_missing", "top", "mid", "tail"]
    series_names = ["overall (excl. missing)", "top", "mid", "tail"]
    colors = [figstyle.ACCENT_2, figstyle.TIER["top"],
              figstyle.TIER["mid"], figstyle.TIER["tail"]]
    markers = ["s", "o", "D", "^"]

    x = np.arange(len(months))
    fig, ax = plt.subplots(figsize=plt.rcParams["figure.figsize"])
    for t, name, col, mk in zip(series_order, series_names, colors, markers):
        m_to_r = dict(series[t])
        y = np.array([m_to_r.get(m, np.nan) for m in months])
        ax.plot(x, y, marker=mk, color=col, label=name, markersize=4)
    if "2025-12" in months:
        dx = months.index("2025-12")
        ax.axvline(dx, linestyle="--", color=figstyle.REF, alpha=0.5, linewidth=0.9)
        ax.text(dx + 0.12, 69, "Dec-9", color=figstyle.REF, fontsize=7.5,
                va="top", ha="left")
    ax.set_xlabel("Month")
    ax.set_ylabel("Attestation rate (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(months, rotation=45, ha="right")
    ax.set_ylim(0, 72)
    ax.legend(loc="lower left", ncol=2, framealpha=0.85, edgecolor="0.8")
    _save(fig, "rq1_timeline_depsdev")

    # ---- RQ2 grouped bar under deps.dev-tier ----
    cls: dict[tuple[str, str], str] = {}
    with (DATA_DIR / "classification.csv").open() as f:
        for r in csv.DictReader(f):
            cls[(r["package"], r["version"])] = r["class"]
    pkg_tier: dict[str, str] = {}
    with (DATA_DIR / "depsdev_dependents_2026_05_05.csv").open() as f:
        for r in csv.DictReader(f):
            try:
                d = int(r["dependents_count"])
                pkg_tier[r["name"]] = "top" if d > 500 else ("mid" if d > 10 else "tail")
            except ValueError:
                continue

    tier_class: dict[str, Counter] = {"top": Counter(), "mid": Counter(), "tail": Counter()}
    for k, c in cls.items():
        t = pkg_tier.get(k[0])
        if t in tier_class:
            tier_class[t][c] += 1

    classes_ae = list("ABCDE")
    tiers = ["top", "mid", "tail"]
    data2 = np.zeros((len(tiers), len(classes_ae)))
    for i, t in enumerate(tiers):
        ae_total = sum(tier_class[t][c] for c in classes_ae)
        if ae_total == 0:
            continue
        for j, c in enumerate(classes_ae):
            data2[i, j] = tier_class[t][c] / ae_total * 100

    def grouped_bar(mat, cats, seriesnames, xlabel, ylabel, stem, ymax=None,
                    ymin=0.0):
        fig, ax = plt.subplots(figsize=plt.rcParams["figure.figsize"])
        ncat, nser = mat.shape
        span, xc = 0.72, np.arange(ncat)
        bw = span / nser
        for j, sname in enumerate(seriesnames):
            off = -span / 2 + bw * (j + 0.5)
            ax.bar(xc + off, mat[:, j], width=bw * 0.92,
                   color=figstyle.NPG[j], label=sname, edgecolor="white",
                   linewidth=0.4)
        ax.set_xticks(xc)
        ax.set_xticklabels(cats)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if ymax is not None:
            ax.set_ylim(ymin, ymax)
        ax.axhline(0, color="0.6", linewidth=0.6)
        ax.legend(loc="upper center", ncol=nser, framealpha=0.85,
                  edgecolor="0.8", handlelength=1.2, columnspacing=0.9)
        _save(fig, stem)

    grouped_bar(data2, tiers, list("ABCDE"),
                "Tier (deps.dev DPC; missing rows excluded)", "% of A-E pool",
                "rq2_classification_depsdev", ymax=max(80, data2.max() + 16))

    # ---- Comparison panel: eco vs dd, A-E pool % per (tier, class) ----
    eco_class: dict[str, Counter] = {"top": Counter(), "mid": Counter(), "tail": Counter()}
    with (DATA_DIR / "rq2_class_distribution.csv").open() as f:
        for r in csv.DictReader(f):
            if r["dimension"] == "tier" and r["slice"] in eco_class:
                eco_class[r["slice"]][r["class"]] = int(r["n"])

    delta = np.zeros((len(tiers), len(classes_ae)))
    for i, t in enumerate(tiers):
        eco_ae = sum(v for k, v in eco_class[t].items() if k != "F")
        dd_ae = sum(tier_class[t][c] for c in classes_ae)
        for j, c in enumerate(classes_ae):
            eco_p = (eco_class[t].get(c, 0) / eco_ae * 100) if eco_ae else 0
            dd_p = (tier_class[t].get(c, 0) / dd_ae * 100) if dd_ae else 0
            delta[i, j] = dd_p - eco_p

    dmax = np.abs(delta).max() * 1.35
    grouped_bar(delta, tiers, list("ABCDE"), "Tier",
                "$\\Delta$ percentage points (deps.dev $-$ ecosyste.ms)",
                "tier_comparison_eco_vs_depsdev", ymax=dmax, ymin=-dmax)
    return 0


if __name__ == "__main__":
    sys.exit(main())
