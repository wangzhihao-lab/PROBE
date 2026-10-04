"""
TSE R1 revision, T4.3 — RQ2 tabulation on the sampling frame (reviewer 1, R1-2a).

Reviewer 1 asked for a tabulation of the A--F distribution by the frame the
sample was actually drawn from (ecosyste.ms strata), with deps.dev kept as
sensitivity. Because the inverse-inclusion-probability weights are constant
WITHIN an eco stratum (top census 1.00; mid 2.58; tail 3.77), the design-
weighted within-stratum package-level shares equal the unweighted package-level
shares within that stratum -- so this table needs no reweighting, only the
correct grouping, and it is exact.

Inputs:  data/classification.csv, data/non_provenance_sample.csv
Outputs: data/tse_r1_rq2_eco_frame.csv, reports/tse_r1_rq2_eco_frame.md

Run: conda run --no-capture-output -n stage2c python scripts/67_rq2_eco_frame_table.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
DATA, REPORTS = ROOT / "data", ROOT / "reports"
CLASSES = ["A", "B", "C", "D", "E"]


def main() -> int:
    c = pd.read_csv(DATA / "classification.csv")
    eco = pd.read_csv(DATA / "non_provenance_sample.csv")[["package", "tier"]] \
        .drop_duplicates("package").rename(columns={"tier": "eco_tier"})
    pkg = c.groupby("package")["class"].first().reset_index().merge(eco, on="package")
    assert len(pkg) == c["package"].nunique() == 381
    rows = []
    for t in ("top", "mid", "tail"):
        sub = pkg[(pkg["eco_tier"] == t) & (pkg["class"] != "F")]
        n = len(sub)
        for cl in CLASSES:
            k = int((sub["class"] == cl).sum())
            rows.append({"eco_tier": t, "class": cl, "packages": k,
                         "share_pct": round(100 * k / n, 1) if n else 0.0, "n_AE": n})
    df = pd.DataFrame(rows)
    df.to_csv(DATA / "tse_r1_rq2_eco_frame.csv", index=False)
    L = ["# T4.3 — RQ2 按抽样帧（ecosyste.ms 分层）的包级 A–E 分布\n",
         "权重在 eco 层内为常数（top census 1.00 / mid 2.58 / tail 3.77），故层内包级份额",
         "免权重且精确。deps.dev 表（Table II / scripts/50 设计加权）保留为报告轴。\n",
         "| eco 层 | A | B | C | D | E | A–E 包数 |", "|---|---|---|---|---|---|---|"]
    for t in ("top", "mid", "tail"):
        sub = df[df["eco_tier"] == t]
        cells = " | ".join(f"{r['share_pct']}% ({r['packages']})" for _, r in sub.iterrows())
        L.append(f"| {t} | {cells} | {int(sub['n_AE'].iloc[0])} |")
    (REPORTS / "tse_r1_rq2_eco_frame.md").write_text("\n".join(L) + "\n")
    print(df.pivot(index="class", columns="eco_tier", values="share_pct").to_string(), flush=True)
    print("✅ T4.3 table complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
