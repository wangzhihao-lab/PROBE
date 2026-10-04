"""
TSE R1 revision, T2.8 — Sampling-flow accounting (reviewer 3, R3-r2; AE, A-1).

Reviewer 3: "The authors should make the sampling flow much clearer: candidate
retrieval, exclusions, missing packuments, inactive packages, missing tier
information, and final analysis sets should be reported in one diagram or
table." This script produces that table from the frozen artefacts themselves --
every count is re-derived from the file that defines the stage, not transcribed
from the paper -- and cross-checks each stage against the value the manuscript
states, so the redrawn Figure 1 (T5.1) has a single audited data source.

Inputs:  data/packages_full.txt, data/packuments_raw/, data/versions{,_extended}.csv,
         data/depsdev_dependents_2026_05_05.csv, data/non_provenance_sample.csv,
         data/classification.csv, data/repo_status.csv, data/workflows_raw/,
         scripts/52 (frame builders, imported)
Outputs: data/tse_r1_sampling_flow.csv, reports/tse_r1_sampling_flow.md

Run: conda run --no-capture-output -n stage2c python scripts/66_sampling_flow_accounting.py
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
DATA, REPORTS = ROOT / "data", ROOT / "reports"
OUT_CSV = DATA / "tse_r1_sampling_flow.csv"
OUT_MD = REPORTS / "tse_r1_sampling_flow.md"

_sp = importlib.util.spec_from_file_location("s52", Path(__file__).parent / "52_rq3_sensitivity.py")
s52 = importlib.util.module_from_spec(_sp)
_sp.loader.exec_module(s52)


def dd_tier(n) -> str:
    if pd.isna(n):
        return "missing"
    n = int(n)
    return "top" if n > 500 else "mid" if n > 10 else "tail"


def stage(rows: list, label: str, value, expected=None, note: str = "") -> None:
    ok = "" if expected is None else ("✓" if value == expected else f"✗ paper says {expected}")
    rows.append({"stage": label, "count": value, "paper_value": expected, "check": ok,
                 "note": note})
    print(f"  {label:<58}{value:>10,}  {ok}", flush=True)


def main() -> int:
    rows: list[dict] = []

    # 1. frame
    pk = pd.read_csv(DATA / "packages_full.txt", sep="\t")
    stage(rows, "frame: stratified cohort (packages_full.txt)", len(pk), 3000)
    eco = pk["eco_tier"].value_counts() if "eco_tier" in pk.columns else pk.iloc[:, 1].value_counts()
    # 2. packuments
    marker404 = 0
    fetched = 0
    for name in pk["name"]:
        f = DATA / "packuments_raw" / (name.replace("/", "_").replace("@", "at_") + ".json")
        if not f.exists():
            continue
        fetched += 1
        try:
            head = f.read_text()[:60]
            if '"_404"' in head:
                marker404 += 1
        except OSError:
            pass
    stage(rows, "packuments on disk for the frame", fetched, 3000)
    stage(rows, "  of which 404 markers (excluded)", marker404, 3)

    # 3. window activity
    v = pd.read_csv(DATA / "versions.csv")
    stage(rows, "as-submitted window: version rows", len(v), 100168)
    stage(rows, "as-submitted window: active packages", v["package"].nunique(), 1614)
    ve = pd.read_csv(DATA / "versions_extended.csv")
    stage(rows, "extended window: version rows", len(ve), 278638)
    stage(rows, "extended window: active packages", ve["package"].nunique(), 2109)

    # 4. tier availability
    dm = pd.read_csv(DATA / "depsdev_dependents_2026_05_05.csv")
    have = dict(zip(dm["name"], dm["dependents_count"]))
    frame_missing = sum(1 for n in pk["name"] if n not in have or pd.isna(have.get(n)))
    stage(rows, "frame: packages with deps.dev tier missing (of 3,000)",
          frame_missing, 62, "the section-9 figure; frame-level")
    tiers = v["package"].map(have).apply(dd_tier)
    per_pkg = pd.DataFrame({"package": v["package"], "tier": tiers}).drop_duplicates("package")
    stage(rows, "  of the 1,614 window-active packages", int((per_pkg["tier"] == "missing").sum()),
          None, "1,614 - 29 = 1,585 = the RQ3 modelling frame; disclose both "
                "denominators in section 9 (T4.x)")

    # 5. RQ2 chain
    post_pool = v[(v["has_attestation"].astype(str).str.lower() != "true")]
    post_pool = post_pool[pd.to_datetime(post_pool["published_at"], utc=True)
                          > pd.Timestamp("2025-12-09 23:59:59.999999", tz="UTC")]
    stage(rows, "RQ2 pool: post-deadline non-attested rows", len(post_pool), 34041)
    stage(rows, "RQ2 pool: distinct packages", post_pool["package"].nunique(), None)
    samp = pd.read_csv(DATA / "non_provenance_sample.csv")
    stage(rows, "RQ2 sample rows (eco-stratified 250/150/100)", len(samp), 500)
    cls = pd.read_csv(DATA / "classification.csv")
    stage(rows, "RQ2 labelled rows (A-F final rule)", len(cls), 500)
    stage(rows, "RQ2 labelled packages", cls["package"].nunique(), 381)
    stage(rows, "  class F rows (no usable repo URL; excluded from A-E)",
          int((cls["class"] == "F").sum()), None)
    rs = pd.read_csv(DATA / "repo_status.csv")
    stage(rows, "RQ2 rows with repo resolution record", len(rs), None)
    wf = len(list((DATA / "workflows_raw").glob("*")))
    stage(rows, "cached workflow files (194 repos)", wf, 982)

    # 6. RQ3 frames, both windows, from the pipeline's own builders
    prewin = s52.prewindow_release_counts()
    lf = s52.build_version_frame(prewin)
    stage(rows, "RQ3 logistic frame: rows (as-submitted)", len(lf), 97672)
    stage(rows, "RQ3 logistic frame: packages (as-submitted)", lf["package"].nunique(), 1585)
    prev = s52.prevalent_packages()
    sd = s52.build_survival(prev, prewin)
    stage(rows, "RQ3 Cox cohort: packages (as-submitted)", len(sd), None)
    stage(rows, "RQ3 Cox: events (as-submitted)", int(sd["event"].sum()), 662)
    stage(rows, "RQ3 Cox: prevalent adopters excluded", int(sd["prevalent"].sum()), 336)
    inc = sd[sd["prevalent"] == 0]
    stage(rows, "RQ3 Cox incident cohort: packages", len(inc), 1249)
    stage(rows, "RQ3 Cox incident: events", int(inc["event"].sum()), 326)

    s52.configure("2024-01-01", "versions_extended.csv", "_tse_r1_long")
    prewin_l = s52.prewindow_release_counts()
    lf_l = s52.build_version_frame(prewin_l)
    stage(rows, "RQ3 logistic frame: rows (extended)", len(lf_l), 271222)
    stage(rows, "RQ3 logistic frame: packages (extended)", lf_l["package"].nunique(), 2072)
    prev_l = s52.prevalent_packages()
    sd_l = s52.build_survival(prev_l, prewin_l)
    stage(rows, "RQ3 Cox: events (extended)", int(sd_l["event"].sum()), 696)
    stage(rows, "RQ3 Cox: prevalent adopters (extended)", int(sd_l["prevalent"].sum()), 93)
    inc_l = sd_l[sd_l["prevalent"] == 0]
    stage(rows, "RQ3 Cox incident cohort: packages (extended)", len(inc_l), 1979)
    stage(rows, "RQ3 Cox incident: events (extended)", int(inc_l["event"].sum()), 603)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    bad = df[df["check"].str.startswith("✗", na=False)]
    print(f"\ncross-checks failed: {len(bad)}", flush=True)

    L = ["# T2.8 — 抽样流程账（R3-r2 / A-1；T5.1 新 Fig 1 的唯一数据源）\n",
         "每个数字都从定义该阶段的冻结文件**重新推导**（RQ3 帧与 Cox 队列直接调用",
         "`scripts/52` 自己的构建函数），并与论文所载值交叉核对。\n",
         "| 阶段 | 计数 | 论文值 | 校验 | 备注 |", "|---|---|---|---|---|"]
    for _, r in df.iterrows():
        pv = "" if pd.isna(r["paper_value"]) else f"{int(r['paper_value']):,}"
        L.append(f"| {r['stage']} | {int(r['count']):,} | {pv} | {r['check']} | {r['note']} |")
    L += ["", f"**交叉核对不一致项：{len(bad)}**"
          + ("" if len(bad) == 0 else " —— 必须逐项查明后才能画图"), ""]
    OUT_MD.write_text("\n".join(L) + "\n")
    print(f"✓ {OUT_CSV.name}  ✓ {OUT_MD.name}\n✅ T2.8 complete", flush=True)
    return 0 if len(bad) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
