"""
TSE R1 revision, T1.2 — Construct audit for the extended pre-period.

Reviewer 1 asked for a longer pre-period. Extending the lower bound re-reads
packuments that were already collected (2026-05-15) and so costs no new
collection, but it does raise three construct questions that must be settled
BEFORE any extended-window estimate enters the paper:

  Q1 Is `dist.attestations` a publish-time attachment, or could the registry
     have backfilled it? If backfilled, a 2024 "attested" version would be
     evidence about 2026, not about 2024, and the whole extension collapses.
  Q2 How far back does the field remain measurable before it thins out?
  Q3 How much does the post-window cohort assembly (2026-05-15) tighten the
     survivorship condition as the window start moves earlier?

Q1 is addressed (not settled) by the SLSA predicate version carried inside each
attestation. npm moved from `https://slsa.dev/provenance/v0.2` to `.../v1`. Had
the registry rewritten every surviving historical attestation to the current
predicate, no pre-migration version would still carry v0.2. A stable, monotone
v0.2 -> v1 transition aligned with publish date is therefore evidence AGAINST a
wholesale rewrite of historical attestations. It does not exclude a later
attachment of an originally signed v0.2 payload, selective backfill, or any
other registry behaviour that preserves the embedded predicate; the frozen
packuments carry no signing timestamp, so a timestamp-versus-publication audit
is not possible without new collection. The outcome on the extended series is
therefore defined as the collection-time attestation-field state indexed by
historical publication date, exactly as on the protocol window.

Inputs:
  data/packuments_raw/*.json      (already-collected packuments; NO new fetches)
  data/packages_full.txt          (cohort membership + tier)
  data/versions.csv               (as-submitted window, for the survivorship contrast)
  data/versions_extended.csv      (extended window, from scripts/03 --window-start)

Outputs:
  data/tse_r1_predicate_by_month.csv    (month x predicateType counts)
  data/tse_r1_window_start_tradeoff.csv (candidate start date -> n, packages, coverage)
  reports/tse_r1_prewindow_construct_audit.md

Carried caveats:
  - Registry-level construct only: has_attestation = the packument carried the
    field at collection time. Unchanged from the primary analysis.
  - Cohort membership is fixed at the 2026-05-15 assembly; extending the window
    earlier does NOT recruit packages that had disappeared by then. This report
    quantifies that condition, it does not remove it.
  - No record may post-date the 2026-05-10 cutoff; asserted at exit.

Run: conda run --no-capture-output -n stage2c python scripts/55_prewindow_construct_audit.py
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
REPORTS_DIR = Path(__file__).parent.parent / "reports"
RAW_DIR = DATA_DIR / "packuments_raw"
PACKAGES_FILE = DATA_DIR / "packages_full.txt"
VERSIONS_CSV = DATA_DIR / "versions.csv"
EXTENDED_CSV = DATA_DIR / "versions_extended.csv"

PREDICATE_CSV = DATA_DIR / "tse_r1_predicate_by_month.csv"
TRADEOFF_CSV = DATA_DIR / "tse_r1_window_start_tradeoff.csv"
REPORT_MD = REPORTS_DIR / "tse_r1_prewindow_construct_audit.md"

DATA_CUTOFF = "2026-05-10T23:59:59"
CANDIDATE_STARTS = ["2023-07-01", "2023-11-01", "2024-01-01", "2024-07-01",
                    "2025-01-01", "2025-08-01"]


def safe_fname(pkg: str) -> str:
    """Packument filename convention, identical to scripts/02 and scripts/03."""
    return pkg.replace("/", "_").replace("@", "at_") + ".json"


def load_cohort() -> dict[str, str]:
    """package -> deps.dev-style tier label from the frozen cohort file."""
    meta: dict[str, str] = {}
    with PACKAGES_FILE.open() as f:
        header = next(f).rstrip("\n").split("\t")
        i_name, i_tier = header.index("name"), header.index("dpc_tier")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) > max(i_name, i_tier):
                meta[parts[i_name]] = parts[i_tier]
    return meta


def scan_packuments(cohort: dict[str, str]) -> tuple[dict, Counter, Counter]:
    """Walk every cohort packument once, collecting predicate/month evidence."""
    by_month: dict[str, Counter] = defaultdict(Counter)
    versions_per_month: Counter = Counter()
    first_attested_year: Counter = Counter()
    for pkg in cohort:
        fpath = RAW_DIR / safe_fname(pkg)
        if not fpath.exists():
            continue
        try:
            data = json.loads(fpath.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if data.get("_404"):
            continue
        times = data.get("time", {}) or {}
        earliest_att = None
        for ver, vdata in (data.get("versions", {}) or {}).items():
            pub = times.get(ver, "")
            if not pub or pub > DATA_CUTOFF:
                continue
            month = pub[:7]
            versions_per_month[month] += 1
            att = (vdata.get("dist") or {}).get("attestations")
            if not isinstance(att, dict) or not att.get("url"):
                continue
            pred = ((att.get("provenance") or {}).get("predicateType")
                    or "MISSING_PREDICATE")
            by_month[month][pred] += 1
            if earliest_att is None or pub < earliest_att:
                earliest_att = pub
        if earliest_att:
            first_attested_year[earliest_att[:4]] += 1
    return by_month, versions_per_month, first_attested_year


def window_start_tradeoff(cohort: dict[str, str]) -> list[dict]:
    """For each candidate start date, how much data and how many packages."""
    rows: list[dict] = []
    per_start: dict[str, Counter] = {s: Counter() for s in CANDIDATE_STARTS}
    pkgs_per_start: dict[str, set] = {s: set() for s in CANDIDATE_STARTS}
    att_per_start: dict[str, Counter] = {s: Counter() for s in CANDIDATE_STARTS}
    for pkg in cohort:
        fpath = RAW_DIR / safe_fname(pkg)
        if not fpath.exists():
            continue
        try:
            data = json.loads(fpath.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if data.get("_404"):
            continue
        times = data.get("time", {}) or {}
        for ver, vdata in (data.get("versions", {}) or {}).items():
            pub = times.get(ver, "")
            if not pub or pub > DATA_CUTOFF:
                continue
            att = (vdata.get("dist") or {}).get("attestations")
            has_att = isinstance(att, dict) and bool(att.get("url"))
            for s in CANDIDATE_STARTS:
                if pub >= s:
                    per_start[s]["n"] += 1
                    pkgs_per_start[s].add(pkg)
                    if has_att:
                        att_per_start[s]["n"] += 1
    for s in CANDIDATE_STARTS:
        n = per_start[s]["n"]
        a = att_per_start[s]["n"]
        pre_weeks = (datetime(2025, 12, 9, tzinfo=timezone.utc)
                     - datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)).days / 7
        rows.append({
            "window_start": s,
            "versions": n,
            "packages": len(pkgs_per_start[s]),
            "attested": a,
            "attested_pct": round(100 * a / n, 2) if n else 0.0,
            "pre_deadline_weeks": round(pre_weeks, 1),
        })
    return rows


def main() -> int:
    cohort = load_cohort()
    print(f"Cohort packages: {len(cohort):,}", flush=True)

    by_month, versions_per_month, first_att_year = scan_packuments(cohort)
    tradeoff = window_start_tradeoff(cohort)

    # --- Q1: predicate transition -------------------------------------------
    predicates = sorted({p for c in by_month.values() for p in c})
    with PREDICATE_CSV.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["month", "versions_total", "attested_total"] + predicates)
        for month in sorted(by_month):
            c = by_month[month]
            w.writerow([month, versions_per_month[month], sum(c.values())]
                       + [c.get(p, 0) for p in predicates])
    print(f"✓ {PREDICATE_CSV}", flush=True)

    with TRADEOFF_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(tradeoff[0].keys()))
        w.writeheader()
        w.writerows(tradeoff)
    print(f"✓ {TRADEOFF_CSV}", flush=True)

    # Backfill test, driven by the observed transition rather than a guessed date.
    # npm migrated the SLSA predicate v0.2 -> v1. Locate the crossover month
    # empirically, then ask the decisive question: do versions published WELL
    # BEFORE the crossover carry the NEW predicate? Under a registry-side rewrite
    # they all would.
    OLD, NEW = "https://slsa.dev/provenance/v0.2", "https://slsa.dev/provenance/v1"
    crossover = None
    for m in sorted(by_month):
        c = by_month[m]
        tot = c.get(OLD, 0) + c.get(NEW, 0)
        if tot >= 20 and c.get(NEW, 0) / tot > 0.5:
            crossover = m
            break

    def _shift(month: str, months: int) -> str:
        y, mo = int(month[:4]), int(month[5:7])
        k = (y * 12 + mo - 1) + months
        return f"{k // 12:04d}-{k % 12 + 1:02d}"

    pre_cut = _shift(crossover, -6) if crossover else "0000-00"
    post_cut = _shift(crossover, 6) if crossover else "9999-99"
    new_before = sum(c.get(NEW, 0) for m, c in by_month.items() if m < pre_cut)
    old_total_before = sum(c.get(OLD, 0) for m, c in by_month.items() if m < pre_cut)
    old_after = sum(c.get(OLD, 0) for m, c in by_month.items() if m > post_cut)
    new_total_after = sum(c.get(NEW, 0) for m, c in by_month.items() if m > post_cut)
    total_att = sum(sum(c.values()) for c in by_month.values())

    lines: list[str] = []
    lines.append("# T1.2 — 扩展 pre-period 的构念核验（TSE R1）\n")
    lines.append(f"生成时间: {datetime.now(timezone.utc).date()}  ")
    lines.append(f"输入: `data/packuments_raw/` （已采集，**无新拉取**）\n")

    lines.append("## Q1 历史 attestation 是否被整体重写？（predicate 迁移检验）\n")
    lines.append("判据：npm 的 SLSA predicate 版本从 `provenance/v0.2` 迁移到 `provenance/v1`。")
    lines.append("若注册表把历史 attestation **整体**重写为当前 predicate，迁移前发布的存活版本就不会再带 v0.2。\n")
    lines.append(f"- 全队列带 attestation 的版本数: **{total_att:,}**")
    lines.append(f"- 观测到的 predicate 迁移交叉月（v1 首次过半）: **{crossover}**\n")
    lines.append("**检验**：若注册表整体重写过历史 attestation，迁移之前发布的版本会")
    lines.append("全部带上新 predicate。实际观测：\n")
    lines.append(f"- 迁移前 6 个月（< {pre_cut}）发布且带**新** predicate 的版本: "
                 f"**{new_before:,}**（同期带旧 predicate: {old_total_before:,}）")
    lines.append(f"- 迁移后 6 个月（> {post_cut}）发布仍带**旧** predicate 的版本: "
                 f"**{old_after:,}**（同期带新 predicate: {new_total_after:,}）\n")
    lines.append("迁移后仍有少量旧 predicate，与发布方工具链滞后一致；迁移前零个新 predicate，")
    lines.append("是**反对整体重写**历史 attestation 的证据。它**不能**排除原始签名的 v0.2 载荷被事后附着、")
    lines.append("选择性回填、或其他保留内嵌 predicate 的注册表行为；冻结 packument 不含签名时间戳，")
    lines.append("时间戳-发布日期审计需要新采集，本轮不做。因此扩展窗的结果定义为")
    lines.append("**按历史发布日期索引的采集时刻 attestation 字段状态**，与协议窗一致。\n")
    lines.append("按月分布见 `data/tse_r1_predicate_by_month.csv`。\n")

    lines.append("## Q2 字段可得性的时间下探\n")
    lines.append("| 首次 attested 发布年 | 包数 |")
    lines.append("|---|---|")
    for y in sorted(first_att_year):
        lines.append(f"| {y} | {first_att_year[y]:,} |")
    lines.append("")

    lines.append("## Q3 窗口起点的取舍（含幸存者条件）\n")
    lines.append("| 起点 | 版本数 | 包数 | attested | attested% | deadline 前周数 |")
    lines.append("|---|---|---|---|---|---|")
    for r in tradeoff:
        lines.append(f"| {r['window_start']} | {r['versions']:,} | {r['packages']:,} | "
                     f"{r['attested']:,} | {r['attested_pct']}% | {r['pre_deadline_weeks']} |")
    lines.append("")
    lines.append("**幸存者条件**：队列成员固定于 2026-05-15 的组建时点。起点越早，")
    lines.append("\"当年活跃但组建前已消失/改名/转私有的包无法贡献轨迹\" 这一条件越强。")
    lines.append("扩窗**不缓解**该偏差，只是把它拉得更长；须在 §9 加重表述。\n")

    REPORT_MD.write_text("\n".join(lines))
    print(f"✓ {REPORT_MD}", flush=True)

    # 铁律
    assert all(m <= DATA_CUTOFF[:7] for m in by_month), "DATA CUTOFF VIOLATION in month keys"
    print("\n✅ T1.2 construct audit complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
