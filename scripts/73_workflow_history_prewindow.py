"""
TSE R1 round-2 audit arm B2 — Collection-time labels replayed at pre-window dates.

Why: the cross-boundary composition analysis (scripts/59; Section V-B) applies
each package's May 2026 label to releases as early as January 2024. Arm B
(scripts/72) bounded the label gap only for the sampled post-deadline
releases (a gap of at most five months). This arm replays the frozen
classifier on the default-branch workflow files as they stood at three
pre-window probe dates. It re-lists the same repositories' commit histories
in a separate fetch (scripts/72 persists no commit list) and reuses the
workflow-file snapshots scripts/72 cached, so it is the fourth post-cutoff
collection; the gap is measured where the extended-series analysis uses
the labels rather than assumed.

Design (deterministic and declared in this script before the run; not committed to version control before the fetch):
  - Packages: every labelled A/B/C/E package whose repository resolved
    (the same 210 repositories as scripts/72).
  - Probe dates: 2024-04-01, 2024-10-01, 2025-04-01 (equally spaced across
    the pre-window). A package is in scope at a probe date only if the frozen
    version table shows at least one release on or before that date.
  - State at a probe date = workflow files at the last default-branch commit
    touching .github/workflows on or before the date (NOGHA if none yet);
    class = frozen classifier (scripts/06, imported unchanged via scripts/72).
  - Agreement compares the workflow-surface class (A, B, NOGHA for C/E) at the
    probe date with the collection-time label's surface.

Limits: rewritten history invisible; committer dates; the same C/E merge and
UNCERTAIN handling as scripts/72. Nothing here identifies an effect.

Pagination: the 2026-09-02 run inherited the five-page (500-commit) cap of
scripts/72; scripts/74 records that nine repositories exceeded it and that no
package-date decision depended on a dropped commit. The listing now paginates
to exhaustion, and a truncated history with no commit on or before the probe
date is marked UNCERTAIN. Rows now also record commit_list_status and
n_workflow_commits_total (absent from the frozen 2026-09-02 table).

Inputs:
  data/classification.csv, data/repo_status.csv, data/versions_extended.csv
  scripts/72_workflow_history_audit.py (imported; not modified)
Outputs:
  data/workflows_history_raw/<owner>__<repo>__<sha7>/   (shared cache with scripts/72)
  data/tse_r1_workflow_history_prewindow.csv
  reports/tse_r1_workflow_history_prewindow.md

Run: conda run --no-capture-output -n stage2c python scripts/73_workflow_history_prewindow.py
"""
from __future__ import annotations

import csv
import importlib.util
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA, REPORTS, SCRIPTS = ROOT / "data", ROOT / "reports", ROOT / "scripts"
CLASS_CSV, STATUS_CSV, VERSIONS_CSV = DATA / "classification.csv", DATA / "repo_status.csv", DATA / "versions_extended.csv"
OUT_CSV = DATA / "tse_r1_workflow_history_prewindow.csv"
OUT_MD = REPORTS / "tse_r1_workflow_history_prewindow.md"
PROBES = [datetime(2024, 4, 1, tzinfo=timezone.utc), datetime(2024, 10, 1, tzinfo=timezone.utc), datetime(2025, 4, 1, tzinfo=timezone.utc)]

_spec = importlib.util.spec_from_file_location("wf72", SCRIPTS / "72_workflow_history_audit.py")
wf72 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wf72)  # type: ignore[union-attr]


def first_release(versions_csv: Path) -> dict[str, datetime]:
    first: dict[str, datetime] = {}
    with versions_csv.open() as f:
        for r in csv.DictReader(f):
            d = wf72.parse_dt(r["published_at"])
            if d and (r["package"] not in first or d < first[r["package"]]):
                first[r["package"]] = d
    return first


def main() -> int:
    if not os.environ.get("GITHUB_TOKEN", ""):
        print("GITHUB_TOKEN missing", flush=True)
        return 1
    labels = {r["package"]: r["class"] for r in csv.DictReader(CLASS_CSV.open()) if r["class"] in ("A", "B", "C", "E")}
    status = {}
    for r in csv.DictReader(STATUS_CSV.open()):
        if r["package"] in labels and r["repo_status"] in ("ok", "no_workflows"):
            status.setdefault(r["package"], (r["owner"], r["repo"]))
    first = first_release(VERSIONS_CSV)
    pkgs = sorted(p for p in labels if p in status)
    repos = sorted(set(status[p] for p in pkgs))
    print(f"packages: {len(pkgs)}; repositories: {len(repos)}; probes: {[d.date().isoformat() for d in PROBES]}", flush=True)
    collected_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    commits: dict[tuple[str, str], tuple[str, list]] = {}
    for i, (o, rp) in enumerate(repos, 1):
        commits[(o, rp)] = wf72.workflow_commits(o, rp)
        time.sleep(0.2)
        if i % 50 == 0:
            print(f"  commits listed for {i}/{len(repos)} repos", flush=True)
    tmp_root = ROOT / "temp" / "wf_history_tmp"
    out = []
    for p in pkgs:
        o, rp = status[p]
        st, clist = commits[(o, rp)]
        for D in PROBES:
            rec = {"package": p, "owner": o, "repo": rp, "label": labels[p], "probe_date": D.date().isoformat(),
                   "commit_list_status": st, "n_workflow_commits_total": len(clist),
                   "in_scope": "", "state": "", "probe_class": "", "agree": "", "collected_at": collected_at}
            fr = first.get(p)
            if fr is None or fr > D:
                rec["in_scope"] = "no"
                out.append(rec)
                continue
            rec["in_scope"] = "yes"
            if st not in ("ok", "ok_truncated"):
                rec["state"] = f"commits_{st}"
                out.append(rec)
                continue
            before = [(d, sha) for d, sha in clist if d <= D]
            if not before and st == "ok_truncated":
                rec["state"], rec["probe_class"] = "history_truncated", "UNCERTAIN"
            elif not before:
                rec["state"], rec["probe_class"] = "no_workflows_at_probe", "NOGHA"
            else:
                d0, sha0 = max(before)
                fst, texts = wf72.files_at(o, rp, sha0)
                if fst in ("ok", "cached", "no_dir"):
                    rec["state"] = f"refit_at_{sha0[:7]}"
                    rec["probe_class"] = wf72.classify_files(texts, tmp_root / f"{o}__{rp}__{sha0[:7]}")
                else:
                    rec["state"] = f"fetch_{fst}"
            if rec["probe_class"]:
                c = rec["probe_class"]
                rec["agree"] = "yes" if wf72.surface(labels[p]) == wf72.surface(c) else ("uncertain" if c == "UNCERTAIN" else "no")
            out.append(rec)
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    L = ["# 审计臂 B2 — 采集时标签在 pre-window 三个时点的重放（TSE R1 round 2）\n",
         f"采集时间：{collected_at}（第四次截止后采集：提交历史重新列取；workflow 文件快照与 scripts/72 共用缓存）；探测日期 {', '.join(d.date().isoformat() for d in PROBES)}。",
         f"对象：{len(pkgs)} 个已标注 A/B/C/E 包（{len(repos)} 个仓库）；每个时点只计入该日期前已有发布的包。\n",
         "| 探测日期 | 在范围内 | 有裁决 | 一致 | 不一致 | 不确定 | 一致率 |", "|---|---|---|---|---|---|---|"]
    for D in PROBES:
        sub = [r for r in out if r["probe_date"] == D.date().isoformat() and r["in_scope"] == "yes"]
        dec = [r for r in sub if r["agree"]]
        c = Counter(r["agree"] for r in dec)
        L.append(f"| {D.date()} | {len(sub)} | {len(dec)} | {c['yes']} | {c['no']} | {c['uncertain']} | {100*c['yes']/max(1,len(dec)):.1f}% |")
    L += ["", "## 按标签（三个时点合并）\n", "| 标签 | 有裁决 | 一致 | 不一致 | 不确定 | 一致率 |", "|---|---|---|---|---|---|"]
    by = defaultdict(Counter)
    for r in out:
        if r["agree"]:
            by[r["label"]][r["agree"]] += 1
    for lab in ("A", "B", "C", "E"):
        c = by[lab]; n = sum(c.values())
        if n:
            L.append(f"| {lab} | {n} | {c['yes']} | {c['no']} | {c['uncertain']} | {100*c['yes']/n:.1f}% |")
    dis = [r for r in out if r["agree"] == "no"]
    L += ["", f"## 不一致（{len(dis)} 条包×时点）\n", "| 包 | 时点 | 采集时标签 | 时点类别 |", "|---|---|---|---|"]
    for r in dis:
        L.append(f"| `{r['package']}` | {r['probe_date']} | {r['label']} | {r['probe_class']} |")
    L += ["", "## 读法\n",
          "- 这就是跨边界分解所依赖的\"固定标签\"在 pre-window 各时点的经验一致率；它把方法学家 #7 的\"分类器看不到历史变化\"变成可量化的界。",
          "- 与 scripts/72 相同的合并规则（C/E 在 GHA 表面同为无发布 workflow）与限制（改写历史不可见）。"]
    OUT_MD.write_text("\n".join(L) + "\n")
    print(f"✓ {OUT_CSV.name}  ✓ {OUT_MD.name}\n✅ audit arm B2 complete", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
