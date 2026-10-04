"""
TSE R1 round-3 check — Completeness of the workflow-history commit listings.

Why: scripts/72 and scripts/73 listed the default-branch commits touching
.github/workflows with a five-page (500-commit) cap on September 2, 2026 and
did not record whether a repository had older commits beyond the cap
(round-3 PI finding NEW-3; methodologist NEW-3). The listing is ordered by
committer date, newest first (the same field the scripts compare), so a
dropped commit can only matter when no retained commit lies on or before the
reference date (publish date in scripts/72, probe date in scripts/73): the
latest retained commit on or before that date is the true latest one. The cap
can therefore only have produced a spurious "no workflows yet" verdict, never
a wrong refit commit.

Design:
  1. Page check (one request per repository): ask the commit-listing endpoint
     scripts/72 used for page 6 (per_page=100, path=.github/workflows). A
     non-empty page means the repository exceeded the cap. The frozen check
     (run 2026-09-03T00:50Z) used until=2026-05-15T00:00:00Z, one day earlier
     than the 23:59:59Z bound of scripts/72, so a repository that exceeded the
     cap only through commits made on 2026-05-15 could show an empty page 6;
     step 2 therefore also takes every repository whose scripts/72 listing
     returned exactly 500 commits, and the two sets coincide (nine). UNTIL now
     matches scripts/72; the CSV records the bound each run used. The result
     is cached in the output CSV; a re-run with the CSV present is offline and
     needs no token.
  2. Impact (offline, frozen tables): for every sampled row (scripts/72) and
     in-scope package-date (scripts/73) in a repository that reached the cap
     (page 6 non-empty, or n_workflow_commits_total = 500 in the scripts/72
     table), report whether the decision used a retained commit
     (refit_at_<sha> / unchanged_since_publish) or a "no workflows" verdict
     that the cap could have produced.

Inputs:
  data/tse_r1_workflow_history_audit.csv, data/tse_r1_workflow_history_prewindow.csv
Outputs:
  data/tse_r1_workflow_history_pagination_check.csv    (per repository)
  reports/tse_r1_workflow_history_pagination_check.md

Run: conda run --no-capture-output -n stage2c python scripts/74_workflow_history_pagination_check.py
"""
from __future__ import annotations

import csv
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA, REPORTS = ROOT / "data", ROOT / "reports"
AUDIT_CSV = DATA / "tse_r1_workflow_history_audit.csv"
PREWIN_CSV = DATA / "tse_r1_workflow_history_prewindow.csv"
OUT_CSV = DATA / "tse_r1_workflow_history_pagination_check.csv"
OUT_MD = REPORTS / "tse_r1_workflow_history_pagination_check.md"
GITHUB_API = "https://api.github.com"
UNTIL = "2026-05-15T23:59:59+00:00"  # scripts/72 COLLECTION_DATE; the frozen 2026-09-03 check used 00:00:00 (see docstring)
CAP_PAGES = 5
FIELDS = ["owner", "repo", "status", "page6_commits", "exceeded_cap", "until", "checked_at"]


def repositories() -> list[tuple[str, str]]:
    repos: set[tuple[str, str]] = set()
    for p in (AUDIT_CSV, PREWIN_CSV):
        with p.open() as f:
            for r in csv.DictReader(f):
                repos.add((r["owner"], r["repo"]))
    return sorted(repos)


def page_check(repos: list[tuple[str, str]]) -> list[dict]:
    import requests

    tok = os.environ.get("GITHUB_TOKEN", "")
    if not tok:
        raise SystemExit("GITHUB_TOKEN missing (needed only when the cached check is absent)")
    headers = {"Authorization": f"Bearer {tok}", "Accept": "application/vnd.github+json",
               "User-Agent": "PROBE-history-check/1.0 (research; contact via repository)"}
    checked_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    rows = []
    for o, rp in repos:
        n: int | str = ""
        try:
            resp = requests.get(f"{GITHUB_API}/repos/{o}/{rp}/commits",
                                params={"path": ".github/workflows", "until": UNTIL, "per_page": 100, "page": CAP_PAGES + 1},
                                headers=headers, timeout=30)
            body = resp.json() if resp.status_code == 200 else None
            st = "ok" if isinstance(body, list) else f"http_{resp.status_code}"
            if isinstance(body, list):
                n = len(body)
        except requests.RequestException as e:
            st = f"error: {str(e)[:60]}"
        rows.append({"owner": o, "repo": rp, "status": st, "page6_commits": n,
                     "exceeded_cap": "" if n == "" else ("yes" if n > 0 else "no"), "until": UNTIL, "checked_at": checked_at})
        time.sleep(0.15)
    return rows


def main() -> int:
    repos = repositories()
    if OUT_CSV.exists():
        with OUT_CSV.open() as f:
            check = list(csv.DictReader(f))
        print(f"page check cached: {OUT_CSV.name} ({len(check)} repositories; checked_at {check[0]['checked_at']})", flush=True)
    else:
        check = page_check(repos)
        with OUT_CSV.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(check)
        print(f"✓ {OUT_CSV.name} ({len(check)} repositories)", flush=True)
    exceeded = {(r["owner"], r["repo"]) for r in check if r["exceeded_cap"] == "yes"}
    unresolved = [(r["owner"], r["repo"], r["status"]) for r in check if r["exceeded_cap"] == ""]

    with AUDIT_CSV.open() as f:
        audit = list(csv.DictReader(f))
    with PREWIN_CSV.open() as f:
        prewin = [r for r in csv.DictReader(f) if r["in_scope"] == "yes"]
    at_cap = {(r["owner"], r["repo"]) for r in audit if r["n_workflow_commits_total"] == str(CAP_PAGES * 100)}
    capped = sorted(exceeded | at_cap)

    def a_state(r: dict) -> str:
        return "refit_at_<sha>" if r["publish_time_state"].startswith("refit_at_") else (r["publish_time_state"] or "no_decision")

    def b_state(r: dict) -> str:
        return "refit_at_<sha>" if r["state"].startswith("refit_at_") else (r["state"] or "no_decision")

    per_repo: dict[tuple[str, str], dict] = defaultdict(lambda: {"n_total": "", "page6": "", "rows": 0, "rows_spurious": 0, "pd": 0, "pd_spurious": 0})
    for r in audit:
        k = (r["owner"], r["repo"])
        if k in capped:
            per_repo[k]["n_total"] = r["n_workflow_commits_total"]
            per_repo[k]["rows"] += 1
            per_repo[k]["rows_spurious"] += r["publish_time_state"] == "no_workflows_at_publish"
    for r in prewin:
        k = (r["owner"], r["repo"])
        if k in capped:
            per_repo[k]["pd"] += 1
            per_repo[k]["pd_spurious"] += r["state"] == "no_workflows_at_probe"
    for r in check:
        k = (r["owner"], r["repo"])
        if k in per_repo:
            per_repo[k]["page6"] = r["page6_commits"]
    a_sub = [r for r in audit if (r["owner"], r["repo"]) in capped]
    b_sub = [r for r in prewin if (r["owner"], r["repo"]) in capped]
    a_states, b_states = Counter(map(a_state, a_sub)), Counter(map(b_state, b_sub))
    spurious_a = sum(v["rows_spurious"] for v in per_repo.values())
    spurious_b = sum(v["pd_spurious"] for v in per_repo.values())

    L = ["# 分页完整性检查 — scripts/72、73 的提交列取上限（TSE R1 round 3）\n",
         f"页检查：{check[0]['checked_at']}；请求 page {CAP_PAGES + 1}（per_page=100，until {check[0]['until']}）；{len(check)} 个仓库；"
         f"未能判定 {len(unresolved)} 个。",
         f"超过上限（第 {CAP_PAGES + 1} 页非空）的仓库：**{len(exceeded)}**；scripts/72 表中恰好列到 {CAP_PAGES * 100} 条的仓库：{len(at_cap)}；并集 {len(capped)}。\n",
         "**为何只有\"无 workflow\"裁决可能受影响**：列表按最新在前截断，只丢最旧的提交；只要保留的列表中存在参考日期（发布日 / 探测日）之前的任一提交，"
         "其中最晚的一条就是真实的最晚提交，refit 所用 SHA 不受影响。只有\"参考日期前没有任何保留提交\"的情形才可能把截断误判为\"尚无 workflow\"。\n",
         "| 仓库 | 72 列到的提交数 | 第 6 页提交数 | 72 抽样行 | 其中\"无 workflow\"裁决 | 73 package-date | 其中\"无 workflow\"裁决 |",
         "|---|---|---|---|---|---|---|"]
    for k in capped:
        v = per_repo[k]
        L.append(f"| `{k[0]}/{k[1]}` | {v['n_total']} | {v['page6']} | {v['rows']} | {v['rows_spurious']} | {v['pd']} | {v['pd_spurious']} |")
    L += ["", f"合计：scripts/72 抽样行 {len(a_sub)}（状态：{dict(a_states)}）；scripts/73 在范围内 package-date {len(b_sub)}（状态：{dict(b_states)}）。",
          f"可能被截断误判的裁决：72 **{spurious_a}** 行、73 **{spurious_b}** 个 package-date。",
          "", "## 读法\n",
          ("- 受上限影响的仓库里所有裁决都基于保留列表中的提交（refit_at_<sha> 或发布后无变更），没有任何\"无 workflow\"裁决落在这些仓库，"
           "因此 scripts/72 的 460/486（94.7%）与 scripts/73 的 184/257、236/310、253/328 均不依赖被丢弃的提交。" if spurious_a == 0 and spurious_b == 0 else
           "- 存在可能受截断影响的裁决：应以 MAX_PAGES 重新列取受影响仓库并重跑 72/73。"),
          "- scripts/72、73 已改为分页到穷尽（MAX_PAGES 守卫），截断且参考日期前无保留提交时记 UNCERTAIN。",
          "- 本检查本身是一次只读的元数据请求（每仓库一次），不读取任何版本记录；所有对象日期 ≤ 2026-05-15。"]
    OUT_MD.write_text("\n".join(L) + "\n")
    print(f"✓ {OUT_MD.name}\ncapped repositories: {len(capped)}; spurious-risk decisions: 72={spurious_a}, 73={spurious_b}\n✅ pagination check complete", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
