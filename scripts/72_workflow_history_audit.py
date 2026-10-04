"""
TSE R1 round-2 audit arm B — Publish-time versus collection-time workflow state.

Why: every A--F label describes the repository's default-branch workflow files
as cached in mid-May 2026, not the state at the sampled version's publish date
(Sections IV-C, IX-A; methodologist round-1 finding 7). The default branch's
commit history lets us check, per sampled row, whether the `.github/workflows`
directory changed between the publish date and the collection date, and, where
it did, what the classifier would have said on the pre-publish state.

Design (deterministic and declared in this script before the run; not committed to version control before the fetch):
  - Rows: the 500-row classification sample (final labels), restricted to rows
    whose repository resolved (classes A, B, C, E; D and F have no inspectable
    workflow surface).
  - Per repository: list the default-branch commits touching
    `.github/workflows` up to COLLECTION_DATE (the May 15, 2026 cache date).
  - Per row: `n_changes_after` = commits with publish_date < date <= collection
    date. If zero, the cached files ARE the publish-time files and the label
    transfers unchanged. If positive, fetch the workflow files at the last
    commit on or before the publish date and re-run the frozen classifier
    (scripts/06, imported unchanged) on them to obtain the publish-time class;
    `no workflows at publish time` is recorded when the directory did not yet
    exist.
  - Agreement: publish-time class versus collection-time label, per class.
    C and E both mean "no GitHub Actions publish workflow" on the workflow
    surface, so they are compared as one group (external-CI presence is not
    re-probed historically).

Limits: rewritten or force-pushed history is invisible; commits are dated by
committer date; renamed default branches are followed by the API. Nothing here
identifies an effect.

Pagination: the September 2, 2026 run listed at most five pages (500 commits)
per repository and did not record whether older commits existed. Nine
repositories exceeded that cap (scripts/74 records the check); because the
listing is ordered by committer date, newest first, a dropped commit can only
matter when no retained commit lies on or before the reference date (the
latest retained one is then the true latest), and scripts/74 shows that every
sampled-row and package-date decision in those repositories used a retained
commit, so no reported figure depends on the cap. The listing now paginates to
exhaustion (MAX_PAGES guard) and a truncated history that leaves no commit on
or before the publish date is marked UNCERTAIN rather than "no workflows".

Inputs:
  data/classification.csv, data/repo_status.csv, data/non_provenance_sample.csv
  scripts/06_classify_non_provenance.py (imported; not modified)
Outputs:
  data/workflows_history_raw/<owner>__<repo>__<sha7>/<file>   (raw pre-publish files; new directory)
  data/tse_r1_workflow_history_audit.csv
  reports/tse_r1_workflow_history_audit.md

Run: conda run --no-capture-output -n stage2c python scripts/72_workflow_history_audit.py
"""
from __future__ import annotations

import base64
import csv
import importlib.util
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).parent.parent
DATA, REPORTS, SCRIPTS = ROOT / "data", ROOT / "reports", ROOT / "scripts"
CLASS_CSV, STATUS_CSV, SAMPLE_CSV = DATA / "classification.csv", DATA / "repo_status.csv", DATA / "non_provenance_sample.csv"
RAW_DIR = DATA / "workflows_history_raw"
OUT_CSV = DATA / "tse_r1_workflow_history_audit.csv"
OUT_MD = REPORTS / "tse_r1_workflow_history_audit.md"

COLLECTION_DATE = datetime(2026, 5, 15, 23, 59, 59, tzinfo=timezone.utc)
MAX_PAGES = 200  # 20,000 commits per repository; beyond this the listing is reported as ok_truncated (the 2026-09-02 run capped at 5 pages; see docstring)
DATA_CUTOFF = datetime(2026, 5, 10, 23, 59, 59, tzinfo=timezone.utc)
GITHUB_API = "https://api.github.com"
TOKEN = os.environ.get("GITHUB_TOKEN", "")
HEADERS = {"Authorization": f"Bearer {TOKEN}", "Accept": "application/vnd.github+json",
           "User-Agent": "PROBE-workflow-history-audit/1.0"}

_spec = importlib.util.spec_from_file_location("clf06", SCRIPTS / "06_classify_non_provenance.py")
clf06 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(clf06)  # type: ignore[union-attr]


def parse_dt(s: str) -> datetime | None:
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def api_get(url: str, params: dict | None = None) -> tuple[int, object]:
    for backoff in (0, 5, 20, 60):
        if backoff:
            time.sleep(backoff)
        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=30)
        except Exception:
            continue
        if r.status_code in (403, 429) and int(r.headers.get("X-RateLimit-Remaining", "1")) == 0:
            wait = max(5, int(r.headers.get("X-RateLimit-Reset", "0")) - int(time.time()))
            print(f"  rate limit; sleeping {wait}s", flush=True)
            time.sleep(wait)
            continue
        if r.status_code >= 500:
            continue
        try:
            return r.status_code, r.json()
        except Exception:
            return r.status_code, None
    return 0, None


def workflow_commits(owner: str, repo: str) -> tuple[str, list[tuple[datetime, str]]]:
    """All default-branch commits touching .github/workflows up to the collection date, newest first.

    Returns ("ok", commits) when the listing was exhausted and ("ok_truncated", commits)
    when MAX_PAGES full pages were read, in which case older commits may be missing."""
    out: list[tuple[datetime, str]] = []
    page = 1
    while True:
        st, j = api_get(f"{GITHUB_API}/repos/{owner}/{repo}/commits",
                        {"path": ".github/workflows", "until": COLLECTION_DATE.isoformat(), "per_page": 100, "page": page})
        if st == 404:
            return "not_found", out
        if st != 200 or not isinstance(j, list):
            return f"http_{st}", out
        for c in j:
            d = parse_dt(((c.get("commit") or {}).get("committer") or {}).get("date", ""))
            if d:
                out.append((d, c["sha"]))
        if len(j) < 100:
            break
        if page >= MAX_PAGES:
            return "ok_truncated", out
        page += 1
        time.sleep(0.2)
    return "ok", out


def files_at(owner: str, repo: str, sha: str) -> tuple[str, dict[str, str]]:
    dest = RAW_DIR / f"{owner}__{repo}__{sha[:7]}"
    if dest.exists() and any(dest.iterdir()):
        return "cached", {p.name: p.read_text(errors="replace") for p in dest.iterdir() if p.is_file()}
    st, j = api_get(f"{GITHUB_API}/repos/{owner}/{repo}/contents/.github/workflows", {"ref": sha})
    if st == 404:
        return "no_dir", {}
    if st != 200 or not isinstance(j, list):
        return f"http_{st}", {}
    dest.mkdir(parents=True, exist_ok=True)
    texts: dict[str, str] = {}
    for item in j:
        if item.get("type") != "file" or not item["name"].endswith((".yml", ".yaml")):
            continue
        st2, f = api_get(item["url"])  # contents API with ref baked in
        if st2 == 200 and isinstance(f, dict) and f.get("encoding") == "base64":
            texts[item["name"]] = base64.b64decode(f["content"]).decode("utf-8", "replace")
            (dest / clf06.safe_name(item["name"]) if hasattr(clf06, "safe_name") else dest / item["name"].replace("/", "_")).write_text(texts[item["name"]])
        time.sleep(0.15)
    return "ok", texts


def classify_files(texts: dict[str, str], tmp: Path) -> str:
    """Reproduce scripts/06's per-repo decision on a set of workflow files (GHA surface only)."""
    if not texts:
        return "NOGHA"
    tmp.mkdir(parents=True, exist_ok=True)
    results = []
    for name, text in texts.items():
        p = tmp / name.replace("/", "_")
        p.write_text(text)
        c, _ = clf06.classify_publish_workflow(p)
        if c != "SKIP":
            results.append(c)
    if not results:
        return "NOGHA"          # no GHA publish workflow -> C or E at collection time
    return results[0] if len(set(results)) == 1 else "UNCERTAIN"


def surface(label: str) -> str:
    return "NOGHA" if label in ("C", "E") else label


def main() -> int:
    if not TOKEN:
        print("GITHUB_TOKEN missing", flush=True)
        return 1
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    labels = list(csv.DictReader(CLASS_CSV.open()))
    status = {(r["package"], r["version"]): r for r in csv.DictReader(STATUS_CSV.open())}
    pub = {(r["package"], r["version"]): parse_dt(r["published_at"]) for r in csv.DictReader(SAMPLE_CSV.open())}
    rows = [r for r in labels if r["class"] in ("A", "B", "C", "E") and (r["package"], r["version"]) in status
            and status[(r["package"], r["version"])]["repo_status"] in ("ok", "no_workflows")]
    repos = sorted({(status[(r["package"], r["version"])]["owner"], status[(r["package"], r["version"])]["repo"]) for r in rows})
    print(f"rows in scope: {len(rows)} of {len(labels)}; repositories: {len(repos)}", flush=True)
    collected_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    commits: dict[tuple[str, str], tuple[str, list[tuple[datetime, str]]]] = {}
    for i, (o, rp) in enumerate(repos, 1):
        commits[(o, rp)] = workflow_commits(o, rp)
        time.sleep(0.2)
        if i % 50 == 0:
            print(f"  commits listed for {i}/{len(repos)} repos", flush=True)
    out = []
    tmp_root = ROOT / "temp" / "wf_history_tmp"
    for r in rows:
        key = (r["package"], r["version"])
        s = status[key]
        o, rp = s["owner"], s["repo"]
        pdate = pub.get(key)
        st, clist = commits[(o, rp)]
        rec = {"package": r["package"], "version": r["version"], "published_at": pdate.isoformat() if pdate else "",
               "owner": o, "repo": rp, "label": r["class"], "commit_list_status": st,
               "n_workflow_commits_total": len(clist), "n_changes_after_publish": "", "last_change_before_publish": "",
               "days_from_last_change_to_publish": "", "publish_time_state": "", "publish_time_class": "",
               "agree": "", "collected_at": collected_at}
        if st in ("ok", "ok_truncated") and pdate:
            assert pdate <= DATA_CUTOFF
            after = [d for d, _ in clist if pdate < d <= COLLECTION_DATE]
            before = [(d, sha) for d, sha in clist if d <= pdate]
            rec["n_changes_after_publish"] = len(after)
            if before:
                d0, sha0 = max(before)
                rec["last_change_before_publish"] = d0.isoformat()
                rec["days_from_last_change_to_publish"] = round((pdate - d0).total_seconds() / 86400, 1)
            if not after:
                rec["publish_time_state"] = "unchanged_since_publish"
                rec["publish_time_class"] = r["class"]
                rec["agree"] = "yes"
            elif not before and st == "ok_truncated":
                rec["publish_time_state"] = "history_truncated"
                rec["publish_time_class"] = "UNCERTAIN"
                rec["agree"] = "uncertain"
            elif not before:
                rec["publish_time_state"] = "no_workflows_at_publish"
                rec["publish_time_class"] = "NOGHA"
                rec["agree"] = "yes" if surface(r["class"]) == "NOGHA" else "no"
            else:
                fst, texts = files_at(o, rp, sha0)
                if fst in ("ok", "cached", "no_dir"):
                    c = classify_files(texts, tmp_root / f"{o}__{rp}__{sha0[:7]}")
                    rec["publish_time_state"] = f"refit_at_{sha0[:7]}"
                    rec["publish_time_class"] = c
                    rec["agree"] = "yes" if surface(r["class"]) == surface(c) else ("uncertain" if c == "UNCERTAIN" else "no")
                else:
                    rec["publish_time_state"] = f"fetch_{fst}"
        out.append(rec)
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    # ---- report ----
    covered = [r for r in out if r["agree"] != ""]
    by_label: dict[str, Counter] = defaultdict(Counter)
    for r in covered:
        by_label[r["label"]][r["agree"]] += 1
    changed = [r for r in covered if r["publish_time_state"] != "unchanged_since_publish"]
    L = ["# 审计臂 B — 发布时刻 vs 采集时刻的 workflow 状态（TSE R1 round 2）\n",
         f"采集时间（git 历史抓取）：{collected_at}；提交记录截止 {COLLECTION_DATE.date()}（原缓存采集日）。",
         f"范围：{len(rows)} 个可核验行（A/B/C/E，仓库可达）/ {len(repos)} 个仓库；覆盖（得到裁决）{len(covered)} 行（{100*len(covered)/max(1,len(rows)):.1f}%）。\n",
         f"- 发布后至采集日 workflow 目录**无任何提交**的行：{sum(1 for r in covered if r['publish_time_state']=='unchanged_since_publish')}（标签即发布时状态）。",
         f"- 发布后有变更、需在发布前提交上重跑分类器的行：{len(changed)}；其中一致 {sum(1 for r in changed if r['agree']=='yes')}、不一致 {sum(1 for r in changed if r['agree']=='no')}、不确定 {sum(1 for r in changed if r['agree']=='uncertain')}。\n",
         "| 标签 | 覆盖行 | 一致 | 不一致 | 不确定 | 一致率 |", "|---|---|---|---|---|---|"]
    for lab in ("A", "B", "C", "E"):
        c = by_label[lab]; n = sum(c.values())
        if n:
            L.append(f"| {lab} | {n} | {c['yes']} | {c['no']} | {c['uncertain']} | {100*c['yes']/n:.1f}% |")
    tot = Counter(r["agree"] for r in covered)
    L.append(f"| 合计 | {len(covered)} | {tot['yes']} | {tot['no']} | {tot['uncertain']} | {100*tot['yes']/max(1,len(covered)):.1f}% |")
    dis = [r for r in covered if r["agree"] == "no"]
    L += ["", "## 不一致行\n", "| 包 | 版本 | 采集时标签 | 发布时类别 | 发布后变更数 |", "|---|---|---|---|---|"]
    for r in dis:
        L.append(f"| `{r['package']}` | {r['version']} | {r['label']} | {r['publish_time_class']} | {r['n_changes_after_publish']} |")
    L += ["", "## 读法\n",
          "- 一致率回答方法学家 #7 与 A-3：采集时刻标签在多大比例的行上等于发布时刻状态。",
          "- C/E 在 GHA 表面上同为\"无发布 workflow\"，合并比较；外部 CI 的历史存在性未回溯。",
          "- 改写/强推的历史不可见；提交按 committer 日期；标签主表不改动，本臂只量化构念差距。"]
    OUT_MD.write_text("\n".join(L) + "\n")
    print(f"✓ {OUT_CSV.name}  ✓ {OUT_MD.name}\n✅ audit arm B complete", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
