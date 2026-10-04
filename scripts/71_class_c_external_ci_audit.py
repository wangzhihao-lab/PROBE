"""
TSE R1 round-2 audit arm C — Class C external-CI configuration verification.

Why: Class C ("possible external CI") is presence-only: the classifier records
that a `.circleci/config.yml` or `.gitlab-ci.yml` exists but never parses it
(Reviewer 2, item R2-1b). This arm fetches the external-CI configuration files
of every Class C repository and checks whether they contain an npm publish
step and, if so, which credential path it uses.

Design (declared in this script before the run; the publish-command detector below was refined ONCE after a manual keyword check of the fetched files, which had shown the first regex missing a `make publish-nightly` job with `npmAuthToken`, so the detector is exploratory rather than pre-fixed; the probe file list and verdict categories were not changed):
  - Rows: every Class C row in data/classification.csv (final labels); repos
    from data/repo_status.csv (owner/repo).
  - Files probed at the repository's current default branch through the GitHub
    Contents API: .circleci/config.yml, .gitlab-ci.yml, .travis.yml,
    Jenkinsfile, azure-pipelines.yml, .drone.yml, bitbucket-pipelines.yml.
  - Verdict per repository (then propagated to rows):
      publish_token      publish command present AND a token credential visible
      publish_oidc       publish command present AND provenance/OIDC markers
      publish_unresolved publish command present, credential not visible
      no_publish         config present, no npm publish command
      config_missing     none of the probed files exist (404) at collection time
      error              API/network error
  - This is a later collection time (September 2026) than the May 2026 cache
    that produced the labels; the verdicts therefore verify the C premise as of
    the audit date and are reported as such, never as publish-time facts.

Inputs:
  data/classification.csv, data/repo_status.csv
Outputs:
  data/external_ci_raw/<owner>__<repo>__<file>   (raw configs; new directory)
  data/tse_r1_class_c_external_ci_audit.csv
  reports/tse_r1_class_c_external_ci_audit.md

Run: conda run --no-capture-output -n stage2c python scripts/71_class_c_external_ci_audit.py
"""
from __future__ import annotations

import base64
import csv
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).parent.parent
DATA, REPORTS = ROOT / "data", ROOT / "reports"
CLASS_CSV, STATUS_CSV = DATA / "classification.csv", DATA / "repo_status.csv"
RAW_DIR = DATA / "external_ci_raw"
OUT_CSV = DATA / "tse_r1_class_c_external_ci_audit.csv"
OUT_MD = REPORTS / "tse_r1_class_c_external_ci_audit.md"

GITHUB_API = "https://api.github.com"
TOKEN = os.environ.get("GITHUB_TOKEN", "")
HEADERS = {"Authorization": f"Bearer {TOKEN}", "Accept": "application/vnd.github+json",
           "User-Agent": "PROBE-classC-audit/1.0"}
PROBE_FILES = [".circleci/config.yml", ".gitlab-ci.yml", ".travis.yml", "Jenkinsfile",
               "azure-pipelines.yml", ".drone.yml", "bitbucket-pipelines.yml"]
PUBLISH_RE = re.compile(r"\b(npm|pnpm|yarn)\s+publish\b|\bsemantic-release\b|\blerna\s+publish\b|\bchangeset\s+publish\b|"
                        r"\bnp\b\s|\brelease-it\b|\bnpm\s+run\s+(release|publish)\b|\bpublish-npm\b|\bnpm-publish\b|"
                        r"\bmake\s+publish\b|npmAuthToken|_authToken=|^\s*publish[\w-]*:\s*$|name:\s*publish\b", re.I | re.M)
DYNAMIC_RE = re.compile(r"^\s*setup:\s*true\s*$", re.M)
TOKEN_RE = re.compile(r"NPM_TOKEN|NODE_AUTH_TOKEN|npm_token|_authToken|NPM_AUTH", re.I)
OIDC_RE = re.compile(r"--provenance|id-token|oidc|openid|trusted.?publish|sigstore", re.I)


def get_file(owner: str, repo: str, path: str) -> tuple[str, str]:
    cached = RAW_DIR / f"{owner}__{repo}__{path.replace('/', '_')}"
    if cached.exists():
        return "ok", cached.read_text(errors="replace")
    url = f"{GITHUB_API}/repos/{owner}/{repo}/contents/{path}"
    for backoff in (0, 5, 20):
        if backoff:
            time.sleep(backoff)
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
        except Exception as e:
            last = f"error:{str(e)[:40]}"
            continue
        if r.status_code == 200:
            j = r.json()
            if isinstance(j, dict) and j.get("encoding") == "base64":
                return "ok", base64.b64decode(j["content"]).decode("utf-8", "replace")
            return "ok", ""
        if r.status_code == 404:
            return "404", ""
        if r.status_code in (403, 429) and int(r.headers.get("X-RateLimit-Remaining", "1")) == 0:
            wait = max(5, int(r.headers.get("X-RateLimit-Reset", "0")) - int(time.time()))
            print(f"  rate limit; sleeping {wait}s", flush=True)
            time.sleep(wait)
            continue
        last = f"http_{r.status_code}"
    return last, ""


def verdict(texts: dict[str, str]) -> tuple[str, str]:
    pub = {n for n, t in texts.items() if PUBLISH_RE.search(t)}
    if not pub:
        if any(DYNAMIC_RE.search(t) for t in texts.values()):
            return "dynamic_config", "setup/continuation config; publish path not resolvable from the committed file"
        return "no_publish", "config present, no npm publish command (external scripts not followed)"
    joined = "\n".join(texts[n] for n in pub)
    tok, oidc = bool(TOKEN_RE.search(joined)), bool(OIDC_RE.search(joined))
    files = ",".join(sorted(pub))
    if oidc:
        return "publish_oidc", f"publish step with provenance/OIDC markers [{files}]"
    if tok:
        return "publish_token", f"publish step with token credential [{files}]"
    return "publish_unresolved", f"publish step, credential not visible [{files}]"


def main() -> int:
    if not TOKEN:
        print("GITHUB_TOKEN missing", flush=True)
        return 1
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    rows_c = [r for r in csv.DictReader(CLASS_CSV.open()) if r["class"] == "C"]
    status = {r["package"]: r for r in csv.DictReader(STATUS_CSV.open())}
    repos = sorted({(status[r["package"]]["owner"], status[r["package"]]["repo"]) for r in rows_c})
    print(f"Class C: {len(rows_c)} rows, {len({r['package'] for r in rows_c})} packages, {len(repos)} repos", flush=True)
    collected_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    repo_verdict: dict[tuple[str, str], dict] = {}
    for owner, repo in repos:
        texts: dict[str, str] = {}
        statuses = []
        for path in PROBE_FILES:
            st, text = get_file(owner, repo, path)
            statuses.append(f"{path}:{st}")
            if st == "ok":
                texts[path] = text
                (RAW_DIR / f"{owner}__{repo}__{path.replace('/', '_')}").write_text(text)
            time.sleep(0.2)
        if texts:
            v, why = verdict(texts)
        elif all(s.endswith(":404") for s in statuses):
            v, why = "config_missing", "no external-CI configuration at collection time"
        else:
            v, why = "error", ";".join(s for s in statuses if not s.endswith(":404"))[:120]
        repo_verdict[(owner, repo)] = {"verdict": v, "why": why, "files_found": "|".join(sorted(texts)), "probe": ";".join(statuses)}
        print(f"  {owner}/{repo}: {v} ({why[:60]})", flush=True)
    out = []
    for r in rows_c:
        s = status[r["package"]]
        rv = repo_verdict[(s["owner"], s["repo"])]
        out.append({"package": r["package"], "version": r["version"], "owner": s["owner"], "repo": s["repo"],
                    "label_reason": r["reason"], "audit_verdict": rv["verdict"], "audit_detail": rv["why"],
                    "files_found": rv["files_found"], "collected_at": collected_at})
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    rc, pc = Counter(r["audit_verdict"] for r in out), Counter(v["verdict"] for v in repo_verdict.values())
    L = ["# 审计臂 C — Class C 外部 CI 配置核验（TSE R1 round 2）\n",
         f"采集时间：{collected_at}（当前默认分支；晚于产生标签的 2026-05 缓存）。",
         f"对象：{len(out)} 个 C 行 / {len(repos)} 个仓库；探测文件：{', '.join(PROBE_FILES)}。\n",
         "| 裁决 | 行数 | 仓库数 |", "|---|---|---|"]
    for k in ("publish_token", "publish_oidc", "publish_unresolved", "no_publish", "dynamic_config", "config_missing", "error"):
        L.append(f"| {k} | {rc.get(k, 0)} | {pc.get(k, 0)} |")
    L += ["", "## 逐仓库\n", "| 仓库 | 裁决 | 说明 | 文件 |", "|---|---|---|---|"]
    for (o, rp), v in sorted(repo_verdict.items()):
        L.append(f"| `{o}/{rp}` | {v['verdict']} | {v['why']} | {v['files_found']} |")
    n_pub = rc.get("publish_token", 0) + rc.get("publish_oidc", 0) + rc.get("publish_unresolved", 0)
    L += ["", "## 读法\n",
          f"- C 的前提是\"外部 CI 可能承担发布\"。在 {len(out)} 行中，{n_pub} 行的外部 CI 配置含 npm 发布步骤（前提在审计时刻成立），",
          f"  {rc.get('no_publish', 0)} 行在提交的配置文件中未观察到字面 npm 发布命令（被调用的脚本未跟踪，不据此判定前提不成立），"
          f"{rc.get('dynamic_config', 0)} 行为动态 setup 配置（发布路径在生成的配置里，从提交文件不可判定），{rc.get('config_missing', 0)} 行配置已不存在。",
          "- 裁决描述审计时刻的仓库状态，不是发布时刻；与标签的 2026-05 采集时刻之间可能有变化。",
          "- 该臂把 R2-1b 的\"误分类风险\"从区间陈述变为按行核验，但不改变主表标签（标签定义为 2026-05 采集时刻状态）。"]
    OUT_MD.write_text("\n".join(L) + "\n")
    print(f"✓ {OUT_CSV.name}  ✓ {OUT_MD.name}\n✅ audit arm C complete", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
