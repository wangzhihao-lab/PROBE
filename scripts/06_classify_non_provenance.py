from __future__ import annotations

import csv
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import requests
import yaml

DATA_DIR = Path(__file__).parent.parent / "data"
STEP21_CSV = DATA_DIR / "classification_step21.csv"
STATUS_CSV = DATA_DIR / "repo_status.csv"
WORKFLOWS_DIR = DATA_DIR / "workflows_raw"
OUT_CSV = DATA_DIR / "classification.csv"
UNCERTAIN_CSV = DATA_DIR / "uncertain_cases.csv"
CIRCLECI_PROBE_CACHE = DATA_DIR / ".other_ci_probes.csv"

GITHUB_API = "https://api.github.com"
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")

HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}" if GITHUB_TOKEN else "",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
    "User-Agent": "stage2c-research/1.0",
}


STRONG_PUBLISH_PATTERNS = [
    r"\bnpm\s+publish\b(?![-_])",
    r"\bpnpm\s+publish\b(?![-_])",
    r"\byarn\s+publish\b(?![-_])",
    r"\blerna\s+publish\b(?![-_])",
    r"\brelease-tool\s+publish\b(?![-_])",
    r"JS-DevTools/npm-publish",
    r"wagoid/npm-publish-action",
    r"actions/publish-package",
    r"changesets/action",
    r"semantic-release",
]
STRONG_PUBLISH_RE = re.compile("|".join(STRONG_PUBLISH_PATTERNS), re.IGNORECASE)


WEAK_RELEASE_PATTERNS = [
    r"google-github-actions/release-please-action",
    r"googleapis/release-please-action",
    r"release-please",
]
WEAK_RELEASE_RE = re.compile("|".join(WEAK_RELEASE_PATTERNS), re.IGNORECASE)


AUXILIARY_PATTERNS = [
    r"softprops/action-gh-release",
    r"\bgh\s+release\s+create\b",
    r"actions/upload-pages-artifact",
    r"actions/deploy-pages",
    r"\bvercel/action",
    r"amondnet/vercel-action",
    r"cloudflare/pages-action",
    r"nwtgck/actions-netlify",
    r"netlify/actions",
    r"Azure/static-web-apps-deploy",
    r"chromaui/action",
]
AUXILIARY_RE = re.compile("|".join(AUXILIARY_PATTERNS), re.IGNORECASE)


LOCAL_REGISTRY_PATTERNS = [
    r"\bverdaccio\b",
    r"--registry\s+['\"]?http://(localhost|127\.0\.0\.1|verdaccio)",
    r"registry-url:\s*['\"]?http://(localhost|127\.0\.0\.1|verdaccio)",
]
LOCAL_REGISTRY_RE = re.compile("|".join(LOCAL_REGISTRY_PATTERNS), re.IGNORECASE)


AUX_FILENAME_TOKENS = ("sandbox", "website", "docs", "deploy", "page", "preview")


PUBLISH_FILENAME_HINTS = ("publish", "release")


PUBLISH_RE = re.compile(
    "|".join(STRONG_PUBLISH_PATTERNS + WEAK_RELEASE_PATTERNS), re.IGNORECASE,
)

NPM_TOKEN_NAMES = ["NPM_TOKEN", "NODE_AUTH_TOKEN", "NPM_AUTH_TOKEN", "NPMJS_TOKEN"]
NPM_TOKEN_RE = re.compile(
    r"(NPM_TOKEN|NODE_AUTH_TOKEN|NPM_AUTH_TOKEN|NPMJS_TOKEN)",
    re.IGNORECASE,
)
ID_TOKEN_RE = re.compile(r"id-token\s*:\s*write", re.IGNORECASE)


def detect_publish_in_yaml(text: str, filename: str = "") -> tuple[bool, str]:
    name_lc = filename.lower()
    has_local_registry = bool(LOCAL_REGISTRY_RE.search(text))
    has_strong = bool(STRONG_PUBLISH_RE.search(text))
    has_weak = bool(WEAK_RELEASE_RE.search(text))
    has_token = bool(NPM_TOKEN_RE.search(text)) or bool(ID_TOKEN_RE.search(text))
    has_aux = bool(AUXILIARY_RE.search(text))


    if has_local_registry:
        return False, ""

    if any(t in name_lc for t in AUX_FILENAME_TOKENS):
        return False, ""

    if has_aux and not has_strong:
        return False, ""


    if has_strong:
        return True, "strong-keyword"


    if has_weak and has_token:
        return True, "release-please+token"
    if has_weak and not has_token:
        return False, ""


    if any(h in name_lc for h in PUBLISH_FILENAME_HINTS):
        if has_aux:
            return False, ""
        if has_token:
            return True, "filename-hint+token"
    return False, ""


def find_publish_job(yaml_obj: dict | None) -> str | None:
    if not isinstance(yaml_obj, dict):
        return None
    jobs = yaml_obj.get("jobs", {})
    if not isinstance(jobs, dict):
        return None
    for job_name, job in jobs.items():
        if not isinstance(job, dict):
            continue

        steps = job.get("steps", [])
        if not isinstance(steps, list):
            continue
        for step in steps:
            if not isinstance(step, dict):
                continue
            run_cmd = str(step.get("run") or "")
            uses_action = str(step.get("uses") or "")
            if STRONG_PUBLISH_RE.search(run_cmd) or STRONG_PUBLISH_RE.search(uses_action):
                return str(job_name)

            if WEAK_RELEASE_RE.search(uses_action) and (
                NPM_TOKEN_RE.search(str(step)) or ID_TOKEN_RE.search(str(step))
            ):
                return str(job_name)
    return None


def id_token_at_publish_job(yaml_obj: dict | None, publish_job: str | None,
                             raw_text: str) -> tuple[bool, str]:
    if isinstance(yaml_obj, dict) and publish_job:
        jobs = yaml_obj.get("jobs", {})
        if isinstance(jobs, dict):
            job = jobs.get(publish_job)
            if isinstance(job, dict):
                p = job.get("permissions")
                if isinstance(p, dict):
                    if str(p.get("id-token", "")).lower() == "write":
                        return True, "job"

                    return False, "job-no-id-token"

    if isinstance(yaml_obj, dict):
        perms = yaml_obj.get("permissions")
        if isinstance(perms, dict) and str(perms.get("id-token", "")).lower() == "write":
            return True, "workflow"

    if publish_job is None and ID_TOKEN_RE.search(raw_text):
        return True, "raw-fallback"
    return False, "none"


def npm_token_at_publish_job(yaml_obj: dict | None, publish_job: str | None,
                              raw_text: str) -> bool:
    if isinstance(yaml_obj, dict) and publish_job:
        jobs = yaml_obj.get("jobs", {})
        if isinstance(jobs, dict):
            job = jobs.get(publish_job)
            if isinstance(job, dict):
                blob = str(job)
                if NPM_TOKEN_RE.search(blob):
                    return True


    if isinstance(yaml_obj, dict):
        env = yaml_obj.get("env")
        if isinstance(env, dict) and NPM_TOKEN_RE.search(str(env)):
            return True

    return bool(NPM_TOKEN_RE.search(raw_text))


def classify_publish_workflow(wf_path: Path) -> tuple[str, str]:
    try:
        text = wf_path.read_text(errors="replace")
    except Exception as e:
        return "UNCERTAIN", f"read error: {str(e)[:80]}"
    is_pub, hint = detect_publish_in_yaml(text, wf_path.name)
    if not is_pub:
        return "SKIP", "no publish pattern"
    hint_suffix = f" ({hint})" if hint else ""
    try:
        parsed = yaml.safe_load(text)
    except Exception:
        parsed = None


    pub_job = find_publish_job(parsed)
    id_tok, id_scope = id_token_at_publish_job(parsed, pub_job, text)
    npm_tok = npm_token_at_publish_job(parsed, pub_job, text)
    scope_suffix = f"; id-token scope={id_scope}"
    if pub_job is None and is_pub:
        scope_suffix += "; publish-job unresolved (YAML fallback)"

    if id_tok and npm_tok:
        return "B", f"id-token:write + NPM_TOKEN co-present (OIDC mis-config){hint_suffix}{scope_suffix}"
    if id_tok and not npm_tok:
        return "B", f"id-token:write w/o NPM_TOKEN; sampled version lacks attestation  OIDC failed{hint_suffix}{scope_suffix}"
    if not id_tok and npm_tok:
        return "A", f"NPM_TOKEN env + no id-token:write (classic publish){hint_suffix}{scope_suffix}"
    return "UNCERTAIN", f"publish detected but neither id-token nor NPM_TOKEN visible (composite/external action){hint_suffix}{scope_suffix}"


def list_other_ci(owner: str, repo: str, cache: dict[tuple[str, str], str]) -> str:
    key = (owner, repo)
    if key in cache:
        return cache[key]
    found = ""
    for path, label in [(".circleci/config.yml", "circleci"),
                        (".gitlab-ci.yml", "gitlab")]:
        url = f"{GITHUB_API}/repos/{owner}/{repo}/contents/{path}"
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            if r.status_code == 200:
                found = label
                break
        except Exception:
            continue
        time.sleep(0.3)
    cache[key] = found
    return found


def load_other_ci_cache() -> dict[tuple[str, str], str]:
    cache: dict[tuple[str, str], str] = {}
    if CIRCLECI_PROBE_CACHE.exists():
        with CIRCLECI_PROBE_CACHE.open() as f:
            for row in csv.reader(f):
                if len(row) >= 3:
                    cache[(row[0], row[1])] = row[2]
    return cache


def save_other_ci_cache(cache: dict[tuple[str, str], str]) -> None:
    with CIRCLECI_PROBE_CACHE.open("w", newline="") as f:
        w = csv.writer(f)
        for (owner, repo), label in sorted(cache.items()):
            w.writerow([owner, repo, label])


def main() -> int:

    step21: dict[tuple[str, str], dict] = {}
    with STEP21_CSV.open() as f:
        for r in csv.DictReader(f):
            step21[(r["package"], r["version"])] = r
    print(f"Loaded {len(step21)} step-2.1 rows", flush=True)


    repo_status: dict[tuple[str, str], dict] = {}
    if STATUS_CSV.exists():
        with STATUS_CSV.open() as f:
            for r in csv.DictReader(f):
                repo_status[(r["package"], r["version"])] = r
        print(f"Loaded {len(repo_status)} repo_status rows", flush=True)
    else:
        print(f" {STATUS_CSV} missing — only F class will be assignable", flush=True)


    wf_index: dict[tuple[str, str], list[Path]] = {}
    for p in WORKFLOWS_DIR.glob("*"):
        name = p.name

        parts = name.split("__", 2)
        if len(parts) == 3:
            wf_index.setdefault((parts[0], parts[1]), []).append(p)
    print(f"Indexed workflows for {len(wf_index)} repos", flush=True)

    other_ci_cache = load_other_ci_cache()


    final_rows: list[dict] = []
    uncertain_rows: list[dict] = []
    class_counts: Counter = Counter()
    other_ci_probes_made = 0

    for (pkg, ver), s21 in step21.items():
        prelim = s21["preliminary_class"]
        owner = s21["github_owner"]
        repo = s21["github_repo"]

        if prelim == "F":
            cls, reason = "F", "no/non-github repo url"
        else:
            stat_row = repo_status.get((pkg, ver))
            if not stat_row:
                cls, reason = "UNCERTAIN", "no repo_status row (run scripts/05 first)"
            else:
                status = stat_row["repo_status"]
                if status == "not_found":
                    cls, reason = "D", "repo 404 (deleted)"
                elif status == "private_or_blocked":
                    cls, reason = "D", "repo private or blocked"
                elif status == "error":
                    cls, reason = "UNCERTAIN", f"repo probe error: {stat_row.get('error_detail', '')[:80]}"
                elif status == "no_workflows":

                    label = list_other_ci(owner, repo, other_ci_cache)
                    if label:
                        other_ci_probes_made += 1
                        cls, reason = "C", f"no GHA; {label} config present"
                    else:
                        cls, reason = "E", "no GHA workflows; no .circleci/.gitlab-ci"
                elif status == "ok":

                    wfs = wf_index.get((owner, repo), [])
                    publish_results: list[tuple[str, str, str]] = []
                    for wf in wfs:
                        c, r = classify_publish_workflow(wf)
                        if c != "SKIP":
                            publish_results.append((wf.name, c, r))
                    if not publish_results:

                        label = list_other_ci(owner, repo, other_ci_cache)
                        if label:
                            other_ci_probes_made += 1
                            cls, reason = "C", f"GHA has no publish; {label} config present"
                        else:
                            cls, reason = "E", "GHA has no publish workflow; no other CI"
                    elif len(publish_results) == 1:
                        wf_name, c, r = publish_results[0]
                        cls, reason = c, f"{r} [{wf_name}]"
                    else:

                        classes = {c for _, c, _ in publish_results}
                        if len(classes) == 1:
                            c = classes.pop()
                            wf_names = ",".join(wn for wn, _, _ in publish_results)
                            cls, reason = c, f"{publish_results[0][2]} [{wf_names}]"
                        else:
                            cls = "UNCERTAIN"
                            reason = "multiple publish workflows with conflicting classes: " + \
                                "; ".join(f"{wn}={c}" for wn, c, _ in publish_results)
                else:
                    cls, reason = "UNCERTAIN", f"unknown repo_status: {status}"

        class_counts[cls] += 1
        final_rows.append({
            "package": pkg,
            "version": ver,
            "class": cls,
            "reason": reason,
            "rater": "rater_A",
        })
        if cls == "UNCERTAIN":
            uncertain_rows.append({
                "package": pkg,
                "version": ver,
                "owner": owner,
                "repo": repo,
                "current_class": cls,
                "reason": reason,
            })

    save_other_ci_cache(other_ci_cache)
    print(f"Other CI probes made this run: {other_ci_probes_made}", flush=True)


    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["package", "version", "class", "reason", "rater"])
        w.writeheader()
        for r in final_rows:
            w.writerow(r)
    print(f"\n Saved {OUT_CSV} ({len(final_rows)} rows)", flush=True)

    with UNCERTAIN_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["package", "version", "owner", "repo", "current_class", "reason"])
        w.writeheader()
        for r in uncertain_rows:
            w.writerow(r)
    print(f" Saved {UNCERTAIN_CSV} ({len(uncertain_rows)} rows)", flush=True)


    print(f"\n=== Class distribution ===", flush=True)
    for c in ("A", "B", "C", "D", "E", "F", "UNCERTAIN"):
        cnt = class_counts.get(c, 0)
        pct = (cnt / len(final_rows) * 100) if final_rows else 0
        print(f"  {c}: {cnt:>4} ({pct:.1f}%)", flush=True)


    print(f"\n=== Class × tier ===", flush=True)
    tier_class: dict[str, Counter] = {"top": Counter(), "mid": Counter(), "tail": Counter()}
    with (DATA_DIR / "non_provenance_sample.csv").open() as f:
        for r in csv.DictReader(f):
            cr = next((fr for fr in final_rows if fr["package"] == r["package"] and fr["version"] == r["version"]), None)
            if cr:
                tier_class[r["tier"]][cr["class"]] += 1
    header_row = "  tier   " + "  ".join(f"{c:>4}" for c in "ABCDEF") + "  UNC"
    print(header_row, flush=True)
    for tier in ("top", "mid", "tail"):
        cnts = tier_class[tier]
        cells = [f"{cnts.get(c, 0):>4}" for c in "ABCDEF"] + [f"{cnts.get('UNCERTAIN', 0):>4}"]
        print(f"  {tier:6s} " + "  ".join(cells), flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
