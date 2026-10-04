"""
Task 2 Step 2.2 — Fetch GitHub `.github/workflows/*.yml` for non-F packages.

Input:  data/classification_step21.csv (preliminary_class == 'pending' rows)
Output:
  data/workflows_raw/<owner>__<repo>__<wfname>  (raw workflow content; ext preserved)
  data/repo_status.csv                          (per (package,version): repo status, workflow count)

Algorithm:
1. Read pending rows; dedup unique (owner, repo) so we make one network probe per repo
2. For each unique repo:
   - GET /repos/{owner}/{repo}/contents/.github/workflows  (1 API call)
   - 404 → workflow dir absent (repo lives but no workflows) → status='no_workflows' → E later
   - 403 (rate limit / blocked) → backoff + retry
   - 404 on /repos/{owner}/{repo} (resolved via separate probe) → status='not_found' → D
   - private → 404 on the contents call too → status='private_or_missing' → D
   - 200 → list each entry; for each .yml/.yaml file: download via download_url (raw.gh)
3. Cache workflows; idempotent: skip repo if status row exists from a prior run
4. Backoff on secondary rate limit (X-RateLimit-Remaining low / 429 / 403 with abuse marker)

Token: $GITHUB_TOKEN from ~/.zshenv (must be sourced before running)

Run: source ~/.zshenv && conda run --no-capture-output -n stage2c python scripts/05_fetch_workflows.py
"""
from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path

import requests

DATA_DIR = Path(__file__).parent.parent / "data"
STEP21_CSV = DATA_DIR / "classification_step21.csv"
WORKFLOWS_DIR = DATA_DIR / "workflows_raw"
STATUS_CSV = DATA_DIR / "repo_status.csv"

WORKFLOWS_DIR.mkdir(parents=True, exist_ok=True)

GITHUB_API = "https://api.github.com"
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
if not GITHUB_TOKEN:
    print("ERROR: GITHUB_TOKEN not set; run `source ~/.zshenv` first", file=sys.stderr)
    sys.exit(2)

HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
    "User-Agent": "stage2c-research/1.0 (Task 2 classification)",
}

SLEEP = 0.4  # between calls (~150 req/min, well under 5000/hr)
BACKOFFS = [5, 15, 45]  # for 5xx / secondary rate limit


def check_rate_limit() -> tuple[int, int]:
    """Return (remaining, reset_epoch)."""
    r = requests.get(f"{GITHUB_API}/rate_limit", headers=HEADERS, timeout=30)
    r.raise_for_status()
    c = r.json()["resources"]["core"]
    return c["remaining"], c["reset"]


def safe_workflow_name(name: str) -> str:
    """Sanitize for filename (strip path separators)."""
    return name.replace("/", "_").replace("\\", "_")


def list_workflows(owner: str, repo: str) -> tuple[str, list[dict], str]:
    """
    Returns (status, workflow_entries, error_detail).
    status ∈ {'ok', 'no_workflows', 'not_found', 'private_or_blocked', 'error'}
    """
    url = f"{GITHUB_API}/repos/{owner}/{repo}/contents/.github/workflows"
    last_err = ""
    for attempt, backoff in enumerate([0, *BACKOFFS]):
        if backoff > 0:
            time.sleep(backoff)
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, list):
                    return "ok", data, ""
                # Sometimes returns dict for a single file (path is actually a file)
                return "no_workflows", [], "contents returned dict not list"
            if r.status_code == 404:
                # Could be: repo gone, repo private, or .github/workflows absent
                # Probe parent to disambiguate
                probe = requests.get(f"{GITHUB_API}/repos/{owner}/{repo}",
                                     headers=HEADERS, timeout=30)
                time.sleep(SLEEP)
                if probe.status_code == 404:
                    return "not_found", [], "repo 404"
                if probe.status_code == 200:
                    return "no_workflows", [], "repo exists but no .github/workflows"
                return "private_or_blocked", [], f"probe status {probe.status_code}"
            if r.status_code in (403, 429):
                # Secondary rate limit / abuse — backoff
                last_err = f"{r.status_code}: {r.text[:200]}"
                # Check remaining
                rem = r.headers.get("X-RateLimit-Remaining", "?")
                if rem == "0":
                    reset = int(r.headers.get("X-RateLimit-Reset", "0"))
                    wait = max(reset - int(time.time()) + 5, 5)
                    print(f"  primary rate limit hit; sleeping {wait}s", flush=True)
                    time.sleep(wait)
                    continue
                # Otherwise secondary; backoff loop
                continue
            last_err = f"{r.status_code}: {r.text[:200]}"
        except Exception as e:
            last_err = f"{type(e).__name__}: {str(e)[:200]}"
    return "error", [], last_err


def fetch_workflow_content(download_url: str) -> str:
    """Direct raw fetch — doesn't count against API rate limit."""
    r = requests.get(download_url, timeout=30)
    r.raise_for_status()
    return r.text


def main() -> int:
    # Load existing status (idempotency)
    done_repos: dict[tuple[str, str], dict] = {}
    if STATUS_CSV.exists():
        with STATUS_CSV.open() as f:
            for row in csv.DictReader(f):
                done_repos[(row["owner"], row["repo"])] = row
        print(f"Loaded {len(done_repos)} pre-existing repo status entries (resume mode)", flush=True)

    # Load pending rows
    pending_rows: list[dict] = []
    with STEP21_CSV.open() as f:
        for row in csv.DictReader(f):
            if row["preliminary_class"] == "pending":
                pending_rows.append(row)
    print(f"Loaded {len(pending_rows)} pending (GitHub) rows", flush=True)

    # Dedup by (owner, repo) — many @types/* etc share repos
    unique_repos: dict[tuple[str, str], list[dict]] = {}
    for r in pending_rows:
        key = (r["github_owner"], r["github_repo"])
        unique_repos.setdefault(key, []).append(r)
    print(f"Unique (owner, repo) pairs: {len(unique_repos)}", flush=True)

    # Pre-flight rate limit check
    remaining, reset = check_rate_limit()
    need = len(unique_repos) * 2  # worst case: 1 list + 1 404 probe
    print(f"Rate limit: {remaining}/5000 remaining; budget needed ~{need}", flush=True)
    if remaining < need:
        wait = max(reset - int(time.time()), 0) + 5
        print(f"⚠ insufficient; would need {wait}s wait; continuing anyway (may stall mid-run)", flush=True)

    # Fetch loop
    results: list[dict] = []
    counts = {"ok": 0, "no_workflows": 0, "not_found": 0, "private_or_blocked": 0,
              "error": 0, "skipped_cached": 0}
    total_workflows = 0

    for i, ((owner, repo), pkg_rows) in enumerate(sorted(unique_repos.items()), start=1):
        if (owner, repo) in done_repos:
            counts["skipped_cached"] += 1
            cached = done_repos[(owner, repo)]
            # Re-emit one row per (pkg,ver) for joins
            for pkg_row in pkg_rows:
                results.append({
                    "package": pkg_row["package"],
                    "version": pkg_row["version"],
                    "owner": owner,
                    "repo": repo,
                    "repo_status": cached["repo_status"],
                    "workflow_count": cached["workflow_count"],
                    "error_detail": cached.get("error_detail", ""),
                })
            continue

        status, entries, err = list_workflows(owner, repo)
        counts[status] = counts.get(status, 0) + 1

        # If ok, download each workflow file
        wf_files: list[str] = []
        if status == "ok":
            for entry in entries:
                if entry.get("type") != "file":
                    continue
                name = entry.get("name", "")
                if not (name.endswith(".yml") or name.endswith(".yaml")):
                    continue
                download_url = entry.get("download_url")
                if not download_url:
                    continue
                out_name = f"{owner}__{repo}__{safe_workflow_name(name)}"
                out_path = WORKFLOWS_DIR / out_name
                if not out_path.exists():
                    try:
                        content = fetch_workflow_content(download_url)
                        out_path.write_text(content)
                        time.sleep(0.1)  # gentle on raw.gh
                    except Exception as e:
                        # Non-fatal; skip this workflow
                        print(f"    ⚠ {owner}/{repo}/{name} dl fail: {str(e)[:80]}", flush=True)
                        continue
                wf_files.append(out_name)
            total_workflows += len(wf_files)

        # Emit result rows for every (pkg, ver) under this repo
        for pkg_row in pkg_rows:
            results.append({
                "package": pkg_row["package"],
                "version": pkg_row["version"],
                "owner": owner,
                "repo": repo,
                "repo_status": status,
                "workflow_count": str(len(wf_files)) if status == "ok" else "0",
                "error_detail": err[:200] if err else "",
            })

        if i % 25 == 0 or i == len(unique_repos):
            print(f"  [{i:>3}/{len(unique_repos)}] {owner}/{repo} → {status} (wfs: {len(wf_files)})", flush=True)

        time.sleep(SLEEP)

    # Save status CSV (one row per (package, version))
    fields = ["package", "version", "owner", "repo", "repo_status", "workflow_count", "error_detail"]
    with STATUS_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow(r)
    print(f"\n✓ Saved {STATUS_CSV} ({len(results)} rows)", flush=True)

    # Summary
    print(f"\n=== Repo status summary (unique repos) ===", flush=True)
    for k in ("ok", "no_workflows", "not_found", "private_or_blocked", "error", "skipped_cached"):
        v = counts.get(k, 0)
        if v:
            print(f"  {k}: {v}", flush=True)
    print(f"  workflows downloaded: {total_workflows}", flush=True)
    print(f"  cache size: {len(list(WORKFLOWS_DIR.glob('*')))} files", flush=True)

    # Final rate limit
    try:
        rem, _ = check_rate_limit()
        print(f"  rate limit after run: {rem}/5000", flush=True)
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
