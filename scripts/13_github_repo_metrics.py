from __future__ import annotations

import csv
import os
import re
import sys
import time
from pathlib import Path

import requests

DATA_DIR = Path(__file__).parent.parent / "data"
PACKAGES_FILE = DATA_DIR / "packages_full.txt"
RAW_DIR = DATA_DIR / "packuments_raw"
OUT_CSV = DATA_DIR / "github_repo_meta.csv"

GITHUB_API = "https://api.github.com"
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
if not GITHUB_TOKEN:
    print("ERROR: GITHUB_TOKEN not set; run `source ~/.zshenv` first", file=sys.stderr)
    sys.exit(2)

HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
    "User-Agent": "stage2c-research/1.0",
}

SLEEP = 0.3
BACKOFFS = [10, 30, 60]


GITHUB_RE = re.compile(
    r"github\.com[/:]([A-Za-z0-9_.\-]+)/([A-Za-z0-9_.\-]+?)(?:\.git)?(?:[/#?].*)?$",
    re.IGNORECASE,
)


def safe_fname(pkg: str) -> str:
    return pkg.replace("/", "_").replace("@", "at_") + ".json"


def parse_github(url: str) -> tuple[str, str] | None:
    if not url:
        return None
    s = url.strip()
    if s.startswith("git+"):
        s = s[4:]
    m = GITHUB_RE.search(s)
    if m:
        owner = m.group(1)
        repo = m.group(2).removesuffix(".git")
        return owner, repo
    return None


def packument_repo(pkg: str) -> str:
    import json as _json
    fpath = RAW_DIR / safe_fname(pkg)
    if not fpath.exists():
        return ""
    try:
        data = _json.loads(fpath.read_text())
    except Exception:
        return ""
    if data.get("_404"):
        return ""
    repo_obj = data.get("repository")
    if isinstance(repo_obj, dict):
        return str(repo_obj.get("url") or "").strip()
    if isinstance(repo_obj, str):
        return repo_obj.strip()
    return ""


def github_request(url: str) -> tuple[int, dict | None, str]:
    last_err = ""
    for attempt, backoff in enumerate([0, *BACKOFFS]):
        if backoff > 0:
            time.sleep(backoff)
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            if r.status_code == 200:
                try:
                    return 200, r.json(), ""
                except Exception:
                    return 200, None, ""
            if r.status_code == 404:
                return 404, None, ""
            if r.status_code == 403:

                rem = r.headers.get("X-RateLimit-Remaining", "?")
                if rem == "0":
                    reset = int(r.headers.get("X-RateLimit-Reset", "0"))
                    wait = max(reset - int(time.time()) + 5, 5)
                    print(f"  primary rate limit; sleeping {wait}s", flush=True)
                    time.sleep(wait)
                    continue
                last_err = "secondary rate limit"
                continue
            if r.status_code == 429:
                last_err = "429"
                continue
            if r.status_code >= 500:
                last_err = f"{r.status_code} server error"
                continue
            last_err = f"{r.status_code}: {r.text[:200]}"
            return r.status_code, None, last_err
        except Exception as e:
            last_err = f"{type(e).__name__}: {str(e)[:200]}"
            continue
    return 0, None, last_err


def main() -> int:

    done: set[str] = set()
    existing_rows: list[dict] = []
    if OUT_CSV.exists():
        with OUT_CSV.open() as f:
            for row in csv.DictReader(f):
                done.add(row["package"])
                existing_rows.append(row)
        print(f"Loaded {len(done)} pre-existing rows (resume)", flush=True)


    pkg_repo: dict[str, str] = {}
    with PACKAGES_FILE.open() as f:
        header = next(f).rstrip("\n").split("\t")
        idx_name = header.index("name")
        idx_repo = header.index("repository_url")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) > idx_repo:
                pkg_repo[parts[idx_name]] = parts[idx_repo]
    print(f"Loaded {len(pkg_repo)} packagerepo_url mappings", flush=True)


    sc, payload, _ = github_request(f"{GITHUB_API}/rate_limit")
    if sc == 200 and isinstance(payload, dict):
        c = payload["resources"]["core"]
        print(f"Rate limit pre-flight: {c['remaining']}/{c['limit']}", flush=True)

    new_rows: list[dict] = []
    counts = {"ok": 0, "no_github": 0, "repo_404": 0, "repo_error": 0, "skipped": 0}

    fields = ["package", "owner", "repo", "stargazers_count", "forks_count",
              "has_dependabot", "has_issue_template", "status", "error_detail"]

    write_header = not OUT_CSV.exists()
    fh = OUT_CSV.open("a", newline="")
    writer = csv.DictWriter(fh, fieldnames=fields)
    if write_header:
        writer.writeheader()

    try:
        for i, (pkg, url) in enumerate(sorted(pkg_repo.items()), start=1):
            if pkg in done:
                counts["skipped"] += 1
                continue


            parsed = parse_github(url)
            if parsed is None:
                pkmt = packument_repo(pkg)
                parsed = parse_github(pkmt)
            if parsed is None:
                row = {
                    "package": pkg, "owner": "", "repo": "",
                    "stargazers_count": "", "forks_count": "",
                    "has_dependabot": "", "has_issue_template": "",
                    "status": "no_github", "error_detail": "",
                }
                writer.writerow(row)
                fh.flush()
                counts["no_github"] += 1
                new_rows.append(row)
                continue

            owner, repo = parsed

            sc, payload, err = github_request(f"{GITHUB_API}/repos/{owner}/{repo}")
            if sc == 404:
                row = {
                    "package": pkg, "owner": owner, "repo": repo,
                    "stargazers_count": "", "forks_count": "",
                    "has_dependabot": "", "has_issue_template": "",
                    "status": "repo_404", "error_detail": "",
                }
                writer.writerow(row)
                fh.flush()
                counts["repo_404"] += 1
                new_rows.append(row)
                time.sleep(SLEEP)
                continue
            if sc != 200 or not isinstance(payload, dict):
                row = {
                    "package": pkg, "owner": owner, "repo": repo,
                    "stargazers_count": "", "forks_count": "",
                    "has_dependabot": "", "has_issue_template": "",
                    "status": "repo_error", "error_detail": err[:150],
                }
                writer.writerow(row)
                fh.flush()
                counts["repo_error"] += 1
                new_rows.append(row)
                time.sleep(SLEEP)
                continue

            stars = payload.get("stargazers_count", 0)
            forks = payload.get("forks_count", 0)
            time.sleep(SLEEP)


            sc, _, _ = github_request(f"{GITHUB_API}/repos/{owner}/{repo}/contents/.github/dependabot.yml")
            has_dependabot = "true" if sc == 200 else "false"
            time.sleep(SLEEP)


            sc, _, _ = github_request(f"{GITHUB_API}/repos/{owner}/{repo}/contents/.github/ISSUE_TEMPLATE")
            if sc == 200:
                has_issue_template = "true"
            else:

                sc2, _, _ = github_request(f"{GITHUB_API}/repos/{owner}/{repo}/contents/.github/ISSUE_TEMPLATE.md")
                has_issue_template = "true" if sc2 == 200 else "false"
            time.sleep(SLEEP)

            row = {
                "package": pkg, "owner": owner, "repo": repo,
                "stargazers_count": str(stars), "forks_count": str(forks),
                "has_dependabot": has_dependabot,
                "has_issue_template": has_issue_template,
                "status": "ok", "error_detail": "",
            }
            writer.writerow(row)
            fh.flush()
            counts["ok"] += 1
            new_rows.append(row)

            if i % 50 == 0 or i == len(pkg_repo):
                print(f"  [{i:>4}/{len(pkg_repo)}] {pkg[:40]:<40s}  "
                      f"ok={counts['ok']} 404={counts['repo_404']} "
                      f"no_gh={counts['no_github']} err={counts['repo_error']}", flush=True)
    finally:
        fh.close()

    print(f"\n=== Status counts ===", flush=True)
    for k in ("ok", "no_github", "repo_404", "repo_error", "skipped"):
        if counts.get(k, 0):
            print(f"  {k}: {counts[k]}", flush=True)


    all_rows = existing_rows + new_rows
    ok_rows = [r for r in all_rows if r["status"] == "ok"]
    if ok_rows:
        stars_vals = sorted(int(r["stargazers_count"]) for r in ok_rows if r["stargazers_count"])
        forks_vals = sorted(int(r["forks_count"]) for r in ok_rows if r["forks_count"])
        n = len(stars_vals)
        if n:
            print(f"\n=== Distributions over {n} ok rows ===", flush=True)
            print(f"  stars:  min={stars_vals[0]:>6}  med={stars_vals[n//2]:>6}  max={stars_vals[-1]:>7}", flush=True)
            print(f"  forks:  min={forks_vals[0]:>6}  med={forks_vals[n//2]:>6}  max={forks_vals[-1]:>7}", flush=True)
        dep = sum(1 for r in ok_rows if r["has_dependabot"] == "true")
        tpl = sum(1 for r in ok_rows if r["has_issue_template"] == "true")
        print(f"  has_dependabot=true:      {dep}/{n} ({dep/n*100:.1f}%)", flush=True)
        print(f"  has_issue_template=true:  {tpl}/{n} ({tpl/n*100:.1f}%)", flush=True)


    sc, payload, _ = github_request(f"{GITHUB_API}/rate_limit")
    if sc == 200 and isinstance(payload, dict):
        c = payload["resources"]["core"]
        print(f"\n  rate limit final: {c['remaining']}/{c['limit']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
