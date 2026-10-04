"""
Task 3 Round 1 PI revision supplementary fetches:

(1) .github/dependabot.yaml fallback (PI minor m-4):
    For rows with has_dependabot=false, also probe .github/dependabot.yaml.
    Update github_repo_meta.csv has_dependabot=true if found.

(2) github_owner_type (PI Major #3):
    Distinct owners from github_repo_meta.csv → /users/{owner} → returns
    `type` ∈ {User, Organization}. Output: data/github_owner_type.csv
    (owner, type). Used by 14_merge to add `github_owner_type` column.

Cache: re-uses GitHub API; resume-safe via row-by-row append (dependabot)
and small per-owner cache (owner_type).

Run: source ~/.zshenv && conda run --no-capture-output -n stage2c python scripts/15_round1_followups.py
"""
from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path

import requests

DATA_DIR = Path(__file__).parent.parent / "data"
GITHUB_CSV = DATA_DIR / "github_repo_meta.csv"
OWNER_TYPE_CSV = DATA_DIR / "github_owner_type.csv"

GITHUB_API = "https://api.github.com"
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
if not GITHUB_TOKEN:
    print("ERROR: GITHUB_TOKEN not set", file=sys.stderr)
    sys.exit(2)

HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
    "User-Agent": "stage2c-research/1.0 (Task 3 Round 1 followups)",
}

SLEEP = 0.3
BACKOFFS = [10, 30, 60]


def github_request(url: str) -> tuple[int, dict | None]:
    for backoff in [0, *BACKOFFS]:
        if backoff:
            time.sleep(backoff)
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            if r.status_code == 200:
                return 200, (r.json() if r.headers.get("Content-Type", "").startswith("application/json") else None)
            if r.status_code == 404:
                return 404, None
            if r.status_code == 403:
                rem = r.headers.get("X-RateLimit-Remaining", "?")
                if rem == "0":
                    reset = int(r.headers.get("X-RateLimit-Reset", "0"))
                    wait = max(reset - int(time.time()) + 5, 5)
                    print(f"  rate limit; sleeping {wait}s", flush=True)
                    time.sleep(wait)
                    continue
                continue
            if r.status_code >= 500:
                continue
            return r.status_code, None
        except Exception:
            continue
    return 0, None


def step_dependabot_yaml() -> int:
    """Probe .github/dependabot.yaml for rows currently has_dependabot=false."""
    print("\n=== Step 1: dependabot.yaml fallback ===", flush=True)
    rows: list[dict] = []
    with GITHUB_CSV.open() as f:
        rows = list(csv.DictReader(f))

    targets = [r for r in rows if r["status"] == "ok" and r["has_dependabot"] == "false"]
    print(f"  {len(targets)} candidates (has_dependabot=false; status=ok)", flush=True)

    updated = 0
    for i, r in enumerate(targets, start=1):
        url = f"{GITHUB_API}/repos/{r['owner']}/{r['repo']}/contents/.github/dependabot.yaml"
        sc, _ = github_request(url)
        if sc == 200:
            r["has_dependabot"] = "true"
            updated += 1
        if i % 200 == 0 or i == len(targets):
            print(f"  [{i:>4}/{len(targets)}] updated={updated}", flush=True)
        time.sleep(SLEEP)

    # Write back full file (preserve order)
    with GITHUB_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  ✓ {updated} rows flipped has_dependabot false→true via .yaml fallback", flush=True)
    return updated


def step_owner_type() -> int:
    """Fetch /users/{owner} type for distinct owners."""
    print("\n=== Step 2: github_owner_type via /users/{owner} ===", flush=True)
    owners: set[str] = set()
    with GITHUB_CSV.open() as f:
        for r in csv.DictReader(f):
            if r["owner"]:
                owners.add(r["owner"])
    print(f"  Distinct owners: {len(owners)}", flush=True)

    # Load cache
    cache: dict[str, str] = {}
    if OWNER_TYPE_CSV.exists():
        with OWNER_TYPE_CSV.open() as f:
            for r in csv.DictReader(f):
                cache[r["owner"]] = r["type"]
        print(f"  Loaded {len(cache)} pre-existing entries (resume)", flush=True)

    todo = sorted(owners - set(cache.keys()))
    print(f"  Remaining: {len(todo)}", flush=True)

    fh = OWNER_TYPE_CSV.open("a", newline="")
    writer = csv.DictWriter(fh, fieldnames=["owner", "type"])
    if not cache:
        writer.writeheader()

    counts = {"User": 0, "Organization": 0, "unknown": 0}
    for i, owner in enumerate(todo, start=1):
        url = f"{GITHUB_API}/users/{owner}"
        sc, payload = github_request(url)
        otype = ""
        if sc == 200 and isinstance(payload, dict):
            otype = payload.get("type", "unknown")
        else:
            otype = "unknown"
        cache[owner] = otype
        writer.writerow({"owner": owner, "type": otype})
        fh.flush()
        counts[otype] = counts.get(otype, 0) + 1
        if i % 100 == 0 or i == len(todo):
            print(f"  [{i:>4}/{len(todo)}] User={counts['User']} Org={counts['Organization']} unknown={counts['unknown']}", flush=True)
        time.sleep(SLEEP)
    fh.close()

    # Final distribution over all owners
    final_counts = {"User": 0, "Organization": 0, "unknown": 0}
    for o, t in cache.items():
        final_counts[t] = final_counts.get(t, 0) + 1
    print(f"\n  Final: User={final_counts['User']} Organization={final_counts['Organization']} unknown={final_counts['unknown']}", flush=True)
    return len(cache)


def main() -> int:
    upd = step_dependabot_yaml()
    n = step_owner_type()
    print(f"\n=== Summary ===", flush=True)
    print(f"  has_dependabot flipped: {upd}", flush=True)
    print(f"  owner_type entries: {n}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
