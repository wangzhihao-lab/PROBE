"""
Task 1 Step 1.1 — Stratified package sample WITHOUT BigQuery.

Per anomaly #4 (2026-05-14 BQ Sandbox quota exhausted): BQ deferred to post-2026-06-01
enrichment. Sampling pivots to ecosyste.ms which exposes `dependent_packages_count` per
package even though the API only allows `sort=downloads` (not sort by dpc).

Method:
1. Fetch pages 1-50 (sort=downloads desc, ranks 1-5000) — dense coverage of popular packages
2. Fetch deep pages 100/150/200/300/500/1000 (sparse) — to probe long tail
3. Retry each failed page up to 3 times with exponential backoff (ecosyste.ms returns 500s transiently)
4. Bucket every package by `dependent_packages_count`:
   - top:  dpc > 500
   - mid:  10 < dpc <= 500    (note: dpc==10 is NOT mid; fixed 2026-05-15 per methods review round 1, Major-1)
   - tail: dpc <= 10
5. Cap each tier to spec target (top 500, mid 1500, tail 1000), random subsample if over

Output: data/packages_full.txt — tab-separated:
  name, dpc_tier, rank_by_downloads, downloads, dependent_packages_count,
  repository_url, latest_release_at_collection_audit, depsdev_dependents_count_pending
  (`latest_release_at_collection_audit` renamed 2026-05-15 per methods review round 1, Critical-1;
   audit-only field, values > 2026-05-10 redacted to `post_cutoff_redacted`)

`depsdev_dependents_count_pending` is literally "pending" — to be filled by BQ
enrichment after free tier resets (2026-06-01) per anomaly #4 policy.

Run: conda run --no-capture-output -n stage2c python scripts/01_sample_packages.py
"""
from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import requests

OUTPUT_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ECOSYSTE_BASE = "https://packages.ecosyste.ms/api/v1/registries/npmjs.org/packages"
HEADERS = {"User-Agent": "stage2c-research/1.0 (Task 1 stratified sample)", "Accept": "application/json"}

PER_PAGE = 100
SLEEP_BETWEEN_REQUESTS = 0.5
RETRY_BACKOFFS = [2, 5, 10]  # seconds; total 3 retries

# Dense pages for top + mid
DENSE_PAGES = list(range(1, 51))  # 1-50 → ranks 1-5000

# Sparse deep pages to probe long tail
DEEP_PAGES = [100, 150, 200, 300, 500, 1000]

SPEC_TARGETS = {"top": 500, "mid": 1500, "tail": 1000}

RNG = random.Random(42)  # deterministic


def fetch_page_with_retry(page: int, per_page: int = PER_PAGE) -> tuple[list[dict] | None, str]:
    url = f"{ECOSYSTE_BASE}?sort=downloads&order=desc&page={page}&per_page={per_page}"
    last_err = ""
    for attempt, backoff in enumerate([0, *RETRY_BACKOFFS]):
        if backoff > 0:
            time.sleep(backoff)
        try:
            r = requests.get(url, headers=HEADERS, timeout=60)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, list):
                return data, ""
            last_err = f"unexpected payload type: {type(data).__name__}"
        except Exception as e:
            last_err = f"{type(e).__name__}: {str(e)[:200]}"
    return None, last_err


def dpc_tier(dpc: int) -> str:
    # Spec (kickoff lines 194-197):
    #   top:  dependents > 500
    #   mid:  10 < dependents <= 500
    #   tail: dependents <= 10
    # Fixed 2026-05-15 after methods review round 1 (was previously dpc>=10 → mid, putting dpc==10 in mid; spec says dpc==10 → tail)
    if dpc > 500:
        return "top"
    if dpc > 10:
        return "mid"
    return "tail"


def main() -> int:
    raw: dict[str, dict] = {}  # name → package row dict (dedup by name)
    page_rank_by_name: dict[str, int] = {}  # name → its rank position in download-sorted list
    fail_pages: list[tuple[int, str]] = []

    all_pages = DENSE_PAGES + DEEP_PAGES

    print(f"=== Fetching {len(all_pages)} pages ({len(DENSE_PAGES)} dense + {len(DEEP_PAGES)} deep) ===", flush=True)
    for page in all_pages:
        items, err = fetch_page_with_retry(page)
        if items is None:
            fail_pages.append((page, err))
            print(f"  page {page:>4} ❌ {err[:80]}", flush=True)
            continue
        for idx, p in enumerate(items):
            name = p.get("name")
            if not name or name in raw:
                continue
            raw[name] = p
            # Rank position = (page-1) * per_page + idx + 1
            page_rank_by_name[name] = (page - 1) * PER_PAGE + idx + 1
        print(f"  page {page:>4} ✓ +{len(items)} (raw cumulative {len(raw)})", flush=True)
        time.sleep(SLEEP_BETWEEN_REQUESTS)

    # Bucket by dpc threshold
    buckets: dict[str, list] = {"top": [], "mid": [], "tail": []}
    no_dpc = 0
    for name, p in raw.items():
        dpc_raw = p.get("dependent_packages_count")
        if dpc_raw is None:
            no_dpc += 1
            continue
        dpc = int(dpc_raw)
        tier = dpc_tier(dpc)
        buckets[tier].append((name, p, dpc))

    print(f"\n=== Pre-cap distribution by dependent_packages_count ===", flush=True)
    for t in ("top", "mid", "tail"):
        print(f"  {t}: {len(buckets[t])} (target {SPEC_TARGETS[t]})", flush=True)
    print(f"  packages with no dpc field: {no_dpc}", flush=True)

    # Cap each tier to target
    final: list[tuple] = []
    for tier in ("top", "mid", "tail"):
        items = buckets[tier]
        target = SPEC_TARGETS[tier]
        if len(items) > target:
            # Random subsample
            items = RNG.sample(items, target)
            note = "(random subsampled)"
        else:
            note = "(taken all)"
        print(f"  cap {tier}: {len(items)} {note}", flush=True)
        for name, p, dpc in items:
            rank = page_rank_by_name.get(name, "")
            # Audit-only field: ecosyste.ms's reported latest_release_published_at at collection time.
            # Per methods review round 1, Critical-1: blank out any value > 2026-05-10 (post-cutoff)
            # to keep packages_full.txt as analysis-safe (no post-cutoff date in analysis input).
            DATA_CUTOFF_ISO = "2026-05-10T23:59:59"
            raw_latest = str(p.get("latest_release_published_at") or "")
            latest_audit = raw_latest if (not raw_latest or raw_latest <= DATA_CUTOFF_ISO) else "post_cutoff_redacted"
            row = (
                str(name),
                tier,
                str(rank),
                str(p.get("downloads", "")),
                str(dpc),
                str(p.get("repository_url") or "").replace("\t", " "),
                latest_audit,
                "pending",
            )
            final.append(row)

    # Save
    out_path = OUTPUT_DIR / "packages_full.txt"
    # NOTE: latest_release_at_collection_audit is NOT analysis input — it is metadata-at-collection-time
    # from ecosyste.ms (2026-05-15). Values > 2026-05-10 are redacted to "post_cutoff_redacted" per methods review round 1, Critical-1.
    header = "name\tdpc_tier\trank_by_downloads\tdownloads\tdependent_packages_count\trepository_url\tlatest_release_at_collection_audit\tdepsdev_dependents_count_pending"
    with out_path.open("w") as f:
        f.write(header + "\n")
        for r in final:
            f.write("\t".join(r) + "\n")

    # Summary
    final_by_tier: dict[str, int] = {}
    for r in final:
        final_by_tier[r[1]] = final_by_tier.get(r[1], 0) + 1

    print(f"\n=== Final ===", flush=True)
    print(f"Total: {len(final)}", flush=True)
    for t in ("top", "mid", "tail"):
        target = SPEC_TARGETS[t]
        actual = final_by_tier.get(t, 0)
        status = "✓" if actual >= target else f"⚠ short by {target - actual}"
        print(f"  {t}: {actual} / {target} target {status}", flush=True)

    if fail_pages:
        print(f"\nFinal failed pages: {len(fail_pages)}", flush=True)
        for p, m in fail_pages[:10]:
            print(f"  page {p}: {m[:100]}", flush=True)

    # Quick sanity samples
    print(f"\n=== Sanity samples ===", flush=True)
    for t in ("top", "mid", "tail"):
        tier_rows = [r for r in final if r[1] == t]
        print(f"--- {t} (3 random) ---", flush=True)
        for r in RNG.sample(tier_rows, min(3, len(tier_rows))):
            name, _, rank, dl, dpc, repo, last, _ = r
            print(f"  {name:30s} rank={rank:>5s} dl={dl:>12s} dpc={dpc:>6s} repo={(repo or '-')[:50]}", flush=True)

    print(f"\nOutput: {out_path}", flush=True)

    # PASS criterion: spec wanted 3000-5000 total. Threshold 2700 = 90% of 3000 lower bound (methods review round 1, minor-1).
    if len(final) < 2700:
        print(f"\n⚠ Only {len(final)} packages — below 2700 (90% of 3000 lower bound)", flush=True)
        return 1
    print(f"\n✅ Task 1 Step 1.1 stratified sample saved", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
