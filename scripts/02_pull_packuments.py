from __future__ import annotations

import json
import sys
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

DATA_DIR = Path(__file__).parent.parent / "data"
PACKAGES_FILE = DATA_DIR / "packages_full.txt"
RAW_DIR = DATA_DIR / "packuments_raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "stage2c-research/1.0 (Task 1.2 packument pull)", "Accept": "application/json"}
N_WORKERS = 10
MAX_RETRIES = 3


def safe_fname(pkg: str) -> str:

    return pkg.replace("/", "_").replace("@", "at_") + ".json"


def fetch_one(pkg: str) -> tuple[str, str, int]:
    out_path = RAW_DIR / safe_fname(pkg)
    if out_path.exists() and out_path.stat().st_size > 100:
        return pkg, "cached", out_path.stat().st_size

    url = f"https://registry.npmjs.org/{urllib.parse.quote(pkg, safe='@')}"
    last_err = ""
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(url, headers=HEADERS, timeout=60)
            if r.status_code == 404:
                out_path.write_text(json.dumps({"_404": True, "name": pkg}))
                return pkg, "404", 0
            r.raise_for_status()
            content = r.content
            out_path.write_bytes(content)
            return pkg, "ok", len(content)
        except Exception as e:
            last_err = f"{type(e).__name__}: {str(e)[:100]}"
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
                continue
    return pkg, f"fail: {last_err}", 0


def main() -> int:
    if not PACKAGES_FILE.exists():
        print(f"ERROR: {PACKAGES_FILE} not found; run 01_sample_packages.py first", flush=True)
        return 2

    packages: list[str] = []
    with PACKAGES_FILE.open() as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if parts and parts[0]:
                packages.append(parts[0])
    print(f"Loaded {len(packages)} packages from {PACKAGES_FILE}", flush=True)
    print(f"Output: {RAW_DIR}", flush=True)
    print(f"Workers: {N_WORKERS}", flush=True)

    stats = {"ok": 0, "cached": 0, "404": 0, "fail": 0}
    total_bytes = 0
    fail_pkgs: list[tuple[str, str]] = []
    start = time.time()

    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(fetch_one, pkg): pkg for pkg in packages}
        for i, fut in enumerate(as_completed(futures), 1):
            pkg, status, size = fut.result()
            if status == "ok":
                stats["ok"] += 1
                total_bytes += size
            elif status == "cached":
                stats["cached"] += 1
            elif status == "404":
                stats["404"] += 1
            else:
                stats["fail"] += 1
                fail_pkgs.append((pkg, status))
            if i % 200 == 0 or i == len(packages):
                elapsed = time.time() - start
                rate = i / elapsed if elapsed > 0 else 0
                eta = (len(packages) - i) / rate if rate > 0 else 0
                print(
                    f"  [{i:>4}/{len(packages)}] ok={stats['ok']} cached={stats['cached']} 404={stats['404']} fail={stats['fail']} "
                    f"rate={rate:.1f}/s eta={eta:.0f}s",
                    flush=True,
                )

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s", flush=True)
    for k, v in stats.items():
        print(f"  {k}: {v}", flush=True)
    print(f"  total downloaded: {total_bytes/1e6:.1f} MB", flush=True)

    if fail_pkgs:
        print(f"\nFailures ({len(fail_pkgs)} total, first 10):", flush=True)
        for p, s in fail_pkgs[:10]:
            print(f"  {p}: {s}", flush=True)

    success_rate = (stats["ok"] + stats["cached"] + stats["404"]) / len(packages)
    print(f"\nSuccess rate (incl. 404 as 'reached'): {success_rate*100:.1f}%", flush=True)
    if success_rate < 0.95:
        print(f" Below 95% spec target", flush=True)
        return 1
    print(f" ≥ 95% reached", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
