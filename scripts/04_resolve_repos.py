from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

DATA_DIR = Path(__file__).parent.parent / "data"
PACKAGES_FILE = DATA_DIR / "packages_full.txt"
SAMPLE_CSV = DATA_DIR / "non_provenance_sample.csv"
RAW_DIR = DATA_DIR / "packuments_raw"
OUT_CSV = DATA_DIR / "classification_step21.csv"


GITHUB_RE = re.compile(
    r"github\.com[/:]([A-Za-z0-9_.\-]+)/([A-Za-z0-9_.\-]+?)(?:\.git)?(?:[/#?].*)?$",
    re.IGNORECASE,
)


def parse_repo_url(raw: str) -> tuple[str, str, str]:
    if not raw:
        return "", "", ""
    s = raw.strip()
    if s.lower() in {"none", "null", "n/a"}:
        return "", "", ""


    if s.startswith("git+"):
        s = s[4:]


    m = GITHUB_RE.search(s)
    if m:
        owner = m.group(1)
        repo = m.group(2).removesuffix(".git")
        return "github.com", owner, repo


    if s.startswith("ssh://"):
        s_for_parse = s
    elif s.startswith(("git://", "git@", "http://", "https://")):
        s_for_parse = s
    else:

        return "unknown", "", ""

    try:
        parsed = urlparse(s_for_parse)
        host = (parsed.hostname or "").lower()
        return host or "unknown", "", ""
    except Exception:
        return "unknown", "", ""


def safe_fname(pkg: str) -> str:
    return pkg.replace("/", "_").replace("@", "at_") + ".json"


def packument_repo(pkg: str) -> str:
    fpath = RAW_DIR / safe_fname(pkg)
    if not fpath.exists():
        return ""
    try:
        data = json.loads(fpath.read_text())
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


def main() -> int:

    pkg_repo: dict[str, str] = {}
    with PACKAGES_FILE.open() as f:
        header = next(f).rstrip("\n").split("\t")
        idx_name = header.index("name")
        idx_repo = header.index("repository_url")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) > idx_repo:
                pkg_repo[parts[idx_name]] = parts[idx_repo]
    print(f"Loaded {len(pkg_repo):,} packagerepo_url mappings", flush=True)


    rows: list[dict] = []
    with SAMPLE_CSV.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    print(f"Loaded {len(rows)} sample rows", flush=True)


    out_rows: list[dict] = []
    miss_lookup = 0
    fallback_used = 0
    counts = {"F_empty": 0, "F_non_github": 0, "F_malformed": 0, "pending_github": 0}

    for r in rows:
        pkg = r["package"]
        eco_url = pkg_repo.get(pkg, "")
        if pkg not in pkg_repo:
            miss_lookup += 1
        host, owner, repo = parse_repo_url(eco_url)


        repo_url = eco_url
        source = "ecosyste.ms" if eco_url else "missing"
        if host != "github.com":
            pkmt_url = packument_repo(pkg)
            pkmt_host, pkmt_owner, pkmt_repo = parse_repo_url(pkmt_url)
            if pkmt_host == "github.com":
                repo_url = pkmt_url
                host, owner, repo = pkmt_host, pkmt_owner, pkmt_repo
                source = "packument"
                fallback_used += 1

        if not repo_url or host == "":
            prelim = "F"
            counts["F_empty"] += 1
        elif host == "unknown":
            prelim = "F"
            counts["F_malformed"] += 1
        elif host == "github.com":
            prelim = "pending"
            counts["pending_github"] += 1
        else:
            prelim = "F"
            counts["F_non_github"] += 1

        out_rows.append({
            "package": pkg,
            "version": r["version"],
            "repo_url": repo_url,
            "repo_url_source": source,
            "host": host,
            "github_owner": owner,
            "github_repo": repo,
            "is_github": "true" if host == "github.com" else "false",
            "preliminary_class": prelim,
        })

    if miss_lookup:
        print(f" {miss_lookup} sample rows had no matching package in packages_full.txt", flush=True)
    if fallback_used:
        print(f"  packument fallback salvaged {fallback_used} packages", flush=True)


    fields = ["package", "version", "repo_url", "repo_url_source", "host",
              "github_owner", "github_repo", "is_github", "preliminary_class"]
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in out_rows:
            w.writerow(r)
    print(f"\n Saved {OUT_CSV} ({len(out_rows)} rows)", flush=True)


    print(f"\n=== Summary ===", flush=True)
    print(f"  F (empty repo_url):   {counts['F_empty']}", flush=True)
    print(f"  F (non-github host):  {counts['F_non_github']}", flush=True)
    print(f"  F (malformed URL):    {counts['F_malformed']}", flush=True)
    total_f = counts["F_empty"] + counts["F_non_github"] + counts["F_malformed"]
    print(f"  F total:              {total_f}", flush=True)
    print(f"  pending (github):     {counts['pending_github']}", flush=True)


    sample_pkg_tier: dict[str, str] = {}
    with SAMPLE_CSV.open() as f:
        for r in csv.DictReader(f):
            sample_pkg_tier[(r["package"], r["version"])] = r["tier"]
    tier_f = {"top": 0, "mid": 0, "tail": 0}
    tier_g = {"top": 0, "mid": 0, "tail": 0}
    for r in out_rows:
        t = sample_pkg_tier.get((r["package"], r["version"]), "?")
        if r["preliminary_class"] == "F":
            tier_f[t] = tier_f.get(t, 0) + 1
        else:
            tier_g[t] = tier_g.get(t, 0) + 1
    print(f"\n=== F-rate by tier ===", flush=True)
    for t in ("top", "mid", "tail"):
        total = tier_f[t] + tier_g[t]
        pct = (tier_f[t] / total * 100) if total else 0
        print(f"  {t}: F={tier_f[t]:>3} / total={total:>3} ({pct:.1f}%)", flush=True)


    f_rows = [r for r in out_rows if r["preliminary_class"] == "F"][:5]
    if f_rows:
        print(f"\n=== F sample preview (5) ===", flush=True)
        for r in f_rows:
            url_preview = (r["repo_url"][:60] + "…") if len(r["repo_url"]) > 60 else r["repo_url"]
            print(f"  {r['package']:35s} host={r['host']:12s} url={url_preview!r}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
