"""
Pre-submission Batch-4 — Deterministic sub-typing of the 286 Class-E rows
(acting-PI batch, 2026-07-08; frozen artifacts only, no new collection).

Class E = "no visible package-publish workflow in inspected sources". This
script splits the E pool by two deterministic frozen signals, WITHOUT any
new rater judgment (so no kappa applies; this is signal counting, not a
validated construct):

  Signal 1 — monorepo directory marker: the sampled version's packument
    entry (or the packument top level) carries repository.directory,
    indicating the package lives in a subdirectory of a larger repository
    whose publish tooling may sit outside the package root.
  Signal 2 — release-automation hint: the repository's cached workflow YAML
    (data/workflows_raw/, frozen at collection time) mentions a
    release-automation tool (release-please, changesets, semantic-release,
    lerna) that the layered publish detector deliberately does NOT count as
    publish evidence (weak-hint tier).

Priority partition (mutually exclusive): E-dir (signal 1, regardless of 2)
> E-auto (signal 2 only) > E-plain (neither). Overlap reported separately.

Inputs:
  data/classification.csv            (500-row final labels)
  data/non_provenance_sample.csv     (sampled rows)
  data/packuments_raw/*.json         (frozen packuments)
  data/workflows_raw/*.yml|yaml      (frozen workflow cache)
  data/depsdev_dependents_2026_05_05.csv (primary tier frame)

Outputs:
  data/class_e_subtypes.csv          (per-row subtype + signals + tier)
  reports/presub_class_e_subtyping.md

Caveats carried: presence-only signals (same epistemic level as Class C);
sub-types are hypothesis-generating descriptions, not verified publish
paths; anti-causal firewall intact.

Run: conda run --no-capture-output -n stage2c python scripts/49_class_e_subtyping.py
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
REPORTS_DIR = Path(__file__).parent.parent / "reports"
PACKUMENTS = DATA_DIR / "packuments_raw"
WORKFLOWS = DATA_DIR / "workflows_raw"
OUT_CSV = DATA_DIR / "class_e_subtypes.csv"
OUT_MD = REPORTS_DIR / "presub_class_e_subtyping.md"

AUTOMATION_PAT = re.compile(
    r"release-please|changesets/action|semantic-release|lerna\b", re.IGNORECASE
)
GITHUB_URL_PAT = re.compile(r"github\.com[:/]+([^/]+)/([^/#?\s]+)", re.IGNORECASE)


def cache_filename(package: str) -> str:
    """Mirror scripts/34 packument cache naming (@scope/name -> at_scope_name)."""
    return package.replace("@", "at_").replace("/", "_") + ".json"


def dpc_tier(dpc: int) -> str:
    if dpc > 500:
        return "top"
    if dpc > 10:
        return "mid"
    return "tail"


def load_packument(package: str) -> dict | None:
    p = PACKUMENTS / cache_filename(package)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def repo_directory(pack: dict, version: str) -> str:
    """repository.directory from the sampled version first, top level fallback."""
    for repo in (
        (pack.get("versions", {}).get(version, {}) or {}).get("repository"),
        pack.get("repository"),
    ):
        if isinstance(repo, dict) and repo.get("directory"):
            return str(repo["directory"])
    return ""


def repo_slug(pack: dict, version: str) -> str:
    """owner__repo slug from the packument repository URL (version first)."""
    for repo in (
        (pack.get("versions", {}).get(version, {}) or {}).get("repository"),
        pack.get("repository"),
    ):
        url = repo.get("url", "") if isinstance(repo, dict) else (repo or "")
        m = GITHUB_URL_PAT.search(str(url))
        if m:
            owner, name = m.group(1), re.sub(r"\.git$", "", m.group(2))
            return f"{owner}__{name}"
    return ""


def automation_hint(slug: str, wf_index: dict[str, list[Path]]) -> str:
    hits = []
    for f in wf_index.get(slug.lower(), []):
        try:
            text = f.read_text(errors="ignore")
        except OSError:
            continue
        m = AUTOMATION_PAT.search(f.name) or AUTOMATION_PAT.search(text)
        if m:
            hits.append(m.group(0).lower())
    return ";".join(sorted(set(hits)))


def main() -> int:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = [r for r in csv.DictReader(open(DATA_DIR / "classification.csv"))
            if r["class"] == "E"]
    assert len(rows) == 286, f"expected 286 Class-E rows, got {len(rows)}"

    depsdev = {r["name"]: int(r["dependents_count"])
               for r in csv.DictReader(open(DATA_DIR / "depsdev_dependents_2026_05_05.csv"))}

    wf_index: dict[str, list[Path]] = {}
    for f in list(WORKFLOWS.glob("*.yml")) + list(WORKFLOWS.glob("*.yaml")):
        parts = f.name.split("__")
        if len(parts) >= 3:
            wf_index.setdefault(f"{parts[0]}__{parts[1]}".lower(), []).append(f)

    out_rows = []
    for r in rows:
        pkg, ver = r["package"], r["version"]
        pack = load_packument(pkg)
        directory = repo_directory(pack, ver) if pack else ""
        slug = repo_slug(pack, ver) if pack else ""
        hints = automation_hint(slug, wf_index) if slug else ""
        subtype = ("E-dir" if directory else
                   "E-auto" if hints else "E-plain")
        tier = dpc_tier(depsdev[pkg]) if pkg in depsdev else "missing"
        out_rows.append({
            "package": pkg, "version": ver, "tier": tier,
            "subtype": subtype, "repo_directory": directory,
            "automation_hints": hints, "repo_slug": slug,
            "packument_found": bool(pack),
        })

    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"✓ Saved {OUT_CSV}", flush=True)

    def dist(rs):
        n = len(rs)
        c = {s: sum(1 for x in rs if x["subtype"] == s)
             for s in ("E-dir", "E-auto", "E-plain")}
        return n, c

    n_all, c_all = dist(out_rows)
    n_top, c_top = dist([x for x in out_rows if x["tier"] == "top"])
    overlap = sum(1 for x in out_rows if x["repo_directory"] and x["automation_hints"])
    missing_pack = sum(1 for x in out_rows if not x["packument_found"])

    lines = [
        "# Pre-submission — Deterministic Class-E sub-typing (frozen signals)",
        "",
        "**Date**: 2026-07-08 (acting-PI Batch-4; frozen artifacts only)",
        "",
        f"- Class-E rows: {n_all} (missing packument: {missing_pack})",
        f"- **All tiers**: E-dir {c_all['E-dir']} ({c_all['E-dir']/n_all*100:.1f}%), "
        f"E-auto {c_all['E-auto']} ({c_all['E-auto']/n_all*100:.1f}%), "
        f"E-plain {c_all['E-plain']} ({c_all['E-plain']/n_all*100:.1f}%)",
        f"- **Top tier** (n={n_top}): E-dir {c_top['E-dir']} ({c_top['E-dir']/n_top*100:.1f}%), "
        f"E-auto {c_top['E-auto']} ({c_top['E-auto']/n_top*100:.1f}%), "
        f"E-plain {c_top['E-plain']} ({c_top['E-plain']/n_top*100:.1f}%)",
        f"- Overlap (directory AND automation hint): {overlap} rows (counted as E-dir)",
        "",
        "Signals are presence-only and deterministic; no rater judgment involved,",
        "so no kappa applies. E-dir marks packument `repository.directory` (monorepo",
        "subdirectory layout); E-auto marks weak release-automation hints",
        "(release-please / changesets / semantic-release / lerna) in the frozen",
        "workflow cache that the publish detector deliberately does not count as",
        "publish evidence. Sub-types are hypothesis-generating, not verified paths.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[4:10]), flush=True)
    print(f"✓ Saved {OUT_MD}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
