"""
Task 4 Step 4.2 — RQ2 class A-F distribution.

From data/classification.csv (500-row non-prov sample with v3 rules + selective PI):
  - Overall class distribution (A-F + UNCERTAIN=0 expected; F preserved but flagged)
  - Tier × class crosstab
  - Class × influence-metric stratification (downloads, stars, owner_type)

Output:
  data/rq2_class_distribution.csv (long-form: dimension, slice, class, n, pct)
  reports/task_4_rq2_summary.txt (human-readable summary)

Construct caveats carried forward (Task 2 PI Round 2):
  - C = "possible external CI publish path" (NOT verified external publish)
  - E = "no visible package publish workflow in inspected sources"
    (NOT asserted manual publish; monorepo masking limitation)
  - Class F is excluded from analytic A-E pool when computing %.

Run: conda run --no-capture-output -n stage2c python scripts/18_rq2_class_distribution.py
"""
from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
REPORTS_DIR = Path(__file__).parent.parent / "reports"
CLASS_CSV = DATA_DIR / "classification.csv"
SAMPLE_CSV = DATA_DIR / "non_provenance_sample.csv"
INFLUENCE_CSV = DATA_DIR / "influence_metrics.csv"
OUT_CSV = DATA_DIR / "rq2_class_distribution.csv"
OUT_TXT = REPORTS_DIR / "task_4_rq2_summary.txt"

CLASSES = ("A", "B", "C", "D", "E", "F", "UNCERTAIN")


def main() -> int:
    # Load classifications
    cls: dict[tuple[str, str], str] = {}
    with CLASS_CSV.open() as f:
        for r in csv.DictReader(f):
            cls[(r["package"], r["version"])] = r["class"]
    print(f"Loaded {len(cls)} classifications", flush=True)

    # Load sample tier
    tier: dict[tuple[str, str], str] = {}
    with SAMPLE_CSV.open() as f:
        for r in csv.DictReader(f):
            tier[(r["package"], r["version"])] = r["tier"]

    # Load influence metrics for stratification
    inf: dict[str, dict] = {}
    with INFLUENCE_CSV.open() as f:
        for r in csv.DictReader(f):
            inf[r["package"]] = r

    # Build long-form rows
    rows: list[dict] = []

    # 1. Overall
    overall = Counter()
    for k, c in cls.items():
        overall[c] += 1
    total = sum(overall.values())
    total_excl_f = sum(v for k, v in overall.items() if k != "F")
    for c in CLASSES:
        n = overall.get(c, 0)
        pct = n / total * 100 if total else 0
        pct_excl_f = (n / total_excl_f * 100) if (c != "F" and total_excl_f) else 0
        rows.append({"dimension": "overall", "slice": "all", "class": c,
                     "n": n, "pct_of_total": f"{pct:.2f}",
                     "pct_excl_F": f"{pct_excl_f:.2f}" if c != "F" else "—"})

    # 2. By tier (use tier-local variables; do NOT mutate outer total_excl_f)
    tier_class: dict[str, Counter] = {"top": Counter(), "mid": Counter(), "tail": Counter()}
    for k, c in cls.items():
        t = tier.get(k, "?")
        if t in tier_class:
            tier_class[t][c] += 1
    for t in ("top", "mid", "tail"):
        tier_total = sum(tier_class[t].values())
        tier_excl_f = sum(v for k, v in tier_class[t].items() if k != "F")
        for c in CLASSES:
            n = tier_class[t].get(c, 0)
            pct = n / tier_total * 100 if tier_total else 0
            pct_excl_f = (n / tier_excl_f * 100) if (c != "F" and tier_excl_f) else 0
            rows.append({"dimension": "tier", "slice": t, "class": c,
                         "n": n, "pct_of_total": f"{pct:.2f}",
                         "pct_excl_F": f"{pct_excl_f:.2f}" if c != "F" else "—"})

    # 3. By github_owner_type
    owner_class: dict[str, Counter] = {"Organization": Counter(), "User": Counter(), "unknown": Counter(), "": Counter()}
    for (pkg, _ver), c in cls.items():
        ot = inf.get(pkg, {}).get("github_owner_type", "")
        owner_class.setdefault(ot, Counter())[c] += 1
    for ot in ("Organization", "User", "unknown", ""):
        ot_total = sum(owner_class[ot].values())
        ot_excl_f = sum(v for k, v in owner_class[ot].items() if k != "F")
        if ot_total == 0:
            continue
        for c in CLASSES:
            n = owner_class[ot].get(c, 0)
            pct = n / ot_total * 100 if ot_total else 0
            pct_excl_f = (n / ot_excl_f * 100) if (c != "F" and ot_excl_f) else 0
            rows.append({"dimension": "owner_type", "slice": ot or "no_github",
                         "class": c, "n": n, "pct_of_total": f"{pct:.2f}",
                         "pct_excl_F": f"{pct_excl_f:.2f}" if c != "F" else "—"})

    # 4. By has_dependabot
    dep_class: dict[str, Counter] = {"true": Counter(), "false": Counter(), "": Counter()}
    for (pkg, _ver), c in cls.items():
        d = inf.get(pkg, {}).get("has_dependabot", "")
        dep_class.setdefault(d, Counter())[c] += 1
    for d in ("true", "false", ""):
        dep_total = sum(dep_class[d].values())
        dep_excl_f = sum(v for k, v in dep_class[d].items() if k != "F")
        if dep_total == 0:
            continue
        for c in CLASSES:
            n = dep_class[d].get(c, 0)
            pct = n / dep_total * 100 if dep_total else 0
            pct_excl_f = (n / dep_excl_f * 100) if (c != "F" and dep_excl_f) else 0
            rows.append({"dimension": "has_dependabot", "slice": d or "no_github",
                         "class": c, "n": n, "pct_of_total": f"{pct:.2f}",
                         "pct_excl_F": f"{pct_excl_f:.2f}" if c != "F" else "—"})

    # Write CSV
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dimension", "slice", "class", "n", "pct_of_total", "pct_excl_F"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n✓ Saved {OUT_CSV} ({len(rows)} rows)", flush=True)

    # Human-readable summary
    # Re-derive overall totals here (the outer `total_excl_f` may have been mutated by
    # earlier tier/owner_type/dependabot loops; we use FRESH local names for safety).
    overall_total = sum(overall.values())
    overall_excl_f = sum(v for k, v in overall.items() if k != "F")
    lines = ["# Task 4 RQ2 — Class distribution summary",
             "",
             "## Overall (excluding F per kickoff)",
             "",
             "| Class | n | % of total | % of A-E pool |",
             "|---|---:|---:|---:|"]
    for c in CLASSES:
        n = overall.get(c, 0)
        pct = n / overall_total * 100 if overall_total else 0
        pct_excl_f = (n / overall_excl_f * 100) if (c != "F" and overall_excl_f) else None
        excl_str = f"{pct_excl_f:.1f}%" if pct_excl_f is not None else "—"
        lines.append(f"| {c} | {n} | {pct:.1f}% | {excl_str} |")
    lines += ["",
              f"Total: 500 sampled non-prov versions (F={overall.get('F', 0)} excluded → "
              f"{overall_excl_f} in A-E analytic pool)",
              "",
              "## Tier × class (% of tier total; F excluded from pct denominator)",
              "",
              f"| tier | " + " | ".join(CLASSES[:-1]) + " | total |",
              "|---|" + "---|" * 7]
    for t in ("top", "mid", "tail"):
        total_tier = sum(tier_class[t].values())
        cells = []
        for c in CLASSES[:-1]:
            n = tier_class[t].get(c, 0)
            denom = total_tier if c == "F" else (total_tier - tier_class[t].get("F", 0))
            pct = (n / denom * 100) if denom else 0
            cells.append(f"{n} ({pct:.1f}%)")
        lines.append(f"| {t} | " + " | ".join(cells) + f" | {total_tier} |")

    lines += ["",
              "## owner_type × class (% of subgroup; F excluded from denominator)",
              "",
              "| owner_type | A | B | C | D | E | F | total |",
              "|---|---|---|---|---|---|---|---|"]
    for ot in ("Organization", "User", "unknown", ""):
        total_sub = sum(owner_class[ot].values())
        if total_sub == 0:
            continue
        cells = []
        for c in CLASSES[:-1]:
            n = owner_class[ot].get(c, 0)
            denom = total_sub if c == "F" else (total_sub - owner_class[ot].get("F", 0))
            pct = (n / denom * 100) if denom else 0
            cells.append(f"{n} ({pct:.1f}%)")
        lines.append(f"| {ot or 'no_github'} | " + " | ".join(cells) + f" | {total_sub} |")

    lines += ["",
              "## Construct caveats (carried forward from Task 2 PI Round 2 mandate)",
              "",
              "- Class C = 'possible external CI publish path' (CircleCI/GitLab config detected; not verified publish to npm)",
              "- Class E = 'no visible package publish workflow in inspected sources' (NOT asserted manual publish; monorepo masking applies)",
              "- Class F (no github / non-public repo) excluded from analytic A-E pool",
              "",
              "## Tier-source note (Round 3 reweight)",
              "",
              "This file is computed under ecosyste.ms-tier (Task 1 sampling source); it is RETAINED as source-sensitivity in Round 3.",
              "The PRIMARY tier-stratified analysis is in `reports/task_4_rq2_summary_depsdev.txt` (deps.dev-tier snapshot 2026-05-05).",
              "Per Task 1 strata-shift check (`reports/task_3_strata_shift.md`): all 3 tiers shift > 5% between sources; PI Round 2 mandated full reweight under deps.dev-tier.",
              ]
    OUT_TXT.write_text("\n".join(lines) + "\n")
    print(f"✓ Saved {OUT_TXT}", flush=True)

    # Print summary
    print(f"\n=== Overall class distribution (excluding F) ===", flush=True)
    print(f"  total non-F: {total_excl_f}", flush=True)
    for c in CLASSES[:-1]:  # skip UNCERTAIN (which is 0)
        n = overall.get(c, 0)
        if c == "F":
            print(f"  {c}: {n} (excluded from A-E pool)", flush=True)
        else:
            pct = n / total_excl_f * 100
            print(f"  {c}: {n} ({pct:.1f}%)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
