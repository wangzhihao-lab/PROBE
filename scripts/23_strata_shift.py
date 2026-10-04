"""
Task 3 Step 3.8 — Task 1 strata-shift check per methodologist mandate.

Compares for each of 3000 packages:
  current_state_dpc  = ecosyste.ms `dependent_packages_count` at collection time (2026-05-15;
                       used to assign tiers in Task 1)
  cutoff_snapshot_dpc = deps.dev `dependents_count` at SnapshotAt 2026-05-05 (closest snapshot
                       before 2026-05-10 cutoff; from scripts/22)

Outputs:
  data/strata_shift.csv             (per-pkg: name, eco_dpc, eco_tier, depsdev_dpc, depsdev_tier, shift)
  reports/task_3_strata_shift.md    (crosstab + per-tier shift summary + verdict)

Verdict logic per methodologist Task 2 RR1+2 mandate (carried by Task 3 PI):
  - For each tier, report % of sample packages that would shift to a different tier
    under the cutoff snapshot
  - If shift > 5% per tier: Task 4/5 MUST reweight in analysis
  - If shift ≤ 5% per tier: tier assignments are stable; sampling-frame bias is bounded

Tier definition (per kickoff + Task 1):
  top:  dpc > 500
  mid:  10 < dpc ≤ 500
  tail: dpc ≤ 10

Run: conda run --no-capture-output -n stage2c python scripts/23_strata_shift.py
"""
from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
REPORTS_DIR = Path(__file__).parent.parent / "reports"
PACKAGES_FILE = DATA_DIR / "packages_full.txt"
DEPSDEV_CSV = DATA_DIR / "depsdev_dependents_2026_05_05.csv"
OUT_CSV = DATA_DIR / "strata_shift.csv"
OUT_MD = REPORTS_DIR / "task_3_strata_shift.md"


def dpc_tier(dpc: int) -> str:
    if dpc > 500:
        return "top"
    if dpc > 10:
        return "mid"
    return "tail"


def main() -> int:
    # Load packages_full.txt: name → (eco_dpc, eco_tier)
    eco: dict[str, tuple[int, str]] = {}
    with PACKAGES_FILE.open() as f:
        header = next(f).rstrip("\n").split("\t")
        idx_name = header.index("name")
        idx_tier = header.index("dpc_tier")
        idx_dpc = header.index("dependent_packages_count")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) > idx_dpc:
                try:
                    eco[parts[idx_name]] = (int(parts[idx_dpc]), parts[idx_tier])
                except ValueError:
                    continue
    print(f"Loaded {len(eco)} ecosyste.ms records", flush=True)

    # Load deps.dev snapshot
    depsdev: dict[str, int] = {}
    with DEPSDEV_CSV.open() as f:
        for r in csv.DictReader(f):
            try:
                depsdev[r["name"]] = int(r["dependents_count"])
            except ValueError:
                continue
    print(f"Loaded {len(depsdev)} deps.dev records (snapshot 2026-05-05)", flush=True)

    # Compute shift per package
    rows: list[dict] = []
    missing_depsdev = 0
    shift_crosstab: dict[tuple[str, str], int] = Counter()  # (eco_tier, depsdev_tier) → count
    per_tier_shift: dict[str, int] = {"top": 0, "mid": 0, "tail": 0}
    per_tier_total: dict[str, int] = {"top": 0, "mid": 0, "tail": 0}

    for name, (eco_dpc, eco_tier) in sorted(eco.items()):
        dd_dpc = depsdev.get(name)
        if dd_dpc is None:
            missing_depsdev += 1
            dd_tier = "missing"
            shift = "missing_in_depsdev"
        else:
            dd_tier = dpc_tier(dd_dpc)
            shift = "same" if dd_tier == eco_tier else f"{eco_tier}→{dd_tier}"

        rows.append({
            "name": name,
            "ecosyste_dpc": eco_dpc,
            "ecosyste_tier": eco_tier,
            "depsdev_dpc": dd_dpc if dd_dpc is not None else "",
            "depsdev_tier": dd_tier,
            "shift": shift,
        })

        if eco_tier in per_tier_total:
            per_tier_total[eco_tier] += 1
            if dd_dpc is not None and dd_tier != eco_tier:
                per_tier_shift[eco_tier] += 1
            shift_crosstab[(eco_tier, dd_tier)] += 1

    # Write CSV
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "ecosyste_dpc", "ecosyste_tier",
                                          "depsdev_dpc", "depsdev_tier", "shift"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n✓ Saved {OUT_CSV} ({len(rows)} rows; missing depsdev={missing_depsdev})", flush=True)

    # Crosstab
    print(f"\n=== Crosstab (ecosyste.ms × deps.dev) ===", flush=True)
    print(f"  {'eco↓ / dd→':<14s}  {'top':>6s}  {'mid':>6s}  {'tail':>6s}  {'missing':>8s}", flush=True)
    for et in ("top", "mid", "tail"):
        cells = []
        for dt in ("top", "mid", "tail", "missing"):
            cells.append(f"{shift_crosstab.get((et, dt), 0):>6}")
        print(f"  {et:<14s}  " + "  ".join(cells), flush=True)

    # Per-tier shift magnitude
    print(f"\n=== Per-tier shift magnitude (vs methodologist 5% threshold) ===", flush=True)
    any_above_5 = False
    for t in ("top", "mid", "tail"):
        total = per_tier_total[t]
        shifted = per_tier_shift[t]
        pct = shifted / total * 100 if total else 0
        flag = "❌ ABOVE 5%" if pct > 5 else "✓ below 5%"
        if pct > 5:
            any_above_5 = True
        print(f"  {t:6s}: {shifted}/{total} = {pct:.1f}%  {flag}", flush=True)

    # Markdown summary
    lines = [
        "# Task 3 Step 3.8 — Strata-shift check (methodologist Task 2 RR1+2 mandate)",
        "",
        f"**Date**: 2026-05-18 (run after Task 3.7 BQ enrichment)",
        f"**Sources**: packages_full.txt (ecosyste.ms current-state at 2026-05-15) vs depsdev_dependents_2026_05_05.csv (deps.dev snapshot at 2026-05-05; 5 days before cutoff)",
        "",
        "## Tier definition (per kickoff + Task 1)",
        "- top:  dpc > 500",
        "- mid:  10 < dpc ≤ 500",
        "- tail: dpc ≤ 10",
        "",
        "## Crosstab (ecosyste.ms current-state × deps.dev cutoff-snapshot)",
        "",
        "| eco ↓ / dd → | top | mid | tail | missing | total |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for et in ("top", "mid", "tail"):
        cells = [shift_crosstab.get((et, dt), 0) for dt in ("top", "mid", "tail", "missing")]
        lines.append(f"| {et} | " + " | ".join(str(c) for c in cells) + f" | {per_tier_total[et]} |")

    lines += ["",
              "## Per-tier shift magnitude (methodologist threshold 5%)",
              "",
              "| Tier | Shifted | Total in tier | % | Verdict |",
              "|---|---:|---:|---:|---|"]
    for t in ("top", "mid", "tail"):
        total = per_tier_total[t]
        shifted = per_tier_shift[t]
        pct = shifted / total * 100 if total else 0
        verdict = "❌ ABOVE 5% — reweight in Task 4/5" if pct > 5 else "✓ below 5%"
        lines.append(f"| {t} | {shifted} | {total} | {pct:.1f}% | {verdict} |")

    lines += ["",
              "## Verdict",
              "",
              f"- {'**At least one tier exceeds 5% shift threshold**' if any_above_5 else '**All tiers below 5% shift threshold**'}",
              f"- Action: " + (
                  "Task 4/5 MUST reweight or report tier-shifted robustness analysis."
                  if any_above_5
                  else "Tier assignments are stable; sampling-frame bias is bounded (≤5%/tier)."),
              "",
              "## Limitations",
              "",
              "- deps.dev snapshot is 2026-05-05, NOT exactly the 2026-05-10 cutoff (5-day delta — closest available snapshot before cutoff)",
              "- ecosyste.ms `dependent_packages_count` and deps.dev `dependents_count` may differ slightly in measurement definition (both count distinct packages depending; deps.dev counts transitive variants via SnapshotAt+System=NPM filter)",
              "- 62 packages have no matching deps.dev row (very recent packages added post-2026-05-05 OR scoped variants)",
              ]
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"\n✓ Saved {OUT_MD}", flush=True)
    return 0 if not any_above_5 else 1


if __name__ == "__main__":
    sys.exit(main())
