from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path
from statistics import median

DATA_DIR = Path(__file__).parent.parent / "data"
REPORTS_DIR = Path(__file__).parent.parent / "reports"
VERSIONS_CSV = DATA_DIR / "versions.csv"
CLASS_CSV = DATA_DIR / "classification.csv"
INFLUENCE_CSV = DATA_DIR / "influence_metrics.csv"
SAMPLE_CSV = DATA_DIR / "non_provenance_sample.csv"
OUT_MD = REPORTS_DIR / "task_4_basic_stats.md"
OUT_CORR_CSV = DATA_DIR / "rq2_class_influence_corr.csv"


NUMERIC_FIELDS = [
    "weekly_downloads_avg", "github_stars", "github_forks",
    "package_age_days", "maintainers_count", "release_freq_12m",
]
BINARY_FIELDS = ["is_scoped_package", "has_dependabot", "has_issue_template"]
CAT_FIELDS = ["github_owner_type"]


def median_safe(values: list[float]) -> float:
    return median(values) if values else 0


def main() -> int:

    inf: dict[str, dict] = {}
    with INFLUENCE_CSV.open() as f:
        for r in csv.DictReader(f):
            inf[r["package"]] = r
    print(f"Loaded {len(inf)} influence rows", flush=True)


    cls: dict[tuple[str, str], str] = {}
    with CLASS_CSV.open() as f:
        for r in csv.DictReader(f):
            cls[(r["package"], r["version"])] = r["class"]


    tier: dict[tuple[str, str], str] = {}
    with SAMPLE_CSV.open() as f:
        for r in csv.DictReader(f):
            tier[(r["package"], r["version"])] = r["tier"]

    lines = ["# Basic stats + coverage + class × influence correlations",
             "",
             f"Source files: `data/versions.csv` (100,168 rows) + `data/classification.csv` (500 rows) + `data/influence_metrics.csv` (3,000 × 16 cols)",
             ""]


    lines.append("## Influence metrics coverage (3000 packages × 16 cols)")
    lines.append("")
    lines.append("| Field | Non-empty | % | Min / Med / Max (numeric) or True% |")
    lines.append("|---|---:|---:|---|")
    for fld in NUMERIC_FIELDS:
        vals: list[float] = []
        for r in inf.values():
            v = r.get(fld, "")
            if v == "" or v is None:
                continue
            try:
                vals.append(float(v))
            except ValueError:
                continue
        n = len(vals)
        pct = n / 3000 * 100
        if vals:
            s = sorted(vals)
            mn, md, mx = s[0], s[n // 2], s[-1]
            sum_str = f"{mn:,.0f} / {md:,.0f} / {mx:,.0f}"
        else:
            sum_str = "—"
        lines.append(f"| `{fld}` | {n} | {pct:.1f}% | {sum_str} |")
    for fld in BINARY_FIELDS:
        true_n = sum(1 for r in inf.values() if r.get(fld) == "true")
        false_n = sum(1 for r in inf.values() if r.get(fld) == "false")
        n = true_n + false_n
        pct = n / 3000 * 100
        true_pct = (true_n / n * 100) if n else 0
        lines.append(f"| `{fld}` | {n} | {pct:.1f}% | {true_n} true ({true_pct:.1f}%) / {false_n} false |")
    for fld in CAT_FIELDS:
        ct = Counter(r.get(fld, "") for r in inf.values())
        non_empty = sum(v for k, v in ct.items() if k)
        pct = non_empty / 3000 * 100
        breakdown = ", ".join(f"{v} {k or 'empty'}" for k, v in ct.most_common())
        lines.append(f"| `{fld}` | {non_empty} | {pct:.1f}% | {breakdown} |")


    lines += ["", "## Tier × influence-metric distribution (median across 3000 packages)",
              "",
              "| metric | top (med) | mid (med) | tail (med) |",
              "|---|---:|---:|---:|"]
    tier_vals: dict[str, dict[str, list[float]]] = {"top": {f: [] for f in NUMERIC_FIELDS},
                                                     "mid": {f: [] for f in NUMERIC_FIELDS},
                                                     "tail": {f: [] for f in NUMERIC_FIELDS}}
    for pkg, r in inf.items():
        t = r["tier"]
        if t not in tier_vals:
            continue
        for fld in NUMERIC_FIELDS:
            v = r.get(fld, "")
            if v == "" or v is None:
                continue
            try:
                tier_vals[t][fld].append(float(v))
            except ValueError:
                continue
    for fld in NUMERIC_FIELDS:
        cells = [f"{median_safe(tier_vals[t][fld]):,.0f}" for t in ("top", "mid", "tail")]
        lines.append(f"| `{fld}` | " + " | ".join(cells) + " |")


    lines += ["", "## Class × influence-metric distribution (median per class, F excluded)",
              "",
              "| metric | A | B | C | D | E |",
              "|---|---:|---:|---:|---:|---:|"]
    class_vals: dict[str, dict[str, list[float]]] = {c: {f: [] for f in NUMERIC_FIELDS} for c in "ABCDE"}
    for (pkg, ver), c in cls.items():
        if c not in "ABCDE":
            continue
        irow = inf.get(pkg, {})
        for fld in NUMERIC_FIELDS:
            v = irow.get(fld, "")
            if v == "" or v is None:
                continue
            try:
                class_vals[c][fld].append(float(v))
            except ValueError:
                continue
    for fld in NUMERIC_FIELDS:
        cells = [f"{median_safe(class_vals[c][fld]):,.0f}" for c in "ABCDE"]
        lines.append(f"| `{fld}` | " + " | ".join(cells) + " |")


    lines += ["", "## Construct caveats (carried forward)",
              "",
              "- Tier from ecosyste.ms current-state metadata; post-2026-06-01 BQ strata-shift required",
              "- Class C = possible external CI publish path (not verified)",
              "- Class E = no visible package publish workflow (not asserted manual publish)",
              "- `is_scoped_package` ≠ org-owner; for org-owner use `github_owner_type`",
              "- `depsdev_dependents_count = pending` — correlations involving it deferred until post-2026-06-01 BQ enrichment",
              ""]

    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"\n Saved {OUT_MD}", flush=True)
    print(f"  Headline tier-medians:", flush=True)
    for fld in NUMERIC_FIELDS:
        cells = " ".join(f"{t}:{median_safe(tier_vals[t][fld]):,.0f}" for t in ("top", "mid", "tail"))
        print(f"    {fld}: {cells}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
