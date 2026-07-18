from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
REPORTS_DIR = Path(__file__).parent.parent / "reports"
CLASS_CSV = DATA_DIR / "classification.csv"
SAMPLE_CSV = DATA_DIR / "non_provenance_sample.csv"
DEPSDEV_CSV = DATA_DIR / "depsdev_dependents_2026_05_05.csv"
OUT_CSV = DATA_DIR / "rq2_class_distribution_depsdev.csv"
OUT_TXT = REPORTS_DIR / "task_4_rq2_summary_depsdev.txt"

CLASSES = ("A", "B", "C", "D", "E", "F", "UNCERTAIN")


def dpc_tier(dpc: int) -> str:
    if dpc > 500:
        return "top"
    if dpc > 10:
        return "mid"
    return "tail"


def main() -> int:

    pkg_tier: dict[str, str] = {}
    with DEPSDEV_CSV.open() as f:
        for r in csv.DictReader(f):
            try:
                pkg_tier[r["name"]] = dpc_tier(int(r["dependents_count"]))
            except ValueError:
                continue


    cls: dict[tuple[str, str], str] = {}
    with CLASS_CSV.open() as f:
        for r in csv.DictReader(f):
            cls[(r["package"], r["version"])] = r["class"]


    rows: list[dict] = []
    overall: Counter = Counter()
    tier_class: dict[str, Counter] = {"top": Counter(), "mid": Counter(),
                                       "tail": Counter(), "missing": Counter()}
    for k, c in cls.items():
        overall[c] += 1
        t = pkg_tier.get(k[0], "missing")
        tier_class[t][c] += 1

    print(f"Loaded {len(cls)} classifications", flush=True)
    print(f"  with deps.dev tier (non-missing): "
          f"{sum(sum(tier_class[t].values()) for t in ('top','mid','tail'))}", flush=True)
    print(f"  with missing tier: {sum(tier_class['missing'].values())}", flush=True)


    overall_total = sum(overall.values())
    overall_excl_f = sum(v for k, v in overall.items() if k != "F")


    for c in CLASSES:
        n = overall.get(c, 0)
        pct = n / overall_total * 100 if overall_total else 0
        pct_excl_f = (n / overall_excl_f * 100) if (c != "F" and overall_excl_f) else None
        rows.append({"dimension": "overall", "slice": "all", "class": c,
                     "n": n, "pct_of_total": f"{pct:.2f}",
                     "pct_excl_F": f"{pct_excl_f:.2f}" if pct_excl_f is not None else "—"})


    for t in ("top", "mid", "tail", "missing"):
        tier_total = sum(tier_class[t].values())
        tier_excl_f = sum(v for k, v in tier_class[t].items() if k != "F")
        for c in CLASSES:
            n = tier_class[t].get(c, 0)
            pct = n / tier_total * 100 if tier_total else 0
            pct_excl_f = (n / tier_excl_f * 100) if (c != "F" and tier_excl_f) else None
            rows.append({"dimension": "depsdev_tier", "slice": t, "class": c,
                         "n": n, "pct_of_total": f"{pct:.2f}",
                         "pct_excl_F": f"{pct_excl_f:.2f}" if pct_excl_f is not None else "—"})

    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dimension", "slice", "class", "n", "pct_of_total", "pct_excl_F"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n Saved {OUT_CSV} ({len(rows)} rows)", flush=True)


    lines = [
        "# RQ2 — Class distribution under deps.dev-tier",
        "",
        f"Source: data/classification.csv (500 non-prov sample, v3 rules + selective PI) joined to",
        f"deps.dev `dependents_count` snapshot 2026-05-05 (Task 3.7).",
        "",
        "Tier definition (same thresholds; applied to deps.dev DPC):",
        "  top:  dpc > 500    mid:  10 < dpc ≤ 500    tail:  dpc ≤ 10    missing: not in snapshot",
        "",
        "## Overall (excluding F per kickoff — identical to ecosyste.ms-tier baseline)",
        "",
        "| Class | n | % of total | % of A-E pool |",
        "|---|---:|---:|---:|",
    ]
    for c in CLASSES:
        n = overall.get(c, 0)
        pct = n / overall_total * 100 if overall_total else 0
        pct_excl_f = (n / overall_excl_f * 100) if (c != "F" and overall_excl_f) else None
        excl_str = f"{pct_excl_f:.1f}%" if pct_excl_f is not None else "—"
        lines.append(f"| {c} | {n} | {pct:.1f}% | {excl_str} |")
    lines.append("")
    lines.append(f"Total: 500 versions (F={overall.get('F', 0)} excluded  {overall_excl_f} in A-E pool)")
    lines += [
        "",
        "## depsdev-tier × class (% of tier's A-E pool; F excluded from denominator)",
        "",
        "| tier | A | B | C | D | E | F | total (A-E pool / +F) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for t in ("top", "mid", "tail", "missing"):
        tier_total = sum(tier_class[t].values())
        tier_excl_f = sum(v for k, v in tier_class[t].items() if k != "F")
        cells = []
        for c in CLASSES[:-1]:
            n = tier_class[t].get(c, 0)
            if c == "F":
                cells.append(f"{n}")
            else:
                pct = (n / tier_excl_f * 100) if tier_excl_f else 0
                cells.append(f"{n} ({pct:.1f}%)")
        lines.append(f"| {t} | " + " | ".join(cells) + f" | {tier_excl_f} / {tier_total} |")


    eco_class: dict[str, Counter] = {"top": Counter(), "mid": Counter(), "tail": Counter()}
    try:
        with (DATA_DIR / "rq2_class_distribution.csv").open() as f:
            for r in csv.DictReader(f):
                if r["dimension"] == "tier" and r["slice"] in eco_class:
                    eco_class[r["slice"]][r["class"]] = int(r["n"])
    except FileNotFoundError:
        pass

    lines += [
        "",
        "## Side-by-side: ecosyste.ms-tier vs deps.dev-tier (A-E pool %, F excluded)",
        "",
        "| tier | class | eco-tier % | dd-tier % | Δ (pp) | Direction |",
        "|---|---|---:|---:|---:|---|",
    ]
    for t in ("top", "mid", "tail"):
        eco_total_excl_f = sum(v for k, v in eco_class[t].items() if k != "F")
        dd_total_excl_f = sum(v for k, v in tier_class[t].items() if k != "F")
        for c in "ABCDE":
            eco_pct = (eco_class[t].get(c, 0) / eco_total_excl_f * 100) if eco_total_excl_f else 0
            dd_pct = (tier_class[t].get(c, 0) / dd_total_excl_f * 100) if dd_total_excl_f else 0
            delta = dd_pct - eco_pct
            direction = "▲" if delta > 1 else ("▼" if delta < -1 else "≈")
            lines.append(f"| {t} | {c} | {eco_pct:.1f}% | {dd_pct:.1f}% | {delta:+.1f} | {direction} |")


    lines += [
        "",
        "## Headline interpretation (verified against review prediction)",
        "",
    ]

    b_eco = {t: (eco_class[t].get("B", 0) / max(1, sum(v for k, v in eco_class[t].items() if k != "F")) * 100) for t in ("top", "mid", "tail")}
    b_dd = {t: (tier_class[t].get("B", 0) / max(1, sum(v for k, v in tier_class[t].items() if k != "F")) * 100) for t in ("top", "mid", "tail")}
    e_dd = {t: (tier_class[t].get("E", 0) / max(1, sum(v for k, v in tier_class[t].items() if k != "F")) * 100) for t in ("top", "mid", "tail")}
    c_dd = {t: (tier_class[t].get("C", 0) / max(1, sum(v for k, v in tier_class[t].items() if k != "F")) * 100) for t in ("top", "mid", "tail")}

    lines.append(f"- B share under deps.dev-tier: top {b_dd['top']:.1f}% / mid {b_dd['mid']:.1f}% / tail {b_dd['tail']:.1f}% — {'monotone (top<mid<tail)' if b_dd['top'] < b_dd['mid'] < b_dd['tail'] else 'NOT monotone'}")
    lines.append(f"- E share under deps.dev-tier: top {e_dd['top']:.1f}% / mid {e_dd['mid']:.1f}% / tail {e_dd['tail']:.1f}% — {'monotone (top>mid>tail)' if e_dd['top'] > e_dd['mid'] > e_dd['tail'] else 'NOT monotone (PI prediction verified)'}")
    lines.append(f"- C share under deps.dev-tier: top {c_dd['top']:.1f}% / mid {c_dd['mid']:.1f}% / tail {c_dd['tail']:.1f}% — {'mid-tier dominant (PI prediction verified)' if c_dd['mid'] > c_dd['top'] and c_dd['mid'] > c_dd['tail'] else 'other pattern'}")

    OUT_TXT.write_text("\n".join(lines) + "\n")
    print(f" Saved {OUT_TXT}", flush=True)


    print(f"\n=== depsdev-tier × class headline ===", flush=True)
    for t in ("top", "mid", "tail", "missing"):
        ae = sum(v for k, v in tier_class[t].items() if k != "F")
        if ae == 0:
            continue
        cells = " / ".join(f"{c}: {tier_class[t].get(c,0)/ae*100:.1f}%" for c in "ABCDE")
        print(f"  {t:<8s} (n={ae}): {cells}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
