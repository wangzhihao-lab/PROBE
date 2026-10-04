"""
Task 2 Step 2.7 — Cohen's Kappa for rater A (rater1) vs rater B (rater2).

Inputs:
  data/classification.csv          (rater='rater_A'; 500 rows)
  data/classification_rater2.csv   (rater='rater_B'; 100 rows expected, from the rater-B handoff)
                                    schema: package, version, class, reason, rater

Outputs:
  reports/kappa_result.json        (kappa, confusion matrix, disagreement detail)
  reports/kappa_summary.txt        (human-readable summary)

Verdict:
  kappa ≥ 0.7 → PASS (proceed to Task 3)
  kappa < 0.7 → FAIL (analyze disagreement classes, revise rules, full re-classify)

Run: conda run --no-capture-output -n stage2c python scripts/07_compute_kappa.py
"""
from __future__ import annotations

import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

from sklearn.metrics import cohen_kappa_score, confusion_matrix

DATA_DIR = Path(__file__).parent.parent / "data"
REPORTS_DIR = Path(__file__).parent.parent / "reports"
RATER_A_CSV = DATA_DIR / "classification.csv"
RATER2_CSV = DATA_DIR / "classification_rater2.csv"
OUT_JSON = REPORTS_DIR / "kappa_result.json"
OUT_TXT = REPORTS_DIR / "kappa_summary.txt"

KAPPA_THRESHOLD = 0.7
ALL_CLASSES = ["A", "B", "C", "D", "E", "F", "UNCERTAIN"]


def load_classifications(path: Path) -> dict[tuple[str, str], str]:
    if not path.exists():
        print(f"ERROR: {path} missing", file=sys.stderr)
        sys.exit(2)
    out: dict[tuple[str, str], str] = {}
    with path.open() as f:
        for r in csv.DictReader(f):
            out[(r["package"], r["version"])] = r["class"]
    return out


def main() -> int:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    rater_a = load_classifications(RATER_A_CSV)
    rater2 = load_classifications(RATER2_CSV)
    print(f"Rater-A rows:  {len(rater_a)}", flush=True)
    print(f"Rater2 rows:  {len(rater2)}", flush=True)

    # Intersection
    common = sorted(set(rater_a.keys()) & set(rater2.keys()))
    if len(common) < 50:
        print(f"⚠ intersection only {len(common)}; need ≥ 50 for meaningful kappa",
              file=sys.stderr)
    print(f"Intersection: {len(common)}", flush=True)

    y1 = [rater_a[k] for k in common]
    y2 = [rater2[k] for k in common]

    # Labels: only those that actually appear (avoid sklearn warning)
    labels = [c for c in ALL_CLASSES if c in (set(y1) | set(y2))]

    kappa = cohen_kappa_score(y1, y2, labels=labels)
    cm = confusion_matrix(y1, y2, labels=labels).tolist()

    # Methodologist RR6 additions: Gwet AC1 + bootstrap CI for κ.
    def gwet_ac1(a, b, lbls):
        n = len(a)
        if n == 0:
            return None
        q = len(lbls)
        if q < 2:
            return None
        p_a = sum(1 for x, y in zip(a, b) if x == y) / n
        pi = []
        for lbl in lbls:
            p1 = sum(1 for x in a if x == lbl) / n
            p2 = sum(1 for y in b if y == lbl) / n
            pi.append((p1 + p2) / 2)
        p_e = sum(p * (1 - p) for p in pi) / (q - 1)
        if p_e >= 1:
            return None
        return (p_a - p_e) / (1 - p_e)

    ac1 = gwet_ac1(y1, y2, labels)

    # Bootstrap CI for kappa (RNG seed=42; 1000 resamples)
    rng = random.Random(42)
    n_boot = 1000
    kappas: list[float] = []
    for _ in range(n_boot):
        idx = [rng.randrange(len(y1)) for _ in range(len(y1))]
        boot_y1 = [y1[i] for i in idx]
        boot_y2 = [y2[i] for i in idx]
        try:
            kappas.append(cohen_kappa_score(boot_y1, boot_y2, labels=labels))
        except Exception:
            continue
    kappas.sort()
    ci_lo = kappas[int(0.025 * len(kappas))] if kappas else None
    ci_hi = kappas[int(0.975 * len(kappas))] if kappas else None

    # Prevalence index (max marginal proportion)
    marginals = [
        max(sum(1 for x in y1 if x == lbl) / len(y1),
            sum(1 for y in y2 if y == lbl) / len(y2))
        for lbl in labels
    ]
    prevalence_max = max(marginals) if marginals else None

    # Per-class agreement
    per_class: dict[str, dict[str, int]] = {}
    for lbl in labels:
        c1 = sum(1 for a, b in zip(y1, y2) if a == lbl and b == lbl)
        rater1_total = sum(1 for a in y1 if a == lbl)
        rater2_total = sum(1 for b in y2 if b == lbl)
        per_class[lbl] = {
            "agree": c1,
            "rater_a_count": rater1_total,
            "rater2_count": rater2_total,
        }

    # Disagreement detail
    disagreements = []
    for k, (a, b) in zip(common, zip(y1, y2)):
        if a != b:
            disagreements.append({
                "package": k[0],
                "version": k[1],
                "rater_a": a,
                "rater2": b,
            })

    # Verdict
    pass_ = kappa >= KAPPA_THRESHOLD
    verdict = "PASS" if pass_ else "FAIL"

    result = {
        "kappa": kappa,
        "kappa_ci_95": [ci_lo, ci_hi],
        "gwet_ac1": ac1,
        "max_marginal_prevalence": prevalence_max,
        "threshold": KAPPA_THRESHOLD,
        "verdict": verdict,
        "intersection_size": len(common),
        "labels": labels,
        "confusion_matrix": cm,
        "per_class": per_class,
        "disagreement_count": len(disagreements),
        "disagreements": disagreements,
        "bootstrap_n": n_boot,
    }
    OUT_JSON.write_text(json.dumps(result, indent=2))
    print(f"\n✓ Saved {OUT_JSON}", flush=True)

    # Human summary
    ci_lo_str = f"{ci_lo:.4f}" if ci_lo is not None else "n/a"
    ci_hi_str = f"{ci_hi:.4f}" if ci_hi is not None else "n/a"
    ac1_str = f"{ac1:.4f}" if ac1 is not None else "n/a"
    prev_str = f"{prevalence_max:.4f}" if prevalence_max is not None else "n/a"
    lines = [
        "=" * 60,
        f"Cohen's Kappa: {kappa:.4f}   (threshold {KAPPA_THRESHOLD})",
        f"  bootstrap 95% CI: [{ci_lo_str}, {ci_hi_str}] (n={n_boot})",
        f"Gwet AC1:         {ac1_str}   (prevalence-robust sensitivity)",
        f"Max marginal prevalence: {prev_str}",
        f"Verdict: {verdict}",
        f"Intersection: {len(common)} rows",
        f"Disagreements: {len(disagreements)} / {len(common)} ({len(disagreements)/len(common)*100:.1f}%)",
        "=" * 60,
        "",
        "Confusion matrix (rows: rater A, cols: rater B):",
        "        " + "  ".join(f"{l:>5}" for l in labels),
    ]
    for i, lbl in enumerate(labels):
        lines.append(f"  {lbl:5s} " + "  ".join(f"{cm[i][j]:>5}" for j in range(len(labels))))
    lines.append("")
    lines.append("Per-class counts:")
    for lbl, pc in per_class.items():
        lines.append(f"  {lbl}: rater_a={pc['rater_a_count']:>3} rater2={pc['rater2_count']:>3} agree={pc['agree']:>3}")
    summary_txt = "\n".join(lines)
    OUT_TXT.write_text(summary_txt + "\n")
    print("\n" + summary_txt, flush=True)
    return 0 if pass_ else 1


if __name__ == "__main__":
    sys.exit(main())
