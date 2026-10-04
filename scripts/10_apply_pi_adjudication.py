"""
Task 2 HUMAN CHECKPOINT-2 — Apply PI adjudication to classification.csv.

Reads data/uncertain_pi_adjudication_input.csv (after PI fills `final_class`)
and updates data/classification.csv with a **selective override policy**
(Round 2 methodologist Major-4 resolution):

  - When the current `class` in classification.csv is UNCERTAIN AND PI gave a
    non-UNCERTAIN `final_class`, overwrite class; append "; PI-adjudicated
    YYYY-MM-DD from UNCERTAIN to X" to reason.
  - When the current `class` is non-UNCERTAIN AND differs from PI's `final_class`,
    KEEP the classifier verdict and LOG the conflict to data/pi_conflicts.csv
    (the classifier may have access to evidence the PI lacked — e.g.,
    `.circleci/config.yml` probe results that drove class C).
  - When PI `final_class` matches the current class, or both are UNCERTAIN, no-op.

Validates: every PI `final_class` value ∈ {A,B,C,D,E,F,UNCERTAIN} (sanity).

After running, invoke `scripts/07_compute_kappa.py` to get post-adjudication κ
+ bootstrap CI + Gwet AC1 (writes to `reports/kappa_result.json` /
`reports/kappa_summary.txt`). The post-adjudication κ is concordance against
rater2 under the new selective policy, NOT a fresh independent reliability
estimate (the primary independent κ remains the v1-raw vs rater2 0.8099 in
`reports/kappa_independent.json`).

Run: conda run --no-capture-output -n stage2c python scripts/10_apply_pi_adjudication.py
"""
from __future__ import annotations

import csv
import datetime
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
PI_INPUT = DATA_DIR / "uncertain_pi_adjudication_input.csv"
CLASSIFICATION = DATA_DIR / "classification.csv"

VALID_CLASSES = {"A", "B", "C", "D", "E", "F", "UNCERTAIN"}


def main() -> int:
    today = datetime.datetime.now().strftime("%Y-%m-%d")

    # Read PI adjudication
    pi_map: dict[tuple[str, str], tuple[str, str]] = {}  # (pkg, ver) → (final_class, rater_b_reason)
    empty_count = 0
    bad_class = []
    with PI_INPUT.open() as f:
        for r in csv.DictReader(f):
            key = (r["package"], r["version"])
            fc = r["final_class"].strip()
            if not fc:
                empty_count += 1
                continue
            if fc not in VALID_CLASSES:
                bad_class.append((key, fc))
                continue
            pi_map[key] = (fc, r.get("rater_b_reason", ""))

    if bad_class:
        print("✗ Invalid `final_class` values; aborting:", file=sys.stderr)
        for key, fc in bad_class:
            print(f"  {key} → {fc!r}", file=sys.stderr)
        return 2

    if empty_count > 0:
        print(f"⚠ {empty_count} UNCERTAIN rows still have empty `final_class` — left as UNCERTAIN", flush=True)

    print(f"Loaded {len(pi_map)} PI adjudications from {PI_INPUT.name}", flush=True)

    # Read + update classification.
    # Update policy (Round 2 methodologist Major-4 alignment):
    #   - When the current classifier output is UNCERTAIN, PI's verdict is the only signal
    #     available → apply override.
    #   - When the current classifier output is non-UNCERTAIN AND matches PI → no change.
    #   - When the current classifier output is non-UNCERTAIN AND CONFLICTS with PI →
    #     keep the classifier's verdict (the rules have access to evidence the PI lacked
    #     during adjudication — e.g., `.circleci/config.yml` probe results that informed
    #     class C). Log the conflict for the report's precision audit.
    rows: list[dict] = []
    updated = 0
    conflicts: list[tuple[str, str, str, str]] = []  # (pkg, ver, classifier_class, pi_class)
    with CLASSIFICATION.open() as f:
        for r in csv.DictReader(f):
            key = (r["package"], r["version"])
            if key in pi_map:
                cur_class = r["class"]
                pi_class, _ = pi_map[key]
                if cur_class == "UNCERTAIN" and pi_class != "UNCERTAIN":
                    r["class"] = pi_class
                    r["reason"] = (r["reason"] + f"; PI-adjudicated {today} from UNCERTAIN to {pi_class}")[:500]
                    updated += 1
                elif cur_class != "UNCERTAIN" and pi_class != cur_class:
                    conflicts.append((key[0], key[1], cur_class, pi_class))
                    # NOTE: keep classifier verdict; flag for precision audit
                # else: cur_class == pi_class, or both UNCERTAIN — no-op
            rows.append(r)

    # Write back
    with CLASSIFICATION.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["package", "version", "class", "reason", "rater"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"✓ Updated {updated} UNCERTAIN→PI-class rows in {CLASSIFICATION.name}", flush=True)
    if conflicts:
        print(f"\n=== {len(conflicts)} classifier vs PI conflicts (classifier kept; PI verdict logged) ===", flush=True)
        for pkg, ver, cur, pi_c in conflicts[:20]:
            print(f"  {pkg:35s} ver={ver:20s} classifier={cur} pi={pi_c}", flush=True)
        if len(conflicts) > 20:
            print(f"  ... and {len(conflicts) - 20} more", flush=True)
        # Persist conflict log for the precision-audit table
        from pathlib import Path
        log_path = Path(__file__).parent.parent / "data" / "pi_conflicts.csv"
        with log_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["package", "version", "classifier_class", "pi_class"])
            for row in conflicts:
                w.writerow(row)
        print(f"  Saved {log_path}", flush=True)

    # Summary distribution after adjudication
    cnt: dict[str, int] = {}
    for r in rows:
        cnt[r["class"]] = cnt.get(r["class"], 0) + 1
    print(f"\n=== Class distribution post-adjudication ===", flush=True)
    for c in ("A", "B", "C", "D", "E", "F", "UNCERTAIN"):
        v = cnt.get(c, 0)
        print(f"  {c}: {v:>4} ({v/len(rows)*100:.1f}%)", flush=True)
    print(f"\nNext: run `python scripts/07_compute_kappa.py` for post-adjudication κ + CI + AC1.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
