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


    pi_map: dict[tuple[str, str], tuple[str, str]] = {}
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
            pi_map[key] = (fc, r.get("rater2_reason", ""))

    if bad_class:
        print(" Invalid `final_class` values; aborting:", file=sys.stderr)
        for key, fc in bad_class:
            print(f"  {key}  {fc!r}", file=sys.stderr)
        return 2

    if empty_count > 0:
        print(f" {empty_count} UNCERTAIN rows still have empty `final_class` — left as UNCERTAIN", flush=True)

    print(f"Loaded {len(pi_map)} PI adjudications from {PI_INPUT.name}", flush=True)


    rows: list[dict] = []
    updated = 0
    conflicts: list[tuple[str, str, str, str]] = []
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


            rows.append(r)


    with CLASSIFICATION.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["package", "version", "class", "reason", "rater"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f" Updated {updated} UNCERTAINPI-class rows in {CLASSIFICATION.name}", flush=True)
    if conflicts:
        print(f"\n=== {len(conflicts)} classifier vs PI conflicts (classifier kept; PI verdict logged) ===", flush=True)
        for pkg, ver, cur, pi_c in conflicts[:20]:
            print(f"  {pkg:35s} ver={ver:20s} classifier={cur} pi={pi_c}", flush=True)
        if len(conflicts) > 20:
            print(f"  ... and {len(conflicts) - 20} more", flush=True)

        from pathlib import Path
        log_path = Path(__file__).parent.parent / "data" / "pi_conflicts.csv"
        with log_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["package", "version", "classifier_class", "pi_class"])
            for row in conflicts:
                w.writerow(row)
        print(f"  Saved {log_path}", flush=True)


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
