from __future__ import annotations

import csv
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
RATER1_CSV = DATA_DIR / "classification.csv"
STEP21_CSV = DATA_DIR / "classification_step21.csv"
STATUS_CSV = DATA_DIR / "repo_status.csv"
WORKFLOWS_DIR = DATA_DIR / "workflows_raw"
OUT_CSV = DATA_DIR / "classification_rater2_input.csv"
OUT_WF_DIR = DATA_DIR / "workflows_for_rater2"

SAMPLE_SIZE = 100
MIN_PER_CLASS = 3
RNG = random.Random(42)


def main() -> int:

    rater1_class: dict[tuple[str, str], str] = {}
    with RATER1_CSV.open() as f:
        for r in csv.DictReader(f):
            rater1_class[(r["package"], r["version"])] = r["class"]
    print(f"Loaded {len(rater1_class)} primary-rater labels", flush=True)


    step21: dict[tuple[str, str], dict] = {}
    with STEP21_CSV.open() as f:
        for r in csv.DictReader(f):
            step21[(r["package"], r["version"])] = r


    repo_status: dict[tuple[str, str], dict] = {}
    if STATUS_CSV.exists():
        with STATUS_CSV.open() as f:
            for r in csv.DictReader(f):
                repo_status[(r["package"], r["version"])] = r


    by_class: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for k, c in rater1_class.items():
        by_class[c].append(k)

    print(f"\n=== Class distribution (rater_A) ===", flush=True)
    for c, items in sorted(by_class.items()):
        print(f"  {c}: {len(items)}", flush=True)


    sample_keys: list[tuple[str, str]] = []
    available_after_min: dict[str, list[tuple[str, str]]] = {}
    for cls, items in by_class.items():
        shuffled = items[:]
        RNG.shuffle(shuffled)
        take = min(MIN_PER_CLASS, len(shuffled))
        sample_keys.extend(shuffled[:take])
        available_after_min[cls] = shuffled[take:]

    remaining_budget = SAMPLE_SIZE - len(sample_keys)

    if remaining_budget > 0:

        pool: list[tuple[str, str]] = []
        for cls, items in available_after_min.items():
            pool.extend(items)
        RNG.shuffle(pool)
        sample_keys.extend(pool[:remaining_budget])


    if len(sample_keys) > SAMPLE_SIZE:
        sample_keys = sample_keys[:SAMPLE_SIZE]


    sample_keys = sorted(set(sample_keys))
    print(f"\nSelected {len(sample_keys)} rows for rater2", flush=True)


    if OUT_WF_DIR.exists():
        shutil.rmtree(OUT_WF_DIR)
    OUT_WF_DIR.mkdir(parents=True, exist_ok=True)

    out_rows: list[dict] = []
    copied = 0
    for (pkg, ver) in sample_keys:
        s21 = step21.get((pkg, ver), {})
        st = repo_status.get((pkg, ver), {})
        owner = s21.get("github_owner", "")
        repo = s21.get("github_repo", "")


        wf_files: list[str] = []
        if owner and repo:
            prefix = f"{owner}__{repo}__"
            for wf_path in WORKFLOWS_DIR.glob(f"{prefix}*"):
                rel = wf_path.name
                wf_files.append(rel)

                target = OUT_WF_DIR / rel
                if not target.exists():
                    shutil.copy2(wf_path, target)
                    copied += 1
        out_rows.append({
            "package": pkg,
            "version": ver,
            "repo_url": s21.get("repo_url", ""),
            "github_owner": owner,
            "github_repo": repo,
            "repo_status": st.get("repo_status", ""),
            "workflow_files": ";".join(sorted(wf_files)),
        })

    fields = ["package", "version", "repo_url", "github_owner", "github_repo",
              "repo_status", "workflow_files"]
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in out_rows:
            w.writerow(r)
    print(f"\n Saved {OUT_CSV} ({len(out_rows)} rows; NO class/reason)", flush=True)
    print(f" Copied {copied} unique workflow YAMLs to {OUT_WF_DIR}", flush=True)


    sample_cls = [rater1_class[k] for k in sample_keys]
    print(f"\n=== Sample class breakdown (audit only — not shown to rater2) ===", flush=True)
    seen: dict[str, int] = {}
    for c in sample_cls:
        seen[c] = seen.get(c, 0) + 1
    for c in sorted(seen):
        print(f"  {c}: {seen[c]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
