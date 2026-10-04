"""
Task 2 HUMAN CHECKPOINT-2 — Prep PI adjudication input for 43 UNCERTAIN cases.

For each of rater A's 43 UNCERTAIN cases, gather:
- package, version, repo info (owner, repo, repo_status)
- rater A's reason for UNCERTAIN
- rater B's verdict (if seen in the 100-sample) + reason
- list of workflow YAML files in this repo's cache (PI can read them locally)

Output: data/uncertain_pi_adjudication_input.csv
  fields: package, version, repo_url, owner, repo, repo_status,
          workflow_files, rater_a_reason, rater_b_verdict, rater_b_reason,
          final_class (BLANK — PI fills in)
"""
from __future__ import annotations

import csv
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
UNCERTAIN_CSV = DATA_DIR / "uncertain_cases.csv"
RATER2_CSV = DATA_DIR / "classification_rater2.csv"
STEP21_CSV = DATA_DIR / "classification_step21.csv"
STATUS_CSV = DATA_DIR / "repo_status.csv"
WORKFLOWS_DIR = DATA_DIR / "workflows_raw"
OUT_CSV = DATA_DIR / "uncertain_pi_adjudication_input.csv"


def main() -> int:
    # Load uncertain
    uncertain: list[dict] = []
    with UNCERTAIN_CSV.open() as f:
        for r in csv.DictReader(f):
            uncertain.append(r)
    print(f"Loaded {len(uncertain)} UNCERTAIN cases from rater A", flush=True)

    # Load rater2 (only their pkg,ver,class,reason)
    rater2: dict[tuple[str, str], dict] = {}
    with RATER2_CSV.open() as f:
        for r in csv.DictReader(f):
            rater2[(r["package"], r["version"])] = r

    # Load step21 for repo info
    step21: dict[tuple[str, str], dict] = {}
    with STEP21_CSV.open() as f:
        for r in csv.DictReader(f):
            step21[(r["package"], r["version"])] = r

    # Load repo_status
    repo_status: dict[tuple[str, str], dict] = {}
    with STATUS_CSV.open() as f:
        for r in csv.DictReader(f):
            repo_status[(r["package"], r["version"])] = r

    # Build output
    rows: list[dict] = []
    rater_b_resolved = 0
    rater_b_still_uncertain = 0
    not_seen_by_rater_b = 0
    for u in uncertain:
        key = (u["package"], u["version"])
        s21 = step21.get(key, {})
        st = repo_status.get(key, {})
        owner = s21.get("github_owner", "")
        repo = s21.get("github_repo", "")
        wfs: list[str] = []
        if owner and repo:
            prefix = f"{owner}__{repo}__"
            wfs = sorted(p.name for p in WORKFLOWS_DIR.glob(f"{prefix}*"))

        r2 = rater2.get(key)
        if r2:
            rater_b_v = r2["class"]
            rater_b_r = r2["reason"]
            if rater_b_v == "UNCERTAIN":
                rater_b_still_uncertain += 1
            else:
                rater_b_resolved += 1
        else:
            rater_b_v = ""
            rater_b_r = "(not in rater2 100-sample)"
            not_seen_by_rater_b += 1

        rows.append({
            "package": u["package"],
            "version": u["version"],
            "repo_url": s21.get("repo_url", ""),
            "owner": owner,
            "repo": repo,
            "repo_status": st.get("repo_status", ""),
            "workflow_files": ";".join(wfs),
            "rater_a_reason": u["reason"],
            "rater_b_verdict": rater_b_v,
            "rater_b_reason": rater_b_r,
            "final_class": "",   # PI fills
        })

    fields = ["package", "version", "repo_url", "owner", "repo", "repo_status",
              "workflow_files", "rater_a_reason", "rater_b_verdict", "rater_b_reason",
              "final_class"]
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n✓ Saved {OUT_CSV} ({len(rows)} rows)", flush=True)
    print(f"  rater B resolved (suggest A/B/C/D/E/F): {rater_b_resolved}", flush=True)
    print(f"  rater B still UNCERTAIN:                {rater_b_still_uncertain}", flush=True)
    print(f"  not seen by rater B (PI does fresh):    {not_seen_by_rater_b}", flush=True)
    print(f"\nPI: fill `final_class` column for ALL {len(rows)} rows; save in place.", flush=True)
    print(f"After PI signoff: run scripts/10_apply_pi_adjudication.py", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
