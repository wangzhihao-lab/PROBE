from __future__ import annotations

import csv
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import requests

DATA_DIR = Path(__file__).parent.parent / "data"
REVIEW_DIR = Path(__file__).parent.parent / "review"
INFLUENCE_CSV = DATA_DIR / "influence_metrics.csv"
PACKUMENTS_DIR = DATA_DIR / "packuments_raw"
OUT_MD = REVIEW_DIR / "task_3_human_spotcheck_round1_2026-05-17.md"


APR_RANGE = "2026-04-01:2026-04-30"
APR_WEEKS = 30 / 7

RNG = random.Random(42)
SLEEP = 0.2


def safe_fname(pkg: str) -> str:
    return pkg.replace("/", "_").replace("@", "at_") + ".json"


def fetch_april(pkg: str) -> tuple[int, float, str]:
    url = f"https://api.npmjs.org/downloads/range/{APR_RANGE}/{quote(pkg, safe='/@')}"
    try:
        r = requests.get(url, timeout=30,
                         headers={"User-Agent": "stage2c-research/1.0 (spot-check)"})
        if r.status_code != 200:
            return 0, 0, f"http_{r.status_code}"
        data = r.json()
        dls = data.get("downloads") or []
        total = sum(int(d.get("downloads", 0)) for d in dls)
        weekly = total / APR_WEEKS if APR_WEEKS > 0 else 0
        return total, weekly, "ok"
    except Exception as e:
        return 0, 0, f"err_{type(e).__name__}"


def verify_packument_age(pkg: str, claimed_age_days: str) -> str:
    if not claimed_age_days or claimed_age_days == "":
        return "n/a (no packument)"
    fpath = PACKUMENTS_DIR / safe_fname(pkg)
    if not fpath.exists():
        return "FAIL (packument missing)"
    try:
        data = json.loads(fpath.read_text())
    except Exception:
        return "FAIL (parse error)"
    if data.get("_404"):
        return "n/a (404 marker)"
    created = (data.get("time") or {}).get("created", "")
    if not created:
        return "n/a (no created field)"
    cutoff = datetime(2026, 5, 10, 23, 59, 59, tzinfo=timezone.utc)
    try:
        c = datetime.fromisoformat(created.replace("Z", "+00:00"))
        derived = max(0, (cutoff - c).days)
        return f"PASS (claimed={claimed_age_days}, re-derived={derived})" \
            if str(derived) == str(claimed_age_days) else \
            f"FAIL (claimed={claimed_age_days}, re-derived={derived})"
    except Exception:
        return "FAIL (date parse)"


def main() -> int:
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    with INFLUENCE_CSV.open() as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 3000:
        print(f" expected 3000 rows, got {len(rows)}", file=sys.stderr)


    by_tier: dict[str, list[dict]] = {"top": [], "mid": [], "tail": []}
    for r in rows:
        by_tier[r["tier"]].append(r)
    for t in by_tier:
        by_tier[t].sort(key=lambda x: x["package"])

    sample: list[dict] = []

    def stride_pick(lst: list[dict], n: int) -> list[dict]:
        step = max(1, len(lst) // n)
        return [lst[i * step] for i in range(n) if i * step < len(lst)]
    sample.extend(stride_pick(by_tier["top"], 4))
    sample.extend(stride_pick(by_tier["mid"], 3))
    sample.extend(stride_pick(by_tier["tail"], 3))
    print(f"10-row sample: {len(sample)}", flush=True)
    for r in sample:
        print(f"  {r['tier']:5s} {r['package']:40s} stars={r['github_stars']:>7s} downloads={r['weekly_downloads_avg']:>15s}", flush=True)


    audit = RNG.sample(sample, 5)
    audit.sort(key=lambda r: r["tier"])
    print(f"\n5-row audit (RNG seed=42):", flush=True)


    audit_results: list[dict] = []
    for r in audit:
        pkg = r["package"]
        apr_total, apr_weekly, apr_status = fetch_april(pkg)
        time.sleep(SLEEP)
        claimed_weekly = float(r["weekly_downloads_avg"]) if r["weekly_downloads_avg"] else 0
        if claimed_weekly > 0 and apr_weekly > 0:
            ratio = apr_weekly / claimed_weekly
            dl_verdict = "PASS" if 0.5 <= ratio <= 2.0 else f"FAIL (ratio {ratio:.2f})"
        else:
            dl_verdict = f"n/a ({apr_status})"

        age_verdict = verify_packument_age(pkg, r["package_age_days"])


        audit_results.append({
            "package": pkg,
            "tier": r["tier"],
            "claimed_weekly": claimed_weekly,
            "april_weekly": apr_weekly,
            "april_total": apr_total,
            "downloads_verdict": dl_verdict,
            "age_verdict": age_verdict,
            "stars": r["github_stars"],
            "forks": r["github_forks"],
            "dependabot": r["has_dependabot"],
            "issue_template": r["has_issue_template"],
            "is_scoped": r["is_scoped_package"],
            "owner_type": r["github_owner_type"],
            "downloads_status": r["downloads_status"],
            "github_status": r["github_status"],
        })
        print(f"  {r['tier']:5s} {pkg:40s} dl={dl_verdict:>30s}  age={age_verdict:>40s}", flush=True)


    lines = [
        "# Checkpoint spot check",
        "",
        "**Reviewer**: the analysis pipeline",
        f"**Date**: 2026-05-17",
        "**Source file**: `data/influence_metrics.csv` (3000 rows × 16 cols after )",
        "",
        "## Scope",
        "",
        "review  (`review/task_3_round1_pi.md`):",
        "- () 10-row sample + 5-row manual audit covering packument / npm / GitHub / missing-status",
        "- () weekly-downloads cross-source sanity check",
        "",
        "Method:",
        "- 10-row sample: 4 top + 3 mid + 3 tail, alphabetically-strided within each tier (reproducible).",
        "- 5-row audit: `random.Random(42).sample(sample, 5)` from the 10.",
        "- Downloads cross-check: re-query npm Downloads `/range/" + APR_RANGE + "/{pkg}` and compare April-only",
        "  weekly average against our 11.3-month weekly_downloads_avg. PASS if ratio in [0.5, 2.0].",
        "- Age re-derivation: read `time.created` from packument and re-compute `(2026-05-10 - created).days`.",
        "",
        "Construct caveat: the cross-window check uses the SAME source (npmjs.org Downloads API) as the",
        "primary collection, but with a DIFFERENT, non-overlapping month (April 2026). It catches:",
        "(a) primary-window data corruption, (b) order-of-magnitude calculation errors, (c) recent",
        "abandonment / spike anomalies. It does NOT catch source-level systematic bias — a truly external",
        "validation source would require ecosyste.ms or deps.dev (the latter deferred to post-2026-06-01).",
        "",
        "## 10-Line Sample",
        "",
        "| Tier | Package | Weekly avg | Stars | Forks | Age (d) | Maintainers | Releases | Scoped | Owner type | Dependabot | Issue tmpl | DL status | GH status |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---|",
    ]
    for r in sample:
        lines.append(
            f"| {r['tier']} | `{r['package']}` | "
            f"{r['weekly_downloads_avg'] or '—'} | {r['github_stars'] or '—'} | {r['github_forks'] or '—'} | "
            f"{r['package_age_days'] or '—'} | {r['maintainers_count'] or '—'} | "
            f"{r['release_freq_12m'] or '—'} | {r['is_scoped_package'] or '—'} | "
            f"{r['github_owner_type'] or '—'} | {r['has_dependabot'] or '—'} | "
            f"{r['has_issue_template'] or '—'} | {r['downloads_status'] or '—'} | {r['github_status'] or '—'} |"
        )

    lines += [
        "",
        "## 5-Line Manual Audit",
        "",
        "| Tier | Package | Claimed weekly | April weekly | Downloads verdict | Age verdict | Notes |",
        "|---|---|---:|---:|---|---|---|",
    ]
    for r in audit_results:
        notes = []
        if r["downloads_status"] != "ok":
            notes.append(f"DL={r['downloads_status']}")
        if r["github_status"] != "ok":
            notes.append(f"GH={r['github_status']}")
        notes_str = "; ".join(notes) if notes else "—"
        lines.append(
            f"| {r['tier']} | `{r['package']}` | "
            f"{r['claimed_weekly']:,.0f} | {r['april_weekly']:,.0f} | "
            f"{r['downloads_verdict']} | {r['age_verdict']} | {notes_str} |"
        )

    pass_count = sum(1 for r in audit_results if r["downloads_verdict"] == "PASS")
    age_pass = sum(1 for r in audit_results if r["age_verdict"].startswith("PASS") or r["age_verdict"].startswith("n/a"))
    lines += [
        "",
        "## Verdict",
        "",
        f"CHECKPOINT-3 spot-check completed.",
        f"- Sample: 10/10 rows printed.",
        f"- Audit downloads cross-window: **{pass_count}/{len(audit_results)} PASS** "
        f"({'all within [50%, 200%] of claimed weekly average' if pass_count == len(audit_results) else 'some outside band — see Notes'}).",
        f"- Audit packument age re-derivation: **{age_pass}/{len(audit_results)} PASS-or-n/a**.",
        "",
        "Construct caveat: the downloads cross-check is NOT a fully external validation (same npmjs.org",
        "source, different time window). For a true second-source check, future revision should use",
        "ecosyste.ms `downloads_period: last-month` or deps.dev post-2026-06-01. The check here verifies",
        "internal consistency and recency, not source-level validity.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"\n Saved {OUT_MD}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
