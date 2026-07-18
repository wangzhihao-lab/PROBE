from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from xml.etree import ElementTree as ET

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
MONITORING_DIR = ROOT / "monitoring"


ARXIV_API = "https://export.arxiv.org/api/query"


ARXIV_COMBINED_QUERY = (
    '("trusted publishing" OR "npm provenance" OR "classic token" OR '
    '"dist.attestations" OR OIDC OR sigstore OR rekor OR '
    '"release authority" OR "publish path" OR "publishing path" OR '
    '"package attestation")'
)


QUERY_GROUP_RULES = [
    ("trusted_publishing", ["trusted publishing"]),
    ("npm_provenance", ["npm provenance", "provenance npm"]),
    ("classic_token", ["classic token"]),
    ("dist_attestations_oidc", ["dist.attestations", "oidc"]),
    ("sigstore_rekor", ["sigstore", "rekor"]),
    ("release_authority_path", ["release authority", "publish path",
                                "publishing path", "package attestation"]),
]


ESCALATE_TERMS_PRIMARY = ["provenance", "trusted publishing", "classic token", "attestation", "OIDC"]
ESCALATE_TERMS_SECONDARY = ["migration", "adoption", "interrupted time series", "difference-in-differences", "survival"]
ESCALATE_LABS = ["Williams", "Davis", "Torres-Arias", "Enck"]


def arxiv_search(query: str, max_results: int = 50, categories: tuple = ("cs.SE", "cs.CR"),
                  max_retries: int = 3) -> list[dict]:
    cat_filter = " OR ".join(f"cat:{c}" for c in categories)
    full_query = f"({cat_filter}) AND all:{query}"
    params = {
        "search_query": full_query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    url = f"{ARXIV_API}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "stage2c-task8-monitor"})
    xml_bytes = None
    arxiv_search._last_failed = False
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                xml_bytes = resp.read()
            break
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < max_retries - 1:
                backoff = 15 * (attempt + 1)
                logger.info(f"arxiv 429 on '{query}'; backing off {backoff}s (attempt {attempt+1}/{max_retries})")
                time.sleep(backoff)
                continue
            logger.warning(f"arxiv query failed for '{query}': {e}")
            arxiv_search._last_failed = True
            return []
        except Exception as e:
            logger.warning(f"arxiv query failed for '{query}': {e}")
            arxiv_search._last_failed = True
            return []
    if xml_bytes is None:
        arxiv_search._last_failed = True
        return []

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as e:
        logger.warning(f"arxiv XML parse failed for '{query}': {e}")
        return []

    papers = []
    for entry in root.findall("atom:entry", ns):
        pid_full = entry.findtext("atom:id", default="", namespaces=ns).strip()

        m = re.search(r"abs/(\d{4}\.\d{5})", pid_full)
        arxiv_id = m.group(1) if m else pid_full
        title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip().replace("\n", " ")
        summary = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip()
        published = entry.findtext("atom:published", default="", namespaces=ns)
        authors = [a.findtext("atom:name", default="", namespaces=ns)
                   for a in entry.findall("atom:author", ns)]
        papers.append({
            "arxiv_id": arxiv_id,
            "title": title,
            "summary_excerpt": summary[:400],
            "published": published,
            "authors": authors,
            "url": pid_full,
        })
    return papers


def assess_escalation(paper: dict) -> list[str]:
    flags = []
    title_lower = paper["title"].lower()

    if "npm" in title_lower:
        if any(term.lower() in title_lower for term in ESCALATE_TERMS_PRIMARY):
            flags.append("ESCALATE_A: npm + primary-term in title")
        if any(term.lower() in title_lower for term in ESCALATE_TERMS_SECONDARY):
            flags.append("ESCALATE_B: npm + secondary-term in title")

    authors_concat = " ".join(paper.get("authors", [])).lower()
    for lab in ESCALATE_LABS:
        if lab.lower() in authors_concat and ("npm" in title_lower or "sigstore" in title_lower):
            flags.append(f"ESCALATE_D: lab {lab} touching npm/sigstore")
            break
    return flags


def filter_recent(papers: list[dict], days: int = 30) -> list[dict]:
    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=days)
    out = []
    for p in papers:
        try:
            pub = datetime.fromisoformat(p["published"].replace("Z", "+00:00"))
        except Exception:
            continue
        if pub >= cutoff:
            out.append(p)
    return out


def load_prior_arxiv_ids(current_raw_path: Path) -> set[str]:
    seen = set()
    for f in sorted(MONITORING_DIR.glob("arxiv_raw_*.json")):
        if f.resolve() == current_raw_path.resolve():
            continue
        if "FIXTURE" in f.name:
            continue
        try:
            for p in json.loads(f.read_text()):
                seen.add(p.get("arxiv_id", ""))
        except Exception:
            continue
    return seen


def write_readme():
    readme = MONITORING_DIR / "README.md"
    if readme.exists():
        return
    readme.write_text("""# Monitoring (Task 8) — monthly collision watch

Cadence: monthly. Near end of each month, run:

```bash
conda run --no-capture-output -n stage2c python scripts/39_monthly_collision_monitor.py
```

The script issues a single combined arXiv query (cs.SE + cs.CR) covering all
5 D-4 keyword groups (trusted publishing, npm provenance, classic token,
dist.attestations / OIDC, sigstore / rekor) and writes:

- `monitoring/collision_scan_YYYY-MM.md` — human-readable monthly report
  (auto-arxiv + manual checklist + diff vs previous month)
- `monitoring/arxiv_raw_YYYY-MM.json` — machine-readable arXiv hit list

Then **manually** complete the checklist section (Scholar / s3c2 /
conference pages) and sign off by PI. Escalation rules: see
`dryrun_baseline/D-4_collision_scan/monitoring_setup.md`.

## Flags

- `--month YYYY-MM` — override the target month (default: current month)
- `--days N` — lookback window in days for the "recent" cut (default: 30)
- `--max-results N` — max arXiv hits for the single combined query (default: 50)
- `--overwrite` — allow overwriting a non-empty same-month raw file with an
  empty failure result. Without this flag, an arXiv failure preserves the
  previously successful raw file.
- `--fixture` — skip arXiv API and use a hardcoded sample paper list to
  exercise attribution + escalation + diff logic. Writes to
  `monitoring/arxiv_raw_YYYY-MM_FIXTURE.json` + `collision_scan_YYYY-MM.md`
  with the original month label. Fixture files are excluded from
  prior-month diff load.

## Per closure

The dryrun D-4 baseline established 9 adjacent papers (0 collisions) as of
2026-05-14. Subsequent months should diff against the union of all prior
monthly files, not just May 2026. The diff loader strictly excludes the
current month's file and any FIXTURE file.
""")
    logger.info(f"wrote {readme}")


def write_report(month: str, all_papers: list[dict], recent_papers: list[dict],
                 new_in_month: list[dict], escalations: list[dict],
                 arxiv_unavailable: bool = False):
    out = MONITORING_DIR / f"collision_scan_{month}.md"
    lines = [
        f"# Monthly Collision Scan — {month}",
        "",
        f"**Generated**: {datetime.now(tz=timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
        f"**Auto-query source**: arXiv API (cs.SE + cs.CR)",
        f"**Window**: last 30 days submitted",
        f"**Diff baseline**: union of `monitoring/arxiv_raw_*.json` prior to this month",
        "",
        "## Auto-arXiv results (last 30 days)",
        "",
    ]
    if arxiv_unavailable:
        lines += [
            " **arXiv API was unavailable this run** (HTTP 429 / timeout after exponential backoff). "
            "Skip auto-arXiv this cycle; **manual checklist below must cover the gap**. Re-run later "
            "if needed: `conda run --no-capture-output -n stage2c python scripts/39_monthly_collision_monitor.py --month "
            f"{month}`.",
            "",
        ]
    lines += [
        f"- Total arXiv hits (combined query): **{len(all_papers)}**",
        f"- Submitted in last 30 days: **{len(recent_papers)}**",
        f"- New (not seen in any prior month's file): **{len(new_in_month)}**",
        f"- Escalation-flagged: **{len(escalations)}**",
        "",
    ]
    if escalations:
        lines += ["###  Escalations (review immediately)", ""]
        for p in escalations:
            lines.append(f"- arxiv:{p['arxiv_id']} — {p['title']}")
            lines.append(f"  - Authors: {', '.join(p['authors'])}")
            lines.append(f"  - Flags: {', '.join(p['_flags'])}")
            lines.append(f"  - URL: {p['url']}")
            lines.append(f"  - Excerpt: {p['summary_excerpt'][:200]}...")
            lines.append("")
    else:
        lines += ["###  No escalation-flagged papers this month", ""]
    if new_in_month:
        lines += ["### New papers (last 30 days, not in prior monthly union)", ""]
        for p in new_in_month:
            flags = p.get("_flags", [])
            flag_str = " " if flags else ""
            lines.append(f"- arxiv:{p['arxiv_id']}{flag_str} — {p['title']}")
            lines.append(f"  - Authors: {', '.join(p['authors'])}")
            lines.append(f"  - URL: {p['url']}")
            lines.append("")
    lines += [
        "## Manual checklist (must complete each month)",
        "",
        "- [ ] Google Scholar 5 queries (see `dryrun_baseline/D-4_collision_scan/monitoring_setup.md`)",
        "- [ ] s3c2.org/publications (retry if last month was 404)",
        "- [ ] ICSE / FSE / USENIX Security accepted-papers page diffs",
        "- [ ] TOSEM / EMSE / TSE Online First page diff",
        "- [ ] ESEC/FSE journal-first ingestion list",
        "- [ ] Williams / Davis / Torres-Arias / Enck lab pages (highest collision risk)",
        "",
        "## PI sign-off",
        "",
        "- [ ] Reviewed by PI",
        "- [ ] Verdict: NO_COLLISION / NEW_ADJACENT / ESCALATE",
        "- [ ] Sign date: ____",
        "",
        "## NEXT",
        "",
        "Next monthly run: ~30 days from generation date. If any escalation flag fires, "
        "fetch full abstract within 24h and forward to PI immediately (do not wait for "
        "monthly cadence).",
    ]
    out.write_text("\n".join(lines) + "\n")
    logger.info(f"wrote {out}")


def _fixture_papers() -> list[dict]:
    today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return [
        {
            "arxiv_id": "9999.00001",
            "title": "An Empirical Study of npm Trusted Publishing Adoption",
            "summary_excerpt": "We measure trusted publishing adoption across the npm registry.",
            "published": today,
            "authors": ["Doe, Jane"],
            "url": "http://arxiv.org/abs/9999.00001",
        },
        {
            "arxiv_id": "9999.00002",
            "title": "npm Provenance: A Sigstore Adoption Survey",
            "summary_excerpt": "Cross-ecosystem look at sigstore rekor adoption in npm and PyPI.",
            "published": today,
            "authors": ["Smith, John"],
            "url": "http://arxiv.org/abs/9999.00002",
        },
        {
            "arxiv_id": "9999.00003",
            "title": "Supply-chain Migration After npm Classic Token Revocation",
            "summary_excerpt": "Difference-in-differences on npm publishing behavior.",
            "published": today,
            "authors": ["Williams, Laurie"],
            "url": "http://arxiv.org/abs/9999.00003",
        },
        {
            "arxiv_id": "9999.00004",
            "title": "Generic OIDC Hardening for CI Pipelines",
            "summary_excerpt": "OIDC patterns for GitHub Actions, no npm focus.",
            "published": today,
            "authors": ["Generic, Author"],
            "url": "http://arxiv.org/abs/9999.00004",
        },
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", default=date.today().strftime("%Y-%m"),
                    help="YYYY-MM (default: current month)")
    ap.add_argument("--days", type=int, default=30, help="Lookback window for 'recent' (default 30)")
    ap.add_argument("--max-results", type=int, default=50,
                    help="Max arXiv results for the single combined query (default 50)")
    ap.add_argument("--overwrite", action="store_true",
                    help="Allow overwriting a non-empty same-month raw file with an empty "
                    "failure result. Without this flag, an arXiv failure preserves the "
                    "previously successful raw file.")
    ap.add_argument("--fixture", action="store_true",
                    help="Run in fixture mode: skip arXiv API and use a hardcoded sample "
                    "paper list to exercise attribution + escalation + diff. Writes a "
                    "report tagged with [FIXTURE] for review.")
    args = ap.parse_args()

    MONITORING_DIR.mkdir(parents=True, exist_ok=True)
    write_readme()

    if args.fixture:
        logger.info("FIXTURE MODE — skipping arXiv; using hardcoded sample papers")
        all_papers = _fixture_papers()
        arxiv_unavailable = False
    else:
        logger.info(f"arxiv combined query: {ARXIV_COMBINED_QUERY}")
        all_papers = arxiv_search(ARXIV_COMBINED_QUERY, max_results=args.max_results)
        arxiv_unavailable = (len(all_papers) == 0
                              and getattr(arxiv_search, "_last_failed", False))

    for p in all_papers:
        text = (p["title"] + " " + p.get("summary_excerpt", "")).lower()
        matched_groups = [g for g, kws in QUERY_GROUP_RULES
                          if any(kw.lower() in text for kw in kws)]
        p["_query_group"] = ",".join(matched_groups) if matched_groups else "other"

    logger.info(f"arXiv hits: {len(all_papers)}")
    raw_path = MONITORING_DIR / f"arxiv_raw_{args.month}.json"


    if (arxiv_unavailable and not args.fixture and raw_path.exists()
            and not args.overwrite):
        try:
            existing = json.loads(raw_path.read_text())
        except Exception:
            existing = []
        if existing:
            logger.warning(
                f"arXiv unavailable AND existing {raw_path.name} has "
                f"{len(existing)} hits — refusing to overwrite. "
                "Re-run with --overwrite to replace, or fix arXiv connectivity.")
        else:
            raw_path.write_text(json.dumps(all_papers, indent=2))
            logger.info(f"wrote {raw_path}")
    else:
        if args.fixture:
            raw_path = MONITORING_DIR / f"arxiv_raw_{args.month}_FIXTURE.json"
        raw_path.write_text(json.dumps(all_papers, indent=2))
        logger.info(f"wrote {raw_path}")

    recent_papers = filter_recent(all_papers, days=args.days)
    logger.info(f"submitted within last {args.days} days: {len(recent_papers)}")

    prior_ids = load_prior_arxiv_ids(raw_path)
    new_in_month = [p for p in recent_papers if p["arxiv_id"] not in prior_ids]
    for p in new_in_month:
        p["_flags"] = assess_escalation(p)
    escalations = [p for p in new_in_month if p.get("_flags")]

    write_report(args.month, all_papers, recent_papers, new_in_month, escalations,
                 arxiv_unavailable=arxiv_unavailable)

    print(f"\n=== Monthly summary ({args.month}) ===")
    print(f"  arXiv hits total: {len(all_papers)}")
    print(f"  Recent (last {args.days}d): {len(recent_papers)}")
    print(f"  New this month: {len(new_in_month)}")
    print(f"  Escalations: {len(escalations)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
