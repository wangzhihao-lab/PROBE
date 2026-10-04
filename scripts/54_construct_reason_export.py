"""Additive final-construct-language export for the 500-row classification.

External review (round 11) observed that `data/classification.csv` retains the
rule engine's historical `reason` shorthand ("OIDC failed", "OIDC mis-config",
"classic publish"), which predates the paper's final construct language and,
read alone, contradicts it. Rewriting those trace strings would falsify the
record of what the classifier printed, so this script instead emits a NEW
derived table that carries the final construct phrasing per row, keyed to the
same (package, version, class). The original CSV is not modified.

Mapping (paper Section 4.3 / Table 2 semantics; see data/CODEBOOK_classification.md):
  A -> long-lived token secret referenced at publish job; no id-token: write;
       token class (classic vs granular) not observable from the YAML
  B -> workflow-side id-token: write capability present AND sampled version
       non-attested: capability/outcome mismatch state; publish-time path and
       npm-side trusted-publisher registration not observable
  C -> possible external CI publish path (.circleci/config.yml or
       .gitlab-ci.yml present at repo root); presence-only, not verified
  D -> repository deleted, private, or 4xx; publish workflow not inspectable
  E -> no visible package-publish workflow in inspected sources; NOT a
       manual-publish claim
  F -> no repository.url or non-GitHub host; excluded from A--E denominators

Inputs:  data/classification.csv  (frozen; NOT modified)
Outputs: data/classification_construct_reasons.csv
Carried-forward caveats: collection-time evidence states; anti-causal
firewall; data cutoff 2026-05-10 untouched.

Run: conda run --no-capture-output -n stage2c python scripts/54_construct_reason_export.py
"""
from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA = Path(__file__).parent.parent / "data"
IN_CSV = DATA / "classification.csv"
OUT_CSV = DATA / "classification_construct_reasons.csv"

CONSTRUCT = {
    "A": ("long-lived token secret referenced at publish job, no id-token: "
          "write; token class (classic vs granular) not observable from YAML"),
    "B": ("workflow-side id-token: write capability present AND sampled "
          "version non-attested: capability/outcome mismatch state; "
          "publish-time path and npm-side trusted-publisher registration "
          "not observable"),
    "C": ("possible external CI publish path (.circleci/config.yml or "
          ".gitlab-ci.yml at repo root); presence-only hypothesis, config "
          "not parsed to verify npm publishing"),
    "D": ("repository deleted, private, or 4xx from GitHub Contents API; "
          "publish workflow not inspectable"),
    "E": ("no visible package-publish workflow in inspected sources; not a "
          "manual-publish claim"),
    "F": ("no repository.url or non-GitHub host; excluded from A--E "
          "denominators"),
}


def main() -> int:
    rows = list(csv.DictReader(open(IN_CSV, encoding="utf-8")))
    if len(rows) != 500:
        logger.error("expected 500 rows, got %d", len(rows))
        return 1
    unknown = sorted({r["class"] for r in rows} - set(CONSTRUCT))
    if unknown:
        logger.error("unmapped classes: %s", unknown)
        return 1
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["package", "version", "class", "construct_reason",
                    "historical_reason_note"])
        for r in rows:
            w.writerow([r["package"], r["version"], r["class"],
                        CONSTRUCT[r["class"]],
                        "original rule-engine trace retained in "
                        "classification.csv `reason`; see "
                        "CODEBOOK_classification.md"])
    counts = {}
    for r in rows:
        counts[r["class"]] = counts.get(r["class"], 0) + 1
    logger.info("wrote %s (500 rows; class counts %s); classification.csv "
                "untouched", OUT_CSV.name, dict(sorted(counts.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
