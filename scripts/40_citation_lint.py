from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
DEFAULT_PAPER_DIR = ROOT / "paper"


CITE_PATTERN = re.compile(
    r"\\cite[a-zA-Z]*\*?\s*"
    r"(?:\[[^\]]*\]\s*)*"
    r"\{([^}]+)\}"
)

PLACEHOLDER_TOKEN = "_PLACEHOLDER"


def scan_file(path: Path) -> list[tuple[int, str, str]]:
    findings: list[tuple[int, str, str]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"WARN: failed to read {path}: {e}", file=sys.stderr)
        return findings
    for match in CITE_PATTERN.finditer(text):
        key_blob = match.group(1)

        keys = [k.strip() for k in key_blob.split(",")]
        offending = [k for k in keys if PLACEHOLDER_TOKEN in k]
        if not offending:
            continue

        line_no = text[: match.start()].count("\n") + 1
        for k in offending:
            findings.append((line_no, match.group(0), k))
    return findings


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Reject \\cite{*_PLACEHOLDER*} in paper prose.")
    ap.add_argument("--paper-dir", default=str(DEFAULT_PAPER_DIR),
                    help="Root directory to scan (default: paper/)")
    ap.add_argument("--exclude", action="append", default=[],
                    help="Files to skip (relative to paper-dir). May be repeated.")
    args = ap.parse_args(argv)

    paper_dir = Path(args.paper_dir).resolve()
    if not paper_dir.exists():
        print(f"ERROR: paper directory not found: {paper_dir}", file=sys.stderr)
        return 2


    exclude_paths = {(paper_dir / e).resolve() for e in args.exclude}
    exclude_paths.add((paper_dir / "refs.bib").resolve())

    targets = []
    for ext in ("*.tex", "*.md"):
        for p in paper_dir.rglob(ext):
            if p.resolve() in exclude_paths:
                continue
            targets.append(p)

    total_findings = 0
    for path in sorted(targets):
        findings = scan_file(path)
        for line_no, full_cite, key in findings:
            rel = path.relative_to(paper_dir)
            print(f"FAIL  {rel}:{line_no}  placeholder key `{key}` "
                  f"referenced via `{full_cite}`", file=sys.stderr)
            total_findings += 1

    if total_findings:
        print(f"\nLint FAILED: {total_findings} placeholder citation(s) found "
              f"across {len(targets)} scanned file(s).", file=sys.stderr)
        print("Fix: verify the citation programmatically (Semantic Scholar / "
              "CrossRef / arXiv / Google Scholar), update paper/refs.bib (drop "
              "`_PLACEHOLDER` suffix), and move the row from Section B to "
              "Section A in review/citations_task_9_phase2.md.", file=sys.stderr)
        return 1

    print(f"OK: scanned {len(targets)} file(s); no placeholder citations found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
