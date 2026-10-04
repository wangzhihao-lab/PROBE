"""Pre-submission B1 — Composition decomposition of the pre/post rate change.

Locked question (PI pre-submission batch, 2026-06-10): does the tier-level
pre/post attestation-rate change come from WITHIN-package rate changes or
from BETWEEN-package composition (volume-mix shifts among stayers, plus
package entry/exit)? Strictly descriptive; this script makes NO causal
claim and cannot promote any Section-5 verdict.

Decomposition (exact additive identity, per tier and window):
    dR = R_post - R_pre
       = d_within + d_mix + d_entryexit
  where, over stayer packages S (>=1 version in both windows) with
  stayer-normalized version-share weights w_p and package rates r_p:
    d_within    = sum_S  mean(w_p) * (r_p_post - r_p_pre)   (Bennet)
    d_mix       = sum_S  mean(r_p) * (w_p_post - w_p_pre)   (Bennet)
    d_entryexit = (R_post - R_post_S) - (R_pre - R_pre_S)
  The Bennet midpoint identity makes d_within + d_mix exactly equal to
  R_post_S - R_pre_S, with no interaction residual.

Windows (both pre-declared; no post-hoc selection):
  prepost_8wk (primary)  — replicates Task 7 B4c exactly, including its
                           pre_end = DEADLINE - 1s boundary, so the
                           version counts must reproduce the published
                           6,755 / 6,486 top-tier sample sizes.
  full_window (secondary) — [ORIGIN, DEADLINE) vs [DEADLINE, CENSOR].

Inputs:
  data/versions.csv                          (frozen; cutoff 2026-05-10)
  data/depsdev_dependents_2026_05_05.csv     (primary tier frame)

Outputs:
  data/composition_decomposition.csv
  reports/presub_composition_decomposition.md

Carried-forward caveats: data cutoff 2026-05-10 (untouched); deps.dev
primary tier frame (top >500, mid >10, tail <=10 dependents); anti-causal
firewall applies — outputs are descriptive decompositions only.

Run: conda run --no-capture-output -n stage2c python scripts/45_composition_decomposition.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"

VERSIONS_CSV = DATA_DIR / "versions.csv"
DEPSDEV_CSV = DATA_DIR / "depsdev_dependents_2026_05_05.csv"
B4C_CSV = DATA_DIR / "baselines" / "task_7_b4_pre_post_contrast.csv"

OUT_CSV = DATA_DIR / "composition_decomposition.csv"
OUT_MD = REPORTS_DIR / "presub_composition_decomposition.md"

# Constants identical to scripts/37_baseline_time_models.py.
DEADLINE = pd.Timestamp("2025-12-09", tz="UTC")
ORIGIN = pd.Timestamp("2025-08-01", tz="UTC")
CENSOR = pd.Timestamp("2026-05-10 23:59:59", tz="UTC")
PREPOST_WEEKS = 8

TIERS = ["top", "mid", "tail"]


def dd_tier(n) -> str:
    if pd.isna(n):
        return "missing"
    n = int(n)
    return "top" if n > 500 else "mid" if n > 10 else "tail"


def load_versions_with_tier() -> pd.DataFrame:
    """Identical loading path to scripts/37 (Task 7 B4)."""
    versions = pd.read_csv(VERSIONS_CSV)
    versions["published_at"] = pd.to_datetime(versions["published_at"], utc=True, errors="coerce")
    versions = versions.dropna(subset=["published_at"])
    versions = versions[(versions["published_at"] >= ORIGIN) & (versions["published_at"] <= CENSOR)]
    versions["has_attestation_int"] = (
        versions["has_attestation"].astype(str).str.lower() == "true"
    ).astype(int)
    depsdev = pd.read_csv(DEPSDEV_CSV)
    dd_map = dict(zip(depsdev["name"], depsdev["dependents_count"]))
    versions["dd_tier"] = versions["package"].map(dd_map).apply(dd_tier)
    return versions


def window_bounds() -> Dict[str, Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]]:
    """Pre-declared window definitions; prepost_8wk replicates B4c exactly."""
    return {
        "prepost_8wk": {
            "pre": (DEADLINE - pd.Timedelta(weeks=PREPOST_WEEKS),
                    DEADLINE - pd.Timedelta(seconds=1)),
            "post": (DEADLINE, DEADLINE + pd.Timedelta(weeks=PREPOST_WEEKS)),
        },
        "full_window": {
            "pre": (ORIGIN, DEADLINE),
            "post": (DEADLINE, CENSOR + pd.Timedelta(seconds=1)),
        },
    }


def package_table(sub: pd.DataFrame) -> pd.DataFrame:
    """Per-package version counts and attested counts within one window."""
    g = (sub.groupby("package")
            .agg(n=("has_attestation_int", "size"),
                 k=("has_attestation_int", "sum"))
            .reset_index())
    g["rate"] = g["k"] / g["n"]
    return g


def decompose(pre: pd.DataFrame, post: pd.DataFrame) -> dict:
    """Exact within / mix / entry-exit decomposition of dR in pp units."""
    n_pre, n_post = int(pre["n"].sum()), int(post["n"].sum())
    r_pre = 100 * pre["k"].sum() / n_pre
    r_post = 100 * post["k"].sum() / n_post

    stayers = sorted(set(pre["package"]) & set(post["package"]))
    pre_s = pre[pre["package"].isin(stayers)].set_index("package")
    post_s = post[post["package"].isin(stayers)].set_index("package")
    n_pre_s, n_post_s = int(pre_s["n"].sum()), int(post_s["n"].sum())
    r_pre_s = 100 * pre_s["k"].sum() / n_pre_s
    r_post_s = 100 * post_s["k"].sum() / n_post_s

    w_pre = pre_s["n"] / n_pre_s
    w_post = post_s["n"] / n_post_s
    rr_pre = 100 * pre_s["rate"]
    rr_post = 100 * post_s["rate"]
    d_within = float((((w_pre + w_post) / 2) * (rr_post - rr_pre)).sum())
    d_mix = float((((rr_pre + rr_post) / 2) * (w_post - w_pre)).sum())
    d_entryexit = float((r_post - r_post_s) - (r_pre - r_pre_s))

    d_total = r_post - r_pre
    residual = d_total - (d_within + d_mix + d_entryexit)
    if abs(residual) > 1e-9:
        raise AssertionError(f"decomposition identity violated: residual={residual}")

    return {
        "n_pre_versions": n_pre, "n_post_versions": n_post,
        "rate_pre_pct": r_pre, "rate_post_pct": r_post,
        "delta_total_pp": d_total,
        "delta_within_pp": d_within,
        "delta_mix_pp": d_mix,
        "delta_entryexit_pp": d_entryexit,
        "n_stayers": len(stayers),
        "n_exiters": int((~pre["package"].isin(stayers)).sum()),
        "n_entrants": int((~post["package"].isin(stayers)).sum()),
        "stayer_rate_pre_pct": r_pre_s,
        "stayer_rate_post_pct": r_post_s,
        "stayer_share_pre_versions": n_pre_s / n_pre,
        "stayer_share_post_versions": n_post_s / n_post,
    }


def check_b4c_consistency(rows: List[dict]) -> None:
    """The prepost_8wk window must reproduce the published B4c counts."""
    b4c = pd.read_csv(B4C_CSV)
    for row in rows:
        if row["window"] != "prepost_8wk":
            continue
        ref_pre = b4c[(b4c["tier"] == row["tier"]) & (b4c["period"] == "pre_8wk")].iloc[0]
        ref_post = b4c[(b4c["tier"] == row["tier"]) & (b4c["period"] == "post_8wk")].iloc[0]
        if int(ref_pre["n_versions"]) != row["n_pre_versions"] or \
           int(ref_post["n_versions"]) != row["n_post_versions"]:
            raise AssertionError(
                f"B4c replication mismatch for {row['tier']}: "
                f"got {row['n_pre_versions']}/{row['n_post_versions']}, "
                f"expected {int(ref_pre['n_versions'])}/{int(ref_post['n_versions'])}"
            )
    logger.info("B4c consistency check passed (version counts reproduce).")


def main() -> int:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    versions = load_versions_with_tier()
    bounds = window_bounds()

    rows: List[dict] = []
    for window, wb in bounds.items():
        for tier in TIERS:
            tv = versions[versions["dd_tier"] == tier]
            pre = package_table(tv[(tv["published_at"] >= wb["pre"][0]) &
                                   (tv["published_at"] < wb["pre"][1])])
            post = package_table(tv[(tv["published_at"] >= wb["post"][0]) &
                                    (tv["published_at"] < wb["post"][1])])
            r = decompose(pre, post)
            r["window"], r["tier"] = window, tier
            rows.append(r)
            logger.info(
                "%s/%s: dR=%+.2f pp | within=%+.2f mix=%+.2f entry/exit=%+.2f "
                "(stayers=%d, entrants=%d, exiters=%d)",
                window, tier, r["delta_total_pp"], r["delta_within_pp"],
                r["delta_mix_pp"], r["delta_entryexit_pp"],
                r["n_stayers"], r["n_entrants"], r["n_exiters"],
            )

    check_b4c_consistency(rows)

    fields = ["window", "tier", "n_pre_versions", "n_post_versions",
              "rate_pre_pct", "rate_post_pct", "delta_total_pp",
              "delta_within_pp", "delta_mix_pp", "delta_entryexit_pp",
              "n_stayers", "n_exiters", "n_entrants",
              "stayer_rate_pre_pct", "stayer_rate_post_pct",
              "stayer_share_pre_versions", "stayer_share_post_versions"]
    pd.DataFrame(rows)[fields].to_csv(OUT_CSV, index=False)
    logger.info("Saved %s", OUT_CSV)

    lines = [
        "# Pre-submission B1 — Composition decomposition of the pre/post rate change",
        "",
        "Locked question: within-package rate change vs between-package composition",
        "(stayer volume-mix + entry/exit). Exact additive identity; descriptive only;",
        "no causal claim; data cutoff 2026-05-10 untouched.",
        "",
        "| Window | Tier | ΔR (pp) | within (pp) | mix (pp) | entry/exit (pp) | stayers | entrants | exiters |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['window']} | {r['tier']} | {r['delta_total_pp']:+.2f} | "
            f"{r['delta_within_pp']:+.2f} | {r['delta_mix_pp']:+.2f} | "
            f"{r['delta_entryexit_pp']:+.2f} | {r['n_stayers']} | "
            f"{r['n_entrants']} | {r['n_exiters']} |"
        )
    lines += [
        "",
        "Stayer-only aggregate rates (balanced-panel check):",
        "",
        "| Window | Tier | stayer pre % | stayer post % | aggregate pre % | aggregate post % |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['window']} | {r['tier']} | {r['stayer_rate_pre_pct']:.2f} | "
            f"{r['stayer_rate_post_pct']:.2f} | {r['rate_pre_pct']:.2f} | "
            f"{r['rate_post_pct']:.2f} |"
        )
    lines += [
        "",
        "Consistency: prepost_8wk reproduces the Task 7 B4c version counts exactly",
        "(assertion-checked against data/baselines/task_7_b4_pre_post_contrast.csv).",
        "",
        "Interpretation boundary: these components describe WHERE the aggregate",
        "rate change sits (same packages changing vs the publishing mix changing).",
        "They do not identify any cause of either component.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")
    logger.info("Saved %s", OUT_MD)
    return 0


if __name__ == "__main__":
    sys.exit(main())
