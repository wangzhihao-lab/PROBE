"""Pre-submission — Estimand sensitivity of the tier-level attestation rates
(acting-PI batch, 2026-07-11; frozen rows only, no new collection).

Motivation: the RQ1/Task-7 headline rates are VERSION-weighted, so a handful
of high-velocity publishers can dominate a tier (top tier: the @opencode-ai/*
pair alone contributes ~44% of version rows). This script recomputes the
tier-level adoption picture under alternative estimands and asks whether the
8-week pre/post contrast around the 2025-12-09 deadline is estimand-robust.

Estimands (per deps.dev tier; full window AND 8-week pre vs 8-week post):
  (1) version-weighted attestation rate          (current paper estimand)
  (2) equal-package mean of per-package rates    (each package weighted 1)
  (3) any-attested package share                 (>=1 attested version)
  (4) stable-only version-weighted rate          (drop semver prereleases,
                                                  i.e. version contains '-')
  (5) top tier only: version-weighted rate excluding the 5 packages with the
      most top-tier rows (exclusion set fixed on full-window counts).

Contrasts: 8-week post-minus-pre difference per tier under estimands
(1)/(2)/(4), each with a package-cluster bootstrap 95% percentile CI
(resample packages with replacement, B=2000, seed 42; identical package
resamples reused across the three estimands within a tier).

Windows: pre = [2025-10-14, 2025-12-09), post = [2025-12-09, 2026-02-03);
these reproduce the Task 7 B4c version counts exactly (assertion-checked
against data/baselines/task_7_b4_pre_post_contrast.csv). Full window = all
frozen rows [2025-08-01, 2026-05-10].

Inputs:
  data/versions.csv                              (frozen; cutoff 2026-05-10)
  data/depsdev_dependents_2026_05_05.csv         (primary tier frame)
  data/baselines/task_7_b4_pre_post_contrast.csv (B4c consistency check)

Outputs:
  data/estimand_sensitivity.csv
  reports/presub_estimand_sensitivity.md

Caveats carried: deps.dev primary tier frame (top >500, mid 11-500, tail
<=10 dependents; missing excluded); all quantities are descriptive — no
estimand is causal and none identifies a Dec-9 effect (anti-causal firewall
intact); data cutoff 2026-05-10 untouched.

Run: conda run --no-capture-output -n stage2c python scripts/51_estimand_sensitivity.py
"""
from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"

VERSIONS_CSV = DATA_DIR / "versions.csv"
DEPSDEV_CSV = DATA_DIR / "depsdev_dependents_2026_05_05.csv"
B4C_CSV = DATA_DIR / "baselines" / "task_7_b4_pre_post_contrast.csv"

OUT_CSV = DATA_DIR / "estimand_sensitivity.csv"
OUT_MD = REPORTS_DIR / "presub_estimand_sensitivity.md"

# Constants identical to scripts/37 and scripts/45.
DEADLINE = pd.Timestamp("2025-12-09", tz="UTC")
ORIGIN = pd.Timestamp("2025-08-01", tz="UTC")
CENSOR = pd.Timestamp("2026-05-10 23:59:59", tz="UTC")
PREPOST_WEEKS = 8

TIERS = ["top", "mid", "tail"]
N_BOOT = 2000
SEED = 42
N_EXCLUDE_TOP = 5

EST_LABELS = {
    "1_version_weighted": "version-weighted rate (paper estimand)",
    "2_equal_package": "equal-package mean of per-package rates",
    "3_any_attested_share": "any-attested package share",
    "4_stable_only": "stable-only version-weighted rate",
    "5_top_excl_top5rows": "top tier excl. 5 highest-volume packages",
}


def dd_tier(n) -> str:
    if pd.isna(n):
        return "missing"
    n = int(n)
    return "top" if n > 500 else "mid" if n > 10 else "tail"


def load_versions_with_tier() -> pd.DataFrame:
    """Identical loading path to scripts/45 (Task 7 B4 conventions)."""
    versions = pd.read_csv(VERSIONS_CSV)
    versions["published_at"] = pd.to_datetime(versions["published_at"], utc=True, errors="coerce")
    versions = versions.dropna(subset=["published_at"])
    versions = versions[(versions["published_at"] >= ORIGIN) & (versions["published_at"] <= CENSOR)]
    versions["y"] = (versions["has_attestation"].astype(str).str.lower() == "true").astype(int)
    versions["is_prerelease"] = versions["version"].astype(str).str.contains("-", regex=False)
    depsdev = pd.read_csv(DEPSDEV_CSV)
    dd_map = dict(zip(depsdev["name"], depsdev["dependents_count"]))
    versions["dd_tier"] = versions["package"].map(dd_map).apply(dd_tier)
    return versions[versions["dd_tier"].isin(TIERS)].copy()


def window_bounds() -> Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]:
    """Pre-declared windows; pre/post replicate the Task 7 B4c 8-week frame."""
    return {
        "full": (ORIGIN, CENSOR + pd.Timedelta(seconds=1)),
        "pre_8wk": (DEADLINE - pd.Timedelta(weeks=PREPOST_WEEKS), DEADLINE),
        "post_8wk": (DEADLINE, DEADLINE + pd.Timedelta(weeks=PREPOST_WEEKS)),
    }


def check_b4c_consistency(counts: Dict[Tuple[str, str], int]) -> None:
    """The 8wk windows must reproduce the published B4c version counts."""
    b4c = pd.read_csv(B4C_CSV)
    for _, ref in b4c.iterrows():
        got = counts[(ref["tier"], ref["period"])]
        if got != int(ref["n_versions"]):
            raise AssertionError(
                f"B4c replication mismatch {ref['tier']}/{ref['period']}: "
                f"got {got}, expected {int(ref['n_versions'])}"
            )
    logger.info("B4c consistency check passed (8wk version counts reproduce).")


def level_estimands(g: pd.DataFrame) -> Dict[str, float]:
    """Estimands (1)-(4) in percent over one tier x window frame."""
    per_pkg = g.groupby("package")["y"]
    stable = g[~g["is_prerelease"]]
    return {
        "1_version_weighted": 100 * g["y"].mean(),
        "2_equal_package": 100 * per_pkg.mean().mean(),
        "3_any_attested_share": 100 * per_pkg.max().mean(),
        "4_stable_only": 100 * stable["y"].mean() if len(stable) else float("nan"),
    }


def package_pre_post_counts(pre: pd.DataFrame, post: pd.DataFrame) -> pd.DataFrame:
    """Per-package n/k (all rows and stable-only) for the two 8wk windows."""
    def agg(sub: pd.DataFrame, tag: str) -> pd.DataFrame:
        out = sub.groupby("package").agg(**{
            f"n_{tag}": ("y", "size"), f"k_{tag}": ("y", "sum")})
        st = sub[~sub["is_prerelease"]].groupby("package").agg(**{
            f"ns_{tag}": ("y", "size"), f"ks_{tag}": ("y", "sum")})
        return out.join(st, how="left")

    tbl = agg(pre, "pre").join(agg(post, "post"), how="outer").fillna(0).astype(int)
    return tbl.reset_index()


def bootstrap_contrasts(tbl: pd.DataFrame, rng: np.random.Generator) -> Dict[str, dict]:
    """8wk post-minus-pre contrast + package-cluster bootstrap percentile CI.

    One (B x n_pkg) package-index matrix is drawn per tier and reused across
    estimands (1), (2), (4) so the estimand comparison is on identical
    resamples. Replicates with an empty pre or post side are dropped (NaN).
    """
    n_pkg = len(tbl)
    idx = rng.integers(0, n_pkg, size=(N_BOOT, n_pkg))
    cols = {c: tbl[c].to_numpy(dtype=float) for c in
            ["n_pre", "k_pre", "n_post", "k_post", "ns_pre", "ks_pre", "ns_post", "ks_post"]}

    def ratio_diff(nb: str, kb: str, na: str, ka: str) -> Tuple[float, np.ndarray]:
        point = 100 * (cols[kb].sum() / cols[nb].sum() - cols[ka].sum() / cols[na].sum())
        with np.errstate(divide="ignore", invalid="ignore"):
            npost, kpost = cols[nb][idx].sum(axis=1), cols[kb][idx].sum(axis=1)
            npre, kpre = cols[na][idx].sum(axis=1), cols[ka][idx].sum(axis=1)
            diffs = 100 * (kpost / npost - kpre / npre)
        return point, diffs

    def equal_pkg_diff() -> Tuple[float, np.ndarray]:
        with np.errstate(divide="ignore", invalid="ignore"):
            r_pre = np.where(cols["n_pre"] > 0, cols["k_pre"] / cols["n_pre"], np.nan)
            r_post = np.where(cols["n_post"] > 0, cols["k_post"] / cols["n_post"], np.nan)
        point = 100 * (np.nanmean(r_post) - np.nanmean(r_pre))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            diffs = 100 * (np.nanmean(r_post[idx], axis=1) - np.nanmean(r_pre[idx], axis=1))
        return point, diffs

    out: Dict[str, dict] = {}
    for est, (point, diffs) in {
        "1_version_weighted": ratio_diff("n_post", "k_post", "n_pre", "k_pre"),
        "2_equal_package": equal_pkg_diff(),
        "4_stable_only": ratio_diff("ns_post", "ks_post", "ns_pre", "ks_pre"),
    }.items():
        valid = int(np.isfinite(diffs).sum())
        lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
        out[est] = {"diff_pp": point, "ci_low_pp": float(lo), "ci_high_pp": float(hi),
                    "boot_valid": valid, "n_packages": n_pkg}
    return out


def main() -> int:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    versions = load_versions_with_tier()
    logger.info("Frozen frame with dd tier: %d rows / %d packages "
                "(prerelease rows: %d = %.1f%%)", len(versions),
                versions["package"].nunique(), versions["is_prerelease"].sum(),
                100 * versions["is_prerelease"].mean())
    bounds = window_bounds()

    # ---- Estimand-5 exclusion set: 5 packages with most top-tier rows (full window).
    top_counts = (versions[versions["dd_tier"] == "top"]
                  .groupby("package").size().sort_values(ascending=False))
    excluded = top_counts.head(N_EXCLUDE_TOP)
    excl_note = "; ".join(f"{p} ({n:,} rows)" for p, n in excluded.items())
    excl_share = 100 * excluded.sum() / top_counts.sum()
    logger.info("Estimand-5 exclusion set (%.1f%% of top-tier rows): %s",
                excl_share, excl_note)

    rows: List[dict] = []
    counts_8wk: Dict[Tuple[str, str], int] = {}

    # ---- Levels: estimands (1)-(4) per tier x window, (5) top tier only.
    for window, (lo, hi) in bounds.items():
        wv = versions[(versions["published_at"] >= lo) & (versions["published_at"] < hi)]
        for tier in TIERS:
            g = wv[wv["dd_tier"] == tier]
            if window != "full":
                counts_8wk[(tier, window.replace("_8wk", "") + "_8wk")] = len(g)
            est = level_estimands(g)
            n_stable = int((~g["is_prerelease"]).sum())
            for name, val in est.items():
                rows.append({
                    "block": "level", "window": window, "tier": tier, "estimand": name,
                    "label": EST_LABELS[name], "value_pct": round(float(val), 4),
                    "n_versions": n_stable if name == "4_stable_only" else len(g),
                    "n_packages": g["package"].nunique(),
                    "ci_low_pp": "", "ci_high_pp": "", "boot_B": "", "boot_valid": "",
                    "note": "",
                })
            if tier == "top":
                g5 = g[~g["package"].isin(excluded.index)]
                rows.append({
                    "block": "level", "window": window, "tier": "top",
                    "estimand": "5_top_excl_top5rows",
                    "label": EST_LABELS["5_top_excl_top5rows"],
                    "value_pct": round(100 * float(g5["y"].mean()), 4),
                    "n_versions": len(g5), "n_packages": g5["package"].nunique(),
                    "ci_low_pp": "", "ci_high_pp": "", "boot_B": "", "boot_valid": "",
                    "note": f"excluded: {excl_note}",
                })

    check_b4c_consistency(counts_8wk)

    # ---- 8wk contrasts with package-cluster bootstrap CIs.
    rng = np.random.default_rng(SEED)
    contrasts: Dict[str, Dict[str, dict]] = {}
    for tier in TIERS:
        tv = versions[versions["dd_tier"] == tier]
        pre = tv[(tv["published_at"] >= bounds["pre_8wk"][0]) &
                 (tv["published_at"] < bounds["pre_8wk"][1])]
        post = tv[(tv["published_at"] >= bounds["post_8wk"][0]) &
                  (tv["published_at"] < bounds["post_8wk"][1])]
        tbl = package_pre_post_counts(pre, post)
        contrasts[tier] = bootstrap_contrasts(tbl, rng)
        for est, c in contrasts[tier].items():
            rows.append({
                "block": "contrast_8wk", "window": "post_minus_pre", "tier": tier,
                "estimand": est, "label": EST_LABELS[est],
                "value_pct": round(c["diff_pp"], 4),
                "n_versions": len(pre) + len(post), "n_packages": c["n_packages"],
                "ci_low_pp": round(c["ci_low_pp"], 4), "ci_high_pp": round(c["ci_high_pp"], 4),
                "boot_B": N_BOOT, "boot_valid": c["boot_valid"],
                "note": "package-cluster bootstrap percentile CI, seed 42",
            })
            logger.info("8wk contrast %s/%s: %+0.2f pp [%+.2f, %+.2f] (valid %d/%d)",
                        tier, est, c["diff_pp"], c["ci_low_pp"], c["ci_high_pp"],
                        c["boot_valid"], N_BOOT)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    logger.info("Saved %s", OUT_CSV)

    write_report(out, contrasts, excl_note, excl_share)
    print_headline(out, contrasts, excl_note)
    return 0


def _lvl(out: pd.DataFrame, window: str, tier: str, est: str) -> float:
    m = out[(out["block"] == "level") & (out["window"] == window) &
            (out["tier"] == tier) & (out["estimand"] == est)]
    return float(m["value_pct"].iloc[0])


def write_report(out: pd.DataFrame, contrasts: Dict[str, Dict[str, dict]],
                 excl_note: str, excl_share: float) -> None:
    lines = [
        "# Pre-submission — Estimand sensitivity of tier-level attestation rates",
        "",
        "**Date**: 2026-07-11 (acting-PI pre-submission batch; frozen rows only, "
        "no new collection). Descriptive sensitivity only — no estimand is causal; "
        "anti-causal firewall intact; data cutoff 2026-05-10 untouched.",
        "",
        "The paper's headline rates are version-weighted, so high-velocity",
        "publishers dominate: the 5 highest-volume top-tier packages contribute",
        f"{excl_share:.1f}% of top-tier version rows ({excl_note}).",
        "This report recomputes tier-level adoption under alternative estimands.",
        "",
        "## Levels by window (percent)",
        "",
    ]
    for window, title in [("full", "Full window [2025-08-01, 2026-05-10]"),
                          ("pre_8wk", "8-week pre [2025-10-14, 2025-12-09)"),
                          ("post_8wk", "8-week post [2025-12-09, 2026-02-03)")]:
        lines += [f"### {title}", "",
                  "| Tier | (1) version-weighted | (2) equal-package | "
                  "(3) any-attested share | (4) stable-only | (5) top excl. top-5 |",
                  "|---|---:|---:|---:|---:|---:|"]
        for tier in TIERS:
            e5 = (f"{_lvl(out, window, 'top', '5_top_excl_top5rows'):.2f}"
                  if tier == "top" else "—")
            lines.append(
                f"| {tier} | {_lvl(out, window, tier, '1_version_weighted'):.2f} | "
                f"{_lvl(out, window, tier, '2_equal_package'):.2f} | "
                f"{_lvl(out, window, tier, '3_any_attested_share'):.2f} | "
                f"{_lvl(out, window, tier, '4_stable_only'):.2f} | {e5} |")
        lines.append("")
    lines += [
        "## 8-week post-minus-pre contrasts (pp)",
        "",
        "Package-cluster bootstrap 95% percentile CIs (resample packages with",
        f"replacement, B={N_BOOT}, seed {SEED}; identical package resamples across",
        "estimands within a tier).",
        "",
        "| Tier | Estimand | Δ (pp) | 95% CI | valid replicates |",
        "|---|---|---:|---|---:|",
    ]
    for tier in TIERS:
        for est in ["1_version_weighted", "2_equal_package", "4_stable_only"]:
            c = contrasts[tier][est]
            lines.append(
                f"| {tier} | {EST_LABELS[est]} | {c['diff_pp']:+.2f} | "
                f"[{c['ci_low_pp']:+.2f}, {c['ci_high_pp']:+.2f}] | "
                f"{c['boot_valid']}/{N_BOOT} |")
    def cfmt(tier: str, est: str) -> str:
        c = contrasts[tier][est]
        spans = c["ci_low_pp"] <= 0 <= c["ci_high_pp"]
        return (f"{c['diff_pp']:+.2f} pp [{c['ci_low_pp']:+.2f}, {c['ci_high_pp']:+.2f}] "
                f"(CI {'spans' if spans else 'excludes'} 0)")

    lines += [
        "",
        "## Reading (descriptive only)",
        "",
        f"- Top tier, full window: the version-weighted rate "
        f"({_lvl(out, 'full', 'top', '1_version_weighted'):.1f}%) sits below the "
        f"equal-package mean ({_lvl(out, 'full', 'top', '2_equal_package'):.1f}%) and "
        f"the any-attested package share ({_lvl(out, 'full', 'top', '3_any_attested_share'):.1f}%): "
        "the paper estimand is the most conservative of the three, pulled down by "
        "high-volume non-attesting publishers.",
        f"- Excluding the 5 highest-volume top-tier packages ({excl_share:.1f}% of "
        f"top-tier rows) raises the top-tier version-weighted rate from "
        f"{_lvl(out, 'full', 'top', '1_version_weighted'):.1f}% to "
        f"{_lvl(out, 'full', 'top', '5_top_excl_top5rows'):.1f}% (full window).",
        f"- The top-tier 8wk contrast is estimand-dependent: version-weighted "
        f"{cfmt('top', '1_version_weighted')}; equal-package flips sign, "
        f"{cfmt('top', '2_equal_package')}; stable-only {cfmt('top', '4_stable_only')} "
        "— the only top-tier estimand whose CI excludes 0 (no multiplicity "
        "adjustment across the 9 contrasts). The equal-package sign flip is "
        "consistent with the composition decomposition (scripts/45): the aggregate "
        "top-tier decline is volume-mix driven, not a broad within-package retreat.",
        f"- Mid and tail move the other way: under equal-package and stable-only "
        f"estimands the 8wk contrasts are positive with CIs excluding 0 (mid "
        f"{cfmt('mid', '2_equal_package')}, {cfmt('mid', '4_stable_only')}; tail "
        f"{cfmt('tail', '2_equal_package')}, {cfmt('tail', '4_stable_only')}), while "
        "the version-weighted CIs span 0.",
        "- No estimand identifies a Dec-9 effect; a bootstrap CI excluding zero is",
        "  a precision statement about a descriptive pre/post difference, not a",
        "  causal claim — anti-causal firewall applies to every cell above.",
        "",
        "Artifacts: `data/estimand_sensitivity.csv`; windows replicate Task 7 B4c",
        "version counts exactly (assertion-checked).",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")
    logger.info("Saved %s", OUT_MD)


def print_headline(out: pd.DataFrame, contrasts: Dict[str, Dict[str, dict]],
                   excl_note: str) -> None:
    print("\n================ ESTIMAND SENSITIVITY — HEADLINE ================")
    print("Full window, top tier: "
          f"(1) version-weighted {_lvl(out, 'full', 'top', '1_version_weighted'):.2f}% | "
          f"(2) equal-package {_lvl(out, 'full', 'top', '2_equal_package'):.2f}% | "
          f"(3) any-attested {_lvl(out, 'full', 'top', '3_any_attested_share'):.2f}% | "
          f"(4) stable-only {_lvl(out, 'full', 'top', '4_stable_only'):.2f}% | "
          f"(5) excl-top5 {_lvl(out, 'full', 'top', '5_top_excl_top5rows'):.2f}%")
    print(f"8wk post-minus-pre (pp; package-cluster bootstrap 95% CI, B={N_BOOT}, seed {SEED}):")
    for tier in TIERS:
        parts = []
        for tag, est in [("(1)", "1_version_weighted"), ("(2)", "2_equal_package"),
                         ("(4)", "4_stable_only")]:
            c = contrasts[tier][est]
            parts.append(f"{tag} {c['diff_pp']:+.2f} "
                         f"[{c['ci_low_pp']:+.2f}, {c['ci_high_pp']:+.2f}]")
        print(f"  {tier:>4}: " + " | ".join(parts))
    print(f"Estimand-5 exclusions: {excl_note}")
    print("==================================================================\n")


if __name__ == "__main__":
    sys.exit(main())
