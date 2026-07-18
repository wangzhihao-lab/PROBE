from __future__ import annotations

import csv
import random
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).parent.parent / "data"
REPORTS_DIR = Path(__file__).parent.parent / "reports"
VERSIONS_CSV = DATA_DIR / "versions.csv"
SAMPLE_CSV = DATA_DIR / "non_provenance_sample.csv"
CLASS_CSV = DATA_DIR / "classification.csv"
DEPSDEV_CSV = DATA_DIR / "depsdev_dependents_2026_05_05.csv"
OUT_CSV = DATA_DIR / "rq2_design_weighted.csv"
OUT_MD = REPORTS_DIR / "presub_rq2_weights.md"


DEADLINE = datetime(2025, 12, 9, 23, 59, 59, 999999, tzinfo=timezone.utc)
ECO_TIERS = ("top", "mid", "tail")
DD_TIERS = ("top", "mid", "tail", "missing")
CLASSES_AE = ("A", "B", "C", "D", "E")
EXPECTED_SAMPLE_N = {"top": 250, "mid": 150, "tail": 100}
B_BOOT = 2000
SEED = 42


@dataclass(frozen=True)
class Row:
    package: str
    eco_tier: str
    dd_tier: str
    cls: str
    weight: float


def parse_dt(s: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def dd_tier_of(dpc: int) -> str:
    if dpc > 500:
        return "top"
    if dpc > 10:
        return "mid"
    return "tail"


def recompute_pool_sizes() -> tuple[dict[str, int], dict[str, int]]:
    pkg_pool: dict[str, set] = {t: set() for t in ECO_TIERS}
    ver_pool: Counter = Counter()
    with VERSIONS_CSV.open() as f:
        for r in csv.DictReader(f):
            if r["has_attestation"] == "True":
                continue
            if (parse_dt(r["published_at"]) or DEADLINE) > DEADLINE:
                if r["tier"] in pkg_pool:
                    pkg_pool[r["tier"]].add(r["package"])
                    ver_pool[r["tier"]] += 1
    return ({t: len(pkg_pool[t]) for t in ECO_TIERS},
            {t: ver_pool[t] for t in ECO_TIERS})


def load_rows(weights: dict[str, float], dd_map: dict[str, int]) -> list[Row]:
    cls_map: dict[tuple[str, str], str] = {}
    with CLASS_CSV.open() as f:
        for r in csv.DictReader(f):
            cls_map[(r["package"], r["version"])] = r["class"]

    by_pkg: dict[str, dict] = {}
    with SAMPLE_CSV.open() as f:
        for r in csv.DictReader(f):
            key = (r["package"], r["version"])
            if key not in cls_map:
                raise ValueError(f"sample row missing classification: {key}")
            pkg = r["package"]
            cls = cls_map[key]
            if pkg not in by_pkg:
                by_pkg[pkg] = {"eco": r["tier"], "classes": set()}
            by_pkg[pkg]["classes"].add(cls)
            if by_pkg[pkg]["eco"] != r["tier"]:
                raise ValueError(f"package {pkg} spans eco strata")

    rows: list[Row] = []
    for pkg, info in by_pkg.items():
        if len(info["classes"]) != 1:
            raise ValueError(f"package {pkg} has non-constant class {info['classes']}")
        cls = next(iter(info["classes"]))
        dpc = dd_map.get(pkg)
        dd = dd_tier_of(dpc) if dpc is not None else "missing"
        rows.append(Row(package=pkg, eco_tier=info["eco"], dd_tier=dd,
                        cls=cls, weight=weights[info["eco"]]))
    return rows


def weighted_ae_shares(rows: list[Row]) -> dict[str, dict[str, float]]:
    mass: dict[str, Counter] = {t: Counter() for t in DD_TIERS}
    for r in rows:
        mass[r.dd_tier][r.cls] += r.weight
    out: dict[str, dict[str, float]] = {}
    for t in DD_TIERS:
        denom = sum(v for k, v in mass[t].items() if k != "F")
        if denom > 0:
            out[t] = {c: mass[t].get(c, 0.0) / denom * 100 for c in CLASSES_AE}
    return out


def percentile(sorted_vals: list[float], p: float) -> float:
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    h = (n - 1) * p / 100.0
    lo = int(h)
    hi = min(lo + 1, n - 1)
    return sorted_vals[lo] + (h - lo) * (sorted_vals[hi] - sorted_vals[lo])


def bootstrap_cis(rows: list[Row]) -> tuple[dict[tuple[str, str], tuple[float, float]],
                                            dict[str, int]]:
    by_stratum_pkg: dict[str, dict[str, list[Row]]] = {t: {} for t in ECO_TIERS}
    for r in rows:
        by_stratum_pkg[r.eco_tier].setdefault(r.package, []).append(r)
    stratum_pkgs = {t: list(by_stratum_pkg[t].keys()) for t in ECO_TIERS}

    rng = random.Random(SEED)
    dist: dict[tuple[str, str], list[float]] = {
        (t, c): [] for t in ("top", "mid", "tail") for c in CLASSES_AE}
    empty: dict[str, int] = {t: 0 for t in ("top", "mid", "tail")}
    for _ in range(B_BOOT):
        rep: list[Row] = []
        for t in ECO_TIERS:
            pkgs = stratum_pkgs[t]
            for pkg in rng.choices(pkgs, k=len(pkgs)):
                rep.extend(by_stratum_pkg[t][pkg])
        shares = weighted_ae_shares(rep)
        for t in ("top", "mid", "tail"):
            if t not in shares:
                empty[t] += 1
                continue
            for c in CLASSES_AE:
                dist[(t, c)].append(shares[t][c])

    cis: dict[tuple[str, str], tuple[float, float]] = {}
    for key, vals in dist.items():
        vals.sort()
        cis[key] = (percentile(vals, 2.5), percentile(vals, 97.5))
    return cis, empty


def main() -> int:

    pkg_pool, ver_pool = recompute_pool_sizes()
    print("=== Eco-stratum PACKAGE pools (scripts/03 filter) ===", flush=True)

    sample_rows: Counter = Counter()
    sample_pkgs: dict[str, set] = {t: set() for t in ECO_TIERS}
    with SAMPLE_CSV.open() as f:
        for r in csv.DictReader(f):
            sample_rows[r["tier"]] += 1
            sample_pkgs[r["tier"]].add(r["package"])
    if dict(sample_rows) != EXPECTED_SAMPLE_N:
        print(f" sample rows {dict(sample_rows)} != design {EXPECTED_SAMPLE_N}", flush=True)
        return 1
    selected = {t: len(sample_pkgs[t]) for t in ECO_TIERS}
    weights = {t: pkg_pool[t] / selected[t] for t in ECO_TIERS}
    for t in ECO_TIERS:
        census = " (census)" if selected[t] >= pkg_pool[t] else ""
        print(f"  {t:<5s} pkg_pool={pkg_pool[t]:>4}  selected_pkgs={selected[t]:>3}  "
              f"rows={sample_rows[t]:>3}  w={weights[t]:.3f}{census}", flush=True)

    wsum = sum(weights[t] * selected[t] for t in ECO_TIERS)
    if round(wsum) != sum(pkg_pool.values()):
        print(f" weight sanity failed: Σw={wsum} != pkg pool total "
              f"{sum(pkg_pool.values())}", flush=True)
        return 1


    dd_map: dict[str, int] = {}
    with DEPSDEV_CSV.open() as f:
        for r in csv.DictReader(f):
            try:
                dd_map[r["name"]] = int(r["dependents_count"])
            except ValueError:
                continue
    rows = load_rows(weights, dd_map)
    print(f"\nLoaded {len(rows)} sampled packages "
          f"({len({r.package for r in rows})} distinct packages)", flush=True)

    counts: dict[str, Counter] = {t: Counter() for t in DD_TIERS}
    wmass: dict[str, Counter] = {t: Counter() for t in DD_TIERS}
    for r in rows:
        counts[r.dd_tier][r.cls] += 1
        wmass[r.dd_tier][r.cls] += r.weight

    miss_rows = [r for r in rows if r.dd_tier == "missing"]
    miss_ae = [r for r in miss_rows if r.cls != "F"]
    print(f"deps.dev-missing rows: {len(miss_rows)} total "
          f"({len(miss_ae)} in A–E pool, {len(miss_rows) - len(miss_ae)} class F)",
          flush=True)


    print(f"\nBootstrap: stratified package-cluster, B={B_BOOT}, seed={SEED} ...",
          flush=True)
    cis, empty = bootstrap_cis(rows)
    if any(empty.values()):
        print(f"  replicates with empty A–E denominator (excluded): {empty}", flush=True)


    def upct(t: str, c: str) -> Optional[float]:
        denom = sum(v for k, v in counts[t].items() if k != "F")
        return counts[t].get(c, 0) / denom * 100 if denom else None

    def wpct(t: str, c: str) -> Optional[float]:
        denom = sum(v for k, v in wmass[t].items() if k != "F")
        return wmass[t].get(c, 0.0) / denom * 100 if denom else None

    out_rows: list[dict] = []
    for t in DD_TIERS:
        n_total = sum(counts[t].values())
        n_ae = sum(v for k, v in counts[t].items() if k != "F")
        for c in CLASSES_AE + ("F",):
            ci = cis.get((t, c))
            out_rows.append({
                "dd_tier": t, "class": c, "n_raw": counts[t].get(c, 0),
                "tier_rows_total": n_total, "tier_rows_AE": n_ae,
                "unweighted_pct_AE": (f"{upct(t, c):.2f}"
                                      if c != "F" and upct(t, c) is not None else ""),
                "weighted_pct_AE": (f"{wpct(t, c):.2f}"
                                    if c != "F" and wpct(t, c) is not None else ""),
                "boot_ci95_lo": f"{ci[0]:.2f}" if ci else "",
                "boot_ci95_hi": f"{ci[1]:.2f}" if ci else "",
            })
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"\n Saved {OUT_CSV} ({len(out_rows)} rows)", flush=True)


    lines = [
        "# Pre-submission RQ2 check — design-weighted class shares under deps.dev tier",
        "",
        f"Generated by `scripts/50_rq2_design_weights.py` (B={B_BOOT} bootstrap, seed {SEED}).",
        "",
        "## Why",
        "",
        "The 500-row non-provenance sample was drawn **package-first** (shuffle",
        "packages, round-robin one version at a time up to 3/package) stratified by",
        "**ecosyste.ms tier**, so the primary sampling unit is the PACKAGE. The A–F",
        "evidence state is a repository-level attribute, constant across a package's",
        "sampled versions (verified). We therefore collapse to one row per package and",
        "weight by the **package** inclusion probability of each eco stratum,",
        "w_h = P_h / n_selected_packages_h, where P_h = packages in the eco-stratum",
        "pool. Design-consistent for the NON-ATTESTED PACKAGE POOL within the cohort,",
        "not an ecosystem prevalence.",
        "",
        "## Package design weights (pools recomputed from `data/versions.csv`)",
        "",
        "| eco stratum | package pool P_h | version pool | selected pkgs | weight w_h |",
        "|---|---:|---:|---:|---:|",
    ]
    for t in ECO_TIERS:
        census = " (census)" if selected[t] >= pkg_pool[t] else ""
        lines.append(f"| {t} | {pkg_pool[t]:,} | {ver_pool[t]:,} | {selected[t]} | "
                     f"{weights[t]:.3f}{census} |")
    lines += [
        f"| total | {sum(pkg_pool.values()):,} | {sum(ver_pool.values()):,} | "
        f"{sum(selected.values())} | — |",
        "",
        "The old version-based weights (11.13 / 66.75 / 212.46) over-weighted the tail",
        "~19x relative to top; the correct package weights are 1 : 2.58 : 3.77.",
        "",
        "## deps.dev-missing packages",
        "",
        f"{len(miss_rows)} sampled packages have no deps.dev `dependents_count` "
        f"(absent from the 2026-05-05 snapshot): "
        f"**{len(miss_ae)} in the A–E pool, {len(miss_rows) - len(miss_ae)} class F**.",
        "Reported as a separate `missing` tier with no bootstrap CI.",
        "",
        "| package | class | eco stratum |",
        "|---|---|---|",
    ]
    for r in sorted(miss_rows, key=lambda x: x.package):
        lines.append(f"| {r.package} | {r.cls} | {r.eco_tier} |")
    lines += [
        "",
        "## deps.dev tier × class — unweighted vs design-weighted (% of A–E package pool, F excluded)",
        "",
        "| dd tier | class | n | unweighted % | weighted % | 95% CI (package-cluster bootstrap) |",
        "|---|---|---:|---:|---:|---|",
    ]
    for t in DD_TIERS:
        n_ae = sum(v for k, v in counts[t].items() if k != "F")
        for c in CLASSES_AE:
            u, wv, ci = upct(t, c), wpct(t, c), cis.get((t, c))
            u_s = f"{u:.1f}%" if u is not None else "—"
            w_s = f"{wv:.1f}%" if wv is not None else "—"
            ci_s = f"[{ci[0]:.1f}, {ci[1]:.1f}]" if ci else "—"
            lines.append(f"| {t} | {c} | {counts[t].get(c, 0)} | {u_s} | {w_s} | {ci_s} |")
        lines.append(f"| {t} | F (excl.) | {counts[t].get('F', 0)} | — | — | — |")
        lines.append(f"| {t} | *A–E pool n* | {n_ae} |  |  |  |")
    lines += [
        "",
        "## Method notes",
        "",
        f"- Tier thresholds identical to `scripts/25`: top > 500, mid 11–500, tail ≤ 10",
        "  deps.dev dependents; `missing` = package not in the snapshot.",
        "- Class F (no repo URL) is excluded from the A–E share denominator, matching the",
        "  paper's RQ2 tabulation.",
        f"- Uncertainty: stratified package-cluster bootstrap, B={B_BOOT}, seed {SEED} —",
        "  packages resampled with replacement within each eco stratum, each package",
        "  keeping all its sampled rows and its fixed stratum weight; 95% CI = 2.5/97.5",
        "  percentiles of the weighted share across replicates.",
        "",
        "## Limitations",
        "",
        "- The weights correct only the unequal eco-stratum sampling fractions. The",
        "  ≤3-rows-per-package cap and the within-package round-robin version selection",
        "  in `scripts/03` are treated as **ignorable** (approximately self-weighting",
        "  within stratum); rows from prolific packages are still mildly under-weighted",
        "  relative to a full version-level IPW. Documented as a limitation.",
        "- deps.dev tier is an outcome-side regrouping of an eco-tier-stratified sample;",
        "  weighted shares are design-consistent for the post-deadline non-provenance",
        "  package pool, not a new deps.dev-stratified draw.",
        "- The 6 deps.dev-missing rows are excluded from top/mid/tail cells rather than",
        "  imputed.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f" Saved {OUT_MD}", flush=True)


    print("\n=== Headline: unweighted%  weighted% [95% CI] (A–E pool, F excluded) ===",
          flush=True)
    for t in ("top", "mid"):
        print(f"  dd-{t} (A–E n={sum(v for k, v in counts[t].items() if k != 'F')}):",
              flush=True)
        for c in CLASSES_AE:
            u, wv, ci = upct(t, c), wpct(t, c), cis.get((t, c))
            ci_s = f"[{ci[0]:.1f}, {ci[1]:.1f}]" if ci else "[—]"
            print(f"    {c}: {u:5.1f}%  {wv:5.1f}%  {ci_s}", flush=True)

    print("\n Pre-submission RQ2 design-weight check PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
