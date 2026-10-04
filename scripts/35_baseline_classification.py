"""Task 7 Baseline 2 (B2) — Classification method sensitivity.

Tests robustness of Task 2 A-F class distribution + Task 6 case anchors to
alternative classification rules. The optional multinomial class-prediction
model is a NEW Task-7 artifact and is NOT a Task 5 logistic re-run (PI Round 0
fix carried).

Sub-baselines:
  B2a — A-F final (reference)
  B2b — Conservative: keep B only if reason has "(strong-keyword)" AND
        "id-token scope=job"; otherwise demote to UNCERTAIN. Demote all C
        to UNCERTAIN (single weak heuristic per Task 2).
  B2c — Strict binary: per-package adoption rate from versions.csv (attested
        / total in window); A/B/C/E rows of the 500 sample all map to
        "non-attested" by construction; comparison is at package-level not
        version-level.
  B2d — C/E swap-test: relabel all C as "external-CI hypothesis only" and all
        E as "publish path not inferable"; report Task 6 case-anchor changes.

Optional B2e: multinomial class-prediction model (class ~ Task 5 covariates)
  under each rule (kept as a NEW Task-7 artifact; skipped silently if
  separation/sparsity makes the fit uninformative).

Outputs:
  data/baselines/task_7_b2_classification_variants.csv
  data/baselines/task_7_b2_case_anchor_changes.csv
  data/baselines/task_7_b2_rq2_distribution.csv
  data/baselines/task_7_b2_classmodel_or_grid.csv   (optional)
  reports/task_7_baseline_2.md

Run: conda run --no-capture-output -n stage2c python scripts/35_baseline_classification.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
BASELINES_DIR = DATA_DIR / "baselines"
REPORTS_DIR = ROOT / "reports"

CLASS_CSV = DATA_DIR / "classification.csv"
VERSIONS_CSV = DATA_DIR / "versions.csv"
INFLUENCE_CSV = DATA_DIR / "influence_metrics.csv"
DEPSDEV_CSV = DATA_DIR / "depsdev_dependents_2026_05_05.csv"

OUT_VARIANTS = BASELINES_DIR / "task_7_b2_classification_variants.csv"
OUT_CASES = BASELINES_DIR / "task_7_b2_case_anchor_changes.csv"
OUT_RQ2 = BASELINES_DIR / "task_7_b2_rq2_distribution.csv"
OUT_CLASSMODEL = BASELINES_DIR / "task_7_b2_classmodel_or_grid.csv"
OUT_REPORT = REPORTS_DIR / "task_7_baseline_2.md"

TASK6_CASES = ["axios", "lodash", "typescript", "webpack", "@types/sinon"]


# --- variant constructors -----------------------------------------------------

def variant_final(cls: pd.DataFrame) -> pd.Series:
    """B2a — reference. Returns the per-row class as-is."""
    return cls["class"].astype(str)


def variant_conservative(cls: pd.DataFrame) -> pd.Series:
    """B2b — keep B only if strong-keyword AND job-scope id-token; collapse
    others to UNCERTAIN. Demote all C to UNCERTAIN (single weak heuristic).
    A/E/F/D unchanged.
    """
    out = cls["class"].astype(str).copy()
    reason = cls["reason"].fillna("").astype(str)
    is_b = out == "B"
    strong = reason.str.contains(r"\(strong-keyword\)", regex=True)
    scope_job = reason.str.contains(r"id-token scope=job", regex=False)
    demoted_b = is_b & ~(strong & scope_job)
    out[demoted_b] = "UNCERTAIN"
    out[out == "C"] = "UNCERTAIN"
    return out


def variant_strict_binary(versions: pd.DataFrame) -> pd.DataFrame:
    """B2c — per-package binary: any-attested-in-window vs none-attested-in-window.
    Operates on the FULL 3000-package versions.csv (not just the 500-row sample).
    """
    v = versions.copy()
    v["has_attestation_int"] = (v["has_attestation"].astype(str).str.lower() == "true").astype(int)
    agg = (v.groupby("package")
            .agg(n_versions=("has_attestation_int", "size"),
                 n_attested=("has_attestation_int", "sum"))
            .reset_index())
    agg["strict_binary"] = np.where(agg["n_attested"] > 0, "attested", "none-attested")
    agg["attestation_rate"] = agg["n_attested"] / agg["n_versions"]
    return agg


def variant_ce_swap(cls: pd.DataFrame) -> pd.Series:
    """B2d — relabel C and E. C → 'C_external-CI-hypothesis'; E → 'E_publish-path-uninferable'."""
    out = cls["class"].astype(str).copy()
    out[out == "C"] = "C_external-CI-hypothesis"
    out[out == "E"] = "E_publish-path-uninferable"
    return out


# --- optional multinomial class-prediction model -----------------------------

def fit_multinomial_classmodel(cls: pd.DataFrame, inf: pd.DataFrame, variant_name: str
                                ) -> Optional[pd.DataFrame]:
    """Fit a multinomial logistic class~covariates model. Returns a DataFrame
    of ORs (vs reference class). Returns None if the fit is uninformative
    (sparse class, separation issue, model fails).
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        logger.info(f"[{variant_name}] sklearn missing; skipping multinomial model")
        return None

    # Build covariates frame at package level (cls is at sampled-version level;
    # one package can appear multiple times but covariates are at package level)
    base = cls.drop_duplicates(subset="package")[["package", "class"]].rename(
        columns={"class": variant_name})
    df = base.merge(inf, on="package", how="left")
    df = df.dropna(subset=["weekly_downloads_avg", "package_age_days"])
    df["log_downloads"] = np.log1p(df["weekly_downloads_avg"].astype(float))
    df["log_age_days"] = np.log1p(df["package_age_days"].astype(float))

    # Build the Task 5 covariate set
    df["owner_org_int"] = (df["github_owner_type"] == "Organization").astype(int)
    for src, dst in [
        ("has_dependabot", "has_dependabot_int"),
        ("has_issue_template", "has_issue_template_int"),
        ("is_scoped_package", "is_scoped_int"),
    ]:
        df[dst] = (df[src].astype(str).str.lower() == "true").astype(int)

    df["tier_top"] = (df["tier"] == "top").astype(int)
    df["tier_mid"] = (df["tier"] == "mid").astype(int)

    feats = ["log_downloads", "log_age_days", "maintainers_count",
             "owner_org_int", "has_dependabot_int", "has_issue_template_int",
             "is_scoped_int", "tier_top", "tier_mid"]
    df = df.dropna(subset=feats)
    if len(df) < 50:
        logger.info(f"[{variant_name}] only {len(df)} rows after dropna; skipping")
        return None

    classes, counts = np.unique(df[variant_name], return_counts=True)
    # Drop classes with < 5 obs (sparse)
    keep_classes = classes[counts >= 5]
    df = df[df[variant_name].isin(keep_classes)]
    if df[variant_name].nunique() < 3:
        logger.info(f"[{variant_name}] only {df[variant_name].nunique()} viable classes; "
                    "skipping (model would be ~binary)")
        return None

    X = df[feats].values
    y = df[variant_name].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    try:
        clf = LogisticRegression(solver="lbfgs", max_iter=2000)
        clf.fit(Xs, y)
    except Exception as e:
        logger.warning(f"[{variant_name}] fit failed: {e}")
        return None

    # sklearn multinomial coefs are per-class; map to OR per unit std change
    rows = []
    for ci, cls_label in enumerate(clf.classes_):
        for fi, fname in enumerate(feats):
            coef = clf.coef_[ci, fi]
            rows.append({
                "variant": variant_name,
                "class": str(cls_label),
                "feature": fname,
                "coef_per_std": coef,
                "OR_per_std": float(np.exp(coef)),
                "n_obs_fit": int(len(df)),
            })
    return pd.DataFrame(rows)


# --- RQ2 distribution table ---------------------------------------------------

def rq2_distribution(variants_df: pd.DataFrame, depsdev: pd.DataFrame) -> pd.DataFrame:
    """RQ2 class distribution per variant per tier (deps.dev-tier PRIMARY)."""
    dd_map = dict(zip(depsdev["name"], depsdev["dependents_count"]))
    def dd_tier(n):
        if pd.isna(n): return "missing"
        n = int(n)
        return "top" if n > 500 else "mid" if n > 10 else "tail"
    variants_df = variants_df.copy()
    variants_df["dd_tier"] = variants_df["package"].map(dd_map).apply(dd_tier)

    variant_cols = [c for c in variants_df.columns
                    if c.startswith("class_") or c == "strict_binary"]
    rows = []
    for var_col in variant_cols:
        for tier in ["top", "mid", "tail", "missing"]:
            sub = variants_df[variants_df["dd_tier"] == tier]
            if sub.empty:
                continue
            counts = sub[var_col].value_counts().to_dict()
            total = len(sub)
            for cls_label, n in counts.items():
                rows.append({
                    "variant": var_col,
                    "dd_tier": tier,
                    "class": str(cls_label),
                    "n": int(n),
                    "share_in_tier": n / total,
                })
    return pd.DataFrame(rows)


# --- case anchor changes ------------------------------------------------------

def case_anchor_changes(variants_df: pd.DataFrame) -> pd.DataFrame:
    """Per Task 6 case, what's its class under each variant?"""
    rows = []
    variant_cols = [c for c in variants_df.columns
                    if c.startswith("class_") or c == "strict_binary"]
    for case in TASK6_CASES:
        sub = variants_df[variants_df["package"] == case]
        if sub.empty:
            rows.append({"package": case, "in_500_sample": False})
            continue
        for var_col in variant_cols:
            vals = sub[var_col].dropna().unique().tolist()
            rows.append({
                "package": case,
                "in_500_sample": True,
                "variant": var_col,
                "labels": ",".join(str(v) for v in vals),
            })
    return pd.DataFrame(rows)


# --- main ---------------------------------------------------------------------

def write_report(variants_df: pd.DataFrame, rq2_df: pd.DataFrame,
                 cases_df: pd.DataFrame, classmodel_df: Optional[pd.DataFrame]) -> None:
    lines = [
        "# Task 7 Baseline 2 — Classification Method Sensitivity",
        "",
        "**Date**: auto-generated 2026-05-18 by `scripts/35_baseline_classification.py`",
        "**Scope**: B2 tests robustness of Task 2 A-F class distribution + Task 6 case "
        "anchors to alternative classification rules. The optional multinomial class-"
        "prediction model is a NEW Task-7 artifact and NOT a Task 5 logistic re-run.",
        "**Framing**: descriptive only; no causal claims (Task 5 PI Round 2 mandate carried).",
        "",
        "## B2 variants",
        "",
        "- **B2a — Final (reference)**: A-F as closed in Task 2 (v3 + selective PI override)",
        "- **B2b — Conservative**: keep B only if reason has `(strong-keyword)` AND "
        "`id-token scope=job`; otherwise demote to UNCERTAIN. Demote all C to UNCERTAIN.",
        "- **B2c — Strict binary** (package-level): per-package any-attested vs none-attested "
        "over versions.csv (full 3000 sample, not the 500-row classification sample)",
        "- **B2d — C/E swap**: relabel C → `C_external-CI-hypothesis`; E → "
        "`E_publish-path-uninferable`",
        "",
        "## RQ2 class distribution under each variant (deps.dev-tier PRIMARY)",
        "",
    ]
    if rq2_df.empty:
        lines.append("- (no rows)")
    else:
        # Pivot for readability: variant × class × dd_tier
        for var in sorted(rq2_df["variant"].unique()):
            sub = rq2_df[rq2_df["variant"] == var].sort_values(["dd_tier", "class"])
            lines += [f"### Variant: `{var}`", "",
                      "| dd_tier | class | n | share in tier |",
                      "|---|---|---:|---:|"]
            for _, r in sub.iterrows():
                lines.append(f"| {r['dd_tier']} | {r['class']} | {int(r['n'])} | "
                             f"{100*r['share_in_tier']:.2f}% |")
            lines += [""]
    lines += ["## Task 6 case-anchor changes", "",
              "Each of the 5 Task 6 cases under each B2 variant:", ""]
    if cases_df.empty:
        lines.append("- (no rows)")
    else:
        lines.append("| Package | variant | labels |")
        lines.append("|---|---|---|")
        for _, r in cases_df.iterrows():
            if not r.get("in_500_sample"):
                lines.append(f"| {r['package']} | (not in 500 sample) | (not classified) |")
            else:
                lines.append(f"| {r['package']} | {r.get('variant','')} | "
                             f"{r.get('labels','')} |")
    lines += ["", "**Reading**: if a Task 6 case anchor shifts class label "
              "across variants, the Task 6 narrative interpretation must be checked.",
              "Conservative rule may demote some B (OIDC drift) cases to UNCERTAIN; "
              "C/E swap renames labels but does not change the underlying class membership."]
    lines += ["", "## Optional multinomial class-prediction model (NEW Task-7 artifact)", ""]
    if classmodel_df is None or classmodel_df.empty:
        lines.append("- Skipped: sparse class counts or separation issue (per PI Round 0.5 minor — "
                     "optional and may be skipped if uninformative).")
    else:
        for var in sorted(classmodel_df["variant"].unique()):
            sub = classmodel_df[classmodel_df["variant"] == var]
            lines += [f"### Variant: `{var}`", "",
                      "**Disclaimer**: This is a NEW Task-7 multinomial model "
                      "(class ~ Task 5 covariates), NOT a re-run of the Task 5 logistic "
                      "(which predicts has_attestation, not class).",
                      "",
                      "| Class | Feature | coef (per std) | OR (per std) |",
                      "|---|---|---:|---:|"]
            for _, r in sub.iterrows():
                lines.append(f"| {r['class']} | {r['feature']} | {r['coef_per_std']:.3f} | "
                             f"{r['OR_per_std']:.3f} |")
            n_obs = int(sub["n_obs_fit"].iloc[0])
            lines += ["", f"_n_obs in fit = {n_obs}_", ""]
    lines += ["", "## Decision gate for downstream baselines",
              "",
              "- If conservative rule (B2b) reclassifies > 20% of B as UNCERTAIN, "
              "flag this in Task 7 summary; otherwise downstream uses A-F final.",
              "- If any Task 6 case-anchor changes substantively (e.g. webpack B → UNCERTAIN), "
              "Task 6 narrative interpretation gets a follow-up note (NOT a re-write).",
              "- C/E swap is presentational only; downstream uses A-F final labels.",
              "",
              "## NEXT",
              "1. Review B2 outputs; confirm no surprise reclassification.",
              "2. Proceed to scripts/36 (B3 influence: tier-frame + covariates merged)."]
    OUT_REPORT.write_text("\n".join(lines) + "\n")
    logger.info(f"wrote {OUT_REPORT}")


def main() -> int:
    BASELINES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("loading data...")
    cls = pd.read_csv(CLASS_CSV)
    versions = pd.read_csv(VERSIONS_CSV)
    inf = pd.read_csv(INFLUENCE_CSV)
    depsdev = pd.read_csv(DEPSDEV_CSV)

    # Build per-row variant columns
    logger.info("computing variants...")
    cls["class_final"] = variant_final(cls)
    cls["class_conservative"] = variant_conservative(cls)
    cls["class_ce_swap"] = variant_ce_swap(cls)
    logger.info(f"  final: {cls['class_final'].value_counts().to_dict()}")
    logger.info(f"  conservative: {cls['class_conservative'].value_counts().to_dict()}")
    logger.info(f"  ce_swap: {cls['class_ce_swap'].value_counts().to_dict()}")

    # B2c strict binary at package level
    strict = variant_strict_binary(versions)
    logger.info(f"  strict_binary: {strict['strict_binary'].value_counts().to_dict()}")

    # Merge: cls is at sampled-version level; one row per sampled version
    # For RQ2-style distribution + case anchors we want per-package mapping
    # but classification has one row per (package, version) sampled, so we keep
    # the version-level frame for variants_df.
    variants_df = cls[["package", "version", "class_final", "class_conservative",
                       "class_ce_swap"]].copy()
    # Attach strict binary as package-level
    sb_map = dict(zip(strict["package"], strict["strict_binary"]))
    variants_df["strict_binary"] = variants_df["package"].map(sb_map)

    variants_df.to_csv(OUT_VARIANTS, index=False)
    logger.info(f"wrote {OUT_VARIANTS}")

    # RQ2 distribution
    rq2_df = rq2_distribution(variants_df, depsdev)
    rq2_df.to_csv(OUT_RQ2, index=False)
    logger.info(f"wrote {OUT_RQ2}")

    # Task 6 case anchors
    cases_df = case_anchor_changes(variants_df)
    cases_df.to_csv(OUT_CASES, index=False)
    logger.info(f"wrote {OUT_CASES}")

    # Optional B2e: multinomial class-prediction model under each variant
    cm_parts = []
    for var_col in ["class_final", "class_conservative", "class_ce_swap"]:
        cl_sub = cls[["package", var_col, "reason"]].rename(columns={var_col: "class"})
        part = fit_multinomial_classmodel(cl_sub, inf, var_col)
        if part is not None:
            cm_parts.append(part)
    classmodel_df = pd.concat(cm_parts, ignore_index=True) if cm_parts else pd.DataFrame()
    if not classmodel_df.empty:
        classmodel_df.to_csv(OUT_CLASSMODEL, index=False)
        logger.info(f"wrote {OUT_CLASSMODEL} (rows={len(classmodel_df)})")
    else:
        logger.info("multinomial classmodel skipped (sparse/uninformative)")

    write_report(variants_df, rq2_df, cases_df,
                 classmodel_df if not classmodel_df.empty else None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
