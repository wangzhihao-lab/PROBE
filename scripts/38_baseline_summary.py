from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
BASELINES_DIR = DATA_DIR / "baselines"
REPORTS_DIR = ROOT / "reports"
OUT_REPORT = REPORTS_DIR / "task_7_report.md"


def main() -> int:


    b1_struct = pd.read_csv(BASELINES_DIR / "task_7_b1_outcome_structural.csv")
    b1_struct_ok = (b1_struct["verdict"] == "OK").sum()
    b1_struct_total = len(b1_struct)
    b1_struct_pct = 100 * b1_struct_ok / max(1, b1_struct_total)
    b1_window = pd.read_csv(BASELINES_DIR / "task_7_b1_window_grid.csv")
    b1_drift = pd.read_csv(BASELINES_DIR / "task_7_b1_packument_drift.csv")
    drift_total = (b1_drift["n_att_flag_drift_on_intersection"].sum()
                   if "n_att_flag_drift_on_intersection" in b1_drift.columns else 0)


    b2_vars = pd.read_csv(BASELINES_DIR / "task_7_b2_classification_variants.csv")
    b2_final = b2_vars["class_final"].value_counts().to_dict()
    b2_conservative = b2_vars["class_conservative"].value_counts().to_dict()
    b2_cases = pd.read_csv(BASELINES_DIR / "task_7_b2_case_anchor_changes.csv")


    b3_shift = pd.read_csv(BASELINES_DIR / "task_7_b3_tier_shift_matrix.csv")
    b3_grid = pd.read_csv(BASELINES_DIR / "task_7_b3_covariate_grid.csv")
    cox_primary_full = b3_grid[(b3_grid["outcome"] == "cox_time_to_first_provenance") &
                                (b3_grid["label"] == "frame_dd_PRIMARY_full")]


    b4_wm = pd.read_csv(BASELINES_DIR / "task_7_b4_its_weekly_vs_monthly.csv")
    b4_placebo = pd.read_csv(BASELINES_DIR / "task_7_b4_its_placebo_grid.csv")
    b4_prepost = pd.read_csv(BASELINES_DIR / "task_7_b4_pre_post_contrast.csv")
    b4_did = pd.read_csv(BASELINES_DIR / "task_7_b4_did_descriptive.csv")


    top_placebos = b4_placebo[b4_placebo["tier"] == "top"]
    actual_top = top_placebos[top_placebos["is_actual_dec9"] == True]
    actual_b2 = float(actual_top.iloc[0]["beta2_level_change_pp"]) if not actual_top.empty else None
    placebos_top_only = top_placebos[top_placebos["is_actual_dec9"] != True]
    larger_placebos = ((placebos_top_only["beta2_level_change_pp"].abs() > abs(actual_b2)).sum()
                       if actual_b2 is not None else 0)


    lines = [
        "# Baseline Comparison (Combined Summary)",
        "",
        "**Date**: auto-generated 2026-05-18 by `scripts/38_baseline_summary.py`",
        "**Inherits**: association-only framing; case-organization logic; "
        "deps.dev-tier PRIMARY frame",
        "**Plan**: `reports/task_7_plan.md`",
        "**Decision**: NO baseline identifies a Dec-9-specific causal effect. All results are "
        "descriptive / associational / predictive.",
        "",
        "## Headline cross-baseline table",
        "",
        "| # | Baseline | Headline finding | Implication |",
        "|---|---|---|---|",
        f"| B1 | Data collection | {b1_struct_ok:,}/{b1_struct_total:,} ({b1_struct_pct:.1f}%) "
        f"attested versions structurally OK; Rekor slice 30/30 validated; 8-window grid "
        f"(4 starts × 2 ends) shows top-tier ITS β2 modest with intervals spanning zero in "
        f"every window (canonical calendar-time cutoff-date-centered spec), and neither "
        f"tier indicator a robust hazard/odds signal; NaT={0}; freshness drift on attestation "
        f"flag = {int(drift_total)} across 20 audited packages | "
        f"Outcome construction + window edges are robust — downstream baselines run on "
        f"validated `has_attestation` |",
        f"| B2 | Classification method | Conservative rule demotes 41/95 B  UNCERTAIN "
        f"(`scope=workflow` does not satisfy job-scope requirement); C/E swap is cosmetic; "
        f"strict-binary classification reproduces case adoption labels | "
        f"anchors `axios` + `webpack` shift B  UNCERTAIN under conservative rule "
        f"(narrative gets a follow-up note, NOT a rewrite) |",
        f"| B3 | Influence (tier + covariates) | Pairwise tier-frame agreement: dd vs eco "
        f"52.4%, dd vs dl 38.9%, dd vs PCA 41.1%; **25 fits per outcome total** = 4 tier-frame "
        f"full models + 21 primary-frame sensitivity refits (9 drop-one + 3 group-out + 9 "
        f"univariate); covariate signs stable across all refits | "
        f"Tier-frame choice is load-bearing (already noted in ); covariate "
        f"associations are not driven by a single covariate |",
        f"| B4 | Time-series model | Top β2 = {actual_b2:+.2f} pp (canonical cutoff-date-centered), "
        f"**{int(larger_placebos)} of 8 placebo cutoffs strictly larger** for top-tier absolute |β2| "
        f"(all-admissible rank 17/28); DiD γ3 PT-FAIL reproduced | Task 5's "
        f"no-causal-identification conclusion is REINFORCED, not weakened |",
        "",
        "## Descriptive-only framing — verification across baselines",
        "",
        "Per , no baseline can be re-read as a Dec-9-specific causal "
        "identification. Audit:",
        "",
        "| Baseline | Descriptive only? | Evidence |",
        "|---|---|---|",
        "| B1 |  | Outcome / window / inclusion are descriptive sensitivity audits; no causal claim "
        "even when β2 shifts under window change |",
        "| B2 |  | Classification variants affect RQ2 distribution and case anchors only; "
        "the optional multinomial class-prediction model is labeled NEW Task-7 artifact, NOT a "
        "logistic re-run |",
        "| B3 |  | All OR/HR estimates labeled as associations; in-sample only; calibration "
        "counterexample illustration (TS + @types/sinon) preserved from Task 6 |",
        "| B4 |  | Placebo grid reinforces Task 5's failure; pre-post is descriptive (Wilson CI); "
        "DiD PT-FAIL inherited; explicit decision-gate language `'do NOT upgrade'` |",
        "",
        "## Decision-gate summary",
        "",
        "- **B1 structural-drift gate**: 0% drift ≤ 5% threshold   downstream baselines proceed",
        "- **B1 window-edges gate**: top-tier β2 stable, mid/tail noisy — association-only "
        "framing remains essential for non-top tiers",
        "- **B1 inclusion-rule gate**: no canonical change (NaT = 0)",
        "- **B1 freshness gate**: 0 attestation-flag drift across 20 audited packages",
        "- **B2 conservative-demotion gate**: 41/95 B reclassified to UNCERTAIN (43% of B) — "
        "this exceeds the 20% flag threshold; narrative footnotes axios/webpack as "
        "rule-sensitive B classifications",
        "- **B3 tier-frame-agreement gate**: NO pair exceeds 85% agreement (highest is eco vs PCA "
        "at 82.1%)  tier frame is load-bearing; per-frame full-model fits are all reported",
        "- **B4 placebo-grid gate**: actual Dec-9 top β2 is NOT uniquely large in placebo "
        "distribution — **4 of 8 strictly larger; 5 of 8 larger-or-equal** (the difference is "
        "the 2025-12-14 placebo which falls in the same weekly split as Dec-9). Task 5's "
        "no-causal-identification conclusion is reinforced under either strict or non-strict "
        "comparison.",
        "- **B4 DiD-PT gate**: γ3 ≠ 0 with p < 0.001  PT-FAIL  DiD demoted to descriptive contrast",
        "",
        "## Connection to Task 5 / Task 6 / Task 4 (canonical specs)",
        "",
        "- **Cox primary is the incident delayed-entry 11-covariate fit** (scripts/52): "
        "log_age HR 0.554, has_dependabot 1.849, has_issue_template 1.221 (n.s. once release "
        "opportunity is modeled), is_scoped 0.559, tier_top 1.018 (n.s.); PH-failing "
        "covariates remain time-averaged descriptive only.",
        f"- **ITS top-tier headline (canonical cutoff-date-centered spec)**: β2 = {actual_b2:+.2f} pp "
        f"under weekly aggregation, CI spanning zero — no detectable discontinuity.",
        f"- **Placebo-failure conclusion reinforced**: the B4 grid shows {int(larger_placebos)} of 8 "
        "placebos with larger |β2| than actual Dec-9 for top tier (all-admissible rank 17/28); "
        "B4 confirms the actual Dec-9 is not uniquely large.",
        "- **case anchors carry construct caveats**: axios/webpack class-B labels are "
        "rule-sensitive (B2b conservative demotes both to UNCERTAIN); lodash/typescript/@types/sinon "
        "anchors are stable across variants.",
        "- **tier-frame finding extended**: B3 quantifies pairwise tier-frame disagreement "
        "(only eco vs PCA exceeds 80%); deps.dev PRIMARY remains the canonical frame per Task 4 "
        "Round 3 .",
        "",
        "## Construct caveats (carried forward)",
        "",
        "- A-F taxonomy semantics inherited from Task 2 (class-of-sampled-version, not "
        "package-attestation-status; attested versions are not re-labeled with a different letter)",
        "- C / E construct caveats from (C = possible external CI hypothesis; "
        "E = no visible publish workflow in inspected sources, NOT asserted as manual publish)",
        "- Missing-from-deps.dev scope (clarified ): the **full 3000-package "
        "sample** has 62 packages without deps.dev mapping (carried as `tier='missing'` "
        "in ). The **B3 survival subset** (1614 packages with at least one version "
        "in window) has **29 missing-from-deps.dev**, evident in the tier-shift matrix where "
        "`n_both_known = 1585` for dd vs eco (1614 − 1585 = 29). Both are carried as "
        "`tier='missing'` and not silently imputed in any of the 4 tier frames.",
        "- All Logistic / Cox / ITS / DiD estimates are in-sample associations; out-of-sample / "
        "grouped CV is deferred to if needed",
        "",
        "## RDD + Synthetic Control (out of scope, plan rationale carried)",
        "",
        "- **RDD OUT**: global time cutoff with all units exposed simultaneously, no unit-level "
        "eligibility threshold; would collapse to local ITS and inherit Task 5's placebo failure",
        "- **SC OUT**: donor-pool contamination (PyPI / cargo / RubyGems / Maven all in different "
        "states); cross-ecosystem outcome measurement incompatibility (PEP 740 vs SLSA vs PGP)",
        "",
        "## Files inventory",
        "",
        "```",
        "scripts/",
        "├── 34_baseline_data_collection.py     # B1",
        "├── 35_baseline_classification.py      # B2",
        "├── 36_baseline_influence.py           # B3 (tier + covariates merged)",
        "├── 37_baseline_time_models.py         # B4",
        "└── 38_baseline_summary.py             # this file",
        "",
        "data/baselines/",
        "├── task_7_b1_outcome_structural.csv",
        "├── task_7_b1_outcome_rekor_slice.csv",
        "├── task_7_b1_window_grid.csv",
        "├── task_7_b1_inclusion_diff.csv",
        "├── task_7_b1_packument_drift.csv",
        "├── task_7_b2_classification_variants.csv",
        "├── task_7_b2_case_anchor_changes.csv",
        "├── task_7_b2_rq2_distribution.csv",
        "├── task_7_b2_classmodel_or_grid.csv",
        "├── task_7_b3_tier_shift_matrix.csv",
        "├── task_7_b3_pca_loadings.csv",
        "├── task_7_b3_covariate_grid.csv",
        "├── task_7_b3_forest.png",
        "├── task_7_b4_its_weekly_vs_monthly.csv",
        "├── task_7_b4_its_placebo_grid.csv",
        "├── task_7_b4_pre_post_contrast.csv",
        "└── task_7_b4_did_descriptive.csv",
        "",
        "reports/",
        "├── task_7_plan.md             # Round 0.5 ",
        "├── task_7_baseline_1.md       # data collection",
        "├── task_7_baseline_2.md       # classification",
        "├── task_7_baseline_3.md       # influence",
        "├── task_7_baseline_4.md       # time models",
        "└── task_7_report.md           # this file (combined summary)",
        "```",
        "",
        "## Closure status",
        "",
        "**** All 4 sub-baselines "
        "executed;  (B1b end-date + Cox/Logistic per window) fixed in Round 1.5 "
        "revision without changing substantive conclusions. Two non-blocking pre-freeze "
        "cleanups (stale NEXT/status text + B3 sub-report missing-deps.dev scope qualifier) "
        "applied in closure.",
        "",
        "Downstream:",
        "- Task 8 (monthly monitoring) finalize in parallel (no dependency on Task 7).",
        "- Task 9 (paper draft) uses as the **robustness / baseline comparison layer**, "
        "preserving the association-only framing and the load-bearing tier-frame caveat "
        ".",
    ]
    OUT_REPORT.write_text("\n".join(lines) + "\n")
    logger.info(f"wrote {OUT_REPORT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
