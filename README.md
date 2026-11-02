# PROBE — Replication Artifact

**Provenance Revocation Observational Boundary Evaluation**

Replication package for *"Measuring npm Provenance Adoption
Around the 2025 Classic-Token Revocation: Coincidence Is Not
Identification"*.

## What this artifact reproduces

Every headline number in the paper: the RQ1 interrupted-time-series and
difference-in-differences trajectory results, the RQ2 collection-time
evidence-state distribution, and the RQ3 Cox and version-level logistic
association models, together with the robustness grids and sensitivity
analyses. Every headline number is also present directly in the frozen
`data/` output tables shipped here, independent of re-execution.

## Important: this is a frozen historical snapshot

The paper measures a specific window (2025-08-01 to 2026-05-10) around the
2025-12-09 npm classic-token revocation deadline. The npm registry, GitHub
repositories, and deps.dev / ecosyste.ms are **live systems**: re-fetching
them today returns post-collection state and will **not** reproduce the
paper's numbers (this is a disclosed limitation, paper Section 12). The
analysis therefore ships the **frozen derived data** in `data/`, which the
analysis scripts consume directly. The raw API pulls (~2.6 GB of packument
JSON and workflow YAML) are **not** included here; they are access-gated
(they carry maintainer contact fields) and available to researchers on
request. `SHA256_MANIFEST.txt` identifies the exact snapshot analyzed.

## Layout

```
scripts/         analysis pipeline, scripts/01_*.py … scripts/79_*.py
                 (+ its_spec.py, figstyle.py shared modules; scripts/79_cox_zph.R)
data/            frozen derived tables (CSV/JSON) the scripts consume
                 - versions.csv .............. per-version has_attestation outcome (protocol window)
                 - versions_extended.csv ..... same, 2024-01-01 → 2026-05-10 (revision; re-read from the frozen packuments)
                 - tse_r1_*.csv .............. every revision-added table (grid, placebo scans, seasonality, MDE, RQ2 composition, B/C/E structure, sampling flow)
                 - classification.csv ........ 500-row A–F evidence-state sample
                 - classification_construct_reasons.csv .. final-construct phrasing
                 - CODEBOOK_classification.md . reason-trace codebook
                 - influence_metrics.csv, depsdev_dependents_2026_05_05.csv, …
environment.yml  conda environment (Python 3.11.13) pinned to analysis versions
environment-r.yml conda environment (R 4.5.3, survival 3.8-12) for scripts/79_cox_zph.R
SHA256_MANIFEST.txt   checksums of every file in this artifact
```

## Reproduce

```bash
conda env create -f environment.yml      # creates env "stage2c"
conda env create -f environment-r.yml    # creates env "rsurv" (only for script 79)
conda activate stage2c
```

**Public path (frozen `data/` only — no raw pulls needed).** These scripts
read the frozen derived tables shipped here and regenerate their outputs into
`data/` (and a human-readable summary into `reports/`, which ships empty):

```bash
python scripts/46_placebo_cutoff_rank.py   # RQ1 ITS placebo rank (17/28)
python scripts/50_rq2_design_weights.py    # RQ2 design-weighted distribution
python scripts/07_compute_kappa.py         # post-adjudication concordance (kappa = 0.82)
```

Note on reliability: `scripts/07` compares the two raters on the final, post-adjudication labels and prints the post-adjudication concordance (kappa = 0.8192, rounded to 0.82 in the paper). The paper's *primary* independent inter-rater estimate --- kappa = 0.8099 (rounded to 0.81), the two raters applying the shared *initial* rule --- is the
`classification_v1_raw.csv` pairing, shipped here and also tabulated in
`review9_rater_perclass.csv`; recompute it directly with
`python scripts/53_review9_sensitivities.py --s4-only`, which reads only the
frozen tables (the script's full run additionally needs the access-gated
`data/packuments_raw/` slice for its Cox blocks and will not complete
without it).

**Access-gated path (needs the raw slice).** The RQ3 primary driver
`scripts/52_rq3_sensitivity.py` rebuilds the incident Cox cohort and the
pre-window covariates from the full packument history, so it reads
`data/packuments_raw/` — the access-gated raw pulls that are **not** in this
public package (see below) — and writes model diagnostic figures to a
`paper/figures/` directory. Its results are already shipped frozen in `data/`
(`cox_incident_primary.csv`, `logistic_sensitivity.csv`, and the RQ3 tables),
so the headline RQ3 numbers are available here without re-running it; full
re-execution requires the raw slice through the access-gated channel.


**Revision path (TSE R1; frozen `data/` only).** The revision-added analyses
that run off the shipped tables:

```bash
python scripts/56_long_window_trajectory.py     # 28-month panels + seasonal baseline (cluster bootstrap)
python scripts/57_long_window_its_placebo.py    # extended ITS + 110-candidate placebo scans, 3 estimands
python scripts/60_extended_its_spec_sensitivity.py  # window x specification grid (Table II)
python scripts/61_power_mde_equivalence.py      # MDE / TOST table
python scripts/58_rq3_window_comparison.py      # RQ3 protocol vs extended (reads shipped CSVs)
python scripts/59_rq2_cross_boundary.py         # RQ2 composition across the boundary (package constancy)
python scripts/63_class_b_cause_space.py        # Class B sub-states
python scripts/65_class_e_structure.py          # Class E six sub-states
python scripts/67_rq2_eco_frame_table.py        # RQ2 on the sampling frame
```

Four verification fetches (September 2, 2026) and a pagination check of two
of them (September 3, 2026) are shipped as scripts with their derived tables;
the fetches need network access and a GitHub token, read only
immutable or current-state third-party objects, and are **not** required to
reproduce any number in the frozen tables above (their raw responses join the
access-gated slice):

```bash
python scripts/70_attestation_timestamp_audit.py   # Sigstore bundle signing time vs publish time (300 sampled versions)
python scripts/71_class_c_external_ci_audit.py     # Class C external-CI configuration read (49 rows)
python scripts/72_workflow_history_audit.py        # publish-workflow git history vs cached labels (486 rows)
python scripts/73_workflow_history_prewindow.py    # the same labels replayed at three 2024-2025 dates (separate history read)
python scripts/74_workflow_history_pagination_check.py  # completeness check of the 72/73 commit listings (cached; offline re-run)
```

Two offline follow-ups after an independent audit (no network). Script 75 runs
from the frozen tables shipped here; script 76 re-reads the cached workflow
YAML in `data/workflows_raw/`, which is part of the access-gated raw slice, so
its frozen output (`tse_r1_provenance_flag_read.csv`) is shipped for checking the
reported counts:

```bash
python scripts/75_protocol_window_estimand_its.py   # equal-package / drop-top-5 ITS companions on the protocol window
python scripts/76_provenance_flag_read.py          # explicit --provenance request in the cached publish workflows (needs the raw slice)
```

The proportional-hazards check on the primary delayed-entry Cox model computes
Schoenfeld residuals on the counting-process (entry, stop] risk sets with R's
`survival::cox.zph`. Script 79 rebuilds the analysis frame through `scripts/52`
when the raw packument slice is present and otherwise runs from the frozen
frame shipped here (`cox_primary_frame_tse_r1_long.csv`); it calls the R script
in the `rsurv` environment and cross-checks the R coefficients against the
lifelines fit:

```bash
python scripts/79_cox_ph_primary.py   # Schoenfeld tests on the primary delayed-entry Cox fit (calls scripts/79_cox_zph.R)
```

Scripts that need the access-gated raw slice (`data/packuments_raw/`,
`data/workflows_raw/`) and therefore will not complete from this public
package alone: `55` (construct audit over packuments), `62` and `66` (they
rebuild the RQ3 frame through `scripts/52`), `64` and `76` (they re-read the
cached workflows), and the extended-window runs of `47`/`52`
(`--origin 2024-01-01 --versions-csv versions_extended.csv`). Their outputs
are shipped frozen (`*_tse_r1_long.csv`, `tse_r1_class_c_audit.csv`,
`tse_r1_sampling_flow.csv`, `tse_r1_or_mechanism.csv`). `68` and `69`
regenerate manuscript tables and figures and expect the paper tree; they
exit cleanly without it. Default (no-argument) runs of `03`, `47`, and `52`
reproduce the as-submitted outputs byte-for-byte.

Scripts are numbered in pipeline order; a later script consumes CSVs written by
earlier ones. Stages 01–14 build the dataset from the raw pulls (not shipped)
and are included for inspection. Most later stages run off the frozen `data/`
tables provided here; the exception is the RQ3 primary driver above, which also
needs the access-gated raw packument history.

## Notes on the classification data

- The A–F class labels are the analyzed artifact. Each label is a deterministic
  function of the archived evidence fields (workflow files, packument metadata,
  repository status) and is recomputable from the access-gated raw slice.
- The `reason` column carries the rule engine's historical shorthand
  (e.g. "OIDC failed", "classic publish"), which predates the paper's final
  construct language; `CODEBOOK_classification.md` maps it to the Section-4.3
  semantics, and `classification_construct_reasons.csv` carries the final
  phrasing per row.
- The `rater` column uses neutral labels (`rater_A`, `rater_B`) for the two
  independent applications of the classification rule set.

## License

Original analysis code: MIT. Original figures, tables, and derived data
tables: CC BY 4.0. Third-party raw snapshots (npm / deps.dev /
ecosyste.ms) are not redistributed and remain under their original terms.
