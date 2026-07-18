# PROBE — Replication Artifact

**Provenance Revocation Observational Boundary Evaluation**

Replication package for *"Measuring npm Provenance Adoption
Around the 2025 Classic-Token Revocation: Coincidence Is Not
Identification"* (IEEE TSE submission).

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
(they carry maintainer contact fields) and available to editors/reviewers
through the submission system. `SHA256_MANIFEST.txt` identifies the exact
snapshot analyzed.

## Layout

```
scripts/         analysis pipeline, scripts/01_*.py … scripts/54_*.py
                 (+ its_spec.py, figstyle.py shared modules)
data/            frozen derived tables (CSV/JSON) the scripts consume
                 - versions.csv .............. per-version has_attestation outcome
                 - classification.csv ........ 500-row A–F evidence-state sample
                 - classification_construct_reasons.csv .. final-construct phrasing
                 - CODEBOOK_classification.md . reason-trace codebook
                 - influence_metrics.csv, depsdev_dependents_2026_05_05.csv, …
environment.yml  conda environment (Python 3.11.13) pinned to analysis versions
SHA256_MANIFEST.txt   checksums of every file in this artifact
```

## Reproduce

```bash
conda env create -f environment.yml      # creates env "stage2c"
conda activate stage2c
```

**Public path (frozen `data/` only — no raw pulls needed).** These scripts
read the frozen derived tables shipped here and regenerate their outputs into
`data/` (and a human-readable summary into `reports/`, created on demand):

```bash
python scripts/46_placebo_cutoff_rank.py   # RQ1 ITS placebo rank (17/28)
python scripts/50_rq2_design_weights.py    # RQ2 design-weighted distribution
python scripts/07_compute_kappa.py         # post-adjudication concordance (kappa = 0.82)
```

Note on reliability: `scripts/07` compares the two raters on the final, post-adjudication labels and prints the post-adjudication concordance (kappa = 0.8192, rounded to 0.82 in the paper). The paper's *primary* independent inter-rater estimate --- kappa = 0.8099 (rounded to 0.81), the two raters applying the shared *initial* rule --- is the `classification_v1_raw.csv` pairing, shipped here and also tabulated in `review9_rater_perclass.csv`; `scripts/53_review9_sensitivities.py` (block S4) recomputes it from those frozen tables.

**Access-gated path (needs the raw slice).** The RQ3 primary driver
`scripts/52_rq3_sensitivity.py` rebuilds the incident Cox cohort and the
pre-window covariates from the full packument history, so it reads
`data/packuments_raw/` — the access-gated raw pulls that are **not** in this
public package (see below) — and writes model diagnostic figures to a
`paper/figures/` directory. Its results are already shipped frozen in `data/`
(`cox_incident_primary.csv`, `logistic_sensitivity.csv`, and the RQ3 tables),
so the headline RQ3 numbers are available here without re-running it; full
re-execution requires the raw slice through the access-gated channel.

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
