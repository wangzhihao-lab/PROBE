# Proportional-hazards check on the PRIMARY delayed-entry Cox model, computed on
# counting-process (entry, stop] data so that every Schoenfeld residual uses the
# left-truncated risk set {j : entry_j < t <= stop_j}.
#
# Called by scripts/79_cox_ph_primary.py, which exports the analysis frame with
# scripts/52's own cohort builders (extended series, incident cohort, 11 covariates).
#
# Input:  data/cox_primary_frame_tse_r1_long.csv
# Output: data/cox_zph_primary_tse_r1_long.csv     (cox.zph score tests per term + GLOBAL,
#                                                   for the km, rank and identity transforms)
#         data/cox_primary_R_coef_tse_r1_long.csv   (coefficients, for the cross-check
#                                                   against the lifelines fit)
# Environment: R 4.5.3, survival 3.8-12 (conda-forge r-base, r-survival; see environment-r.yml)
# Run: conda run -n rsurv Rscript scripts/79_cox_zph.R

suppressPackageStartupMessages(library(survival))

args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
root <- normalizePath(file.path(here, ".."))
data_dir <- file.path(root, "data")

covs <- c("log_weekly_downloads", "log_age_days", "log_maintainers", "owner_is_org",
          "has_dependabot_int", "has_issue_template_int", "is_scoped_int",
          "tier_top", "tier_mid", "log_prewin_release_freq", "github_unobserved")

d <- read.csv(file.path(data_dir, "cox_primary_frame_tse_r1_long.csv"))
stopifnot(all(d$entry_days < d$duration_days))

form <- as.formula(paste("Surv(entry_days, duration_days, event) ~",
                         paste(covs, collapse = " + ")))
# One row per package, so clustering on the package gives the per-subject robust variance
# (lifelines robust=True); point estimates and cox.zph do not depend on it.
fit <- coxph(form, data = d, ties = "efron", cluster = package)

tabs <- lapply(c("km", "rank", "identity"), function(tr) {
  z <- cox.zph(fit, transform = tr)
  t <- as.data.frame(z$table)
  data.frame(transform = tr, term = rownames(t), chisq = t$chisq, df = t$df, p = t$p)
})
zph <- do.call(rbind, tabs)
write.csv(zph, file.path(data_dir, "cox_zph_primary_tse_r1_long.csv"), row.names = FALSE)

coefs <- data.frame(term = names(coef(fit)), coef = unname(coef(fit)),
                    robust_se = sqrt(diag(fit$var)), naive_se = sqrt(diag(fit$naive.var)))
write.csv(coefs, file.path(data_dir, "cox_primary_R_coef_tse_r1_long.csv"), row.names = FALSE)

cat(R.version.string, "| survival", as.character(packageVersion("survival")),
    "| n =", fit$n, "| events =", fit$nevent, "\n")
print(zph, row.names = FALSE, digits = 4)
