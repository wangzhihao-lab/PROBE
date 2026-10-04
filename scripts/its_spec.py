"""Canonical ITS specification (Batch-8, 2026-07-11) — shared by all ITS scripts.

One function, one estimand, used verbatim by scripts/27, 34, 37, 46 so the
weekly model, the monthly model, the window grid, the placebo grid, and the
all-cutoff rank scan can never diverge in time-axis convention again.

Estimand: the ACTUAL-DATE discontinuity. The model is

    rate_t = b0 + b1*(t - c) + b2*post_t + b3*(t - c)*post_t + e_t

where t is the TRUE CALENDAR position of each period (weeks or months elapsed
since the series origin — gaps are preserved, never re-indexed away), c is the
calendar position of the cutoff DATE on the same axis, and post_t = 1 iff the
period starts on/after the cutoff date. Periods that straddle the cutoff
(start < cutoff < next start) are excluded, so every retained period is
cleanly pre or post; the excluded period leaves a real gap in t.

Under this parameterization b2 is the vertical distance between the post and
pre fitted lines evaluated AT the cutoff date, and b3 is the slope change.
This replaces the Batch-7 interim spec, which dropped the straddling period
and then re-indexed with arange(), silently compressing the calendar axis by
one step at the seam (ANOMALY_REPORT.md, Anomaly 9).

SEs: Newey-West HAC, maxlags = ceil(n^0.25). Diagnostics: Durbin-Watson +
Ljung-Box (as in the Task 5 spec). b2 AND b3 are both returned with 95% CIs
so placebo/rank analyses can rank either coefficient.

This module fits; it does not read files. Anti-causal firewall applies to
every consumer: b2/b3 are descriptive trajectory coefficients, not causal
effects of any cutoff.
"""
from __future__ import annotations

import calendar
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox

_DATE_MAX = date.max


def _calendar_t(starts: list[date], granularity: str) -> np.ndarray:
    """Calendar position of each period start (weeks or months since origin)."""
    o = starts[0]
    if granularity == "W":
        return np.array([(d - o).days / 7.0 for d in starts])
    if granularity == "MS":
        return np.array([(d.year - o.year) * 12 + (d.month - o.month)
                         for d in starts], dtype=float)
    raise ValueError(granularity)


def _cutoff_c(origin: date, cutoff: date, granularity: str) -> float:
    """Calendar position of the cutoff date on the same axis as _calendar_t."""
    if granularity == "W":
        return (cutoff - origin).days / 7.0
    if granularity == "MS":
        month_len = calendar.monthrange(cutoff.year, cutoff.month)[1]
        return ((cutoff.year - origin.year) * 12 + (cutoff.month - origin.month)
                + (cutoff.day - 1) / month_len)
    raise ValueError(granularity)


def fit_its(sub: pd.DataFrame, cutoff, *, period_col: str = "period_start",
            rate_col: str = "rate", granularity: str = "W",
            min_pre: int = 2, min_post: int = 2) -> Optional[dict]:
    """Fit the canonical cutoff-date-centered ITS on one tier's period series.

    Args:
        sub: rows for ONE tier; needs period_col (datetime-like) and rate_col
            (fraction in [0, 1]; multiplied by 100 to percentage points).
        cutoff: cutoff date (datetime/Timestamp/date).
        granularity: "W" (weekly panel) or "MS" (month-start panel).
        min_pre / min_post: admissibility floor on RETAINED clean periods.

    Returns:
        dict of coefficients/CIs/diagnostics (keys below), or None if the
        partition is inadmissible. 'sub_df' carries x/post/t columns and
        'fit_obj' the statsmodels results object, for plotting.
    """
    sub = sub.sort_values(period_col).reset_index(drop=True).copy()
    if len(sub) < min_pre + min_post:
        return None
    starts = [pd.Timestamp(v).date() for v in sub[period_col]]
    cut = pd.Timestamp(cutoff).date()

    t = _calendar_t(starts, granularity)
    c = _cutoff_c(starts[0], cut, granularity)

    nxt = starts[1:] + [_DATE_MAX]
    keep = np.array([not (s < cut < nx) for s, nx in zip(starts, nxt)])
    sub, t = sub[keep].reset_index(drop=True), t[keep]
    starts = [s for s, k in zip(starts, keep) if k]

    post = np.array([s >= cut for s in starts], dtype=int)
    n_pre, n_post = int((post == 0).sum()), int(post.sum())
    if n_pre < min_pre or n_post < min_post:
        return None

    x = t - c
    y = sub[rate_col].astype(float).values * 100  # percentage points
    X = sm.add_constant(np.column_stack([x, post, x * post]))
    n = len(sub)
    lag = max(1, int(np.ceil(n ** 0.25)))
    try:
        fit = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": lag})
    except Exception:
        return None
    ci = fit.conf_int(alpha=0.05)

    resid = fit.resid
    dw = sm.stats.stattools.durbin_watson(resid)
    lb_lags = sorted({min(4, n // 3), min(8, n // 2), min(12, n - 4)})
    lb_lags = [l for l in lb_lags if l > 0]
    lb_pvals: list[float] = []
    if lb_lags:
        try:
            lb = acorr_ljungbox(resid, lags=lb_lags, return_df=True)
            lb_pvals = [float(p) for p in lb["lb_pvalue"].values]
        except Exception:
            pass

    sub = sub.assign(t=t, x=x, post=post, cutoff_c=c)
    return {
        "n_periods": n, "n_pre": n_pre, "n_post": n_post,
        "hac_maxlags": lag, "cutoff_c": float(c),
        "beta0": float(fit.params[0]),
        "beta1_pre_slope": float(fit.params[1]),
        "beta2_level_change_pp": float(fit.params[2]),
        "beta3_slope_change": float(fit.params[3]),
        "post_slope": float(fit.params[1] + fit.params[3]),
        "se_beta2": float(fit.bse[2]), "se_beta3": float(fit.bse[3]),
        "ci95_beta2_low": float(ci[2, 0]), "ci95_beta2_high": float(ci[2, 1]),
        "ci95_beta3_low": float(ci[3, 0]), "ci95_beta3_high": float(ci[3, 1]),
        "pval_beta2": float(fit.pvalues[2]), "pval_beta3": float(fit.pvalues[3]),
        "DW": float(dw),
        "LB_min_pval": float(min(lb_pvals)) if lb_pvals else None,
        "lb_pvals": lb_pvals,
        "fit_obj": fit, "sub_df": sub,
    }
