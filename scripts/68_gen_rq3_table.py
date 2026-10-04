"""
TSE R1 revision — Generate the Table III (tab:rq3-models) body from the
extended-window CSVs, so no coefficient is ever transcribed by hand.

Injects the rows directly between the RQ3-TABLE-BODY markers inside
paper_tse/sections/05_results.tex (an \\input inside tabular breaks the
alignment scanner -- verified by minimal repro, 2026-09-01).

Inputs:  data/cox_incident_primary_tse_r1_long.csv,
         data/logistic_sensitivity_tse_r1_long.csv
Run: conda run --no-capture-output -n stage2c python scripts/68_gen_rq3_table.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
cox = pd.read_csv(ROOT / "data" / "cox_incident_primary_tse_r1_long.csv").set_index("covariate")
log = pd.read_csv(ROOT / "data" / "logistic_sensitivity_tse_r1_long.csv")
log = log[(log["model"] == "primary") & (log["kind"] == "OR")].set_index("name")

ORDER = ["post_deadline", "log_weekly_downloads", "log_age_days", "log_maintainers",
         "owner_is_org", "has_dependabot_int", "has_issue_template_int", "is_scoped_int",
         "log_prewin_release_freq", "github_unobserved", "tier_top", "tier_mid"]
FLAG_ALPHA = 0.05


def pfmt(p: float) -> str:
    if p < 1e-10:
        return "$<10^{-10}$"
    if p < 1e-4:
        return "$<10^{-4}$"
    if p < 0.001:
        return "$<0.001$"
    return f"{p:.3f}".lstrip("0") if p < 1 else f"{p:.2f}"


def tex(cov: str) -> str:
    return "\\texttt{" + cov.replace("_", "\\_") + "}"


lines = []
for cov in ORDER:
    if cov in cox.index:
        r = cox.loc[cov]
        sch = r.get("schoenfeld_p", float("nan"))
        flag = " \\textbf{(fl.)}" if pd.notna(sch) and sch < FLAG_ALPHA else ""
        cx = (f"{r['HR']:.3f} & $[{r['HR_ci_low']:.3f}, {r['HR_ci_high']:.3f}]$ & "
              f"{pfmt(r['p'])} & {sch:.3f}{flag}")
    else:
        cx = "\\multicolumn{4}{c}{(not in model)}"
    l = log.loc[cov]
    if cov == "github_unobserved":
        lg = "$\\approx 0$ & (near-sep.) & " + pfmt(l["p"])
    else:
        lg = f"{l['value']:.3f} & $[{l['ci_low']:.2f}, {l['ci_high']:.2f}]$ & {pfmt(l['p'])}"
    lines.append(f"    {tex(cov)} & {cx} & {lg} \\\\")

target = ROOT / "paper_tse" / "sections" / "05_results.tex"
if not target.exists():
    print("manuscript tree not present; rows:"); print("\n".join(lines)); raise SystemExit(0)
doc = target.read_text()
BEGIN = "% RQ3-TABLE-BODY-BEGIN (auto-managed by scripts/68_gen_rq3_table.py; do not edit rows by hand)"
END = "% RQ3-TABLE-BODY-END"
i, j = doc.index(BEGIN) + len(BEGIN), doc.index(END)
doc = doc[:i] + "\n" + "\n".join(lines) + "\n" + doc[j:]
target.write_text(doc)
print(f"✓ injected {len(lines)} rows into 05_results.tex between markers")
