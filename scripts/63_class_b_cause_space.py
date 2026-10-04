"""
TSE R1 revision, T2.5 — The cause space behind Class B (reviewer 2, R2-1c).

Reviewer 2: "the space of possible causes for Class B has not been explored."
Class B marks the capability/outcome mismatch state: the collection-time
workflow declares OIDC capability (id-token: write) yet the sampled release
carries no attestation. This script decomposes the 95 B rows into deterministic
sub-states using evidence ALREADY RECORDED at classification time -- the reason
strings written by scripts/06 encode the token co-presence mode, the id-token
permission scope, the detection strength, and whether the publish job resolved.
Nothing is re-classified and no new evidence is fetched; this is a re-tabulation
of the frozen record, so kappa and all counts are untouched.

Sub-states and the causes each is consistent with (all presence-only; none can
be confirmed from repository evidence, and the npm-side trusted-publisher
registration is invisible in principle):

  B-dual   id-token AND a long-lived token secret co-present at the publish
           path. On npm, authentication and provenance are separate: a
           token-authenticated publish produces an attestation when the job can
           mint an identity token and requests provenance explicitly, so
           co-presence is an ordinary arrangement, not a second publish path.
           Consistent with: a token-authenticated publish that never requested
           provenance; a request that failed on an unsupported runner or
           toolchain; configuration post-dating the release; the workflow not
           being the path that published the sampled version.
  B-pure   id-token without any visible token secret. Consistent with: a
           missing npm-side trusted-publisher registration; an npm CLI too old
           for trusted publishing; the same runner, toolchain, and timing
           causes; a publish through another path.
  scope    job-level vs workflow-level vs raw-fallback id-token: job-level is
           the strongest capability evidence; raw-fallback the weakest.

Inputs:
  data/classification.csv
  data/baselines/task_7_b2_classification_variants.csv  (conservative-rule labels)
  data/non_provenance_sample.csv, data/depsdev_dependents_2026_05_05.csv

Outputs:
  data/class_b_subtypes.csv
  reports/tse_r1_class_b_cause_space.md

Run: conda run --no-capture-output -n stage2c python scripts/63_class_b_cause_space.py
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
DATA, REPORTS = ROOT / "data", ROOT / "reports"
OUT_CSV = DATA / "class_b_subtypes.csv"
OUT_MD = REPORTS / "tse_r1_class_b_cause_space.md"


def dd_tier(n) -> str:
    if pd.isna(n):
        return "missing"
    n = int(n)
    return "top" if n > 500 else "mid" if n > 10 else "tail"


def parse_reason(reason: str) -> dict:
    mode = ("B-dual" if "co-present" in reason
            else "B-pure" if "w/o NPM_TOKEN" in reason else "B-other")
    m = re.search(r"id-token scope=([a-z-]+)", reason)
    scope = m.group(1) if m else "unrecorded"
    strength = ("strong-keyword" if "strong-keyword" in reason
                else "filename-hint" if "filename-hint" in reason else "other")
    return {"subtype": mode, "id_token_scope": scope, "detection": strength,
            "publish_job_resolved": "publish-job unresolved" not in reason,
            "workflow_file": (re.search(r"\[([^\]]+)\]", reason) or [None, ""])[1]}


def main() -> int:
    c = pd.read_csv(DATA / "classification.csv")
    b = c[c["class"] == "B"].copy()
    assert len(b) == 95, f"expected 95 B rows, got {len(b)}"
    parsed = pd.DataFrame([parse_reason(r) for r in b["reason"]])
    b = pd.concat([b.reset_index(drop=True), parsed], axis=1)

    dm = pd.read_csv(DATA / "depsdev_dependents_2026_05_05.csv")
    b["dd_tier"] = b["package"].map(dict(zip(dm["name"], dm["dependents_count"]))).apply(dd_tier)

    var = pd.read_csv(DATA / "baselines" / "task_7_b2_classification_variants.csv")
    b = b.merge(var[["package", "version", "class_conservative"]],
                on=["package", "version"], how="left")
    b["demotes_conservative"] = (b["class_conservative"] != "B")

    b.to_csv(OUT_CSV, index=False)
    n_demote = int(b["demotes_conservative"].sum())
    print(f"B rows: {len(b)}  packages: {b['package'].nunique()}  "
          f"conservative demotions: {n_demote} (paper says 41)", flush=True)
    assert n_demote == 41, f"conservative demotion count {n_demote} != published 41"

    def tab(col: str) -> pd.DataFrame:
        t = b.groupby([col, "dd_tier"]).size().unstack(fill_value=0)
        t["rows"] = t.sum(axis=1)
        t["packages"] = b.groupby(col)["package"].nunique()
        t["demote_conservative"] = b.groupby(col)["demotes_conservative"].sum().astype(int)
        return t

    L = ["# T2.5 — Class B 的成因空间（R2-1c）\n",
         "**方法**：对 95 个 B 行做确定性子分型，全部证据取自分类时已写入 reason 字段的记录",
         "（token 共存模式、id-token 权限作用域、检测强度、publish job 是否解析成功）。",
         "**零重分类、零新抓取**：κ、计数、任何已发布数字均不受影响。\n",
         "所有子态都是 presence-only 证据；npm 侧 trusted-publisher 注册在原理上不可见，",
         "因此没有任何子态能升级为已验证的发布路径。\n"]
    n = len(b)
    for col, title in (("subtype", "1. 模式：token 共存 vs 纯 OIDC"),
                       ("id_token_scope", "2. id-token 权限作用域"),
                       ("detection", "3. 检测强度")):
        t = tab(col)
        L.append(f"## {title}\n")
        L.append("| 子态 | top | mid | tail | 行数 | 份额 | 包数 | 保守规则降级 |")
        L.append("|---|---|---|---|---|---|---|---|")
        for idx, r in t.iterrows():
            L.append(f"| {idx} | {r.get('top', 0)} | {r.get('mid', 0)} | {r.get('tail', 0)} | "
                     f"{r['rows']} | {100 * r['rows'] / n:.1f}% | {r['packages']} | {r['demote_conservative']} |")
        L.append("")
    xt = b.groupby(["subtype", "id_token_scope"]).size().unstack(fill_value=0)
    L.append("## 4. 模式 × 作用域交叉表\n")
    L.append("| | " + " | ".join(xt.columns) + " |")
    L.append("|---|" + "---|" * len(xt.columns))
    for idx, r in xt.iterrows():
        L.append(f"| {idx} | " + " | ".join(str(v) for v in r.values) + " |")
    L += ["", "## 5. 每个子态与之相容的成因（写进 §5.2 / supplement 的底稿）\n",
          "- **B-dual**（id-token 与长期 token 秘密共存）：npm 的认证与 provenance 是两套机制，token 认证的发布只要 job 能铸造 id-token 并显式请求 provenance 就会产生 attestation，",
          "  所以共存是常规架构而非\"第二条发布路径\"。相容成因：token 认证的发布从未请求 provenance；请求在不受支持的 runner/工具链上失败；配置晚于该版本发布；该 workflow 不是发布这个版本的路径。",
          "- **B-pure**（有 id-token、无可见 token 秘密）：npm 侧 trusted-publisher 未注册；npm CLI 版本不支持 trusted publishing；同样的 runner/工具链/时间成因；走了别的发布路径。",
          "  各成因从仓库证据不可区分；scripts/76 的显式 provenance 请求读取（B 21/95）补充了成因空间，不裁决任何一行。",
          "- **作用域弱证据**（workflow 级 / raw-fallback）与 **publish job 未解析** 的行，",
          "  正是保守规则降级的主体——降级规则（仅 strong-keyword 且 scope=job 保留）",
          f"  在 95 行中降掉 {n_demote} 行，与论文既有敏感性一致（断言校验）。\n",
          "## 6. 一句结论\n",
          "B 不是单一状态：约定量比例的行是 **token 共现配置**（token 认证与 provenance 请求可以共存，不构成路径歧义），其余是",
          "**纯 OIDC 能力宣示而结果缺失**（npm 侧前置或路径归属问题）。成因空间可以枚举、",
          "不能裁决——这正是 §4 把 B 定义为 mismatch **state** 而非 failure **event** 的原因。"]
    OUT_MD.write_text("\n".join(L) + "\n")
    print(f"✓ {OUT_CSV.name}  ✓ {OUT_MD.name}\n✅ T2.5 complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
