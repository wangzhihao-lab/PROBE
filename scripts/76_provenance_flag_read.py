"""
TSE R1 independent-audit follow-up — Explicit provenance request in the labelled publish workflows.

Why: on npm, authentication and provenance are separate mechanisms. A publish
authenticated with a long-lived token still produces a registry attestation
when the job can mint an OpenID Connect identity token (`id-token: write`) and
the publish requests provenance explicitly (`npm publish --provenance`, or an
equivalent configuration flag); trusted publishing generates the attestation
automatically without the flag (npm Docs, "Generating provenance statements";
GitHub Changelog, 2023-04-19). The A--F labels record only whether the publish
workflow can mint the identity token (Class B) or holds a token secret without
it (Class A). This read records, per labelled row, whether the publish workflow
the classifier named also requests provenance explicitly, so that the Class B
cause space can be stated without treating token/OIDC co-presence as a
"publish path".

Design (offline; no new collection):
  - Rows: every Class A and Class B row of the 500-row sample.
  - The publish workflow files are the ones the classifier's reason string
    names (in brackets, or as `file=class` pairs for adjudicated rows); they
    are read from the frozen cache data/workflows_raw/ collected in mid-May
    2026. Rows whose named files are absent from the cache are reported
    separately as unresolved, not counted as "no request".
  - provenance_request = the file, with YAML comments removed (everything
    from a `#` that starts a line or follows whitespace), matches
    `--provenance`, `provenance: true` / `provenance=true`, or
    `NPM_CONFIG_PROVENANCE` (case-insensitive; `PNPM_CONFIG_PROVENANCE`
    is matched by the same pattern). Comment-only mentions such as
    `id-token: write # for --provenance` do not count. The regex is declared
    here before the read.
  - token_secret = the file references NPM_TOKEN or NODE_AUTH_TOKEN (the
    classifier's own co-presence criterion, re-applied to the same file).
  - A repository-level column records whether ANY cached workflow of the
    repository carries the request, for completeness.

Limits: the flag can also be set in package.json (`publishConfig.provenance`)
or .npmrc, which the cache does not hold, so "no request seen" is not "no
request"; the read says nothing about which path a given publish took.

Inputs:
  data/classification.csv, data/workflows_raw/
Outputs:
  data/tse_r1_provenance_flag_read.csv
  reports/tse_r1_provenance_flag_read.md

Run: conda run --no-capture-output -n stage2c python scripts/76_provenance_flag_read.py
"""
from __future__ import annotations

import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA, REPORTS = ROOT / "data", ROOT / "reports"
CLASS_CSV, RAW_DIR = DATA / "classification.csv", DATA / "workflows_raw"
OUT_CSV = DATA / "tse_r1_provenance_flag_read.csv"
OUT_MD = REPORTS / "tse_r1_provenance_flag_read.md"

PROV_RE = re.compile(r"--provenance|provenance\s*[:=]\s*true|NPM_CONFIG_PROVENANCE", re.I)
TOKEN_RE = re.compile(r"NPM_TOKEN|NODE_AUTH_TOKEN", re.I)
FILE_RE = re.compile(r"([A-Za-z0-9@._-]+__[A-Za-z0-9._-]+__[A-Za-z0-9._-]+\.ya?ml)")
COMMENT_RE = re.compile(r"(^|\s)#.*$", re.M)


def main() -> int:
    rows = [r for r in csv.DictReader(CLASS_CSV.open()) if r["class"] in ("A", "B")]
    repo_files: dict[str, list[Path]] = defaultdict(list)
    for p in RAW_DIR.iterdir():
        if p.is_file():
            m = re.match(r"(.+?__.+?)__.+$", p.name)
            if m:
                repo_files[m.group(1)].append(p)
    out = []
    for r in rows:
        names = list(dict.fromkeys(FILE_RE.findall(r["reason"])))
        texts = []
        for n in names:
            p = RAW_DIR / n
            if p.exists():
                texts.append(p.read_text(errors="replace"))
        repo_key = names[0].rsplit("__", 1)[0] if names else ""
        any_repo = any(PROV_RE.search(COMMENT_RE.sub("", p.read_text(errors="replace"))) for p in repo_files.get(repo_key, []))
        joined = COMMENT_RE.sub("", "\n".join(texts))
        rec = {"package": r["package"], "version": r["version"], "class": r["class"],
               "publish_workflow": "|".join(names), "workflow_cached": "yes" if texts else "no",
               "token_secret": "yes" if TOKEN_RE.search(joined) else "no",
               "provenance_request": ("yes" if PROV_RE.search(joined) else "no") if texts else "unresolved",
               "provenance_request_any_workflow": "yes" if any_repo else "no"}
        rec["mode"] = ("co-presence" if rec["token_secret"] == "yes" else "capability-without-outcome") if r["class"] == "B" else "token-only"
        out.append(rec)
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    c = Counter((r["class"], r["mode"], r["provenance_request"]) for r in out)
    pk = defaultdict(set)
    for r in out:
        pk[(r["class"], r["provenance_request"])].add(r["package"])
    n_uncached = sum(1 for r in out if r["workflow_cached"] == "no")
    L = ["# 标注发布 workflow 中的显式 provenance 请求（TSE R1 独立复核后补；离线重读缓存，无新采集）\n",
         f"对象：500 行样本中的 Class A {sum(1 for r in out if r['class']=='A')} 行与 Class B {sum(1 for r in out if r['class']=='B')} 行；"
         f"读取分类器 reason 中命名的发布 workflow（缓存于 2026-05 中旬）；未命中缓存 {n_uncached} 行。\n",
         "| 类 | 模式 | 显式 provenance 请求 | 行数 | 包数 |", "|---|---|---|---|---|"]
    for (cls, mode, prov), n in sorted(c.items()):
        pkgs = len({r["package"] for r in out if (r["class"], r["mode"], r["provenance_request"]) == (cls, mode, prov)})
        L.append(f"| {cls} | {mode} | {prov} | {n} | {pkgs} |")
    b_yes = sum(n for (cls, _, prov), n in c.items() if cls == "B" and prov == "yes")
    b_tot = sum(n for (cls, _, _), n in c.items() if cls == "B")
    a_yes = sum(n for (cls, _, prov), n in c.items() if cls == "A" and prov == "yes")
    a_tot = sum(n for (cls, _, _), n in c.items() if cls == "A")
    any_b = sum(1 for r in out if r["class"] == "B" and r["provenance_request_any_workflow"] == "yes")
    L += ["", f"合计：B 行 {b_yes}/{b_tot}（{len(pk[('B','yes')])} 包）在其发布 workflow 中显式请求 provenance；A 行 {a_yes}/{a_tot}。"
          f"按仓库任一缓存 workflow 计，B 为 {any_b}/{b_tot}。", "", "## 读法\n",
          "- 认证与 provenance 是两套机制：token 认证的发布只要 job 能铸造 id-token 并显式请求 provenance（`--provenance` 等）就会产生 attestation；"
          "trusted publishing 则自动生成、无需该 flag（npm Docs; GitHub Changelog 2023-04-19）。",
          "- 因此 B 类中 token 与 id-token 共现不是\"两条发布路径\"，而是 token 认证 + 显式 provenance 的常规架构之一；共现本身不能推出半迁移或 fallback。",
          f"- 未见显式请求的 {b_tot - b_yes} 个 B 行：workflow 中看不到请求；attestation 需要 trusted publishing（npm 侧注册与 CLI ≥ 11.5.1），"
          "或者写在 package.json 的 publishConfig / .npmrc 中的请求（缓存均未含），所以\"未见\"不等于\"没有\"。",
          "- 本读取的 token_secret 列（B 共现 27 / 无 secret 68）是对分类器命名的发布 workflow 重新匹配 NPM_TOKEN/NODE_AUTH_TOKEN 的结果，"
          "与 scripts/63 按冻结 reason 划分的 B-dual 29 / B-pure 65 / B-other 1 不是同一操作定义，不应视为逐行相同。",
          "- 本读取只补充 B 的原因空间，不判定任何一次发布实际走了哪条路径；主表标签不变。"]
    OUT_MD.write_text("\n".join(L) + "\n")
    print(f"B: {b_yes}/{b_tot} rows request provenance explicitly ({len(pk[('B','yes')])} packages); A: {a_yes}/{a_tot}; by mode: {dict(c)}")
    print(f"✓ {OUT_CSV.name}  ✓ {OUT_MD.name}\n✅ provenance-flag read complete", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
