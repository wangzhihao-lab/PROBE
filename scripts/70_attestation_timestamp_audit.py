"""
TSE R1 round-2 audit arm A — Attestation signing-time versus publish-time audit.

Why: the predicate-migration audit (scripts/55) only shows that historical
attestations were not rewritten wholesale; it cannot show that each attestation
was attached when its version was published. The Sigstore bundle behind every
`dist.attestations` entry is an immutable object that carries the Rekor
transparency-log `integratedTime` of the signature. Comparing that time with the
registry publish time answers the question directly for a stratified sample.

Design (deterministic and declared in this script before the run; the script was not committed to version control before the fetch, so the ordering rests on this declaration):
  - Universe: attested versions in data/versions_extended.csv published in the
    extended pre-window [2024-01-01, 2025-07-31] (the part of the series the
    extension added) plus a same-size-per-quarter comparison from the protocol
    window [2025-08-01, 2026-05-10].
  - Sample: N_PER_QUARTER versions per calendar quarter, drawn with seed 42
    without replacement, so every quarter of the series is represented equally.
  - For each sampled version: GET its attestation bundle from the registry URL
    recorded in the frozen version table, store the raw JSON, and read the
    `integratedTime` of the SLSA provenance attestation (predicateType containing
    `slsa.dev/provenance`) and of the npm publish attestation.
  - delta_min = (integratedTime - published_at) in minutes. Attachment at publish
    time predicts |delta| of seconds to minutes; registry-side backfill would
    show integratedTime months after publish (or after the data cutoff).

Firewall: every audited version was published on or before the 2026-05-10 data
cutoff (asserted); the bundle is an immutable artefact whose timestamps
pre-date the cutoff (asserted). The fetch itself happened after the cutoff and
is recorded as a separate collection date. Nothing here identifies an effect.

Inputs:
  data/versions_extended.csv           (frozen; attestation_url column)
Outputs:
  data/attestation_bundles_raw/<pkg>@<ver>.json   (raw bundles; new directory)
  data/tse_r1_attestation_timestamp_audit.csv
  reports/tse_r1_attestation_timestamp_audit.md

Run: conda run --no-capture-output -n stage2c python scripts/70_attestation_timestamp_audit.py
"""
from __future__ import annotations

import csv
import json
import random
import statistics
import sys
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).parent.parent
DATA, REPORTS = ROOT / "data", ROOT / "reports"
VERSIONS_CSV = DATA / "versions_extended.csv"
RAW_DIR = DATA / "attestation_bundles_raw"
OUT_CSV = DATA / "tse_r1_attestation_timestamp_audit.csv"
OUT_MD = REPORTS / "tse_r1_attestation_timestamp_audit.md"

DATA_CUTOFF = datetime(2026, 5, 10, 23, 59, 59, tzinfo=timezone.utc)
PRE_START, PRE_END = datetime(2024, 1, 1, tzinfo=timezone.utc), datetime(2025, 7, 31, 23, 59, 59, tzinfo=timezone.utc)
N_PER_QUARTER = 30
SEED = 42
SLEEP = 0.25
HEADERS = {"User-Agent": "PROBE-attestation-audit/1.0 (research; contact via repository)"}


def parse_dt(s: str) -> datetime | None:
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def quarter(d: datetime) -> str:
    return f"{d.year}Q{(d.month - 1) // 3 + 1}"


def load_universe() -> list[dict]:
    rows = []
    with VERSIONS_CSV.open() as f:
        for r in csv.DictReader(f):
            if str(r.get("has_attestation", "")).lower() != "true" or not r.get("attestation_url"):
                continue
            d = parse_dt(r["published_at"])
            if d is None or d > DATA_CUTOFF:
                continue
            r["_dt"] = d
            r["_window"] = "pre_window" if PRE_START <= d <= PRE_END else ("protocol_window" if d > PRE_END else "before_series")
            rows.append(r)
    return [r for r in rows if r["_window"] != "before_series"]


def draw_sample(universe: list[dict]) -> list[dict]:
    rng = random.Random(SEED)
    by_q: dict[str, list[dict]] = {}
    for r in universe:
        by_q.setdefault(quarter(r["_dt"]), []).append(r)
    sample = []
    for q in sorted(by_q):
        pool = sorted(by_q[q], key=lambda r: (r["package"], r["version"]))
        take = pool if len(pool) <= N_PER_QUARTER else rng.sample(pool, N_PER_QUARTER)
        sample.extend(take)
    return sample


def fetch_bundle(url: str, dest: Path) -> tuple[str, dict | None]:
    if dest.exists():
        try:
            return "cached", json.loads(dest.read_text())
        except Exception:
            pass
    for attempt, backoff in enumerate([0, 5, 15, 45]):
        if backoff:
            time.sleep(backoff)
        try:
            resp = requests.get(url, headers=HEADERS, timeout=45)
        except Exception as e:  # network error: retry
            last = f"error: {str(e)[:60]}"
            continue
        if resp.status_code == 200:
            dest.write_text(resp.text)
            return "fetched", resp.json()
        if resp.status_code == 404:
            return "http_404", None
        last = f"http_{resp.status_code}"
        if resp.status_code not in (429, 500, 502, 503, 504):
            return last, None
    return last, None


def integrated_times(bundle: dict) -> dict[str, int | None]:
    out: dict[str, int | None] = {"provenance": None, "publish": None}
    for att in bundle.get("attestations", []) or []:
        ptype = att.get("predicateType", "") or ""
        entries = ((att.get("bundle") or {}).get("verificationMaterial") or {}).get("tlogEntries") or []
        t = min((int(e["integratedTime"]) for e in entries if e.get("integratedTime")), default=None)
        key = "provenance" if "slsa.dev/provenance" in ptype else ("publish" if "npm/attestation" in ptype or "publish" in ptype else None)
        if key and out[key] is None:
            out[key] = t
        att_pred = ptype
        out.setdefault("_predicates", []).append(att_pred)  # type: ignore[arg-type]
    return out


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    universe = load_universe()
    sample = draw_sample(universe)
    print(f"universe: {len(universe):,} attested versions in [2024-01-01, 2026-05-10]; "
          f"sample: {len(sample)} ({N_PER_QUARTER}/quarter, seed {SEED})", flush=True)
    collected_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    rows = []
    for i, r in enumerate(sample, 1):
        pkg, ver, d = r["package"], r["version"], r["_dt"]
        assert d <= DATA_CUTOFF, "audited version post-dates the data cutoff"
        dest = RAW_DIR / f"{urllib.parse.quote(pkg, safe='')}@{ver}.json"
        status, bundle = fetch_bundle(r["attestation_url"], dest)
        rec = {"package": pkg, "version": ver, "published_at": r["published_at"], "window": r["_window"],
               "quarter": quarter(d), "tier": r.get("tier", ""), "fetch_status": status,
               "predicate_types": "", "provenance_integrated_time": "", "publish_integrated_time": "",
               "delta_provenance_min": "", "delta_publish_min": "", "collected_at": collected_at}
        if bundle:
            it = integrated_times(bundle)
            rec["predicate_types"] = "|".join(it.get("_predicates", []))  # type: ignore[arg-type]
            for key in ("provenance", "publish"):
                t = it.get(key)
                if t is not None:
                    ts = datetime.fromtimestamp(t, timezone.utc)
                    assert ts <= DATA_CUTOFF, "bundle timestamp post-dates the data cutoff"
                    rec[f"{key}_integrated_time"] = ts.isoformat()
                    rec[f"delta_{key}_min"] = round((ts - d).total_seconds() / 60, 3)
        rows.append(rec)
        if status == "fetched":
            time.sleep(SLEEP)
        if i % 25 == 0:
            print(f"  {i}/{len(sample)} done", flush=True)
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"✓ {OUT_CSV.name}", flush=True)

    # ---- report ----
    ok = [r for r in rows if r["delta_provenance_min"] != ""]
    L = ["# 审计臂 A — attestation 签名时间 vs 发布时间（TSE R1 round 2）\n",
         f"采集时间（bundle 抓取）：{collected_at}；样本：每季度 {N_PER_QUARTER} 个 attested 版本，seed {SEED}；",
         f"母体 {len(universe):,} 个 attested 版本（2024-01-01 至 2026-05-10）。所有被审计版本发布日 ≤ 2026-05-10（断言）。\n",
         f"抓取成功：{len(ok)}/{len(rows)}（{100*len(ok)/len(rows):.1f}%）；失败状态：" +
         ", ".join(f"{s}={sum(1 for r in rows if r['fetch_status']==s)}" for s in sorted({r['fetch_status'] for r in rows} - {'fetched','cached'})) + "\n",
         "| 窗口 | n | |Δ| 中位数 (min) | p95 (min) | 最大 (min) | ≤1 h | ≤24 h | Δ<0 (签名早于发布) |", "|---|---|---|---|---|---|---|---|"]
    for win in ("pre_window", "protocol_window"):
        sub = [float(r["delta_provenance_min"]) for r in ok if r["window"] == win]
        if not sub:
            continue
        a = sorted(abs(x) for x in sub)
        p95 = a[min(len(a) - 1, int(round(0.95 * len(a))) - 1)] if len(a) > 1 else a[0]
        L.append(f"| {win} | {len(sub)} | {statistics.median(a):.2f} | {p95:.2f} | {max(a):.2f} | "
                 f"{100*sum(1 for x in a if x <= 60)/len(a):.1f}% | {100*sum(1 for x in a if x <= 1440)/len(a):.1f}% | "
                 f"{sum(1 for x in sub if x < 0)} |")
    L += ["", "## 按季度\n", "| 季度 | n | |Δ| 中位数 (min) | 最大 (min) | predicate 类型 |", "|---|---|---|---|---|"]
    for q in sorted({r["quarter"] for r in ok}):
        sub = [r for r in ok if r["quarter"] == q]
        a = [abs(float(r["delta_provenance_min"])) for r in sub]
        preds = sorted({p.split("/")[-1] for r in sub for p in r["predicate_types"].split("|") if "slsa" in p})
        L.append(f"| {q} | {len(sub)} | {statistics.median(a):.2f} | {max(a):.2f} | {', '.join(preds)} |")
    worst = sorted(ok, key=lambda r: -abs(float(r["delta_provenance_min"])))[:5]
    L += ["", "## |Δ| 最大的 5 个版本\n", "| 包 | 版本 | 发布 | 签名 integratedTime | Δ (min) |", "|---|---|---|---|---|"]
    for r in worst:
        L.append(f"| `{r['package']}` | {r['version']} | {r['published_at']} | {r['provenance_integrated_time']} | {float(r['delta_provenance_min']):+.2f} |")
    L += ["", "## 读法\n",
          "- 若 |Δ| 以秒到分钟计且 ≤24 h 的比例 ≥95%，则样本内的 attestation 是与发布**同时签名并登记**的；Rekor 时间不直接时间戳 dist.attestations 字段的首次出现，",
          "  因此它支持但不直接观测\"发布时附着\"；签名层面的注册表事后回填在样本内被排除，这把 scripts/55 的\"反对整体重写\"向前推进一步。",
          "- Δ 略为负值属正常：签名与 Rekor 登记先于注册表写入发布时间。",
          "- 该臂不涉及任何效应识别；它只支撑扩展窗结果变量的构念（按发布日期索引的 attestation 字段状态）。"]
    OUT_MD.write_text("\n".join(L) + "\n")
    print(f"✓ {OUT_MD.name}\n✅ audit arm A complete", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
