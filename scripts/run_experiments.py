"""RQ1/RQ2: paired evaluation of A0-A4 on the Compliant Lie benchmark (smoke / dev / test splits)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import paths
from config import ExperimentConfig
from dataset import generate_benchmark, generate_seeds
from llm_cache import LLMCache
from metrics import summarize_group
from models import SCENARIOS, BenchmarkCase, VerifierResult
from llm_client import LLMClient, get_llm_client
from premise_contract import PREMISE_NAMES
from verifiers import (
    AirGappedCascadeVerifier,
    MultiAgentDebateVerifier,
    NoAirGapReconstructionVerifier,
    PremiseGuidedJudgeVerifier,
    SingleJudgeVerifier,
    StatelessDIMVerifier,
    Verifier,
)

logger: logging.Logger = logging.getLogger("run_experiments")

ALL_ARCHS: tuple[str, ...] = ("A0", "A1", "A2", "A3", "A3x", "A4", "A4-T3")
TABLE_ORDER: tuple[str, ...] = ("A0", "A1", "A2", "A3", "A3x", "A4", "A4-T3", "A5")
SMOKE_SEED: int = 99  # disjoint from dev (0) and test (1..10)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format='{"ts":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","event":"%(message)s"}',
    )


def build_verifier(arch: str, client: LLMClient, model: str, tau: float = 0.0) -> Verifier:
    match arch:
        case "A0":
            return StatelessDIMVerifier()
        case "A1":
            return SingleJudgeVerifier(client, model)
        case "A2":
            return MultiAgentDebateVerifier(client, model)
        case "A3":
            return NoAirGapReconstructionVerifier(client, model, include_context=False, tau=tau)
        case "A3x":
            return NoAirGapReconstructionVerifier(client, model, include_context=True, tau=tau)
        case "A4":
            return AirGappedCascadeVerifier(client, model, tau=tau)
        case "A4-T3":
            return AirGappedCascadeVerifier(client, model, tau=tau, use_gates=False)
        case "A5":
            return PremiseGuidedJudgeVerifier(client, model)
    raise ValueError(f"Unknown architecture {arch}")


def case_seed(case: BenchmarkCase) -> int:
    return int(case.case_id.rsplit("-", 1)[1]) if not case.case_id.startswith("F") else -1


def result_row(case: BenchmarkCase, r: VerifierResult) -> dict[str, Any]:
    row: dict[str, Any] = {
        "case_id": case.case_id,
        "seed": case_seed(case),
        "dfid": case.dfid,
        "scenario": case.scenario,
        "label": case.label,
        "subtype": case.subtype,
        "template_id": case.template_id,
        "policy_kind": case.a.policy_kind,
    }
    row.update(r.model_dump(exclude={"p_obs", "p_hat", "case_id"}))
    for k in PREMISE_NAMES:
        row[f"true_{k}"] = getattr(case.p_true, k)
        row[f"obs_{k}"] = getattr(r.p_obs, k) if r.p_obs is not None else None
        row[f"hat_{k}"] = r.p_hat.get(k) if r.p_hat is not None else None
    return row


def run_architecture(
    verifier: Verifier, cases: Sequence[BenchmarkCase], workers: int = 1
) -> list[dict[str, Any]]:
    """Cases are independent, so they run concurrently; result order matches input order."""
    t0 = time.perf_counter()
    done = 0

    def one(case: BenchmarkCase) -> dict[str, Any]:
        nonlocal done
        row = result_row(case, verifier.verify(case))
        done += 1
        if done % 30 == 0 or done == len(cases):
            logger.info(f"progress arch={verifier.architecture} done={done}/{len(cases)} elapsed_s={time.perf_counter() - t0:.0f}")
        return row

    if workers <= 1:
        return [one(c) for c in cases]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, cases))


def summarize(df: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for arch, g in df.groupby("architecture", sort=False):
        model = g["model"].dropna().iloc[0] if g["model"].notna().any() else None
        for scen in (*SCENARIOS, "ALL"):
            sub = g if scen == "ALL" else g[g["scenario"] == scen]
            records.append({"architecture": arch, "model": model, "scenario": scen, **summarize_group(sub, cfg, model)})
    summary = pd.DataFrame.from_records(records)
    base = summary[summary["architecture"] == "A0"].set_index("scenario")["c_total_eur"]
    summary["net_loss_reduction"] = [
        1.0 - c / base[s] if s in base.index and base[s] > 0 else float("nan")
        for c, s in zip(summary["c_total_eur"], summary["scenario"])
    ]
    order = {a: i for i, a in enumerate(TABLE_ORDER)}
    return summary.sort_values(["architecture", "scenario"], key=lambda col: col.map(order) if col.name == "architecture" else col).reset_index(drop=True)


def print_table(summary: pd.DataFrame) -> None:
    rows = []
    for arch, g in summary.groupby("architecture", sort=False):
        g = g.set_index("scenario")
        rec: dict[str, Any] = {"arch": arch}
        for s, tag in zip(SCENARIOS, ("S1", "S2", "S3")):
            rec[f"{tag} Rec"] = f"{100 * g.loc[s, 'recall']:.1f}"
            rec[f"{tag} FPR"] = f"{100 * g.loc[s, 'fpr']:.1f}"
        a = g.loc["ALL"]
        rec["Rec"] = f"{100 * a['recall']:.1f} [{100 * a['recall_ci_lo']:.1f},{100 * a['recall_ci_hi']:.1f}]"
        rec["FPR"] = f"{100 * a['fpr']:.1f} [{100 * a['fpr_ci_lo']:.1f},{100 * a['fpr_ci_hi']:.1f}]"
        rec["AUC"] = f"{a['auc']:.2f}"
        rec["Tok"] = f"{a['mean_tokens']:.0f}"
        rec["PE"] = int(a["parse_errors"])
        rows.append(rec)
    print("\n=== summary (Recall/FPR in %, Tok = mean verification tokens per flow, PE = parse errors) ===")
    print(pd.DataFrame(rows).to_string(index=False))


def load_split(cfg: ExperimentConfig, split: str) -> list[BenchmarkCase]:
    if split == "smoke":
        return generate_benchmark(SMOKE_SEED, 2)
    if split == "dev":
        return generate_benchmark(cfg.dev_seed, cfg.dev_cases_per_class_per_scenario)
    if split == "test":
        return generate_seeds(cfg.test_seeds, cfg.cases_per_class_per_scenario_per_seed)
    raise ValueError(split)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=("smoke", "dev", "test"), required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--archs", default=",".join(ALL_ARCHS))
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--tag", default="", help="Optional suffix for output files (e.g. prompt variant)")
    args = ap.parse_args()
    configure_logging()

    cfg = ExperimentConfig()
    model = args.model or cfg.default_model
    workers = args.workers or cfg.max_workers
    archs = [a.strip() for a in args.archs.split(",") if a.strip()]
    out_dir = paths.RESULTS_DIR / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    cache = LLMCache(paths.RESULTS_DIR / cfg.cache_filename)
    client = get_llm_client(cfg, cache)
    if hasattr(client, "check_connectivity"):
        client.check_connectivity((model,))

    cases = load_split(cfg, args.split)
    logger.info(f"split={args.split} cases={len(cases)} model={model} archs={archs} workers={workers}")

    rows: list[dict[str, Any]] = []
    for arch in archs:
        rows += run_architecture(build_verifier(arch, client, model), cases, workers)
    cache.close()

    df = pd.DataFrame.from_records(rows)
    df.to_csv(out_dir / f"per_case{args.tag}.csv", index=False)
    summary = summarize(df, cfg)
    summary.to_csv(out_dir / f"summary{args.tag}.csv", index=False)

    a4 = df[df["architecture"] == "A4"]
    meta: dict[str, Any] = {
        "split": args.split,
        "model": model,
        "n_cases": len(cases),
        "live_llm_calls": client.live_calls,
        "usage": client.usage_stats().model_dump() if hasattr(client, "usage_stats") else None,
        "a4_tier_counts": a4.groupby(["scenario", "label", "tier"]).size().reset_index(name="n").to_dict(orient="records"),
        "a4_gb_oracle_agreement": float(a4["gb_agreement"].dropna().astype(bool).mean()) if a4["gb_agreement"].notna().any() else None,
        "config": cfg.model_dump(exclude={"nvidia_api_key"}),
    }
    payload = {"meta": meta, "summary": summary.to_dict(orient="records")}
    (out_dir / f"summary{args.tag}.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print_table(summary)
    print(f"\nA4 g_B vs oracle pre-image agreement: {meta['a4_gb_oracle_agreement']}")
    print(f"Live LLM calls this run: {client.live_calls}")
    if hasattr(client, "usage_stats"):
        u = client.usage_stats()
        print(
            f"Tokens: prompt={u.prompt_tokens:,} completion={u.completion_tokens:,} total={u.total_tokens:,} | "
            f"HTTP requests={u.http_requests:,} | peak RPM={u.peak_requests_per_minute}"
        )


if __name__ == "__main__":
    main()
