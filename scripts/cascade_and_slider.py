"""RQ3: Exp 2 (3-tier cascade on a 1,000-flow stream) and the Financial Slider C_total(tau) from the test-split run."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import paths
from config import ExperimentConfig
from dataset import make_cascade_stream
from llm_cache import LLMCache
from llm_client import LLMClient, get_llm_client
from metrics import fpr, recall
from run_experiments import configure_logging, run_architecture
from verifiers import AirGappedCascadeVerifier

logger: logging.Logger = logging.getLogger("cascade_and_slider")

HALT_TIERS: frozenset[str] = frozenset({"DIM", "TIER1", "TIER2", "PARSE_ERROR"})
TEST_DIR: Path = paths.RESULTS_DIR / "test"


def _tokens(df: pd.DataFrame) -> pd.Series:
    return df["prompt_tokens"] + df["completion_tokens"]


def run_exp2(cfg: ExperimentConfig, client: LLMClient, model: str, n_flows: int) -> dict[str, Any]:
    stream = make_cascade_stream(cfg.dataset_seed, n_flows)
    df = pd.DataFrame.from_records(run_architecture(AirGappedCascadeVerifier(client, model), stream, cfg.max_workers))
    df.to_csv(TEST_DIR / "exp2_cascade_per_flow.csv", index=False)

    faults = df[df["label"] == "COMPLIANT_LIE"]
    tier_by_subtype = faults.groupby(["subtype", "tier"]).size().unstack(fill_value=0)
    n_faults = len(faults)
    caught = faults[faults["escalate"]]
    share_of_faults = {t: float((caught["tier"] == t).sum() / n_faults) for t in ("TIER1", "TIER2", "TIER3", "PARSE_ERROR")}
    share_of_flows = {t: float((df["tier"] == t).sum() / len(df)) for t in ("TIER1", "TIER2")}
    reached_t3 = int((~df["tier"].isin(["DIM", "TIER1", "TIER2"])).sum())

    rq1 = json.loads((TEST_DIR / "summary.json").read_text(encoding="utf-8"))["summary"]
    per_flow = {r["architecture"]: r["mean_tokens"] for r in rq1 if r["scenario"] == "ALL"}
    y = (df["label"] == "COMPLIANT_LIE").to_numpy(dtype=bool)
    pred = df["escalate"].to_numpy(dtype=bool)
    a4_total = int(_tokens(df).sum())
    result: dict[str, Any] = {
        "model": model,
        "n_flows": len(df),
        "composition": df["subtype"].value_counts().to_dict(),
        "tier_by_subtype": {k: {kk: int(vv) for kk, vv in v.items()} for k, v in tier_by_subtype.to_dict(orient="index").items()},
        "share_of_faults_caught_by_tier": share_of_faults,
        "share_of_flows_halted_by_tier": share_of_flows,
        "flows_reaching_tier3": reached_t3,
        "a4_recall": recall(y, pred),
        "a4_fpr": fpr(y, pred),
        "tokens_total": {
            "A4": a4_total,
            "A1_extrapolated": float(per_flow.get("A1", float("nan")) * len(df)),
            "A2_extrapolated": float(per_flow.get("A2", float("nan")) * len(df)),
        },
        "tokens_per_flow_test": per_flow,
    }
    for base in ("A1", "A2"):
        tot = result["tokens_total"][f"{base}_extrapolated"]
        result[f"token_reduction_vs_{base}"] = 1.0 - a4_total / tot if tot and not np.isnan(tot) else float("nan")
    return result


def slider(cfg: ExperimentConfig, model: str) -> dict[str, Any]:
    """C_total(tau) for A4 (and A4-T3 / A3 for reference) plus constant-cost baselines A0, A1, A2, all from the test run."""
    df = pd.read_csv(TEST_DIR / "per_case.csv")
    summary = pd.DataFrame(json.loads((TEST_DIR / "summary.json").read_text(encoding="utf-8"))["summary"])
    n_fine = int(round(1.0 / cfg.tau_fine_step))
    taus = sorted({round(i * cfg.tau_fine_step, 4) for i in range(n_fine + 1)} | set(cfg.tau_grid))
    price = cfg.price_per_token(model)

    curves: dict[str, list[dict[str, float]]] = {}
    for arch in ("A4", "A4-T3", "A3"):
        g = df[df["architecture"] == arch]
        y = (g["label"] == "COMPLIANT_LIE").to_numpy(dtype=bool)
        halted = g["tier"].isin(list(HALT_TIERS)).to_numpy(dtype=bool)
        delta = g["delta"].fillna(0.0).to_numpy(dtype=float)
        c_compute = float(_tokens(g).mean()) * price
        out: list[dict[str, float]] = []
        for tau in taus:
            esc = halted | (delta > tau + 1e-9)
            p_esc, p_fn = float(esc.mean()), float((y & ~esc).mean())
            out.append({
                "tau": tau, "p_esc": p_esc, "p_fn": p_fn, "recall": recall(y, esc), "fpr": fpr(y, esc),
                "c_compute": c_compute, "c_total": c_compute + p_esc * cfg.c_human_eur + p_fn * cfg.l_biz_eur,
            })
        curves[arch] = out
    flat = {r["architecture"]: float(r["c_total_eur"]) for r in summary.to_dict(orient="records") if r["scenario"] == "ALL"}
    best = min(curves["A4"], key=lambda r: r["c_total"])
    return {
        "model": model,
        "taus": taus,
        "curves": curves,
        "flat_baselines": {k: flat[k] for k in ("A0", "A1", "A2") if k in flat},
        "tau_star_a4": best["tau"],
        "c_total_min_a4": best["c_total"],
        "params": {"c_human_eur": cfg.c_human_eur, "l_biz_eur": cfg.l_biz_eur, "eur_per_million_tokens": cfg.eur_per_million_tokens.get(model)},
    }


def print_summary(exp2: dict[str, Any] | None, sl: dict[str, Any] | None) -> None:
    if exp2:
        print("\n=== Exp 2: 3-tier cascade on", exp2["n_flows"], "flows ===")
        print("Faults caught by tier:", {k: f"{100 * v:.1f}%" for k, v in exp2["share_of_faults_caught_by_tier"].items()})
        print("Flows reaching Tier 3:", exp2["flows_reaching_tier3"], f"| A4 recall={100 * exp2['a4_recall']:.1f}% FPR={100 * exp2['a4_fpr']:.1f}%")
        print("Tokens:", {k: f"{v:,.0f}" for k, v in exp2["tokens_total"].items()})
        print(f"Token reduction vs A1={100 * exp2['token_reduction_vs_A1']:.1f}%  vs A2={100 * exp2['token_reduction_vs_A2']:.1f}%")
    if sl:
        print("\n=== Financial Slider (EUR / decision) ===")
        grid = {c["tau"]: c for c in sl["curves"]["A4"]}
        print({f"A4 tau={t}": round(grid[t]["c_total"], 3) for t in (0.0, 0.25, 0.5, 0.75, 1.0)})
        print("flat baselines:", {k: round(v, 3) for k, v in sl["flat_baselines"].items()}, "| tau*:", sl["tau_star_a4"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp", choices=("2", "slider", "all"), default="all")
    ap.add_argument("--flows", type=int, default=None)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    configure_logging()
    cfg = ExperimentConfig()
    model = args.model or cfg.default_model
    TEST_DIR.mkdir(parents=True, exist_ok=True)

    exp2 = sl = None
    client: LLMClient | None = None
    if args.exp in ("2", "all"):
        cache = LLMCache(paths.RESULTS_DIR / cfg.cache_filename)
        client = get_llm_client(cfg, cache)
        exp2 = run_exp2(cfg, client, model, args.flows or cfg.cascade_flows)
        (TEST_DIR / "exp2_cascade.json").write_text(json.dumps(exp2, indent=2), encoding="utf-8")
        cache.close()
    if args.exp in ("slider", "all"):
        sl = slider(cfg, model)
        (TEST_DIR / "slider.json").write_text(json.dumps(sl, indent=2), encoding="utf-8")
    print_summary(exp2, sl)
    if client is not None:
        print(f"Live LLM calls this run: {client.live_calls}")
        if hasattr(client, "usage_stats"):
            u = client.usage_stats()
            print(
                f"Tokens: prompt={u.prompt_tokens:,} completion={u.completion_tokens:,} total={u.total_tokens:,} | "
                f"HTTP requests={u.http_requests:,} | peak RPM={u.peak_requests_per_minute}"
            )


if __name__ == "__main__":
    main()
