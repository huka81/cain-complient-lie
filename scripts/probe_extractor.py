"""Dev-set probe of extractor premise accuracy (seed != evaluation seed)."""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from collections import Counter
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import paths  # noqa: F401

from config import ExperimentConfig
from dataset import generate_benchmark
from llm_cache import LLMCache
from llm_client import get_llm_client
from premise_contract import PREMISE_NAMES, reconstruct_declarative
from verifiers import extract_premises, reconstruct_airgapped_llm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3.2:3b")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--per-class", type=int, default=5)
    ap.add_argument("--mode", choices=("facts", "direct"), default="facts")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    cfg = ExperimentConfig()
    with tempfile.TemporaryDirectory() as tmp:
        cache = LLMCache(Path(tmp) / "probe.sqlite")
        client = get_llm_client(cfg, cache)
        errs: Counter[str] = Counter()
        cases = generate_benchmark(args.seed, args.per_class)
        for c in cases:
            p, _ = extract_premises(client, args.model, c.x, args.mode)
            if p is None:
                errs["PARSE"] += 1
                continue
            for k in PREMISE_NAMES:
                if getattr(p, k) != getattr(c.p_true, k):
                    errs[f"{c.scenario}:{c.label}:{k}"] += 1
        agree = 0
        for c in cases:
            pre, _ = reconstruct_airgapped_llm(client, args.model, c.a)
            agree += int(pre == reconstruct_declarative(c.a.policy_kind))
        print(f"g_B LLM/declarative agreement: {agree}/{len(cases)}")
        cache.close()
    print(f"model={args.model} mode={args.mode} cases={len(cases)} coordinate_errors={sum(errs.values())}")
    for k, v in sorted(errs.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
