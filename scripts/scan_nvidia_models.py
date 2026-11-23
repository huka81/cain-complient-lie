"""Scan NVIDIA NIM for actively available chat models.

Lists the hosted catalog and probes each chat model with a minimal request,
printing a classified report (available / rate-limited / EOL / not-found /
forbidden). Active probing consumes a tiny amount of free credits per model.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import ExperimentConfig
from nvidia_models import DiscoveryResult, find_working_models


def _print_report(result: DiscoveryResult) -> None:
    print(
        f"Scanned {len(result.scanned)} chat models "
        f"(skipped {len(result.skipped)} specialized).\n"
    )
    for model_id in result.working:
        print(f"  [AVAILABLE]        {model_id}")
    for model_id in result.rate_limited:
        print(f"  [RATE-LIMITED 429] {model_id}")
    for model_id in result.eol:
        print(f"  [EOL 410]          {model_id}")
    for model_id in result.not_found:
        print(f"  [NOT FOUND 404]    {model_id}")
    for model_id in result.forbidden:
        print(f"  [FORBIDDEN 401/403]{model_id}")
    for model_id, reason in result.errors:
        print(f"  [ERROR {reason}]    {model_id}")

    print("\n" + "=" * 55)
    print(f"WORKING MODELS ({len(result.working)}):")
    for model_id in result.working:
        print(f"  + {model_id}")
    if result.rate_limited:
        print(f"\nIN-PLAN BUT RATE-LIMITED ({len(result.rate_limited)}):")
        for model_id in result.rate_limited:
            print(f"  * {model_id}")
    print("=" * 55)
    print(
        f"Probe requests: {result.probe_requests} in {result.elapsed_s:.1f}s "
        f"(effective {result.requests_per_minute:.0f} req/min)"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--no-skip-specialized",
        action="store_true",
        help="Also probe embedding/rerank/audio/image/guard models.",
    )
    ap.add_argument("--timeout", type=float, default=None, help="Per-model probe timeout (s).")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(levelname)s %(name)s %(message)s")

    cfg = ExperimentConfig()
    if not cfg.nvidia_api_key:
        print("NVIDIA_API_KEY is not set (configure it in .env).")
        raise SystemExit(1)

    result = find_working_models(
        cfg,
        skip_specialized=not args.no_skip_specialized,
        timeout=args.timeout,
    )
    _print_report(result)


if __name__ == "__main__":
    main()
