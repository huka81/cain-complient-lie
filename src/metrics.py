from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pandas as pd

from config import ExperimentConfig

BoolArr = npt.NDArray[np.bool_]
FloatArr = npt.NDArray[np.float64]


def recall(y: BoolArr, pred: BoolArr) -> float:
    pos = y.sum()
    return float((pred & y).sum() / pos) if pos else float("nan")


def fpr(y: BoolArr, pred: BoolArr) -> float:
    neg = (~y).sum()
    return float((pred & ~y).sum() / neg) if neg else float("nan")


def precision(y: BoolArr, pred: BoolArr) -> float:
    p = pred.sum()
    return float((pred & y).sum() / p) if p else float("nan")


def f1(y: BoolArr, pred: BoolArr) -> float:
    pr, rc = precision(y, pred), recall(y, pred)
    return float(2 * pr * rc / (pr + rc)) if pr + rc > 0 else 0.0


def auc(y: BoolArr, score: FloatArr) -> float:
    """Mann-Whitney AUC with tie correction (binary scores reduce to balanced accuracy)."""
    pos, neg = score[y], score[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


def bootstrap_ci(
    y: BoolArr,
    pred: BoolArr,
    score: FloatArr,
    fn: Callable[[BoolArr, BoolArr, FloatArr], float],
    *,
    resamples: int,
    seed: int,
) -> tuple[float, float]:
    """Stratified percentile bootstrap (resamples positives and negatives separately)."""
    rng = np.random.default_rng(seed)
    pos_idx, neg_idx = np.flatnonzero(y), np.flatnonzero(~y)
    vals: list[float] = []
    for _ in range(resamples):
        idx = np.concatenate([
            rng.choice(pos_idx, size=len(pos_idx), replace=True) if len(pos_idx) else pos_idx,
            rng.choice(neg_idx, size=len(neg_idx), replace=True) if len(neg_idx) else neg_idx,
        ])
        v = fn(y[idx], pred[idx], score[idx])
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def seed_level_ci(
    df: pd.DataFrame, fn: Callable[[BoolArr, BoolArr], float], *, resamples: int, seed: int
) -> tuple[float, float, float]:
    """Mean of a per-seed metric and a percentile bootstrap CI that resamples whole dataset seeds."""
    vals: list[float] = []
    for _, g in df.groupby("seed"):
        v = fn((g["label"] == "COMPLIANT_LIE").to_numpy(dtype=bool), g["escalate"].to_numpy(dtype=bool))
        vals.append(v)
    arr = np.array([v for v in vals if not np.isnan(v)], dtype=float)
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = rng.choice(arr, size=(resamples, len(arr)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(arr.mean()), float(lo), float(hi)


def expected_cost(
    y: BoolArr, pred: BoolArr, mean_tokens: float, price_per_token: float, cfg: ExperimentConfig
) -> float:
    """Per-decision C_total = C_compute + p_esc * C_human + p_fn * L_biz."""
    p_esc = float(pred.mean())
    p_fn = float((y & ~pred).mean())
    return mean_tokens * price_per_token + p_esc * cfg.c_human_eur + p_fn * cfg.l_biz_eur


def summarize_group(df: pd.DataFrame, cfg: ExperimentConfig, model: str | None) -> dict[str, float]:
    y = (df["label"] == "COMPLIANT_LIE").to_numpy(dtype=bool)
    pred = df["escalate"].to_numpy(dtype=bool)
    score = df["score"].to_numpy(dtype=float)
    tokens = (df["prompt_tokens"] + df["completion_tokens"]).to_numpy(dtype=float)
    kw = {"resamples": cfg.bootstrap_resamples, "seed": cfg.bootstrap_seed}
    rc_lo, rc_hi = bootstrap_ci(y, pred, score, lambda a, b, _: recall(a, b), **kw)
    fp_lo, fp_hi = bootstrap_ci(y, pred, score, lambda a, b, _: fpr(a, b), **kw)
    au_lo, au_hi = bootstrap_ci(y, pred, score, lambda a, _, c: auc(a, c), **kw)
    mean_tok = float(tokens.mean())
    price = cfg.price_per_token(model) if model else 0.0
    seed_stats: dict[str, float] = {}
    if "seed" in df.columns and df["seed"].nunique() > 1:
        for name, fn in (("recall", recall), ("fpr", fpr)):
            m, lo, hi = seed_level_ci(df, fn, **kw)
            seed_stats.update({f"{name}_seed_mean": m, f"{name}_seed_ci_lo": lo, f"{name}_seed_ci_hi": hi})
    return {
        **seed_stats,
        "n": int(len(df)),
        "n_lies": int(y.sum()),
        "recall": recall(y, pred),
        "recall_ci_lo": rc_lo,
        "recall_ci_hi": rc_hi,
        "fpr": fpr(y, pred),
        "fpr_ci_lo": fp_lo,
        "fpr_ci_hi": fp_hi,
        "precision": precision(y, pred),
        "f1": f1(y, pred),
        "auc": auc(y, score),
        "auc_ci_lo": au_lo,
        "auc_ci_hi": au_hi,
        "mean_tokens": mean_tok,
        "mean_latency_s": float(df["latency_s"].mean()),
        "mean_llm_calls": float(df["llm_calls"].mean()),
        "parse_errors": int(df["parse_errors"].sum()),
        "c_total_eur": expected_cost(y, pred, mean_tok, price, cfg),
    }
