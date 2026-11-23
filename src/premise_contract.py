"""Kernel-space pure functions: contract pre-image table, premise comparator, Tier 1/2 gates.

No I/O, no LLM calls (DIR-1 / DIR-7).
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from models import CanonicalPremiseVector, CustomerState

PremiseName = Literal["explicit_cancellation_requested", "delay_exceeds_48h", "churn_scope", "primary_driver"]
PREMISE_NAMES: tuple[PremiseName, ...] = (
    "explicit_cancellation_requested",
    "delay_exceeds_48h",
    "churn_scope",
    "primary_driver",
)
UNCONSTRAINED: str = "UNCONSTRAINED"


class PremisePreimage(BaseModel):
    """Assumed premises p_hat; None marks a coordinate the action does not constrain."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    explicit_cancellation_requested: bool | None = None
    delay_exceeds_48h: bool | None = None
    churn_scope: Literal["SINGLE_USER", "TEAM_ENTERPRISE"] | None = None
    primary_driver: Literal["PRICE_SENSITIVITY", "SERVICE_OR_TECH_FAILURE"] | None = None

    def constrained(self) -> tuple[PremiseName, ...]:
        return tuple(k for k in PREMISE_NAMES if getattr(self, k) is not None)

    def as_labels(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for k in PREMISE_NAMES:
            v = getattr(self, k)
            out[k] = UNCONSTRAINED if v is None else str(v).upper() if isinstance(v, bool) else str(v)
        return out


PREIMAGE_TABLE: Mapping[str, PremisePreimage] = MappingProxyType(
    {
        "APPLY_RETENTION_DISCOUNT": PremisePreimage(
            explicit_cancellation_requested=False, primary_driver="PRICE_SENSITIVITY"
        ),
        "ISSUE_SLA_COMPENSATION_REFUND": PremisePreimage(delay_exceeds_48h=True),
        "APPLY_BILLING_COURTESY_CREDIT": PremisePreimage(
            churn_scope="SINGLE_USER", primary_driver="PRICE_SENSITIVITY"
        ),
        "ESCALATE_ENTERPRISE_INCIDENT": PremisePreimage(
            churn_scope="TEAM_ENTERPRISE", primary_driver="SERVICE_OR_TECH_FAILURE"
        ),
    }
)
"""Oracle pre-images: used only to score g_B in tests and diagnostics, never shown to an LLM or used for verdicts."""


class UnknownPolicyKind(ValueError):
    """Action has no registered pre-image; fail closed."""


def reconstruct_declarative(policy_kind: str) -> PremisePreimage:
    try:
        return PREIMAGE_TABLE[policy_kind]
    except KeyError as exc:
        raise UnknownPolicyKind(policy_kind) from exc


def preimage_table_text() -> str:
    lines = ["Policy precondition table (premises that MUST hold for each action; UNCONSTRAINED = not required):"]
    for kind, pre in PREIMAGE_TABLE.items():
        parts = ", ".join(f"{k}={v}" for k, v in pre.as_labels().items())
        lines.append(f"- {kind}: {parts}")
    return "\n".join(lines)


def premise_delta(
    p_obs: CanonicalPremiseVector,
    p_hat: PremisePreimage | CanonicalPremiseVector,
    mask: tuple[PremiseName, ...],
    weights: Mapping[PremiseName, float] | None = None,
) -> float:
    """Weighted Hamming distance over action-constrained coordinates, normalized to [0, 1]."""
    if not mask:
        return 0.0
    w = {k: (weights[k] if weights is not None else 1.0) for k in mask}
    total = sum(w.values())
    if total <= 0:
        return 0.0
    mismatch = sum(w[k] for k in mask if getattr(p_obs, k) != getattr(p_hat, k))
    return mismatch / total


def trips(delta: float, tau: float) -> bool:
    return delta > tau + 1e-9


# ---------------------------------------------------------------- Tier 1 ---


def tier1_fact_check(policy_kind: str, params: Mapping[str, Any], s: CustomerState) -> str | None:
    """Differential SQL heuristics: bind proposal params against the authoritative fact view."""
    if not s.verified_email:
        return "FACT_VIOLATION:UNVERIFIED_IDENTITY"
    if policy_kind == "APPLY_RETENTION_DISCOUNT":
        if float(params.get("discount_pct", 0)) > s.max_discount_pct:
            return "FACT_VIOLATION:TIER_DISCOUNT_LIMIT"
    if policy_kind == "APPLY_BILLING_COURTESY_CREDIT" and s.account_seats > 1:
        return "FACT_VIOLATION:MULTI_SEAT_ACCOUNT"
    if policy_kind == "ISSUE_SLA_COMPENSATION_REFUND":
        if float(params.get("refund_eur", 0)) > s.refund_budget_eur:
            return "FACT_VIOLATION:REFUND_BUDGET"
        if s.shipping_delay_hours is not None and s.shipping_delay_hours <= 48:
            return "FACT_VIOLATION:TRACKED_DELAY_BELOW_48H"
    return None


# ---------------------------------------------------------------- Tier 2 ---

_CANCEL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bcancel(?:l?ing)?\s+(?:my|our|the)\s+(?:\w+\s+)?(?:subscription|account|membership|plan)\b", re.I),
    re.compile(r"\bclose\s+(?:my|our)\s+account\b", re.I),
    re.compile(r"\bterminate\s+(?:my|our)\s+(?:subscription|account|membership|plan)\b", re.I),
    re.compile(r"\bcancel(?:lation)?\s+(?:it\s+)?(?:immediately|now|today)\b", re.I),
)
_NEGATION = re.compile(r"\b(?:not|never|don't|do not|no intention to|not planning to|won't)\b[\w\s,']{0,20}$", re.I)


def detect_explicit_cancellation(x: str) -> bool:
    for pattern in _CANCEL_PATTERNS:
        for m in pattern.finditer(x):
            if not _NEGATION.search(x[max(0, m.start() - 40) : m.start()]):
                return True
    return False


def tier2_intent_guard(policy_kind: str, params: Mapping[str, Any], x: str) -> str | None:
    """Orthogonal keyword automaton for unambiguous directional inversions."""
    if bool(params.get("cancel_account", False)):
        return None
    if policy_kind in ("APPLY_RETENTION_DISCOUNT", "ISSUE_SLA_COMPENSATION_REFUND", "APPLY_BILLING_COURTESY_CREDIT"):
        if detect_explicit_cancellation(x):
            return "EVIDENTIAL_CONFLICT:EXPLICIT_CANCELLATION"
    return None
