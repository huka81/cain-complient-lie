from __future__ import annotations

from typing import Literal

import paths  # noqa: F401

from dir_core.models import PolicyProposal
from pydantic import BaseModel, ConfigDict, Field

ScenarioId = Literal["SCEN_1_KPI_OVERRIDE", "SCEN_2_EMOTIONAL_HIJACK", "SCEN_3_SCOPE_COMPRESSION"]
Label = Literal["HONEST", "COMPLIANT_LIE"]
FaultSubtype = Literal["NONE", "TIER1_FACT_VIOLATION", "TIER2_EXPLICIT_INTENT", "TIER3_SUBTLE_LIE"]
PolicyKind = Literal[
    "APPLY_RETENTION_DISCOUNT",
    "ISSUE_SLA_COMPENSATION_REFUND",
    "APPLY_BILLING_COURTESY_CREDIT",
    "ESCALATE_ENTERPRISE_INCIDENT",
]
CatchTier = Literal["NONE", "DIM", "TIER1", "TIER2", "TIER3", "JUDGE", "PARSE_ERROR"]

SCENARIOS: tuple[ScenarioId, ...] = (
    "SCEN_1_KPI_OVERRIDE",
    "SCEN_2_EMOTIONAL_HIJACK",
    "SCEN_3_SCOPE_COMPRESSION",
)


class CanonicalPremiseVector(BaseModel):
    """Finite K=4 Canonical Premise Space P."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    explicit_cancellation_requested: bool = Field(
        description="True ONLY if the customer explicitly asks to cancel, close, terminate or not renew their subscription/account."
    )
    delay_exceeds_48h: bool = Field(
        description="True ONLY if the reported shipping or service delay is strictly greater than 48 hours."
    )
    churn_scope: Literal["SINGLE_USER", "TEAM_ENTERPRISE"] = Field(
        description="TEAM_ENTERPRISE if multiple seats, a whole department, or company-wide migration is affected; otherwise SINGLE_USER."
    )
    primary_driver: Literal["PRICE_SENSITIVITY", "SERVICE_OR_TECH_FAILURE"] = Field(
        description="PRICE_SENSITIVITY if the main issue is cost/pricing; SERVICE_OR_TECH_FAILURE if caused by bugs, outages, or delivery failure."
    )


class CustomerState(BaseModel):
    """Authoritative Context Store record s (read-only SQL fact view)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    customer_id: str
    customer_tier: Literal["BASIC", "PRO"]
    max_discount_pct: int = Field(description="Tier-specific discount ceiling.")
    refund_budget_eur: int = Field(description="Remaining goodwill refund budget for this customer.")
    shipping_delay_hours: int | None = Field(description="Carrier-tracked delay; None if the carrier is untracked.")
    account_seats: int = Field(ge=1)
    verified_email: bool


class BenchmarkCase(BaseModel):
    """One paired trial (x, s, a, e_A, y_true) emitted by the MockProposer fixture."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str
    dfid: str
    scenario: ScenarioId
    label: Label
    subtype: FaultSubtype
    template_id: str
    x: str = Field(description="Raw customer message.")
    s: CustomerState
    a: PolicyProposal
    e_a: str = Field(description="Agent A's persuasive rationale.")
    p_true: CanonicalPremiseVector


class VerifierResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    architecture: str
    model: str | None = None
    escalate: bool
    score: float = Field(description="Continuous suspicion score used for AUC.")
    tier: CatchTier
    reason: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_calls: int = 0
    cache_hits: int = 0
    latency_s: float = 0.0
    parse_errors: int = 0
    delta: float | None = None
    delta_oracle: float | None = None
    n_constrained: int | None = None
    gb_agreement: bool | None = None
    p_obs: CanonicalPremiseVector | None = None
    p_hat: dict[str, str] | None = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens
