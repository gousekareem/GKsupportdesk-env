"""
Typed Pydantic models — OpenEnv spec compliant.
7 tasks: classify, prioritize, resolve, sentiment, summarise, compliance, deescalate.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class TicketObservation(BaseModel):
    """What the agent observes each step."""
    ticket_id: str
    subject: str
    body: str
    customer_email: str
    customer_name: str = ""
    customer_history: List[Dict[str, Any]] = Field(default_factory=list)
    customer_lifetime_value: int = 0
    available_categories: List[str]
    available_priorities: List[str]
    available_sentiments: List[str] = Field(default_factory=list)
    available_compliance_types: List[str] = Field(default_factory=list)
    task_name: str
    task_description: str
    step: int
    max_steps: int
    # Emotional journey state (de-escalation task)
    emotional_state: Optional[float] = Field(
        default=None,
        description="Current customer anger/frustration level 0.0=calm 1.0=furious"
    )
    # SLA context
    sla_hours_remaining: Optional[int] = None
    # Context (batch tickets, conversation history, etc.)
    context: Dict[str, Any] = Field(default_factory=dict)
    language: str = "en"


class TicketAction(BaseModel):
    """Agent action — all fields optional, used by relevant tasks."""
    # classify + resolve
    category: Optional[str] = None
    priority: Optional[str] = None
    # resolve
    response_draft: Optional[str] = None
    escalate: Optional[bool] = None
    # prioritize
    priority_ranking: Optional[List[str]] = None
    # sentiment
    detected_sentiment: Optional[str] = None
    churn_risk: Optional[bool] = None
    # summarise
    summary: Optional[str] = None
    root_cause: Optional[str] = None
    resolution_status: Optional[str] = None
    # compliance
    compliance_violation_detected: Optional[str] = None
    compliant_response: Optional[str] = None
    # de-escalation
    deescalation_response: Optional[str] = None
    empathy_shown: Optional[bool] = None
    concrete_action_offered: Optional[bool] = None
    # meta
    reasoning: Optional[str] = None


class StepResult(BaseModel):
    """Return value from env.step()."""
    observation: TicketObservation
    reward: float = Field(ge=0.0, le=1.0)
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class ResetResult(BaseModel):
    """Return value from env.reset()."""
    observation: TicketObservation
    session_id: str
    info: Dict[str, Any] = Field(default_factory=dict)


class StateResult(BaseModel):
    """Return value from env.state()."""
    task_name: str
    session_id: str
    step: int
    max_steps: int
    total_reward: float
    avg_reward: float
    done: bool
    emotional_trajectory: List[float] = Field(default_factory=list)
    grader_history: List[Dict[str, Any]] = Field(default_factory=list)
