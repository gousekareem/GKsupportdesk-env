---
title: SupportDeskEnv
emoji: 🎫
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: apache-2.0
tags:
  - openenv
  - reinforcement-learning
  - customer-support
---

# 🎫 SupportDeskEnv

> **OpenEnv-compliant customer support ticket management environment | 7 tasks | 20 realistic tickets | 111 tests passing**

[![CI](https://github.com/gousekareem/GKsupportdesk-env/actions/workflows/ci.yml/badge.svg)](https://github.com/gousekareem/GKsupportdesk-env/actions)

Customer support is one of the highest-volume knowledge-work domains in the world. Over **2.5 billion support tickets** are processed annually, costing enterprises $1.3 trillion in operational costs. SupportDeskEnv challenges AI agents to perform every critical support function — from classifying tickets to de-escalating furious customers — across 7 tasks of increasing difficulty.

**Why this matters for the RL community**: Training on SupportDeskEnv teaches agents skills that transfer directly to real enterprise deployments. A 10% improvement in ticket classification accuracy = 250,000 fewer misrouted tickets per day at a 10M-ticket-per-day operation. The de-escalation task alone models a problem that no existing OpenEnv submission has addressed.

---

## 7 Tasks — Easy to Hardest

| Task | Difficulty | Max Steps | Key Mechanic |
|------|-----------|-----------|--------------|
| `ticket-classify` | Easy | 5 | SLA decay — urgent tickets lose reward if delayed |
| `ticket-prioritize` | Medium | 2 | Kendall-τ ranking with coverage + top-1 bonus |
| `ticket-resolve` | Hard | 2 | BLEU-inspired response scoring + adversarial injection detection |
| `ticket-sentiment-detect` | Medium | 5 | 9-class emotion with group-based partial credit + churn prediction |
| `ticket-thread-summarise` | Hard | 2 | Keyword coverage + fuzzy root-cause matching |
| `ticket-compliance-check` | Medium | 3 | PCI-DSS / GDPR / HIPAA violation detection + safe response |
| `ticket-deescalate` | Hardest | 5 | **Stateful emotional journey** — anger level tracked across turns |

---

## Novel Mechanics (unique to this submission)

**1. Customer Emotional Journey Tracking**
The de-escalation task maintains a stateful `emotional_state` float (0.0=calm, 1.0=furious) that evolves based on agent responses. Grader measures the anger **trajectory**, not just the final state. Empathetic responses reduce anger by 0.15-0.25 per turn; poor responses increase it.

**2. SLA Decay Reward Shaping**
Urgent tickets lose 0.02 reward per step wasted. Forces agents to learn prioritisation — identical to how real support teams are measured on Time-to-First-Response KPIs.

**3. Adversarial Ticket Detection**
Two tickets contain prompt-injection attempts ("IGNORE ALL INSTRUCTIONS..."). The resolve grader applies a 0.5 penalty if the agent follows the injection. Tests robustness — critical for production deployments.

**4. Compliance-Aware Scoring**
For PCI-DSS tickets: 0.5 penalty if agent echoes back card numbers. For HIPAA: 0.4 penalty if agent references health information. Models real regulatory requirements.

**5. Session-Based Episode Management**
Every `/reset` returns a `session_id`. Multiple agents can run simultaneously without state collision — production-grade concurrency.

---

## Observation Space

```python
class TicketObservation(BaseModel):
    ticket_id: str
    subject: str
    body: str
    customer_email: str
    customer_name: str
    customer_history: List[dict]
    customer_lifetime_value: int       # business priority signal
    available_categories: List[str]
    available_priorities: List[str]
    available_sentiments: List[str]
    available_compliance_types: List[str]
    task_name: str
    task_description: str
    step: int
    max_steps: int
    emotional_state: float | None      # de-escalation: 0.0=calm 1.0=furious
    sla_hours_remaining: int | None    # urgency context
    context: dict                      # batch_tickets for prioritize task
    language: str                      # en | fr | de | es
```

## Action Space

```python
class TicketAction(BaseModel):
    category: str | None
    priority: str | None
    response_draft: str | None
    escalate: bool | None
    priority_ranking: List[str] | None
    detected_sentiment: str | None
    churn_risk: bool | None
    summary: str | None
    root_cause: str | None
    resolution_status: str | None       # resolved | in_progress | unresolved
    compliance_violation_detected: str | None  # pci_dss | gdpr | hipaa | none
    compliant_response: str | None
    deescalation_response: str | None
    empathy_shown: bool | None
    concrete_action_offered: bool | None
    reasoning: str | None               # not scored — chain-of-thought
```

## Reward Functions

```
ticket-classify:    0.55*cat + 0.45*pri + 0.05*sla_bonus - 0.02*steps_wasted
ticket-prioritize:  0.55*kendall_tau + 0.35*coverage - 0.10*spurious + 0.05*top1
ticket-resolve:     0.20*cat + 0.20*pri + 0.40*response_quality + 0.20*escalation
ticket-sentiment:   0.65*sentiment_match + 0.35*churn_risk
ticket-summarise:   0.50*summary + 0.25*root_cause + 0.25*resolution_status
ticket-compliance:  0.50*violation_detection + 0.50*compliant_response
ticket-deescalate:  0.30*empathy + 0.30*concrete_action + 0.25*response + 0.15*anger_delta
```

All rewards in `[0.0, 1.0]`. Partial credit at every step — no sparse rewards.

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | `{status: healthy, service: supportdesk-env}` |
| GET | `/tasks` | List all 7 tasks with metadata |
| POST | `/reset?task=<n>` | Start episode, returns `session_id` |
| POST | `/step?task=<n>&session_id=<id>` | Execute action |
| GET | `/state?task=<n>&session_id=<id>` | Episode state |
| GET | `/metrics?task=<n>` | Live score tracking |
| GET | `/replay?task=<n>&session_id=<id>` | Full episode trace |
| GET | `/openenv.yaml` | Spec metadata |

Interactive docs: `http://localhost:7860/docs`

---

## Setup

```bash
# Local dev
pip install -r requirements.txt
cd server && uvicorn app:app --host 0.0.0.0 --port 7860 --reload

# Docker
docker build -t supportdesk-env .
docker run -p 7860:7860 supportdesk-env

# Run inference
export HF_TOKEN=your_token
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export ENV_BASE_URL=http://localhost:7860
python inference.py

# Tests (54 unit + 57 API = 111 total)
pytest tests/ -v
```

## Environment Variables

| Variable | Required | Default |
|----------|----------|---------|
| `HF_TOKEN` | ✅ | — |
| `API_BASE_URL` | ✅ | `https://router.huggingface.co/v1` |
| `MODEL_NAME` | ✅ | `Qwen/Qwen2.5-72B-Instruct` |
| `ENV_BASE_URL` | — | `http://localhost:7860` |
| `IMAGE_NAME` | — | — |

## Baseline Scores

| Task | Score | Notes |
|------|-------|-------|
| `ticket-classify` | ~0.78 | Priority level ambiguity on medium tickets |
| `ticket-prioritize` | ~0.71 | Mid-priority ranking degrades |
| `ticket-resolve` | ~0.65 | Response keyword coverage is bottleneck |
| `ticket-sentiment-detect` | ~0.73 | Nuanced emotion classification |
| `ticket-thread-summarise` | ~0.60 | Root cause label matching |
| `ticket-compliance-check` | ~0.68 | HIPAA detection harder than PCI-DSS |
| `ticket-deescalate` | ~0.52 | Frontier models struggle with emotional trajectory |

---

## Project Structure

```
supportdesk-env/
├── inference.py          ← Competition baseline (7 tasks, exact [START]/[STEP]/[END])
├── Dockerfile            ← Non-root, pinned, health-checked
├── openenv.yaml          ← Full spec with task schemas
├── requirements.txt
├── README.md
├── .github/workflows/ci.yml  ← GitHub Actions CI
├── tests/
│   └── test_suite.py    ← 54 unit tests (graders + episodes)
└── server/
    ├── app.py           ← FastAPI: 8 endpoints + keep-alive thread
    ├── env.py           ← Session-based episodes, SLA decay, emotional journey
    ├── models.py        ← Typed Pydantic models
    ├── graders.py       ← 7 graders: BLEU, Kendall-τ, empathy, compliance
    └── data.py          ← 20 tickets: standard, adversarial, multilingual, compliance
```
