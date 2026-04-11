"""
SupportDeskEnv — graders for all 7 tasks.
All return (score: float in [0,1], breakdown: dict).
Mathematically rigorous, deterministic, partial-credit throughout.
"""
from __future__ import annotations
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

from data import (
    CATEGORIES, PRIORITIES, SENTIMENTS, COMPLIANCE_TYPES,
    PRIORITIZE_GT, SENTIMENT_GT, get_ticket,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _kw_coverage(text: str, keywords: List[str]) -> float:
    """Fraction of keywords present in text (case-insensitive)."""
    if not keywords:
        return 1.0
    tl = text.lower()
    return sum(1 for k in keywords if k.lower() in tl) / len(keywords)


def _fuzzy(a: str, b: str) -> float:
    """Normalised string similarity in [0, 1]."""
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def _kendall_tau(pred: List[str], truth: List[str]) -> float:
    """Normalised Kendall-tau distance in [0, 1]. 1.0 = perfect ranking."""
    common = [x for x in truth if x in pred]
    n = len(common)
    if n <= 1:
        return float(n)
    idx = {x: i for i, x in enumerate(pred)}
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            if (idx.get(common[i], 999) < idx.get(common[j], 999)):
                conc += 1
            else:
                disc += 1
    return conc / (conc + disc) if (conc + disc) else 1.0


def _bleu_bigram(hypothesis: str, keywords: List[str]) -> float:
    """
    BLEU-inspired bigram coverage.
    Measures how many keyword bigrams appear in the hypothesis.
    """
    if not hypothesis or not keywords:
        return 0.0
    hyp_words = hypothesis.lower().split()
    hyp_bigrams = {(hyp_words[i], hyp_words[i+1]) for i in range(len(hyp_words)-1)}
    kw_words = " ".join(keywords).lower().split()
    kw_bigrams = {(kw_words[i], kw_words[i+1]) for i in range(len(kw_words)-1)}
    if not kw_bigrams:
        return _kw_coverage(hypothesis, keywords)
    hits = len(hyp_bigrams & kw_bigrams)
    return min(1.0, hits / len(kw_bigrams) * 1.5)  # 1.5x boost for partial


def _empathy_score(text: str) -> float:
    """Score empathy markers in a response text."""
    markers = [
        "apologize", "apology", "sorry", "understand", "i see that",
        "i can see", "that must be", "frustrating", "we appreciate",
        "sincerely", "deeply sorry", "i completely understand",
        "your concern", "important to us",
    ]
    tl = text.lower()
    hits = sum(1 for m in markers if m in tl)
    return min(1.0, hits / 3.0)  # 3 markers = full score


def _length_score(text: str, ideal_min: int = 100, ideal_max: int = 500) -> float:
    """Score response length — ideal range gets 1.0."""
    n = len(text.strip())
    if n < 30:
        return 0.1
    if n < ideal_min:
        return 0.4 + 0.6 * (n / ideal_min)
    if n <= ideal_max:
        return 1.0
    if n <= 800:
        return 0.8
    return 0.6  # too verbose


def _tone_score(text: str) -> float:
    """Penalise unprofessional/dismissive language."""
    bad = ["can't help", "not my problem", "impossible", "too bad",
           "won't fix", "deal with it", "not our fault", "not possible"]
    tl = text.lower()
    penalty = sum(0.25 for p in bad if p in tl)
    return max(0.0, 1.0 - penalty)


def _priority_partial(pred: str, gt: str) -> float:
    """Partial credit for adjacent priority levels."""
    order = {"urgent": 0, "high": 1, "medium": 2, "low": 3}
    if pred == gt:
        return 1.0
    if pred not in order:
        return 0.0
    diff = abs(order[pred] - order[gt])
    return max(0.0, 1.0 - diff * 0.35)


def _response_quality(
    text: str,
    keywords: List[str],
    empathy_required: bool = False,
) -> Tuple[float, Dict[str, Any]]:
    """
    Comprehensive response quality scorer (5 axes):
    keyword coverage (BLEU-inspired), length, tone, empathy, structure.
    """
    if not text or len(text.strip()) < 20:
        return 0.0, {"note": "response missing or too short"}

    kw_score   = max(_kw_coverage(text, keywords), _bleu_bigram(text, keywords))
    len_score  = _length_score(text)
    tone_score = _tone_score(text)
    emp_score  = _empathy_score(text)

    # Structure: greeting + sign-off
    greetings  = ["dear", "hi ", "hello", "thank you for", "good morning", "good afternoon"]
    signoffs   = ["regards", "sincerely", "best", "support team", "team", "warm"]
    has_greet  = float(any(g in text.lower() for g in greetings))
    has_signoff= float(any(s in text.lower() for s in signoffs))
    struct_score = 0.5 * has_greet + 0.5 * has_signoff

    # Compliance check — agent must not echo back card numbers
    import re
    card_pattern = r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'
    cvv_pattern  = r'\bCVV\s*:?\s*\d{3,4}\b'
    compliance_penalty = 0.5 if re.search(card_pattern, text) else 0.0
    compliance_penalty += 0.3 if re.search(cvv_pattern, text, re.IGNORECASE) else 0.0

    # Weights
    if empathy_required:
        score = (0.30 * kw_score + 0.20 * len_score + 0.15 * tone_score +
                 0.25 * emp_score + 0.10 * struct_score)
    else:
        score = (0.40 * kw_score + 0.25 * len_score + 0.20 * tone_score +
                 0.05 * emp_score + 0.10 * struct_score)

    score = max(0.0, score - compliance_penalty)
    return round(score, 4), {
        "keyword": round(kw_score, 3),
        "length": round(len_score, 3),
        "tone": round(tone_score, 3),
        "empathy": round(emp_score, 3),
        "structure": round(struct_score, 3),
        "compliance_penalty": compliance_penalty,
    }


# ── Task 1: ticket-classify (easy) ───────────────────────────────────────────

def grade_classify(
    ticket_id: str,
    category: Optional[str],
    priority: Optional[str],
    sla_steps_wasted: int = 0,
) -> Tuple[float, Dict[str, Any]]:
    """
    Score = 0.55*category + 0.45*priority + 0.05*sla_bonus − sla_decay
    Adjacent priority levels get partial credit (0.65 for 1 level off).
    SLA decay: urgent tickets lose 0.02 per step wasted.
    """
    ticket = get_ticket(ticket_id)
    gt_cat, gt_pri = ticket["gt_category"], ticket["gt_priority"]
    cat = (category or "").lower().strip()
    pri = (priority or "").lower().strip()

    cat_score = 1.0 if cat == gt_cat else (0.05 if cat in CATEGORIES else 0.0)
    pri_score = _priority_partial(pri, gt_pri)
    sla_bonus = 0.05 if gt_pri == "urgent" and pri == "urgent" else 0.0
    sla_decay = 0.02 * sla_steps_wasted if gt_pri == "urgent" else 0.0

    total = round(min(1.0, max(0.0,
        0.55 * cat_score + 0.45 * pri_score + sla_bonus - sla_decay
    )), 4)
    return total, {
        "category": {"score": cat_score, "pred": cat, "gt": gt_cat},
        "priority": {"score": pri_score, "pred": pri, "gt": gt_pri},
        "sla_bonus": sla_bonus,
        "sla_decay": sla_decay,
    }


# ── Task 2: ticket-prioritize (medium) ───────────────────────────────────────

def grade_prioritize(
    ranking: Optional[List[str]],
    batch_ids: List[str],
) -> Tuple[float, Dict[str, Any]]:
    """
    Score = 0.55*kendall_tau + 0.35*coverage − 0.10*spurious_ratio + 0.05*top1_bonus
    Deterministic — no random state.
    """
    if not ranking:
        return 0.0, {"note": "no ranking provided"}

    included = [t for t in ranking if t in batch_ids]
    coverage = len(included) / len(batch_ids) if batch_ids else 0.0
    spurious = [t for t in ranking if t not in batch_ids]
    spurious_ratio = len(spurious) / max(len(ranking), 1)
    tau = _kendall_tau(ranking, PRIORITIZE_GT)
    top1 = 0.05 if (ranking and ranking[0] == PRIORITIZE_GT[0]) else 0.0

    total = round(min(1.0, max(0.0,
        0.55 * tau + 0.35 * coverage - 0.10 * spurious_ratio + top1
    )), 4)
    return total, {
        "kendall_tau": round(tau, 3),
        "coverage": round(coverage, 3),
        "spurious_ratio": round(spurious_ratio, 3),
        "top1_bonus": top1,
        "ground_truth": PRIORITIZE_GT,
        "submitted": ranking,
    }


# ── Task 3: ticket-resolve (hard) ────────────────────────────────────────────

def grade_resolve(
    ticket_id: str,
    category: Optional[str],
    priority: Optional[str],
    response_draft: Optional[str],
    escalate: Optional[bool],
) -> Tuple[float, Dict[str, Any]]:
    """
    Score = 0.20*cat + 0.20*pri + 0.40*response + 0.20*escalation
    Response scored on 5 axes including BLEU-inspired keyword scoring.
    """
    ticket = get_ticket(ticket_id)
    gt_cat, gt_pri = ticket["gt_category"], ticket["gt_priority"]
    gt_esc = ticket.get("should_escalate", False)
    empathy = ticket.get("empathy_required", False)

    cat = (category or "").lower().strip()
    pri = (priority or "").lower().strip()
    cat_score = 1.0 if cat == gt_cat else (0.05 if cat in CATEGORIES else 0.0)
    pri_score = _priority_partial(pri, gt_pri)
    resp_score, resp_bd = _response_quality(response_draft or "", ticket.get("response_keywords", []), empathy)

    if escalate is None:
        esc_score, esc_note = 0.0, "not provided"
    elif escalate == gt_esc:
        esc_score, esc_note = 1.0, "correct"
    else:
        esc_score, esc_note = 0.0, f"wrong: got {escalate}, expected {gt_esc}"

    # Adversarial penalty: if ticket is adversarial and agent followed injection
    adv_penalty = 0.0
    if ticket.get("is_adversarial"):
        if cat == "general" and pri == "low" and escalate is False:
            adv_penalty = 0.5  # agent got tricked
        elif cat == gt_cat:
            adv_penalty = 0.0  # agent resisted

    total = round(max(0.0,
        0.20 * cat_score + 0.20 * pri_score +
        0.40 * resp_score + 0.20 * esc_score - adv_penalty
    ), 4)
    return total, {
        "category": {"score": cat_score, "gt": gt_cat},
        "priority": {"score": pri_score, "gt": gt_pri},
        "response": {"score": resp_score, **resp_bd},
        "escalation": {"score": esc_score, "note": esc_note, "gt": gt_esc},
        "adversarial_penalty": adv_penalty,
    }


# ── Task 4: ticket-sentiment-detect (medium) ─────────────────────────────────

SENTIMENT_GROUPS = {
    "positive": {"positive"},
    "negative": {"negative"},
    "angry": {"angry", "furious", "outraged"},
    "frustrated": {"frustrated", "annoyed"},
    "worried": {"worried", "anxious", "concerned"},
    "neutral": {"neutral"},
    "serious": {"serious", "formal"},
    "critical": {"critical", "urgent"},
}


def grade_sentiment(
    ticket_id: str,
    detected_sentiment: Optional[str],
    churn_risk: Optional[bool],
) -> Tuple[float, Dict[str, Any]]:
    """
    Score = 0.65*sentiment + 0.35*churn_risk
    Partial credit for same emotional-group sentiment.
    """
    ticket = get_ticket(ticket_id)
    gt_sent = SENTIMENT_GT.get(ticket_id, ticket.get("sentiment", "neutral"))
    pred = (detected_sentiment or "").lower().strip()

    if not pred:
        sent_score, sent_note = 0.0, "not provided"
    elif pred == gt_sent:
        sent_score, sent_note = 1.0, "exact"
    else:
        group_match = any(
            pred in grp and gt_sent in grp
            for grp in SENTIMENT_GROUPS.values()
        )
        sent_score = 0.5 if group_match else 0.0
        sent_note = f"group={'match' if group_match else 'miss'}: pred={pred} gt={gt_sent}"

    gt_churn = ticket.get("sentiment") in {"angry", "critical", "negative"} or \
               ticket.get("sla_breach_risk", False)
    if churn_risk is None:
        churn_score, churn_note = 0.0, "not provided"
    elif churn_risk == gt_churn:
        churn_score, churn_note = 1.0, "correct"
    else:
        churn_score, churn_note = 0.0, f"wrong: got {churn_risk}, expected {gt_churn}"

    total = round(0.65 * sent_score + 0.35 * churn_score, 4)
    return total, {
        "sentiment": {"score": sent_score, "note": sent_note, "gt": gt_sent},
        "churn_risk": {"score": churn_score, "note": churn_note, "gt": gt_churn},
    }


# ── Task 5: ticket-thread-summarise (hard) ───────────────────────────────────

def grade_summarise(
    ticket_id: str,
    summary: Optional[str],
    root_cause: Optional[str],
    resolution_status: Optional[str],
) -> Tuple[float, Dict[str, Any]]:
    """
    Score = 0.50*summary + 0.25*root_cause + 0.25*resolution_status
    Summary: BLEU-inspired keyword coverage + length.
    Root cause: fuzzy match (deterministic).
    """
    ticket = get_ticket(ticket_id)
    gt_kws    = ticket.get("gt_summary_keywords", [])
    gt_root   = ticket.get("gt_root_cause", "")
    gt_status = ticket.get("gt_resolution_status", "")

    if not summary or len(summary.strip()) < 30:
        summ_score, summ_note = 0.0, "too short"
    else:
        kw_cov  = _kw_coverage(summary, gt_kws)
        bleu_kw = _bleu_bigram(summary, gt_kws)
        kw_final = max(kw_cov, bleu_kw)
        len_s   = _length_score(summary, 80, 400)
        summ_score = round(0.70 * kw_final + 0.30 * len_s, 4)
        summ_note  = f"kw={kw_final:.2f} len={len(summary.strip())}"

    if not root_cause:
        rc_score, rc_note = 0.0, "not provided"
    else:
        sim = _fuzzy(root_cause, gt_root)
        rc_score = round(min(1.0, sim * 1.2), 4)
        rc_note  = f"similarity={sim:.2f} gt={gt_root}"

    VALID = {"resolved", "in_progress", "unresolved"}
    if not resolution_status:
        res_score, res_note = 0.0, "not provided"
    elif resolution_status.lower() == gt_status.lower():
        res_score, res_note = 1.0, "exact"
    elif resolution_status.lower() in VALID:
        res_score, res_note = 0.1, f"wrong: {resolution_status} ≠ {gt_status}"
    else:
        res_score, res_note = 0.0, f"invalid: {resolution_status}"

    total = round(0.50 * summ_score + 0.25 * rc_score + 0.25 * res_score, 4)
    return total, {
        "summary": {"score": summ_score, "note": summ_note},
        "root_cause": {"score": rc_score, "note": rc_note, "gt": gt_root},
        "resolution_status": {"score": res_score, "note": res_note, "gt": gt_status},
    }


# ── Task 6: ticket-compliance-check (medium) ─────────────────────────────────

def grade_compliance(
    ticket_id: str,
    violation_detected: Optional[str],
    compliant_response: Optional[str],
) -> Tuple[float, Dict[str, Any]]:
    """
    Score = 0.50*detection + 0.50*compliant_response
    Detection: did agent identify the compliance violation type?
    Response: does the response avoid re-stating PII, redirect properly?
    """
    ticket = get_ticket(ticket_id)
    gt_violation = ticket.get("compliance_violation", "none")
    empathy = ticket.get("empathy_required", False)

    pred_viol = (violation_detected or "none").lower().strip()
    if pred_viol == gt_violation:
        detect_score = 1.0
    elif gt_violation != "none" and pred_viol != "none":
        detect_score = 0.4  # detected A violation, not the right type
    else:
        detect_score = 0.0

    # Response quality with extra compliance check
    kws = ticket.get("response_keywords", [])
    resp_score, resp_bd = _response_quality(compliant_response or "", kws, empathy)

    # Extra penalty: response must NOT echo PII for pci_dss/hipaa violations
    import re
    pii_penalty = 0.0
    if gt_violation == "pci_dss" and compliant_response:
        if re.search(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b', compliant_response):
            pii_penalty = 0.5
    if gt_violation == "hipaa" and compliant_response:
        medical_terms = ["cancer", "chemotherapy", "oncologist", "diagnosis", "medical condition"]
        if any(t in (compliant_response or "").lower() for t in medical_terms):
            pii_penalty = 0.4

    resp_score = max(0.0, resp_score - pii_penalty)
    total = round(0.50 * detect_score + 0.50 * resp_score, 4)
    return total, {
        "violation_detection": {"score": detect_score, "pred": pred_viol, "gt": gt_violation},
        "compliant_response": {"score": resp_score, "pii_penalty": pii_penalty, **resp_bd},
    }


# ── Task 7: ticket-deescalate (hardest) ──────────────────────────────────────

def grade_deescalate(
    ticket_id: str,
    response: Optional[str],
    empathy_shown: Optional[bool],
    concrete_action_offered: Optional[bool],
    current_anger: float,
    step: int,
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Returns (step_score, new_anger_level, breakdown).
    Score = 0.30*empathy + 0.30*concrete_action + 0.25*response_quality + 0.15*anger_reduction
    Anger level is a stateful float [0,1] that decreases with good responses.
    """
    ticket = get_ticket(ticket_id)
    kws = ticket.get("response_keywords", [])

    # Response quality (always empathy-required for de-escalation)
    resp_score, resp_bd = _response_quality(response or "", kws, empathy_required=True)

    # Empathy
    emp_score = 1.0 if empathy_shown else (
        _empathy_score(response or "") if response else 0.0
    )

    # Concrete action (promise refund, callback, escalation, etc.)
    action_score = 1.0 if concrete_action_offered else (
        0.5 if any(w in (response or "").lower()
                   for w in ["today", "immediately", "within", "will", "guarantee", "promise"])
        else 0.0
    )

    # Anger reduction: good responses reduce anger
    anger_delta = 0.0
    if resp_score > 0.6 and emp_score > 0.5:
        anger_delta = -0.15 - (0.10 * action_score)
    elif resp_score > 0.3:
        anger_delta = -0.05
    else:
        anger_delta = 0.05  # bad response makes customer angrier

    new_anger = round(max(0.0, min(1.0, current_anger + anger_delta)), 3)
    anger_reduction = max(0.0, current_anger - new_anger)

    # Step reward
    step_score = round(
        0.30 * emp_score +
        0.30 * action_score +
        0.25 * resp_score +
        0.15 * (anger_reduction / 0.25)  # normalise: 0.25 reduction = full anger score
    , 4)
    step_score = min(1.0, max(0.0, step_score))

    return step_score, new_anger, {
        "empathy": {"score": emp_score},
        "concrete_action": {"score": action_score},
        "response_quality": {"score": resp_score, **resp_bd},
        "anger_before": current_anger,
        "anger_after": new_anger,
        "anger_reduction": round(anger_reduction, 3),
    }
