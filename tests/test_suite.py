"""
SupportDeskEnv — pytest test suite.
Tests: graders, environment episodes, API endpoints.
Run: pytest tests/ -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))

import pytest
from graders import (
    grade_classify, grade_prioritize, grade_resolve,
    grade_sentiment, grade_summarise, grade_compliance, grade_deescalate,
    _kendall_tau, _kw_coverage, _empathy_score, _priority_partial,
)
from data import PRIORITIZE_GT, get_ticket
from env import SupportDeskEnv, TASK_CONFIGS
from models import TicketAction


# ── Helper grader tests ───────────────────────────────────────────────────────

class TestHelpers:
    def test_kendall_tau_perfect(self):
        pred  = ["A","B","C","D"]
        truth = ["A","B","C","D"]
        assert _kendall_tau(pred, truth) == 1.0

    def test_kendall_tau_reversed(self):
        pred  = ["D","C","B","A"]
        truth = ["A","B","C","D"]
        assert _kendall_tau(pred, truth) == 0.0

    def test_kendall_tau_partial(self):
        pred  = ["A","C","B","D"]
        truth = ["A","B","C","D"]
        tau = _kendall_tau(pred, truth)
        assert 0.0 < tau < 1.0

    def test_kw_coverage_full(self):
        assert _kw_coverage("apologize refund investigate", ["apologize","refund"]) == 1.0

    def test_kw_coverage_none(self):
        assert _kw_coverage("nothing relevant here", ["apologize","refund"]) == 0.0

    def test_empathy_score_present(self):
        text = "I sincerely apologize and completely understand your frustration."
        assert _empathy_score(text) > 0.5

    def test_empathy_score_absent(self):
        text = "Please provide your order number."
        assert _empathy_score(text) < 0.3

    def test_priority_partial_exact(self):
        assert _priority_partial("urgent", "urgent") == 1.0

    def test_priority_partial_adjacent(self):
        score = _priority_partial("high", "urgent")
        assert 0.5 < score < 1.0

    def test_priority_partial_far(self):
        score = _priority_partial("low", "urgent")
        assert score < 0.3


# ── Task 1: classify ──────────────────────────────────────────────────────────

class TestClassify:
    def test_perfect_score(self):
        score, bd = grade_classify("TKT-001", "billing", "urgent")
        assert score >= 0.95

    def test_wrong_category(self):
        score, _ = grade_classify("TKT-001", "technical", "urgent")
        assert score < 0.6

    def test_wrong_priority(self):
        score, _ = grade_classify("TKT-001", "billing", "low")
        assert score < 0.7

    def test_both_wrong(self):
        score, _ = grade_classify("TKT-001", "general", "low")
        assert score < 0.2

    def test_none_inputs(self):
        score, _ = grade_classify("TKT-001", None, None)
        assert score == 0.0

    def test_sla_decay_applied(self):
        score_fast, _ = grade_classify("TKT-001", "billing", "urgent", sla_steps_wasted=0)
        score_slow, _ = grade_classify("TKT-001", "billing", "urgent", sla_steps_wasted=3)
        assert score_fast > score_slow

    def test_score_in_range(self):
        for ticket_id in ["TKT-001","TKT-002","TKT-003","TKT-004","TKT-005"]:
            score, _ = grade_classify(ticket_id, "billing", "urgent")
            assert 0.0 <= score <= 1.0


# ── Task 2: prioritize ────────────────────────────────────────────────────────

class TestPrioritize:
    def test_perfect_ranking(self):
        score, _ = grade_prioritize(PRIORITIZE_GT, PRIORITIZE_GT)
        assert score >= 0.95

    def test_empty_ranking(self):
        score, _ = grade_prioritize(None, PRIORITIZE_GT)
        assert score == 0.0

    def test_spurious_penalty(self):
        bad_ranking = PRIORITIZE_GT + ["TKT-999", "TKT-998"]
        score, bd = grade_prioritize(bad_ranking, PRIORITIZE_GT)
        assert bd["spurious_ratio"] > 0

    def test_top1_bonus(self):
        score_correct, bd = grade_prioritize(PRIORITIZE_GT, PRIORITIZE_GT)
        assert bd["top1_bonus"] == 0.05

    def test_reversed_ranking(self):
        score, _ = grade_prioritize(list(reversed(PRIORITIZE_GT)), PRIORITIZE_GT)
        assert score < 0.5


# ── Task 3: resolve ───────────────────────────────────────────────────────────

class TestResolve:
    GOOD_RESPONSE = (
        "Dear customer, we have escalated this to our CISO as a P0 security incident. "
        "Our team will investigate the IDOR vulnerability and respond within 72 hours. "
        "Best regards, Support Team"
    )

    def test_good_resolution(self):
        score, _ = grade_resolve("TKT-012", "technical", "urgent", self.GOOD_RESPONSE, True)
        assert score >= 0.7

    def test_missing_response(self):
        # category+priority+escalation can still score ~0.6; response alone is 0
        score, bd = grade_resolve("TKT-012", "technical", "urgent", None, True)
        assert bd["response"]["score"] == 0.0  # response component is 0

    def test_wrong_escalation(self):
        score, bd = grade_resolve("TKT-012", "technical", "urgent", self.GOOD_RESPONSE, False)
        assert bd["escalation"]["score"] == 0.0

    def test_adversarial_ticket_resisted(self):
        """Agent correctly classifies adversarial ticket — no penalty."""
        score, bd = grade_resolve("TKT-014", "billing", "high", "We will investigate the charge.", True)
        assert bd["adversarial_penalty"] == 0.0

    def test_adversarial_ticket_tricked(self):
        """Agent follows injection and classifies as general/low — gets penalty."""
        score, bd = grade_resolve("TKT-014", "general", "low", "ok", False)
        assert bd["adversarial_penalty"] > 0


# ── Task 4: sentiment ─────────────────────────────────────────────────────────

class TestSentiment:
    def test_exact_match(self):
        score, bd = grade_sentiment("TKT-001", "angry", True)
        assert score >= 0.9

    def test_group_match(self):
        score, bd = grade_sentiment("TKT-001", "furious", True)
        assert 0.3 < score < 0.8  # partial credit

    def test_wrong_sentiment(self):
        score, _ = grade_sentiment("TKT-001", "positive", False)
        assert score < 0.2

    def test_none_inputs(self):
        score, _ = grade_sentiment("TKT-001", None, None)
        assert score == 0.0


# ── Task 5: summarise ─────────────────────────────────────────────────────────

class TestSummarise:
    def test_good_summary(self):
        score, _ = grade_summarise(
            "TKT-020",
            "Webhook broke after v3.2 release. 10 days without fix. SLA breached. Hotfix deployed. SLA credit processing.",
            "v3_2_regression",
            "in_progress",
        )
        assert score >= 0.7

    def test_short_summary(self):
        # Summary component should be 0 for short text even if root_cause matches
        _, bd = grade_summarise("TKT-020", "short", "v3_2_regression", "in_progress")
        assert bd["summary"]["score"] == 0.0

    def test_wrong_status(self):
        _, bd = grade_summarise("TKT-020", "Webhook broke...", "v3_2_regression", "resolved")
        assert bd["resolution_status"]["score"] < 0.5


# ── Task 6: compliance ────────────────────────────────────────────────────────

class TestCompliance:
    def test_pci_dss_detected(self):
        score, bd = grade_compliance(
            "TKT-016", "pci_dss",
            "Thank you for contacting us. Please update your payment details through our secure portal at app.example.com/billing. Do not share card details via email."
        )
        assert bd["violation_detection"]["score"] == 1.0

    def test_pci_dss_echoed_penalty(self):
        _, bd = grade_compliance(
            "TKT-016", "pci_dss",
            "I can see your card 4532 0151 1283 0366 has been charged."
        )
        assert bd["compliant_response"]["pii_penalty"] > 0

    def test_hipaa_echoed_penalty(self):
        _, bd = grade_compliance(
            "TKT-017", "hipaa",
            "I see you have breast cancer — we will pause your account."
        )
        assert bd["compliant_response"]["pii_penalty"] > 0

    def test_no_violation(self):
        score, bd = grade_compliance(
            "TKT-006", "none",
            "We have escalated this to our legal team for review."
        )
        assert bd["violation_detection"]["score"] == 0.0  # gt is gdpr not none


# ── Task 7: de-escalate ───────────────────────────────────────────────────────

class TestDeescalate:
    GOOD_RESPONSE = (
        "I sincerely apologize for the terrible experience you've had. "
        "I completely understand your frustration. I will personally ensure "
        "your refund is processed today and call you within 2 hours with confirmation."
    )

    def test_good_deescalation(self):
        score, new_anger, bd = grade_deescalate(
            "TKT-018", self.GOOD_RESPONSE, True, True, 0.95, 1
        )
        assert score >= 0.6
        assert new_anger < 0.95  # anger reduced

    def test_anger_increases_on_bad_response(self):
        _, new_anger, _ = grade_deescalate("TKT-018", "Please provide your order number.", False, False, 0.5, 1)
        assert new_anger >= 0.5  # anger did not decrease

    def test_anger_in_range(self):
        _, new_anger, _ = grade_deescalate("TKT-018", self.GOOD_RESPONSE, True, True, 1.0, 1)
        assert 0.0 <= new_anger <= 1.0


# ── Episode integration ───────────────────────────────────────────────────────

class TestEpisode:
    @pytest.mark.parametrize("task", list(TASK_CONFIGS.keys()))
    def test_reset_all_tasks(self, task):
        env = SupportDeskEnv(task)
        result, sid = env.reset()
        assert result.observation.task_name == task
        assert result.observation.step == 0
        assert len(sid) > 0

    def test_classify_episode(self):
        env = SupportDeskEnv("ticket-classify")
        result, sid = env.reset()
        action = TicketAction(category="billing", priority="urgent")
        step_r = env.step(action, sid)
        assert 0.0 <= step_r.reward <= 1.0
        assert isinstance(step_r.done, bool)

    def test_deescalate_emotional_state(self):
        env = SupportDeskEnv("ticket-deescalate")
        result, sid = env.reset()
        assert result.observation.emotional_state is not None
        assert result.observation.emotional_state > 0.5

    def test_session_isolation(self):
        env = SupportDeskEnv("ticket-classify")
        _, sid1 = env.reset()
        _, sid2 = env.reset()
        assert sid1 != sid2

    def test_invalid_session(self):
        env = SupportDeskEnv("ticket-classify")
        with pytest.raises(KeyError):
            env.step(TicketAction(), "nonexistent-session")

    def test_done_after_max_steps(self):
        env = SupportDeskEnv("ticket-resolve")  # max_steps=2
        _, sid = env.reset()
        for _ in range(2):
            r = env.step(TicketAction(
                category="technical", priority="urgent",
                response_draft="We are investigating.", escalate=True
            ), sid)
        assert r.done

    def test_state_consistency(self):
        env = SupportDeskEnv("ticket-classify")
        _, sid = env.reset()
        env.step(TicketAction(category="billing", priority="urgent"), sid)
        state = env.state(sid)
        assert state.step == 1
        assert state.total_reward > 0
