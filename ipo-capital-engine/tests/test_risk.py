"""Tests for the risk metrics and the GO / NO-GO decision framework."""

from __future__ import annotations

from dataclasses import replace

import pytest

from calculations import FundingMode, analyze, expected_net_profit
from risk import (
    DecisionThresholds,
    Verdict,
    compare_opportunities,
    compute_risk_metrics,
    evaluate_decision,
    outcome_distribution,
)
from tests.test_calculations import BID_COST, HOLDING_CARRY, frictionless


# ---------------------------------------------------------------------------
# Exact outcome distribution
# ---------------------------------------------------------------------------
def test_outcome_distribution_reproduces_the_analytic_expected_value():
    result = analyze(frictionless(n_accounts=3, probability=0.2))
    distribution = outcome_distribution(result)
    assert distribution.exact
    assert distribution.expected_profit == pytest.approx(
        result.expected_net_profit, abs=1e-4
    )
    assert sum(distribution.probabilities) == pytest.approx(1.0)


def test_single_account_has_two_outcomes_priced_by_hand():
    result = analyze(frictionless(probability=0.25))
    distribution = outcome_distribution(result)
    outcomes = dict(zip(distribution.profits, distribution.probabilities, strict=True))
    loss = -BID_COST
    win = 2_000.0 - BID_COST - HOLDING_CARRY
    assert any(value == pytest.approx(loss, abs=1e-4) for value in outcomes)
    assert any(value == pytest.approx(win, abs=1e-4) for value in outcomes)
    assert distribution.probability_of_loss == pytest.approx(0.75)
    assert distribution.probability_of_profit == pytest.approx(0.25)


def test_worst_case_is_paying_carry_on_nothing_allotted():
    result = analyze(frictionless(n_accounts=3, probability=0.2))
    distribution = outcome_distribution(result)
    assert distribution.worst_case == pytest.approx(result.net_profit_if_no_allotment)
    assert distribution.best_case == pytest.approx(result.net_profit_if_all_allotted)


def test_a_positive_expected_value_can_still_lose_most_of_the_time():
    result = analyze(frictionless(probability=0.1))
    risk = compute_risk_metrics(result)
    assert result.expected_net_profit > 0
    assert risk.probability_of_loss == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# Risk metrics
# ---------------------------------------------------------------------------
def test_expected_loss_and_gain_decompose_the_expected_value():
    result = analyze(frictionless(n_accounts=2, probability=0.2))
    risk = compute_risk_metrics(result)
    assert risk.expected_gain + risk.expected_loss == pytest.approx(
        result.expected_net_profit, abs=1e-4
    )
    assert risk.expected_loss <= 0
    assert risk.expected_gain >= 0
    assert risk.profit_to_loss_ratio == pytest.approx(
        risk.expected_gain / abs(risk.expected_loss)
    )


def test_margins_of_safety_shrink_as_assumptions_get_more_aggressive():
    generous = compute_risk_metrics(analyze(frictionless(gmp=50.0, probability=0.3)))
    thin = compute_risk_metrics(analyze(frictionless(gmp=3.0, probability=0.05)))
    assert generous.gmp_margin_of_safety > thin.gmp_margin_of_safety
    assert generous.probability_margin_of_safety > thin.probability_margin_of_safety


def test_margin_of_safety_is_negative_when_the_strategy_cannot_work():
    risk = compute_risk_metrics(analyze(frictionless(gmp=1.0, probability=0.02)))
    assert risk.gmp_margin_of_safety < 0
    assert risk.probability_margin_of_safety < 0


def test_od_headroom_is_the_gap_to_the_maximum_sustainable_rate():
    result = analyze(frictionless(od_rate=10.0))
    risk = compute_risk_metrics(result)
    assert risk.od_rate_headroom_pct == pytest.approx(
        result.break_even.max_od_rate_pct - 10.0
    )


def test_downside_scenarios_are_reported_for_a_flat_and_falling_listing():
    result = analyze(frictionless())
    risk = compute_risk_metrics(result)
    assert risk.profit_if_lists_flat < 0  # only carry, no gain
    assert risk.profit_if_lists_10pct_below < risk.profit_if_lists_flat
    assert risk.profit_if_lists_flat == pytest.approx(
        expected_net_profit(
            replace(
                result.inputs,
                ipo=replace(result.inputs.ipo, expected_exit_price_override=100.0),
            )
        )
    )


def test_elasticities_show_dependence_on_gmp_and_hit_rate():
    risk = compute_risk_metrics(analyze(frictionless()))
    assert risk.gmp_elasticity > 0
    assert risk.probability_elasticity > 0


def test_flags_call_out_gmp_dependence():
    risk = compute_risk_metrics(analyze(frictionless()))
    assert any("grey" in flag.lower() for flag in risk.flags)


def test_flags_call_out_a_strategy_that_loses_at_a_zero_financing_rate():
    # Own-funded, so the opportunity cost alone already exceeds the tiny edge.
    inputs = frictionless(
        gmp=0.5,
        probability=0.05,
        funding_mode=FundingMode.OWN,
        own_available=20_000.0,
        fd_amount=0.0,
    )
    result = analyze(inputs)
    assert result.break_even.max_od_rate_pct is None
    risk = compute_risk_metrics(result)
    assert any("zero financing rate" in flag for flag in risk.flags)


def test_multi_account_flag_mentions_the_independence_assumption():
    risk = compute_risk_metrics(analyze(frictionless(n_accounts=3)))
    assert any("independent" in flag.lower() for flag in risk.flags)


# ---------------------------------------------------------------------------
# GO / NO-GO
# ---------------------------------------------------------------------------
def decide(inputs, thresholds=None):
    result = analyze(inputs)
    return evaluate_decision(result, compute_risk_metrics(result), thresholds)


def test_negative_expected_value_is_always_a_no_go():
    decision = decide(frictionless(gmp=1.0, probability=0.02))
    assert decision.verdict is Verdict.NO_GO
    assert any(
        c.name.startswith("Expected net profit") and not c.passed
        for c in decision.checks
    )


def test_a_strong_case_with_several_accounts_is_a_go():
    decision = decide(frictionless(gmp=40.0, probability=0.35, n_accounts=5))
    assert decision.verdict is Verdict.GO
    assert all(c.passed for c in decision.checks)


def test_positive_expected_value_alone_is_not_enough():
    # Profitable, but the return does not clear the cost of the money:
    # 5.2% annualised on equity against a 12% overdraft.
    inputs = frictionless(gmp=3.0, probability=0.35, od_rate=12.0, days_blocked=20)
    result = analyze(inputs)
    assert result.expected_net_profit > 0
    decision = evaluate_decision(result, compute_risk_metrics(result))
    assert decision.verdict is Verdict.NO_GO


def test_a_lone_soft_failure_downgrades_a_go_to_borderline():
    decision = decide(frictionless(gmp=40.0, probability=0.1))
    assert decision.verdict is Verdict.BORDERLINE
    failures = [c for c in decision.checks if not c.passed]
    assert failures and all(not c.hard for c in failures)


def test_thresholds_are_configurable_and_change_the_verdict():
    inputs = frictionless(gmp=40.0, probability=0.1)
    strict = decide(inputs, DecisionThresholds(max_probability_of_loss=0.5))
    lenient = decide(inputs, DecisionThresholds(max_probability_of_loss=0.95))
    assert strict.verdict is Verdict.BORDERLINE
    assert lenient.verdict is Verdict.GO


def test_the_framework_disclaims_being_investment_advice():
    decision = decide(frictionless())
    assert any("not investment advice" in line.lower() for line in decision.rationale)


def test_decision_frame_lists_every_check_with_its_severity():
    decision = decide(frictionless())
    frame = decision.to_frame()
    assert len(frame) == len(decision.checks)
    assert set(frame["Severity"]) <= {"Hard rule", "Soft rule"}
    assert set(frame["Result"]) <= {"PASS", "FAIL", "WARN"}


# ---------------------------------------------------------------------------
# Portfolio comparison
# ---------------------------------------------------------------------------
def test_portfolio_comparison_ranks_opportunities():
    opportunities = [
        ("Weak", frictionless(gmp=2.0, probability=0.05)),
        ("Strong", frictionless(gmp=45.0, probability=0.30)),
        ("Middling", frictionless(gmp=15.0, probability=0.15)),
    ]
    frame = compare_opportunities(opportunities)
    assert len(frame) == 3
    for column in (
        "IPO",
        "Application",
        "Expected allotments",
        "Expected net profit",
        "Financing cost",
        "ROI on own equity",
        "Probability of loss",
        "Decision",
    ):
        assert column in frame.columns
    ranked = frame.sort_values("Expected net profit", ascending=False)
    assert ranked.iloc[0]["IPO"] == "Strong"
    assert ranked.iloc[-1]["IPO"] == "Weak"
    assert ranked.iloc[-1]["Decision"] == "NO-GO"
