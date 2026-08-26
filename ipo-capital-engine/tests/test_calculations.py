"""Unit tests for the calculation engine.

Every expected value in this file is derived by hand from the formulas in the
README, not copied from a previous run of the code, so the tests genuinely pin
the arithmetic down.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from calculations import (
    AnalysisInputs,
    ApplicationAccount,
    FinancingAssumptions,
    FundingMode,
    GMPMode,
    IPOAssumptions,
    IPOCategory,
    TaxAssumptions,
    TransactionCostAssumptions,
    allotment_distribution,
    analyze,
    annualize,
    annualize_simple,
    break_even_exit_price,
    build_funding_plan,
    compute_capital_gains_tax,
    compute_transaction_costs,
    conditional_net_profit,
    expected_net_profit,
    gmp_from_listing_price,
    listing_gain_pct,
    listing_price_from_gmp,
    max_sustainable_od_rate,
    min_allotment_probability,
    simple_interest,
)

# ---------------------------------------------------------------------------
# Frictionless fixture: zero costs and zero taxes, so every number below can be
# reproduced with a pocket calculator.
# ---------------------------------------------------------------------------
ZERO_COSTS = TransactionCostAssumptions(
    brokerage_pct_buy=0.0,
    brokerage_flat_buy=0.0,
    brokerage_pct_sell=0.0,
    brokerage_flat_sell=0.0,
    stt_pct_buy=0.0,
    stt_pct_sell=0.0,
    exchange_txn_pct=0.0,
    sebi_turnover_pct=0.0,
    stamp_duty_pct_buy=0.0,
    gst_pct=0.0,
    dp_charges_flat_sell=0.0,
    other_charges_flat=0.0,
)
ZERO_TAXES = TaxAssumptions(stcg_rate_pct=0.0, ltcg_rate_pct=0.0, cess_and_surcharge_pct=0.0)


def frictionless(
    probability: float = 0.25,
    gmp: float = 20.0,
    od_rate: float = 10.0,
    days_blocked: int = 10,
    holding_days: int = 5,
    n_accounts: int = 1,
    funding_mode: FundingMode = FundingMode.OD,
    own_available: float = 0.0,
    own_deployed: float = 0.0,
    fd_amount: float = 100_000.0,
) -> AnalysisInputs:
    """One lot of 100 shares at Rs 100, i.e. an application of Rs 10,000."""
    return AnalysisInputs(
        ipo=IPOAssumptions(
            name="Test",
            issue_price=100.0,
            lot_size=100,
            gmp_value=gmp,
            gmp_mode=GMPMode.ABSOLUTE,
            holding_period_days=holding_days,
        ),
        accounts=tuple(
            ApplicationAccount(
                label=f"PAN {i + 1}",
                category=IPOCategory.RETAIL,
                lots_applied=1,
                allotment_probability=probability,
                lots_allotted_if_successful=1.0,
            )
            for i in range(n_accounts)
        ),
        financing=FinancingAssumptions(
            funding_mode=funding_mode,
            own_capital_available=own_available,
            own_capital_deployed=own_deployed,
            fd_amount=fd_amount,
            fd_rate_pct=7.0,
            od_ltv_pct=90.0,
            od_rate_pct=od_rate,
            days_blocked=days_blocked,
            opportunity_cost_rate_pct=7.0,
            include_opportunity_cost=True,
        ),
        costs=ZERO_COSTS,
        taxes=ZERO_TAXES,
    )


# ---------------------------------------------------------------------------
# Price / GMP arithmetic
# ---------------------------------------------------------------------------
def test_listing_price_from_absolute_gmp():
    assert listing_price_from_gmp(100.0, 25.0) == 125.0
    assert gmp_from_listing_price(100.0, 125.0) == 25.0
    assert listing_gain_pct(100.0, 125.0) == pytest.approx(25.0)


def test_gmp_percent_mode_converts_to_rupees():
    ipo = IPOAssumptions(issue_price=250.0, gmp_value=12.0, gmp_mode=GMPMode.PERCENT)
    assert ipo.gmp_absolute == pytest.approx(30.0)
    assert ipo.expected_listing_price == pytest.approx(280.0)
    assert ipo.expected_listing_gain_pct == pytest.approx(12.0)


def test_exit_price_defaults_to_listing_price_but_can_be_overridden():
    ipo = IPOAssumptions(issue_price=100.0, gmp_value=20.0)
    assert ipo.expected_exit_price == 120.0
    later = replace(ipo, expected_exit_price_override=135.0)
    assert later.expected_exit_price == 135.0
    assert later.expected_listing_price == 120.0  # listing view is unchanged


def test_listing_price_override_used_when_gmp_is_switched_off():
    ipo = IPOAssumptions(
        issue_price=100.0,
        gmp_value=20.0,
        use_gmp_for_listing=False,
        expected_listing_price_override=90.0,
    )
    assert ipo.expected_listing_price == 90.0
    assert ipo.expected_listing_gain_pct == pytest.approx(-10.0)


# ---------------------------------------------------------------------------
# Interest and annualisation
# ---------------------------------------------------------------------------
def test_simple_interest_uses_365_day_basis():
    # 10,000 at 10% for 10 days = 10000 * 0.10 * 10/365
    assert simple_interest(10_000, 10.0, 10) == pytest.approx(27.397260273972602)
    assert simple_interest(10_000, 10.0, 365) == pytest.approx(1000.0)


def test_simple_interest_honours_an_alternative_day_count():
    assert simple_interest(10_000, 10.0, 10, 360) == pytest.approx(27.77777777777778)


def test_simple_interest_is_zero_for_degenerate_inputs():
    assert simple_interest(0, 10.0, 30) == 0.0
    assert simple_interest(10_000, 10.0, 0) == 0.0
    assert simple_interest(10_000, 0.0, 30) == 0.0


def test_annualize_compounds_and_simple_version_does_not():
    # 1% over 30 days: (1.01)^(365/30) - 1 = 12.8695%
    assert annualize(0.01, 30) == pytest.approx((1.01) ** (365 / 30) - 1, rel=1e-12)
    assert annualize(0.01, 30) == pytest.approx(0.1286952941, rel=1e-9)
    assert annualize_simple(0.01, 30) == pytest.approx(0.1216666, rel=1e-6)
    assert annualize(0.05, 0) is None


def test_annualize_handles_total_loss_without_complex_numbers():
    assert annualize(-1.5, 30) == pytest.approx(-1.5 * 365 / 30)


# ---------------------------------------------------------------------------
# Funding plan: own / OD / mixed
# ---------------------------------------------------------------------------
def test_od_limit_is_fd_times_ltv():
    fin = FinancingAssumptions(fd_amount=100_000.0, od_ltv_pct=90.0)
    assert fin.od_limit == pytest.approx(90_000.0)


def test_own_capital_only_never_draws_od():
    fin = FinancingAssumptions(
        funding_mode=FundingMode.OWN,
        own_capital_available=50_000.0,
        fd_amount=100_000.0,
        od_ltv_pct=90.0,
    )
    plan = build_funding_plan(40_000.0, fin)
    assert plan.od_drawn == 0.0
    assert plan.own_capital_deployed == pytest.approx(40_000.0)
    assert plan.shortfall == 0.0
    assert plan.od_share == 0.0


def test_od_mode_is_capped_at_the_sanctioned_limit():
    fin = FinancingAssumptions(
        funding_mode=FundingMode.OD,
        own_capital_available=20_000.0,
        fd_amount=100_000.0,
        od_ltv_pct=90.0,
    )
    plan = build_funding_plan(100_000.0, fin)
    assert plan.od_drawn == pytest.approx(90_000.0)
    assert plan.own_capital_deployed == pytest.approx(10_000.0)
    assert plan.shortfall == 0.0
    assert plan.od_utilisation_pct == pytest.approx(100.0)


def test_shortfall_is_reported_not_silently_borrowed():
    fin = FinancingAssumptions(
        funding_mode=FundingMode.OD,
        own_capital_available=5_000.0,
        fd_amount=10_000.0,
        od_ltv_pct=90.0,
    )
    plan = build_funding_plan(100_000.0, fin)
    assert plan.od_drawn == pytest.approx(9_000.0)
    assert plan.own_capital_deployed == pytest.approx(5_000.0)
    assert plan.shortfall == pytest.approx(86_000.0)


def test_mixed_funding_splits_as_instructed():
    fin = FinancingAssumptions(
        funding_mode=FundingMode.MIXED,
        own_capital_available=10_000.0,
        own_capital_deployed=4_000.0,
        fd_amount=100_000.0,
        od_ltv_pct=90.0,
    )
    plan = build_funding_plan(10_000.0, fin)
    assert plan.own_capital_deployed == pytest.approx(4_000.0)
    assert plan.od_drawn == pytest.approx(6_000.0)
    assert plan.od_share == pytest.approx(0.6)
    assert plan.own_share == pytest.approx(0.4)


def test_fd_collateral_locked_reflects_the_ltv():
    fin = FinancingAssumptions(
        funding_mode=FundingMode.OD,
        own_capital_available=0.0,
        fd_amount=100_000.0,
        od_ltv_pct=90.0,
    )
    plan = build_funding_plan(45_000.0, fin)
    assert plan.fd_collateral_locked == pytest.approx(50_000.0)


# ---------------------------------------------------------------------------
# Transaction costs and taxes
# ---------------------------------------------------------------------------
def test_transaction_costs_match_a_hand_calculation():
    costs = TransactionCostAssumptions(
        brokerage_flat_sell=20.0,
        stt_pct_sell=0.1,
        exchange_txn_pct=0.00297,
        sebi_turnover_pct=0.0001,
        gst_pct=18.0,
        dp_charges_flat_sell=15.93,
        stamp_duty_pct_buy=0.0,
    )
    breakdown = compute_transaction_costs(10_000.0, 12_000.0, costs)
    assert breakdown.brokerage == pytest.approx(20.0)
    assert breakdown.stt == pytest.approx(12.0)  # 0.1% of 12,000
    assert breakdown.exchange_txn_charges == pytest.approx(22_000 * 0.0000297)
    assert breakdown.sebi_turnover_fees == pytest.approx(22_000 * 0.000001)
    assert breakdown.gst == pytest.approx(
        (20.0 + 22_000 * 0.0000297 + 22_000 * 0.000001) * 0.18
    )
    assert breakdown.dp_charges == pytest.approx(15.93)
    assert breakdown.total == pytest.approx(52.326972, rel=1e-9)
    # STT is not deductible against capital gains
    assert breakdown.deductible_from_gain == pytest.approx(breakdown.total - 12.0)


def test_no_sell_leg_means_no_sell_side_charges():
    costs = TransactionCostAssumptions()
    breakdown = compute_transaction_costs(10_000.0, 0.0, costs)
    assert breakdown.dp_charges == 0.0
    assert breakdown.stt == 0.0
    assert breakdown.brokerage == pytest.approx(costs.brokerage_flat_buy)


def test_short_term_capital_gains_tax_with_cess():
    taxes = TaxAssumptions(stcg_rate_pct=20.0, cess_and_surcharge_pct=4.0)
    # 2,000 gain, 40 of deductible costs -> 1,960 * 20% * 1.04
    tax = compute_capital_gains_tax(2_000.0, holding_days=5, taxes=taxes, deductible_costs=40.0)
    assert tax == pytest.approx(1_960.0 * 0.20 * 1.04)


def test_long_term_rate_applies_beyond_the_threshold():
    taxes = TaxAssumptions(stcg_rate_pct=20.0, ltcg_rate_pct=12.5, ltcg_threshold_days=365)
    short = compute_capital_gains_tax(10_000.0, 365, taxes)
    long = compute_capital_gains_tax(10_000.0, 366, taxes)
    assert short == pytest.approx(10_000 * 0.20 * 1.04)
    assert long == pytest.approx(10_000 * 0.125 * 1.04)


def test_losses_attract_no_tax_unless_a_shield_is_requested():
    taxes = TaxAssumptions()
    assert compute_capital_gains_tax(-5_000.0, 5, taxes) == 0.0
    shielded = replace(taxes, recognise_tax_shield_on_loss=True)
    assert compute_capital_gains_tax(-5_000.0, 5, shielded) == pytest.approx(
        -5_000.0 * 0.20 * 1.04
    )


def test_ltcg_exemption_only_applies_when_enabled():
    taxes = TaxAssumptions(apply_ltcg_exemption=True, ltcg_exemption_amount=125_000.0)
    assert compute_capital_gains_tax(100_000.0, 400, taxes) == 0.0
    assert compute_capital_gains_tax(200_000.0, 400, taxes) == pytest.approx(
        75_000.0 * 0.125 * 1.04
    )


# ---------------------------------------------------------------------------
# Allotment probability mathematics
# ---------------------------------------------------------------------------
def test_probability_of_zero_and_at_least_one_allotment():
    dist = allotment_distribution([0.1, 0.2, 0.3])
    assert dist.p_zero == pytest.approx(0.9 * 0.8 * 0.7)  # 0.504
    assert dist.p_at_least_one == pytest.approx(1 - 0.504)
    assert dist.expected_allotments == pytest.approx(0.6)
    assert dist.variance == pytest.approx(0.09 + 0.16 + 0.21)


def test_poisson_binomial_distribution_is_exact_and_sums_to_one():
    dist = allotment_distribution([0.1, 0.2, 0.3])
    assert dist.probabilities == pytest.approx((0.504, 0.398, 0.092, 0.006))
    assert sum(dist.probabilities) == pytest.approx(1.0)


def test_identical_accounts_reduce_to_the_binomial():
    dist = allotment_distribution([0.5, 0.5, 0.5])
    assert dist.probabilities == pytest.approx((0.125, 0.375, 0.375, 0.125))


def test_certain_and_impossible_allotment_edge_cases():
    assert allotment_distribution([1.0, 1.0]).p_zero == 0.0
    assert allotment_distribution([0.0, 0.0]).p_at_least_one == 0.0
    assert allotment_distribution([]).p_zero == 1.0


# ---------------------------------------------------------------------------
# End-to-end expected value, hand-checked
# ---------------------------------------------------------------------------
BID_COST = 10_000 * 0.10 * 10 / 365           # 27.397260273972602
HOLDING_CARRY = 10_000 * 0.10 * 5 / 365       # 13.698630136986301


def test_basic_ipo_profit_if_allotted():
    result = analyze(frictionless())
    account = result.accounts[0]
    assert account.shares_if_allotted == 100
    assert account.investment_if_allotted == pytest.approx(10_000.0)
    assert account.exit_value_if_allotted == pytest.approx(12_000.0)
    assert account.gross_profit_if_allotted == pytest.approx(2_000.0)
    assert account.transaction_costs_if_allotted.total == 0.0
    assert account.tax_if_allotted == 0.0
    # carry = bidding window + holding window on the allotted shares
    assert account.carry_cost_if_allotted == pytest.approx(BID_COST + HOLDING_CARRY)
    assert account.net_profit_if_allotted == pytest.approx(
        2_000.0 - BID_COST - HOLDING_CARRY
    )


def test_expected_profit_is_probability_times_profit_less_unconditional_carry():
    result = analyze(frictionless(probability=0.25))
    assert result.expected_gross_profit == pytest.approx(500.0)
    assert result.expected_financing_cost == pytest.approx(BID_COST + 0.25 * HOLDING_CARRY)
    assert result.expected_net_profit_cash == pytest.approx(
        500.0 - BID_COST - 0.25 * HOLDING_CARRY
    )
    assert result.expected_net_profit_cash == pytest.approx(469.17808219178085)


def test_financing_cost_splits_into_bidding_and_holding_windows():
    result = analyze(frictionless(probability=0.4))
    assert result.financing.od_cost_bidding_window == pytest.approx(BID_COST)
    assert result.financing.expected_od_cost_holding_window == pytest.approx(
        0.4 * HOLDING_CARRY
    )
    # The bidding-window cost is unconditional; only the holding leg is weighted.
    assert result.net_profit_if_no_allotment == pytest.approx(-BID_COST)


def test_account_contributions_reconcile_with_the_headline_number():
    result = analyze(frictionless(n_accounts=4, probability=0.2))
    total = sum(a.expected_net_profit_contribution for a in result.accounts)
    assert total == pytest.approx(result.expected_net_profit_economic)


def test_zero_allotment_probability_leaves_only_the_financing_cost():
    result = analyze(frictionless(probability=0.0))
    assert result.expected_gross_profit == 0.0
    assert result.expected_net_profit_cash == pytest.approx(-BID_COST)
    assert result.allotment.p_zero == 1.0
    assert result.allotment.p_at_least_one == 0.0


def test_certain_allotment_matches_the_conditional_calculation():
    inputs = frictionless(probability=1.0)
    result = analyze(inputs)
    assert result.expected_net_profit_cash == pytest.approx(
        2_000.0 - BID_COST - HOLDING_CARRY
    )
    assert result.expected_net_profit_cash == pytest.approx(
        conditional_net_profit(inputs)
    )
    assert result.allotment.p_zero == 0.0


def test_negative_listing_produces_a_loss_and_no_tax():
    inputs = frictionless(gmp=-20.0, probability=0.25)
    result = analyze(inputs)
    assert inputs.ipo.expected_listing_price == 80.0
    assert result.accounts[0].gross_profit_if_allotted == pytest.approx(-2_000.0)
    assert result.accounts[0].tax_if_allotted == 0.0
    assert result.expected_net_profit_cash == pytest.approx(
        -500.0 - BID_COST - 0.25 * HOLDING_CARRY
    )


def test_multiple_accounts_scale_profit_by_hit_rate_but_cost_by_applications():
    single = analyze(frictionless(n_accounts=1, probability=0.2))
    triple = analyze(frictionless(n_accounts=3, probability=0.2))
    assert triple.capital.total_application_amount == pytest.approx(30_000.0)
    assert triple.expected_gross_profit == pytest.approx(3 * single.expected_gross_profit)
    # Financing scales with every application, not just the allotted ones.
    assert triple.financing.od_cost_bidding_window == pytest.approx(3 * BID_COST)
    assert triple.expected_net_profit_cash == pytest.approx(
        1_200.0 - 3 * BID_COST - 3 * 0.2 * HOLDING_CARRY
    )
    assert triple.allotment.p_zero == pytest.approx(0.8 ** 3)
    assert triple.expected_allotments == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# Funding structures
# ---------------------------------------------------------------------------
def test_od_financed_application_charges_interest_on_the_full_draw():
    result = analyze(frictionless(funding_mode=FundingMode.OD, own_available=0.0))
    assert result.funding.od_drawn == pytest.approx(10_000.0)
    assert result.funding.own_capital_deployed == 0.0
    assert result.expected_opportunity_cost == 0.0
    assert result.financing.od_cost_bidding_window == pytest.approx(BID_COST)
    # Economic capital is the pledged deposit: 10,000 / 0.90
    assert result.capital.economic_capital_at_risk == pytest.approx(10_000 / 0.9)


def test_own_capital_only_charges_opportunity_cost_and_no_interest():
    inputs = frictionless(
        funding_mode=FundingMode.OWN, own_available=20_000.0, fd_amount=0.0
    )
    result = analyze(inputs)
    assert result.funding.od_drawn == 0.0
    assert result.financing.od_cost_bidding_window == 0.0
    expected_opportunity = 10_000 * 0.07 * 10 / 365 + 0.25 * (10_000 * 0.07 * 5 / 365)
    assert result.expected_opportunity_cost == pytest.approx(expected_opportunity)
    assert result.expected_net_profit_economic == pytest.approx(
        500.0 - expected_opportunity
    )


def test_mixed_funding_charges_od_and_opportunity_cost_pro_rata():
    inputs = frictionless(
        funding_mode=FundingMode.MIXED, own_available=10_000.0, own_deployed=4_000.0
    )
    result = analyze(inputs)
    assert result.funding.own_capital_deployed == pytest.approx(4_000.0)
    assert result.funding.od_drawn == pytest.approx(6_000.0)

    bid_od = 6_000 * 0.10 * 10 / 365
    bid_opportunity = 4_000 * 0.07 * 10 / 365
    hold_od = 10_000 * 0.6 * 0.10 * 5 / 365
    hold_opportunity = 10_000 * 0.4 * 0.07 * 5 / 365
    assert result.financing.od_cost_bidding_window == pytest.approx(bid_od)
    assert result.financing.opportunity_cost_bidding_window == pytest.approx(bid_opportunity)
    assert result.expected_net_profit_cash == pytest.approx(
        500.0 - bid_od - 0.25 * hold_od
    )
    assert result.expected_net_profit_economic == pytest.approx(
        500.0 - bid_od - 0.25 * hold_od - bid_opportunity - 0.25 * hold_opportunity
    )
    # Economic capital = own cash + the deposit pledged for the 6,000 drawn
    assert result.capital.economic_capital_at_risk == pytest.approx(4_000 + 6_000 / 0.9)


def test_squaring_off_at_allotment_moves_carry_from_od_to_opportunity_cost():
    inputs = frictionless()
    financed = analyze(inputs)
    squared = analyze(replace(inputs, financing=replace(inputs.financing, finance_holding_period=False)))
    assert financed.financing.expected_od_cost_holding_window > 0
    assert squared.financing.expected_od_cost_holding_window == 0.0
    assert squared.financing.expected_opportunity_cost_holding_window == pytest.approx(
        0.25 * 10_000 * 0.07 * 5 / 365
    )


def test_processing_fees_are_unconditional():
    inputs = frictionless(probability=0.0)
    inputs = replace(
        inputs,
        financing=replace(inputs.financing, processing_fee=500.0, other_financing_charges=250.0),
    )
    result = analyze(inputs)
    assert result.expected_net_profit_cash == pytest.approx(-BID_COST - 750.0)


def test_fd_interest_is_excluded_from_profit_unless_explicitly_counted():
    inputs = frictionless()
    default = analyze(inputs)
    counted = analyze(
        replace(inputs, financing=replace(inputs.financing, count_fd_interest_as_income=True))
    )
    fd_interest = 100_000 * 0.07 * 15 / 365
    assert default.financing.fd_interest_earned == pytest.approx(fd_interest)
    assert default.financing.fd_interest_credit == 0.0
    assert counted.expected_net_profit_cash == pytest.approx(
        default.expected_net_profit_cash + fd_interest
    )


# ---------------------------------------------------------------------------
# Break-even analysis
# ---------------------------------------------------------------------------
def test_break_even_price_if_allotted_covers_only_the_cost_of_carry():
    inputs = frictionless()
    price = break_even_exit_price(inputs, "if_allotted")
    # 100 shares must recover the full carry on the allotted lot:
    # (P - 100) * 100 = bidding carry + holding carry
    expected = 100.0 + (BID_COST + HOLDING_CARRY) / 100.0
    assert price == pytest.approx(expected, rel=1e-9)
    assert price == pytest.approx(100.41095890410959, rel=1e-9)
    assert conditional_net_profit(inputs, price) == pytest.approx(0.0, abs=1e-6)


def test_expected_value_break_even_also_pays_for_the_failed_applications():
    inputs = frictionless(probability=0.25)
    price = break_even_exit_price(inputs, "expected_value")
    # 0.25 * ((P - 100) * 100 - holding carry) = bidding carry
    expected = 100.0 + (BID_COST / 0.25 + HOLDING_CARRY) / 100.0
    assert price == pytest.approx(expected, rel=1e-9)
    assert price == pytest.approx(101.23287671232876, rel=1e-9)
    assert expected_net_profit(
        replace(inputs, ipo=replace(inputs.ipo, expected_exit_price_override=price))
    ) == pytest.approx(0.0, abs=1e-6)


def test_expected_value_break_even_is_always_the_higher_hurdle():
    inputs = frictionless(probability=0.25)
    assert break_even_exit_price(inputs, "expected_value") > break_even_exit_price(
        inputs, "if_allotted"
    )


def test_break_even_gmp_is_the_break_even_price_less_the_issue_price():
    result = analyze(frictionless())
    assert result.break_even.gmp_if_allotted == pytest.approx(
        result.break_even.exit_price_if_allotted - 100.0
    )
    assert result.break_even.gmp_expected_value == pytest.approx(1.2328767123287672, rel=1e-9)
    assert result.break_even.listing_gain_pct_expected_value == pytest.approx(
        1.2328767123287672, rel=1e-9
    )


def test_lower_hit_rate_raises_the_expected_value_break_even():
    high = analyze(frictionless(probability=0.5)).break_even.exit_price_expected_value
    low = analyze(frictionless(probability=0.05)).break_even.exit_price_expected_value
    assert low > high


def test_break_even_rises_once_real_costs_and_taxes_are_switched_on():
    frictionless_price = break_even_exit_price(frictionless(), "if_allotted")
    with_costs = replace(
        frictionless(),
        costs=TransactionCostAssumptions(),
        taxes=TaxAssumptions(),
    )
    assert break_even_exit_price(with_costs, "if_allotted") > frictionless_price


def test_maximum_sustainable_od_rate():
    inputs = frictionless()
    # 0.25 * (2000 - 10000*r*5/365) = 10000*r*10/365  ->  r = 500*365/112500
    expected = 500 * 365 / 112_500 * 100
    rate = max_sustainable_od_rate(inputs)
    assert rate == pytest.approx(expected, rel=1e-6)
    assert rate == pytest.approx(162.2222222, rel=1e-6)
    at_the_limit = replace(inputs, financing=replace(inputs.financing, od_rate_pct=rate))
    assert expected_net_profit(at_the_limit) == pytest.approx(0.0, abs=1e-4)


def test_maximum_sustainable_od_rate_is_undefined_when_the_trade_itself_loses():
    inputs = frictionless(gmp=-5.0)
    assert max_sustainable_od_rate(inputs) is None
    assert expected_net_profit(replace(inputs, financing=replace(inputs.financing, od_rate_pct=0.0))) < 0


def test_minimum_allotment_probability():
    inputs = frictionless()
    # p * (2000 - holding carry) = bidding carry
    expected = BID_COST / (2_000.0 - HOLDING_CARRY)
    probability = min_allotment_probability(inputs)
    assert probability == pytest.approx(expected, rel=1e-6)
    assert probability == pytest.approx(0.0137931034, rel=1e-6)


# ---------------------------------------------------------------------------
# Capital efficiency
# ---------------------------------------------------------------------------
def test_application_capital_and_own_equity_returns_are_not_conflated():
    result = analyze(frictionless())
    capital = result.capital
    net = result.expected_net_profit
    assert capital.total_application_amount == pytest.approx(10_000.0)
    assert capital.economic_capital_at_risk == pytest.approx(10_000 / 0.9)
    assert capital.return_on_application_capital == pytest.approx(net / 10_000.0)
    assert capital.return_on_economic_capital == pytest.approx(net / (10_000 / 0.9))
    # The two denominators differ, so the two returns must differ.
    assert capital.return_on_application_capital != capital.return_on_economic_capital


def test_return_on_own_capital_is_undefined_when_no_own_cash_is_deployed():
    result = analyze(frictionless(funding_mode=FundingMode.OD, own_available=0.0))
    assert result.funding.own_capital_deployed == 0.0
    assert result.capital.return_on_own_capital is None
    assert result.capital.return_on_economic_capital is not None


def test_capital_weighted_days_weights_the_holding_leg_by_the_hit_rate():
    result = analyze(frictionless(probability=0.25, days_blocked=10, holding_days=5))
    # (10,000 x 10 days + 0.25 x 10,000 x 5 days) / 10,000 = 11.25
    assert result.capital.capital_weighted_days == pytest.approx(11.25)
    assert result.capital.cycle_days == 15


def test_annualized_return_uses_capital_weighted_days():
    result = analyze(frictionless())
    roi = result.capital.return_on_economic_capital
    assert result.capital.annualized_return_on_economic_capital == pytest.approx(
        (1 + roi) ** (365 / 11.25) - 1
    )


def test_capital_efficiency_ratios():
    result = analyze(frictionless())
    capital = result.capital
    assert capital.financing_cost_to_gross_profit == pytest.approx(
        result.expected_financing_cost / 500.0
    )
    assert capital.profit_to_financing_cost == pytest.approx(
        result.expected_net_profit / result.expected_financing_cost
    )


# ---------------------------------------------------------------------------
# Precision and unit hygiene
# ---------------------------------------------------------------------------
def test_no_intermediate_rounding_occurs():
    result = analyze(frictionless())
    value = result.expected_net_profit_cash
    assert value != round(value, 2)
    assert result.financing.od_cost_bidding_window != round(
        result.financing.od_cost_bidding_window, 2
    )


def test_doubling_the_holding_period_doubles_the_holding_leg_of_the_carry():
    five = analyze(frictionless(holding_days=5)).financing.expected_od_cost_holding_window
    ten = analyze(frictionless(holding_days=10)).financing.expected_od_cost_holding_window
    assert ten == pytest.approx(2 * five)


def test_an_alternative_day_count_basis_is_respected_end_to_end():
    inputs = frictionless()
    base = analyze(inputs).financing.od_cost_bidding_window
    alt = analyze(
        replace(inputs, financing=replace(inputs.financing, day_count_basis=360))
    ).financing.od_cost_bidding_window
    assert alt == pytest.approx(base * 365 / 360)


def test_analyze_accepts_an_exit_price_override_without_mutating_inputs():
    inputs = frictionless()
    overridden = analyze(inputs, exit_price=150.0)
    assert overridden.accounts[0].exit_value_if_allotted == pytest.approx(15_000.0)
    assert inputs.ipo.expected_exit_price == 120.0  # unchanged
    assert analyze(inputs).accounts[0].exit_value_if_allotted == pytest.approx(12_000.0)


def test_summary_dictionary_exposes_every_headline_metric():
    summary = analyze(frictionless()).summary_dict()
    for key in (
        "Total application amount",
        "Own capital deployed",
        "Borrowed capital (OD)",
        "Expected net profit (economic)",
        "Return on application capital",
        "Return on economic capital",
        "Break-even GMP (expected value)",
        "Max sustainable OD rate (%)",
    ):
        assert key in summary


# ---------------------------------------------------------------------------
# A realistic case with the full cost and tax stack, verified line by line
# ---------------------------------------------------------------------------
def realistic_inputs(**overrides) -> AnalysisInputs:
    """Rs 300 issue, 50-share lot, Rs 105 GMP, 10% hit-rate, six days blocked."""
    financing = FinancingAssumptions(
        funding_mode=FundingMode.OWN,
        own_capital_available=20_000.0,
        fd_amount=0.0,
        od_rate_pct=0.0,
        days_blocked=6,
        opportunity_cost_rate_pct=7.0,
        include_opportunity_cost=True,
    )
    inputs = AnalysisInputs(
        ipo=IPOAssumptions(
            name="Realistic",
            issue_price=300.0,
            lot_size=50,
            gmp_value=105.0,
            holding_period_days=0,
        ),
        accounts=(ApplicationAccount(label="Self", allotment_probability=0.10),),
        financing=financing,
        costs=TransactionCostAssumptions(),  # shipped defaults
        taxes=TaxAssumptions(),
    )
    return replace(inputs, **overrides) if overrides else inputs


def hand_computed_costs() -> tuple:
    """Recompute the whole cost stack independently of the engine."""
    investment = 300.0 * 50
    exit_value = 405.0 * 50
    turnover = investment + exit_value
    brokerage = 20.0
    stt = 0.1 / 100 * exit_value
    exchange = 0.00297 / 100 * turnover
    sebi = 0.0001 / 100 * turnover
    gst = 0.18 * (brokerage + exchange + sebi)
    dp = 15.93
    total = brokerage + stt + exchange + sebi + gst + dp
    return total, total - stt


def test_realistic_case_matches_a_line_by_line_hand_calculation():
    result = analyze(realistic_inputs())
    account = result.accounts[0]
    costs, deductible = hand_computed_costs()
    gross = 5_250.0
    tax = (gross - deductible) * 0.20 * 1.04
    carry = 15_000.0 * 0.07 * 6 / 365

    assert account.gross_profit_if_allotted == pytest.approx(gross)
    assert account.transaction_costs_if_allotted.total == pytest.approx(costs, rel=1e-12)
    assert costs == pytest.approx(61.0569665, rel=1e-9)
    assert account.tax_if_allotted == pytest.approx(tax, rel=1e-12)
    assert tax == pytest.approx(1_083.51215097, rel=1e-9)
    assert account.carry_cost_if_allotted == pytest.approx(carry, rel=1e-12)
    assert account.net_profit_if_allotted == pytest.approx(gross - costs - tax - carry)
    assert result.expected_net_profit == pytest.approx(0.10 * (gross - costs - tax) - carry)
    assert result.expected_net_profit == pytest.approx(393.28281428, rel=1e-9)


def test_the_same_trade_on_borrowed_money_earns_less():
    own_funded = analyze(realistic_inputs())
    borrowed = analyze(
        realistic_inputs(
            financing=FinancingAssumptions(
                funding_mode=FundingMode.OD,
                own_capital_available=0.0,
                fd_amount=20_000.0,
                od_ltv_pct=90.0,
                od_rate_pct=11.0,
                days_blocked=6,
                opportunity_cost_rate_pct=7.0,
                include_opportunity_cost=True,
            )
        )
    )
    interest = 15_000.0 * 0.11 * 6 / 365
    assert borrowed.funding.od_drawn == pytest.approx(15_000.0)
    assert borrowed.financing.od_cost_bidding_window == pytest.approx(interest)
    # Same gross profit, higher cost of money, so a lower net.
    assert borrowed.expected_gross_profit == pytest.approx(own_funded.expected_gross_profit)
    assert borrowed.expected_net_profit < own_funded.expected_net_profit


def test_a_long_holding_period_switches_to_the_long_term_tax_rate():
    short = analyze(realistic_inputs())
    long = analyze(
        replace(
            realistic_inputs(),
            ipo=replace(realistic_inputs().ipo, holding_period_days=400),
        )
    )
    costs, deductible = hand_computed_costs()
    assert long.accounts[0].tax_if_allotted == pytest.approx(
        (5_250.0 - deductible) * 0.125 * 1.04
    )
    assert long.accounts[0].tax_if_allotted < short.accounts[0].tax_if_allotted
    # ... but 400 days of carry costs far more than the tax saved.
    assert long.expected_net_profit < short.expected_net_profit
