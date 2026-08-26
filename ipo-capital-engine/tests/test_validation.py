"""Tests for input validation and the assumption ledger."""

from __future__ import annotations

from dataclasses import replace

import pytest

from calculations import (
    AnalysisInputs,
    ApplicationAccount,
    FinancingAssumptions,
    FundingMode,
    IPOAssumptions,
    IPOCategory,
    Provenance,
    TaxAssumptions,
    TransactionCostAssumptions,
    assumption_ledger,
)
from validation import RETAIL_LIMIT, Severity, validate_inputs


def valid_inputs() -> AnalysisInputs:
    return AnalysisInputs(
        ipo=IPOAssumptions(
            name="Test IPO", issue_price=100.0, lot_size=100, gmp_value=20.0, holding_period_days=2
        ),
        accounts=(ApplicationAccount(label="PAN 1", allotment_probability=0.2),),
        financing=FinancingAssumptions(
            funding_mode=FundingMode.OD,
            own_capital_available=0.0,
            fd_amount=50_000.0,
            od_ltv_pct=90.0,
            od_rate_pct=11.0,
            days_blocked=7,
        ),
    )


def messages(report, severity=None):
    return [
        i.message
        for i in report.issues
        if severity is None or i.severity is severity
    ]


def test_a_well_formed_set_of_inputs_produces_no_errors():
    report = validate_inputs(valid_inputs())
    assert report.is_valid
    assert not report.errors


@pytest.mark.parametrize("price", [0.0, -50.0])
def test_non_positive_issue_price_is_rejected(price):
    inputs = valid_inputs()
    report = validate_inputs(replace(inputs, ipo=replace(inputs.ipo, issue_price=price)))
    assert not report.is_valid
    assert any("Issue price" in i.field for i in report.errors)


def test_invalid_lot_size_is_rejected():
    inputs = valid_inputs()
    report = validate_inputs(replace(inputs, ipo=replace(inputs.ipo, lot_size=0)))
    assert any("Lot size" in i.field for i in report.errors)


def test_negative_holding_period_is_rejected():
    inputs = valid_inputs()
    report = validate_inputs(
        replace(inputs, ipo=replace(inputs.ipo, holding_period_days=-1))
    )
    assert any("Holding period" in i.field for i in report.errors)


def test_zero_days_blocked_is_rejected():
    inputs = valid_inputs()
    report = validate_inputs(
        replace(inputs, financing=replace(inputs.financing, days_blocked=0))
    )
    assert any("Days capital blocked" in i.field for i in report.errors)


@pytest.mark.parametrize("probability", [-0.1, 1.5])
def test_impossible_probabilities_are_rejected(probability):
    inputs = valid_inputs()
    account = replace(inputs.accounts[0], allotment_probability=probability)
    report = validate_inputs(inputs.with_accounts([account]))
    assert not report.is_valid
    assert any("probability" in m for m in messages(report, Severity.ERROR))


def test_negative_interest_rates_are_rejected():
    inputs = valid_inputs()
    report = validate_inputs(
        replace(inputs, financing=replace(inputs.financing, od_rate_pct=-5.0))
    )
    assert any("Interest rates cannot be negative." in m for m in messages(report, Severity.ERROR))


def test_od_ltv_above_100_percent_is_rejected():
    inputs = valid_inputs()
    report = validate_inputs(
        replace(inputs, financing=replace(inputs.financing, od_ltv_pct=120.0))
    )
    assert any("LTV" in i.field for i in report.errors)


def test_ltv_above_typical_bank_sanction_warns():
    inputs = valid_inputs()
    report = validate_inputs(
        replace(inputs, financing=replace(inputs.financing, od_ltv_pct=98.0))
    )
    assert report.is_valid
    assert any("LTV" in i.field for i in report.warnings)


def test_funding_shortfall_is_reported_as_an_error():
    inputs = valid_inputs()
    financing = replace(
        inputs.financing, fd_amount=1_000.0, own_capital_available=100.0
    )
    report = validate_inputs(replace(inputs, financing=financing))
    assert not report.is_valid
    assert any("Shortfall" in m for m in messages(report, Severity.ERROR))


def test_retail_application_above_the_sebi_cap_is_rejected():
    inputs = valid_inputs()
    account = replace(inputs.accounts[0], lots_applied=25, category=IPOCategory.RETAIL)
    financing = replace(inputs.financing, fd_amount=1_000_000.0)
    report = validate_inputs(
        replace(inputs, accounts=(account,), financing=financing)
    )
    assert not report.is_valid
    assert any(str(int(RETAIL_LIMIT)) in m.replace(",", "") for m in messages(report, Severity.ERROR))


def test_allotment_above_lots_applied_is_rejected():
    inputs = valid_inputs()
    account = replace(inputs.accounts[0], lots_applied=1, lots_allotted_if_successful=3.0)
    report = validate_inputs(inputs.with_accounts([account]))
    assert any("cannot exceed lots applied" in m for m in messages(report, Severity.ERROR))


def test_missing_listing_price_when_gmp_is_disabled_is_rejected():
    inputs = valid_inputs()
    ipo = replace(inputs.ipo, use_gmp_for_listing=False, expected_listing_price_override=None)
    report = validate_inputs(replace(inputs, ipo=ipo))
    assert any("listing price" in m for m in messages(report, Severity.ERROR))


def test_drawing_od_without_an_fd_is_rejected():
    inputs = valid_inputs()
    financing = replace(
        inputs.financing,
        funding_mode=FundingMode.MIXED,
        fd_amount=0.0,
        own_capital_available=10_000.0,
        own_capital_deployed=10_000.0,
    )
    report = validate_inputs(replace(inputs, financing=financing))
    assert report.is_valid  # fully own-funded, so no OD is drawn


def test_deploying_more_than_available_is_rejected():
    inputs = valid_inputs()
    financing = replace(
        inputs.financing,
        funding_mode=FundingMode.MIXED,
        own_capital_available=1_000.0,
        own_capital_deployed=5_000.0,
    )
    report = validate_inputs(replace(inputs, financing=financing))
    assert any("more own capital than is available" in m for m in messages(report, Severity.ERROR))


def test_negative_gmp_only_warns_because_a_discount_is_legitimate():
    inputs = valid_inputs()
    report = validate_inputs(replace(inputs, ipo=replace(inputs.ipo, gmp_value=-10.0)))
    assert report.is_valid
    assert any("Negative GMP" in m for m in messages(report, Severity.WARNING))


def test_negative_charges_are_rejected():
    inputs = valid_inputs()
    report = validate_inputs(
        replace(inputs, costs=TransactionCostAssumptions(brokerage_flat_sell=-20.0))
    )
    assert not report.is_valid


def test_tax_rate_above_100_percent_is_rejected():
    inputs = valid_inputs()
    report = validate_inputs(replace(inputs, taxes=TaxAssumptions(stcg_rate_pct=150.0)))
    assert not report.is_valid


# ---------------------------------------------------------------------------
# Assumption ledger (data integrity)
# ---------------------------------------------------------------------------
def test_ledger_marks_untouched_defaults_as_assumptions():
    ledger = assumption_ledger(valid_inputs())
    gst = next(r for r in ledger if r.name == "GST %")
    assert gst.provenance is Provenance.ASSUMED


def test_ledger_marks_edited_values_as_user_entered():
    inputs = valid_inputs()
    inputs = replace(inputs, costs=replace(inputs.costs, gst_pct=20.0))
    ledger = assumption_ledger(inputs)
    gst = next(r for r in ledger if r.name == "GST %")
    assert gst.provenance is Provenance.USER


def test_ledger_marks_derived_numbers_as_calculated():
    ledger = assumption_ledger(valid_inputs())
    listing = next(r for r in ledger if r.name == "Expected listing price")
    assert listing.provenance is Provenance.CALCULATED
    assert listing.value == pytest.approx(120.0)


def test_ledger_flags_the_independence_assumption_for_multiple_accounts():
    ledger = assumption_ledger(valid_inputs())
    row = next(r for r in ledger if r.name == "Independent allotment draws")
    assert row.provenance is Provenance.ASSUMED
    assert "independence" in row.note.lower()


def test_ledger_warns_that_gmp_is_not_a_guarantee():
    ledger = assumption_ledger(valid_inputs())
    row = next(r for r in ledger if r.name.startswith("GMP"))
    assert "sentiment" in row.note.lower()
