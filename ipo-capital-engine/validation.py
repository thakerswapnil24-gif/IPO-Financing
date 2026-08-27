"""Input validation for the IPO capital allocation engine.

Validation is deliberately separate from both the UI and the calculation engine
so that any front-end (Streamlit, CLI, batch job) can run the same checks and
get the same messages.

Two severities are produced:

* ``ERROR``   - the analysis is not meaningful; the caller should refuse to run.
* ``WARNING`` - the analysis will run, but a number looks implausible or a
  regulatory limit appears to be breached and the user must confirm it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum

from calculations import (
    AnalysisInputs,
    ApplicationAccount,
    FinancingAssumptions,
    FundingMode,
    IPOCategory,
    TaxAssumptions,
    TransactionCostAssumptions,
    build_funding_plan,
)

__all__ = [
    "Severity",
    "ValidationIssue",
    "ValidationReport",
    "validate_inputs",
    "RETAIL_LIMIT",
    "SNII_LIMIT",
]

#: SEBI application-size boundaries for individual investors (Rs).
RETAIL_LIMIT = 200_000.0
SNII_LIMIT = 1_000_000.0


class Severity(str, Enum):
    ERROR = "Error"
    WARNING = "Warning"


@dataclass(frozen=True)
class ValidationIssue:
    severity: Severity
    field: str
    message: str

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return f"[{self.severity.value}] {self.field}: {self.message}"


@dataclass
class ValidationReport:
    issues: list[ValidationIssue] = field(default_factory=list)

    def add(self, severity: Severity, field_name: str, message: str) -> None:
        self.issues.append(ValidationIssue(severity, field_name, message))

    def error(self, field_name: str, message: str) -> None:
        self.add(Severity.ERROR, field_name, message)

    def warn(self, field_name: str, message: str) -> None:
        self.add(Severity.WARNING, field_name, message)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity is Severity.ERROR]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity is Severity.WARNING]

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def __bool__(self) -> bool:  # pragma: no cover - convenience only
        return self.is_valid


def _check_ipo(inputs: AnalysisInputs, report: ValidationReport) -> None:
    ipo = inputs.ipo
    if not ipo.name or not ipo.name.strip():
        report.warn(
            "IPO name", "No IPO name supplied - reports will be hard to identify."
        )
    if ipo.issue_price <= 0:
        report.error("Issue price", "Issue price must be greater than zero.")
    if ipo.lot_size <= 0:
        report.error("Lot size", "Lot size must be a positive whole number of shares.")
    elif int(ipo.lot_size) != ipo.lot_size:
        report.error("Lot size", "Lot size must be a whole number of shares.")
    if ipo.holding_period_days < 0:
        report.error("Holding period", "Holding period cannot be negative.")
    if ipo.expected_listing_price < 0:
        report.error("Expected listing price", "Listing price cannot be negative.")
    if ipo.expected_exit_price < 0:
        report.error("Expected exit price", "Exit price cannot be negative.")
    if not ipo.use_gmp_for_listing and ipo.expected_listing_price_override is None:
        report.error(
            "Expected listing price",
            "GMP-derived pricing is switched off but no listing price was entered.",
        )
    if ipo.gmp_absolute < 0:
        report.warn(
            "GMP",
            "Negative GMP implies the market expects a listing below the issue price.",
        )
    if ipo.issue_price > 0 and ipo.expected_listing_gain_pct > 100:
        report.warn(
            "Expected listing gain",
            f"A {ipo.expected_listing_gain_pct:.0f}% listing gain is an extreme "
            "assumption - the result will be highly sensitive to it.",
        )
    if ipo.holding_period_days == 0:
        report.warn(
            "Holding period",
            "A zero-day holding period means selling at the listing price with no "
            "carry cost after allotment.",
        )


def _check_accounts(inputs: AnalysisInputs, report: ValidationReport) -> None:
    accounts: Sequence[ApplicationAccount] = inputs.accounts
    if not accounts:
        report.error("Accounts", "At least one application account is required.")
        return
    labels = [a.label for a in accounts]
    if len(set(labels)) != len(labels):
        report.warn("Accounts", "Duplicate account labels make the report ambiguous.")
    for acct in accounts:
        prefix = f"Account '{acct.label}'"
        if acct.lots_applied <= 0:
            report.error(prefix, "Lots applied must be at least 1.")
        if not 0.0 <= acct.allotment_probability <= 1.0:
            report.error(
                prefix,
                f"Allotment probability {acct.allotment_probability} is outside "
                "the valid range 0 to 1.",
            )
        if acct.lots_allotted_if_successful <= 0:
            report.error(prefix, "Lots allotted on a successful bid must be positive.")
        if acct.lots_allotted_if_successful > acct.lots_applied:
            report.error(
                prefix,
                "Lots allotted cannot exceed lots applied for.",
            )
        amount = acct.application_amount(inputs.ipo.issue_price, inputs.ipo.lot_size)
        if acct.category is IPOCategory.RETAIL and amount > RETAIL_LIMIT:
            report.error(
                prefix,
                f"Retail bids are capped at Rs {RETAIL_LIMIT:,.0f}; this bid is "
                f"Rs {amount:,.0f}. Re-categorise it as sNII or reduce the lots.",
            )
        if acct.category is IPOCategory.SNII and not (
            RETAIL_LIMIT < amount <= SNII_LIMIT
        ):
            report.warn(
                prefix,
                f"sNII bids sit between Rs {RETAIL_LIMIT:,.0f} and Rs "
                f"{SNII_LIMIT:,.0f}; "
                f"this bid is Rs {amount:,.0f}.",
            )
        if acct.category is IPOCategory.BNII and amount <= SNII_LIMIT:
            report.warn(
                prefix,
                f"bNII bids are above Rs {SNII_LIMIT:,.0f}; this bid is Rs "
                f"{amount:,.0f}.",
            )
        if (
            acct.category is IPOCategory.RETAIL
            and acct.lots_allotted_if_successful != 1.0
        ):
            report.warn(
                prefix,
                "In an oversubscribed retail book allotment is one lot per "
                "successful application; values other than 1 need justification.",
            )
        if acct.allotment_probability >= 0.95:
            report.warn(
                prefix,
                "An allotment probability at or above 95% implies an "
                "undersubscribed issue - check the assumption.",
            )


def _check_financing(inputs: AnalysisInputs, report: ValidationReport) -> None:
    fin: FinancingAssumptions = inputs.financing
    total_application = inputs.total_application_amount

    for label, value in (
        ("Own capital available", fin.own_capital_available),
        ("Own capital deployed", fin.own_capital_deployed),
        ("FD amount", fin.fd_amount),
        ("Processing fee", fin.processing_fee),
        ("Other financing charges", fin.other_financing_charges),
    ):
        if value < 0:
            report.error(label, "Cannot be negative.")

    for label, value in (
        ("FD interest rate", fin.fd_rate_pct),
        ("OD interest rate", fin.od_rate_pct),
        ("Opportunity cost rate", fin.opportunity_cost_rate_pct),
    ):
        if value < 0:
            report.error(label, "Interest rates cannot be negative.")
        elif value > 60:
            report.warn(label, f"{value}% p.a. is an extreme rate - please confirm.")

    if not 0 <= fin.od_ltv_pct <= 100:
        report.error(
            "OD LTV", "Loan-to-value against an FD must be between 0% and 100%."
        )
    elif fin.od_ltv_pct > 95:
        report.warn(
            "OD LTV",
            f"{fin.od_ltv_pct}% LTV is above what Indian banks typically sanction "
            "against a fixed deposit (usually 75-90%).",
        )

    if fin.days_blocked <= 0:
        report.error(
            "Days capital blocked",
            "Capital must be blocked for at least one day - the T+3 IPO timetable "
            "means roughly 5 to 7 days in practice.",
        )
    if fin.day_count_basis not in (360, 365, 366):
        report.warn(
            "Day-count basis",
            f"A {fin.day_count_basis}-day basis is unusual; 365 is the Indian norm.",
        )

    if (
        fin.funding_mode is FundingMode.MIXED
        and fin.own_capital_deployed > fin.own_capital_available
    ):
        # Only meaningful in mixed mode: the other modes derive the split.
        report.error(
            "Own capital deployed",
            "Cannot deploy more own capital than is available.",
        )

    plan = build_funding_plan(total_application, fin)
    if plan.shortfall > 1e-6:
        report.error(
            "Funding",
            f"Application needs Rs {total_application:,.0f} but only "
            f"Rs {plan.own_capital_deployed + plan.od_drawn:,.0f} is available "
            f"(own cash + OD limit). Shortfall: Rs {plan.shortfall:,.0f}.",
        )
    if plan.od_drawn > plan.od_limit + 1e-6:
        report.error(
            "OD drawn",
            "Overdraft drawn exceeds the sanctioned limit (FD amount x LTV).",
        )
    if (
        fin.funding_mode is FundingMode.OWN
        and fin.fd_amount > 0
        and fin.od_rate_pct > 0
    ):
        report.warn(
            "Funding mode",
            "Funding mode is 'own capital only', so the OD inputs do not affect "
            "the result.",
        )
    if fin.od_rate_pct > 0 and fin.od_rate_pct < fin.fd_rate_pct:
        report.warn(
            "OD interest rate",
            "An OD rate below the FD rate is unusual - banks normally charge "
            "1-2% above the deposit rate.",
        )
    if plan.od_drawn > 0 and fin.fd_amount <= 0:
        report.error("FD amount", "OD is drawn but no FD collateral was entered.")


def _check_costs_and_taxes(
    costs: TransactionCostAssumptions, taxes: TaxAssumptions, report: ValidationReport
) -> None:
    for label, value in (
        ("Brokerage % (buy)", costs.brokerage_pct_buy),
        ("Brokerage % (sell)", costs.brokerage_pct_sell),
        ("Brokerage flat (buy)", costs.brokerage_flat_buy),
        ("Brokerage flat (sell)", costs.brokerage_flat_sell),
        ("STT % (buy)", costs.stt_pct_buy),
        ("STT % (sell)", costs.stt_pct_sell),
        ("Exchange transaction %", costs.exchange_txn_pct),
        ("SEBI turnover %", costs.sebi_turnover_pct),
        ("Stamp duty % (buy)", costs.stamp_duty_pct_buy),
        ("GST %", costs.gst_pct),
        ("DP charges", costs.dp_charges_flat_sell),
        ("Other charges", costs.other_charges_flat),
    ):
        if value < 0:
            report.error(label, "Charges cannot be negative.")
    if costs.brokerage_pct_sell > 5 or costs.brokerage_pct_buy > 5:
        report.warn("Brokerage %", "Brokerage above 5% of turnover is implausible.")
    if costs.gst_pct > 50:
        report.warn("GST %", "GST above 50% is implausible.")

    for label, value in (
        ("STCG rate %", taxes.stcg_rate_pct),
        ("LTCG rate %", taxes.ltcg_rate_pct),
        ("Cess & surcharge %", taxes.cess_and_surcharge_pct),
        ("LTCG exemption", taxes.ltcg_exemption_amount),
    ):
        if value < 0:
            report.error(label, "Cannot be negative.")
    if taxes.stcg_rate_pct > 100 or taxes.ltcg_rate_pct > 100:
        report.error("Tax rate", "A capital-gains rate above 100% is invalid.")
    if taxes.ltcg_threshold_days <= 0:
        report.error("LTCG threshold", "The long-term threshold must be positive.")
    if taxes.recognise_tax_shield_on_loss:
        report.warn(
            "Tax shield on losses",
            "Losses are being credited at the capital-gains rate. This assumes you "
            "have other realised gains to set them off against.",
        )


def validate_inputs(inputs: AnalysisInputs) -> ValidationReport:
    """Run every check and return a combined report.

    The report is *not* raised as an exception: the UI shows errors and warnings
    together so the user can see everything that needs fixing at once.
    """
    report = ValidationReport()
    _check_ipo(inputs, report)
    _check_accounts(inputs, report)
    _check_financing(inputs, report)
    _check_costs_and_taxes(inputs.costs, inputs.taxes, report)
    return report
