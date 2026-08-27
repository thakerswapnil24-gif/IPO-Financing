"""Core financial engine for the IPO capital allocation & financing decision tool.

This module is deliberately free of any UI dependency (no Streamlit imports) so
that it can be imported, unit-tested and reused from scripts, notebooks or other
front-ends.

Design principles
-----------------
1. **Full precision internally.** Nothing is rounded inside the engine; rounding
   is a presentation concern only.
2. **Explicit money buckets.** The engine never conflates
   *application capital* (money blocked while bidding),
   *own equity deployed* (the analyst's actual cash),
   *borrowed capital* (OD drawn) and
   *economic capital at risk* (own cash + FD collateral pledged).
3. **No silent assumptions.** Every input that materially moves the answer is a
   named, user-settable field, and :func:`assumption_ledger` reports each value
   together with its provenance (user entered / calculated / default assumption).
4. **Two-phase financing.** Capital is blocked on the *whole* application for the
   bidding window, but only the *allotted* portion is carried through the holding
   window. Modelling this as one blob materially overstates the financing cost.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "DAY_COUNT_DEFAULT",
    "IPOCategory",
    "GMPMode",
    "FundingMode",
    "Provenance",
    "IPOAssumptions",
    "ApplicationAccount",
    "FinancingAssumptions",
    "TransactionCostAssumptions",
    "TaxAssumptions",
    "AnalysisInputs",
    "TransactionCostBreakdown",
    "FundingPlan",
    "FinancingBreakdown",
    "AccountOutcome",
    "AllotmentDistribution",
    "CapitalEfficiency",
    "BreakEvenResults",
    "AnalysisResult",
    "AssumptionRecord",
    "listing_price_from_gmp",
    "gmp_from_listing_price",
    "listing_gain_pct",
    "simple_interest",
    "build_funding_plan",
    "compute_transaction_costs",
    "compute_capital_gains_tax",
    "allotment_distribution",
    "analyze",
    "expected_net_profit_for_exit_price",
    "break_even_exit_price",
    "max_sustainable_od_rate",
    "annualize",
    "assumption_ledger",
    "expected_net_profit",
    "conditional_net_profit",
    "annualize_simple",
]

DAY_COUNT_DEFAULT: int = 365
_SOLVER_TOL = 1e-9

try:  # SciPy gives a faster, more robust root find; bisection is the fallback.
    # Imported once at module load so that no user request ever pays the
    # several-hundred-millisecond cost of importing SciPy on its first solve.
    from scipy.optimize import brentq as _brentq
except ImportError:  # pragma: no cover - exercised only without SciPy installed
    _brentq = None


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------
class IPOCategory(str, Enum):
    """SEBI bidding categories relevant to an individual investor."""

    RETAIL = "Retail"
    SNII = "sNII"
    BNII = "bNII"

    @property
    def typical_min_application(self) -> str:
        return {
            "Retail": "up to Rs 2,00,000",
            "sNII": "Rs 2,00,000 to Rs 10,00,000",
            "bNII": "above Rs 10,00,000",
        }[self.value]


class GMPMode(str, Enum):
    """How the user chose to express grey-market premium."""

    ABSOLUTE = "Absolute Rs"
    PERCENT = "Percent of issue price"


class FundingMode(str, Enum):
    OWN = "Own capital only"
    OD = "Maximise OD"
    MIXED = "Mixed (explicit own capital)"


class Provenance(str, Enum):
    """Where a number came from - surfaced in the UI for data integrity."""

    USER = "User entered"
    CALCULATED = "Calculated"
    ASSUMED = "Default assumption"


# ---------------------------------------------------------------------------
# Input dataclasses
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class IPOAssumptions:
    """Everything about the issue itself and the expected price path.

    Attributes
    ----------
    issue_price:
        Price paid per share on allotment (cut-off price for retail).
    lot_size:
        Shares per lot / bid lot.
    gmp_value / gmp_mode:
        Grey market premium, either absolute rupees per share or a percentage of
        the issue price. GMP is an **unregulated, unverifiable sentiment
        indicator**, not a forecast - see :attr:`use_gmp_for_listing`.
    use_gmp_for_listing:
        If True the expected listing price is derived from the GMP. If False the
        user supplies ``expected_listing_price_override`` directly.
    expected_exit_price_override:
        Price at which the position is actually sold. ``None`` means "sell at the
        expected listing price" (i.e. flat on listing day).
    holding_period_days:
        Calendar days from allotment/credit to exit. Drives both the carry cost
        of the retained shares and the capital-gains rate.
    """

    name: str = "Sample IPO"
    issue_price: float = 100.0
    lot_size: int = 100
    gmp_value: float = 0.0
    gmp_mode: GMPMode = GMPMode.ABSOLUTE
    use_gmp_for_listing: bool = True
    expected_listing_price_override: Optional[float] = None
    expected_exit_price_override: Optional[float] = None
    holding_period_days: int = 1

    # -- derived ----------------------------------------------------------
    @property
    def gmp_absolute(self) -> float:
        """GMP expressed in rupees per share."""
        if self.gmp_mode is GMPMode.PERCENT:
            return self.issue_price * self.gmp_value / 100.0
        return self.gmp_value

    @property
    def gmp_percent(self) -> float:
        """GMP expressed as a percentage of the issue price."""
        if self.issue_price <= 0:
            return 0.0
        return self.gmp_absolute / self.issue_price * 100.0

    @property
    def expected_listing_price(self) -> float:
        """Expected price on listing day (an assumption, never a guarantee)."""
        if self.use_gmp_for_listing or self.expected_listing_price_override is None:
            return listing_price_from_gmp(self.issue_price, self.gmp_absolute)
        return float(self.expected_listing_price_override)

    @property
    def expected_exit_price(self) -> float:
        """Price actually realised on exit."""
        if self.expected_exit_price_override is None:
            return self.expected_listing_price
        return float(self.expected_exit_price_override)

    @property
    def expected_listing_gain_pct(self) -> float:
        return listing_gain_pct(self.issue_price, self.expected_listing_price)

    @property
    def expected_exit_gain_pct(self) -> float:
        return listing_gain_pct(self.issue_price, self.expected_exit_price)


@dataclass(frozen=True)
class ApplicationAccount:
    """One PAN / demat account bidding in the issue.

    ``lots_allotted_if_successful`` is the number of lots received *conditional*
    on being allotted at all. For an oversubscribed retail book this is 1.0 by
    construction (SEBI's minimum-lot lottery). For NII categories allotment is
    proportionate, so a fractional value represents an expected proportionate
    allotment and is a modelling assumption.
    """

    label: str = "Account 1"
    category: IPOCategory = IPOCategory.RETAIL
    lots_applied: int = 1
    allotment_probability: float = 0.10
    lots_allotted_if_successful: float = 1.0

    def application_amount(self, issue_price: float, lot_size: int) -> float:
        return issue_price * lot_size * self.lots_applied

    def shares_if_allotted(self, lot_size: int) -> float:
        return lot_size * self.lots_allotted_if_successful

    def investment_if_allotted(self, issue_price: float, lot_size: int) -> float:
        return issue_price * self.shares_if_allotted(lot_size)


@dataclass(frozen=True)
class FinancingAssumptions:
    """FD-backed overdraft and own-capital assumptions."""

    funding_mode: FundingMode = FundingMode.MIXED
    own_capital_available: float = 100_000.0
    own_capital_deployed: float = 100_000.0
    fd_amount: float = 0.0
    fd_rate_pct: float = 7.0
    od_ltv_pct: float = 90.0
    od_rate_pct: float = 10.5
    processing_fee: float = 0.0
    other_financing_charges: float = 0.0
    days_blocked: int = 7
    opportunity_cost_rate_pct: float = 7.0
    include_opportunity_cost: bool = True
    count_fd_interest_as_income: bool = False
    finance_holding_period: bool = True
    day_count_basis: int = DAY_COUNT_DEFAULT

    @property
    def od_limit(self) -> float:
        """Maximum drawable overdraft = FD amount x LTV."""
        return self.fd_amount * self.od_ltv_pct / 100.0


@dataclass(frozen=True)
class TransactionCostAssumptions:
    """Brokerage, statutory and exchange charges.

    Every rate is a *configurable assumption*. The defaults below reflect common
    Indian equity-delivery charges as of the tool's authoring date; they vary by
    broker, exchange and statute and must be verified by the user.
    """

    brokerage_pct_buy: float = 0.0
    brokerage_flat_buy: float = 0.0
    brokerage_pct_sell: float = 0.0
    brokerage_flat_sell: float = 20.0
    stt_pct_buy: float = 0.0
    stt_pct_sell: float = 0.1
    exchange_txn_pct: float = 0.00297
    sebi_turnover_pct: float = 0.0001
    stamp_duty_pct_buy: float = 0.0
    gst_pct: float = 18.0
    dp_charges_flat_sell: float = 15.93
    other_charges_flat: float = 0.0


@dataclass(frozen=True)
class TaxAssumptions:
    """Capital-gains treatment. All rates are configurable assumptions."""

    stcg_rate_pct: float = 20.0
    ltcg_rate_pct: float = 12.5
    ltcg_threshold_days: int = 365
    cess_and_surcharge_pct: float = 4.0
    apply_ltcg_exemption: bool = False
    ltcg_exemption_amount: float = 125_000.0
    deduct_transaction_costs_from_gain: bool = True
    stt_deductible: bool = False
    recognise_tax_shield_on_loss: bool = False


@dataclass(frozen=True)
class AnalysisInputs:
    """Complete, self-contained description of one IPO financing decision."""

    ipo: IPOAssumptions = field(default_factory=IPOAssumptions)
    accounts: Tuple[ApplicationAccount, ...] = field(
        default_factory=lambda: (ApplicationAccount(),)
    )
    financing: FinancingAssumptions = field(default_factory=FinancingAssumptions)
    costs: TransactionCostAssumptions = field(default_factory=TransactionCostAssumptions)
    taxes: TaxAssumptions = field(default_factory=TaxAssumptions)
    assume_independent_allotments: bool = True

    def with_accounts(self, accounts: Sequence[ApplicationAccount]) -> "AnalysisInputs":
        return replace(self, accounts=tuple(accounts))

    @property
    def total_application_amount(self) -> float:
        return sum(
            a.application_amount(self.ipo.issue_price, self.ipo.lot_size)
            for a in self.accounts
        )


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TransactionCostBreakdown:
    brokerage: float
    stt: float
    exchange_txn_charges: float
    sebi_turnover_fees: float
    stamp_duty: float
    gst: float
    dp_charges: float
    other: float

    @property
    def total(self) -> float:
        return (
            self.brokerage
            + self.stt
            + self.exchange_txn_charges
            + self.sebi_turnover_fees
            + self.stamp_duty
            + self.gst
            + self.dp_charges
            + self.other
        )

    @property
    def deductible_from_gain(self) -> float:
        """Transfer expenses allowable against capital gains (STT excluded by law)."""
        return self.total - self.stt

    def as_dict(self) -> Dict[str, float]:
        return {
            "Brokerage": self.brokerage,
            "STT": self.stt,
            "Exchange transaction charges": self.exchange_txn_charges,
            "SEBI turnover fees": self.sebi_turnover_fees,
            "Stamp duty": self.stamp_duty,
            "GST": self.gst,
            "DP charges": self.dp_charges,
            "Other charges": self.other,
            "Total": self.total,
        }


@dataclass(frozen=True)
class FundingPlan:
    """How the application money is split between own cash and borrowed money."""

    application_amount: float
    own_capital_deployed: float
    od_drawn: float
    od_limit: float
    fd_collateral_locked: float
    shortfall: float

    @property
    def od_share(self) -> float:
        """Fraction of the application funded by borrowed money."""
        return _safe_div(self.od_drawn, self.application_amount, 0.0)

    @property
    def own_share(self) -> float:
        return 1.0 - self.od_share

    @property
    def od_utilisation_pct(self) -> float:
        return _safe_div(self.od_drawn, self.od_limit, 0.0) * 100.0


@dataclass(frozen=True)
class FinancingBreakdown:
    """Cost of money, split into an unconditional and an allotment-contingent leg."""

    od_cost_bidding_window: float
    opportunity_cost_bidding_window: float
    expected_od_cost_holding_window: float
    expected_opportunity_cost_holding_window: float
    processing_fee: float
    other_charges: float
    fd_interest_earned: float
    fd_interest_counted: bool

    @property
    def fixed_fees(self) -> float:
        return self.processing_fee + self.other_charges

    @property
    def expected_borrowing_cost(self) -> float:
        """Cash interest on borrowed money plus one-time fees."""
        return (
            self.od_cost_bidding_window
            + self.expected_od_cost_holding_window
            + self.fixed_fees
        )

    @property
    def expected_opportunity_cost(self) -> float:
        return (
            self.opportunity_cost_bidding_window
            + self.expected_opportunity_cost_holding_window
        )

    @property
    def expected_total_cost_of_capital(self) -> float:
        return self.expected_borrowing_cost + self.expected_opportunity_cost

    @property
    def fd_interest_credit(self) -> float:
        return self.fd_interest_earned if self.fd_interest_counted else 0.0


@dataclass(frozen=True)
class AccountOutcome:
    """Per-account economics, both conditional on allotment and in expectation."""

    label: str
    category: IPOCategory
    lots_applied: int
    allotment_probability: float
    application_amount: float
    shares_if_allotted: float
    investment_if_allotted: float
    exit_value_if_allotted: float
    gross_profit_if_allotted: float
    transaction_costs_if_allotted: TransactionCostBreakdown
    tax_if_allotted: float
    carry_cost_if_allotted: float
    net_profit_if_allotted: float
    expected_investment: float
    expected_gross_profit: float
    expected_transaction_costs: float
    expected_tax: float
    expected_carry_cost: float
    expected_net_profit_contribution: float


@dataclass(frozen=True)
class AllotmentDistribution:
    """Exact distribution of the number of allotted accounts (Poisson-binomial)."""

    probabilities: Tuple[float, ...]
    expected_allotments: float
    p_zero: float
    p_at_least_one: float
    variance: float
    independence_assumed: bool

    @property
    def std_dev(self) -> float:
        return math.sqrt(max(self.variance, 0.0))


@dataclass(frozen=True)
class CapitalEfficiency:
    total_application_amount: float
    own_capital_deployed: float
    borrowed_capital: float
    economic_capital_at_risk: float
    expected_net_profit: float
    financing_cost: float
    expected_gross_profit: float
    capital_weighted_days: float
    cycle_days: int

    @property
    def return_on_application_capital(self) -> Optional[float]:
        return _safe_div_opt(self.expected_net_profit, self.total_application_amount)

    @property
    def return_on_own_capital(self) -> Optional[float]:
        return _safe_div_opt(self.expected_net_profit, self.own_capital_deployed)

    @property
    def return_on_economic_capital(self) -> Optional[float]:
        return _safe_div_opt(self.expected_net_profit, self.economic_capital_at_risk)

    @property
    def financing_cost_to_gross_profit(self) -> Optional[float]:
        return _safe_div_opt(self.financing_cost, self.expected_gross_profit)

    @property
    def profit_to_financing_cost(self) -> Optional[float]:
        return _safe_div_opt(self.expected_net_profit, self.financing_cost)

    @property
    def annualized_return_on_economic_capital(self) -> Optional[float]:
        roi = self.return_on_economic_capital
        if roi is None:
            return None
        return annualize(roi, self.capital_weighted_days)

    @property
    def annualized_return_on_application_capital(self) -> Optional[float]:
        roi = self.return_on_application_capital
        if roi is None:
            return None
        return annualize(roi, self.capital_weighted_days)


@dataclass(frozen=True)
class BreakEvenResults:
    exit_price_if_allotted: Optional[float]
    gmp_if_allotted: Optional[float]
    listing_gain_pct_if_allotted: Optional[float]
    exit_price_expected_value: Optional[float]
    gmp_expected_value: Optional[float]
    listing_gain_pct_expected_value: Optional[float]
    min_allotment_probability: Optional[float]
    max_od_rate_pct: Optional[float]


@dataclass(frozen=True)
class AnalysisResult:
    """Everything the UI, the risk module and the exporter need."""

    inputs: AnalysisInputs
    funding: FundingPlan
    financing: FinancingBreakdown
    accounts: Tuple[AccountOutcome, ...]
    allotment: AllotmentDistribution
    expected_gross_profit: float
    expected_transaction_costs: float
    expected_taxes: float
    expected_financing_cost: float
    expected_opportunity_cost: float
    expected_net_profit_cash: float
    expected_net_profit_economic: float
    net_profit_if_all_allotted: float
    net_profit_if_no_allotment: float
    capital: CapitalEfficiency
    break_even: BreakEvenResults

    # convenience -------------------------------------------------------
    @property
    def expected_net_profit(self) -> float:
        """Headline expected profit (economic if opportunity cost is enabled)."""
        if self.inputs.financing.include_opportunity_cost:
            return self.expected_net_profit_economic
        return self.expected_net_profit_cash

    @property
    def expected_allotments(self) -> float:
        return self.allotment.expected_allotments

    def summary_dict(self) -> Dict[str, Any]:
        cap = self.capital
        return {
            "IPO": self.inputs.ipo.name,
            "Issue price": self.inputs.ipo.issue_price,
            "Expected listing price": self.inputs.ipo.expected_listing_price,
            "Expected exit price": self.inputs.ipo.expected_exit_price,
            "GMP (Rs)": self.inputs.ipo.gmp_absolute,
            "GMP (%)": self.inputs.ipo.gmp_percent,
            "Total application amount": cap.total_application_amount,
            "Own capital deployed": cap.own_capital_deployed,
            "Borrowed capital (OD)": cap.borrowed_capital,
            "Economic capital at risk": cap.economic_capital_at_risk,
            "Expected allotments": self.allotment.expected_allotments,
            "P(at least one allotment)": self.allotment.p_at_least_one,
            "P(zero allotment)": self.allotment.p_zero,
            "Expected gross profit": self.expected_gross_profit,
            "Expected transaction costs": self.expected_transaction_costs,
            "Expected taxes": self.expected_taxes,
            "Expected financing cost": self.expected_financing_cost,
            "Expected opportunity cost": self.expected_opportunity_cost,
            "Expected net profit (cash)": self.expected_net_profit_cash,
            "Expected net profit (economic)": self.expected_net_profit_economic,
            "Return on application capital": cap.return_on_application_capital,
            "Return on own capital": cap.return_on_own_capital,
            "Return on economic capital": cap.return_on_economic_capital,
            "Annualized return on economic capital": cap.annualized_return_on_economic_capital,
            "Break-even exit price (if allotted)": self.break_even.exit_price_if_allotted,
            "Break-even GMP (if allotted)": self.break_even.gmp_if_allotted,
            "Break-even exit price (expected value)": self.break_even.exit_price_expected_value,
            "Break-even GMP (expected value)": self.break_even.gmp_expected_value,
            "Minimum allotment probability": self.break_even.min_allotment_probability,
            "Max sustainable OD rate (%)": self.break_even.max_od_rate_pct,
            "Financing cost / expected gross profit": cap.financing_cost_to_gross_profit,
            "Expected profit / financing cost": cap.profit_to_financing_cost,
        }


@dataclass(frozen=True)
class AssumptionRecord:
    section: str
    name: str
    value: Any
    provenance: Provenance
    note: str = ""


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------
def _safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide, returning ``default`` when the denominator is (near) zero."""
    if denominator is None or abs(denominator) < 1e-12:
        return default
    return numerator / denominator


def _safe_div_opt(numerator: float, denominator: float) -> Optional[float]:
    """Divide, returning ``None`` when the denominator is (near) zero."""
    if denominator is None or abs(denominator) < 1e-12:
        return None
    return numerator / denominator


def listing_price_from_gmp(issue_price: float, gmp_absolute: float) -> float:
    """Expected listing price = issue price + GMP.

    GMP is an unregulated grey-market quote. It is a *sentiment proxy*, not a
    forecast, and carries no settlement guarantee.
    """
    return issue_price + gmp_absolute


def gmp_from_listing_price(issue_price: float, listing_price: float) -> float:
    """Inverse of :func:`listing_price_from_gmp`."""
    return listing_price - issue_price


def listing_gain_pct(issue_price: float, listing_price: float) -> float:
    """Listing gain in percent: ``(listing / issue - 1) * 100``."""
    if issue_price <= 0:
        return 0.0
    return (listing_price / issue_price - 1.0) * 100.0


def simple_interest(
    principal: float,
    annual_rate_pct: float,
    days: float,
    day_count_basis: int = DAY_COUNT_DEFAULT,
) -> float:
    """Simple interest: ``principal * rate * days / basis``.

    Overdraft interest in India is charged on the daily outstanding balance and
    debited monthly; for holding periods of a few days to a few months simple
    interest on a 365-day basis is the correct convention.
    """
    if principal <= 0 or days <= 0 or annual_rate_pct == 0:
        return 0.0
    basis = day_count_basis or DAY_COUNT_DEFAULT
    return principal * (annual_rate_pct / 100.0) * (days / basis)


def annualize(
    period_return: float, days: float, day_count_basis: int = DAY_COUNT_DEFAULT
) -> Optional[float]:
    """Compound a holding-period return to an annual equivalent.

    Returns ``None`` when the period is not positive. For a total loss
    (``period_return <= -1``) compounding is undefined, so the simple
    (non-compounded) annualisation is returned instead.
    """
    if days is None or days <= 0:
        return None
    basis = day_count_basis or DAY_COUNT_DEFAULT
    periods_per_year = basis / days
    if period_return <= -1.0:
        return period_return * periods_per_year
    exponent = periods_per_year * math.log1p(period_return)
    if exponent > 700:  # guard against float overflow on very short cycles
        return math.inf
    return math.expm1(exponent)


def annualize_simple(
    period_return: float, days: float, day_count_basis: int = DAY_COUNT_DEFAULT
) -> Optional[float]:
    """Non-compounded annualisation: ``r * basis / days``."""
    if days is None or days <= 0:
        return None
    basis = day_count_basis or DAY_COUNT_DEFAULT
    return period_return * basis / days


def _solve_monotonic(
    func: Callable[[float], float],
    lo: float,
    hi: float,
    tol: float = 1e-7,
    max_iter: int = 200,
) -> Optional[float]:
    """Find a root of a continuous monotonic function on ``[lo, hi]``.

    Uses SciPy's Brent solver when available and falls back to bisection so the
    engine keeps working in a minimal environment.
    """
    f_lo, f_hi = func(lo), func(hi)
    if math.isnan(f_lo) or math.isnan(f_hi):
        return None
    if f_lo == 0.0:
        return lo
    if f_hi == 0.0:
        return hi
    if f_lo * f_hi > 0:
        return None  # no sign change: root not bracketed
    if _brentq is not None:
        try:
            return float(_brentq(func, lo, hi, xtol=tol, maxiter=max_iter))
        except Exception:  # pragma: no cover - fall through to bisection
            pass
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = func(mid)
        if f_mid == 0.0 or (hi - lo) < tol:
            return mid
        if f_lo * f_mid < 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# Funding, costs and taxes
# ---------------------------------------------------------------------------
def build_funding_plan(
    application_amount: float, financing: FinancingAssumptions
) -> FundingPlan:
    """Split the application amount into own cash and drawn overdraft.

    The split never exceeds the OD limit (FD x LTV) or the user's available
    cash; anything that cannot be funded is reported as ``shortfall`` rather
    than being silently borrowed.
    """
    application_amount = max(application_amount, 0.0)
    od_limit = financing.od_limit
    available = max(financing.own_capital_available, 0.0)

    if financing.funding_mode is FundingMode.OWN:
        own = min(available, application_amount)
        od = 0.0
    elif financing.funding_mode is FundingMode.OD:
        od = min(application_amount, od_limit)
        own = min(available, application_amount - od)
    else:  # MIXED - the user states how much of their own cash to commit
        own = min(max(financing.own_capital_deployed, 0.0), available, application_amount)
        od = min(max(application_amount - own, 0.0), od_limit)

    shortfall = max(application_amount - own - od, 0.0)
    collateral = 0.0
    if financing.od_ltv_pct > 0:
        collateral = min(od / (financing.od_ltv_pct / 100.0), financing.fd_amount)
    return FundingPlan(
        application_amount=application_amount,
        own_capital_deployed=own,
        od_drawn=od,
        od_limit=od_limit,
        fd_collateral_locked=collateral,
        shortfall=shortfall,
    )


def compute_transaction_costs(
    buy_value: float, sell_value: float, costs: TransactionCostAssumptions
) -> TransactionCostBreakdown:
    """Statutory + broker charges on the buy (allotment) and sell (exit) legs.

    All percentages are configurable assumptions supplied by the caller; none of
    them are hard-coded inside this function.
    """
    buy_value = max(buy_value, 0.0)
    sell_value = max(sell_value, 0.0)
    traded = buy_value + sell_value
    has_trade = traded > 0

    brokerage = (
        buy_value * costs.brokerage_pct_buy / 100.0
        + sell_value * costs.brokerage_pct_sell / 100.0
    )
    if buy_value > 0:
        brokerage += costs.brokerage_flat_buy
    if sell_value > 0:
        brokerage += costs.brokerage_flat_sell

    stt = (
        buy_value * costs.stt_pct_buy / 100.0
        + sell_value * costs.stt_pct_sell / 100.0
    )
    exchange = traded * costs.exchange_txn_pct / 100.0
    sebi = traded * costs.sebi_turnover_pct / 100.0
    stamp = buy_value * costs.stamp_duty_pct_buy / 100.0
    gst = (brokerage + exchange + sebi) * costs.gst_pct / 100.0
    dp = costs.dp_charges_flat_sell if sell_value > 0 else 0.0
    other = costs.other_charges_flat if has_trade else 0.0

    return TransactionCostBreakdown(
        brokerage=brokerage,
        stt=stt,
        exchange_txn_charges=exchange,
        sebi_turnover_fees=sebi,
        stamp_duty=stamp,
        gst=gst,
        dp_charges=dp,
        other=other,
    )


def compute_capital_gains_tax(
    gross_gain: float,
    holding_days: float,
    taxes: TaxAssumptions,
    deductible_costs: float = 0.0,
) -> float:
    """Capital-gains tax on an equity exit.

    ``holding_days`` is measured from allotment. Above
    ``taxes.ltcg_threshold_days`` the long-term rate applies, otherwise the
    short-term rate. Losses produce zero tax unless the user explicitly asks the
    model to recognise a tax shield (which assumes other gains exist to set the
    loss off against).
    """
    deduction = deductible_costs if taxes.deduct_transaction_costs_from_gain else 0.0
    taxable = gross_gain - deduction
    is_long_term = holding_days > taxes.ltcg_threshold_days
    rate = taxes.ltcg_rate_pct if is_long_term else taxes.stcg_rate_pct
    multiplier = (rate / 100.0) * (1.0 + taxes.cess_and_surcharge_pct / 100.0)

    if taxable <= 0:
        return taxable * multiplier if taxes.recognise_tax_shield_on_loss else 0.0
    if is_long_term and taxes.apply_ltcg_exemption:
        taxable = max(taxable - taxes.ltcg_exemption_amount, 0.0)
    return taxable * multiplier


def allotment_distribution(
    probabilities: Sequence[float], independent: bool = True
) -> AllotmentDistribution:
    """Exact Poisson-binomial distribution of the number of allotted accounts.

    ``P(no allotment) = prod(1 - p_i)`` and ``P(>=1) = 1 - P(none)`` hold only if
    the per-account draws are independent. In a heavily oversubscribed retail
    book the registrar's lottery is run per application, which is close to
    independent; that is nonetheless a **modelling assumption** and is flagged on
    the returned object.
    """
    probs = [min(max(float(p), 0.0), 1.0) for p in probabilities]
    dist: List[float] = [1.0]
    for p in probs:
        nxt = [0.0] * (len(dist) + 1)
        for k, prob in enumerate(dist):
            nxt[k] += prob * (1.0 - p)
            nxt[k + 1] += prob * p
        dist = nxt
    expected = sum(probs)
    variance = sum(p * (1.0 - p) for p in probs)
    p_zero = dist[0] if dist else 1.0
    return AllotmentDistribution(
        probabilities=tuple(dist),
        expected_allotments=expected,
        p_zero=p_zero,
        p_at_least_one=1.0 - p_zero,
        variance=variance,
        independence_assumed=independent,
    )


# ---------------------------------------------------------------------------
# The main engine
# ---------------------------------------------------------------------------
def analyze(inputs: AnalysisInputs, exit_price: Optional[float] = None) -> AnalysisResult:
    """Run the full expected-value analysis for one IPO financing decision.

    Parameters
    ----------
    inputs:
        Complete set of assumptions.
    exit_price:
        Optional override for the realised exit price. Used by the break-even
        and sensitivity routines so that no calculation is duplicated.

    Notes
    -----
    Financing is modelled in two phases:

    * **Bidding window** - ``days_blocked`` days during which the *entire*
      application amount is blocked/drawn. This cost is incurred whether or not
      shares are allotted.
    * **Holding window** - ``holding_period_days`` during which only the
      *allotted* investment is carried. This cost is contingent on allotment and
      is therefore probability-weighted.
    """
    ipo = inputs.ipo
    fin = inputs.financing
    basis = fin.day_count_basis or DAY_COUNT_DEFAULT
    price = ipo.expected_exit_price if exit_price is None else float(exit_price)

    total_application = inputs.total_application_amount
    plan = build_funding_plan(total_application, fin)

    # -- financing shares -------------------------------------------------
    od_share_bid = plan.od_share
    own_share_bid = plan.own_share
    if fin.finance_holding_period:
        od_share_hold, own_share_hold = od_share_bid, own_share_bid
    else:
        # Position is squared off against own funds once the refund is received.
        od_share_hold, own_share_hold = 0.0, 1.0
    hold_days = max(float(ipo.holding_period_days), 0.0)
    bid_days = max(float(fin.days_blocked), 0.0)
    opp_rate = fin.opportunity_cost_rate_pct if fin.include_opportunity_cost else 0.0

    od_cost_bid = simple_interest(plan.od_drawn, fin.od_rate_pct, bid_days, basis)
    opp_cost_bid = simple_interest(
        plan.own_capital_deployed, opp_rate, bid_days, basis
    )

    # -- per account ------------------------------------------------------
    outcomes: List[AccountOutcome] = []
    exp_gross = exp_txn = exp_tax = exp_hold_od = exp_hold_opp = 0.0
    exp_investment = 0.0
    all_allotted_net = 0.0
    for acct in inputs.accounts:
        app_amount = acct.application_amount(ipo.issue_price, ipo.lot_size)
        shares = acct.shares_if_allotted(ipo.lot_size)
        investment = ipo.issue_price * shares
        exit_value = price * shares
        gross = exit_value - investment
        txn = compute_transaction_costs(investment, exit_value, inputs.costs)
        deductible = txn.total if inputs.taxes.stt_deductible else txn.deductible_from_gain
        tax = compute_capital_gains_tax(gross, hold_days, inputs.taxes, deductible)

        hold_od = simple_interest(
            investment * od_share_hold, fin.od_rate_pct, hold_days, basis
        )
        hold_opp = simple_interest(investment * own_share_hold, opp_rate, hold_days, basis)
        # Cost of carrying *these* shares through the bidding window too - used
        # for the conditional (if-allotted) view and its break-even price.
        bid_od_on_shares = simple_interest(
            investment * od_share_bid, fin.od_rate_pct, bid_days, basis
        )
        bid_opp_on_shares = simple_interest(
            investment * own_share_bid, opp_rate, bid_days, basis
        )
        carry_if_allotted = hold_od + hold_opp + bid_od_on_shares + bid_opp_on_shares
        net_if_allotted = gross - txn.total - tax - carry_if_allotted

        p = min(max(acct.allotment_probability, 0.0), 1.0)
        # Bidding-window cost attributable to this account's *whole* application
        # (unconditional - the money is blocked even when nothing is allotted).
        bid_cost_on_application = simple_interest(
            app_amount * od_share_bid, fin.od_rate_pct, bid_days, basis
        ) + simple_interest(app_amount * own_share_bid, opp_rate, bid_days, basis)
        contribution = (
            p * (gross - txn.total - tax - hold_od - hold_opp) - bid_cost_on_application
        )

        outcomes.append(
            AccountOutcome(
                label=acct.label,
                category=acct.category,
                lots_applied=acct.lots_applied,
                allotment_probability=p,
                application_amount=app_amount,
                shares_if_allotted=shares,
                investment_if_allotted=investment,
                exit_value_if_allotted=exit_value,
                gross_profit_if_allotted=gross,
                transaction_costs_if_allotted=txn,
                tax_if_allotted=tax,
                carry_cost_if_allotted=carry_if_allotted,
                net_profit_if_allotted=net_if_allotted,
                expected_investment=p * investment,
                expected_gross_profit=p * gross,
                expected_transaction_costs=p * txn.total,
                expected_tax=p * tax,
                expected_carry_cost=p * (hold_od + hold_opp),
                expected_net_profit_contribution=contribution,
            )
        )
        exp_gross += p * gross
        exp_txn += p * txn.total
        exp_tax += p * tax
        exp_hold_od += p * hold_od
        exp_hold_opp += p * hold_opp
        exp_investment += p * investment
        all_allotted_net += gross - txn.total - tax - hold_od - hold_opp

    # -- financing aggregation -------------------------------------------
    fd_interest = simple_interest(
        fin.fd_amount, fin.fd_rate_pct, bid_days + hold_days, basis
    )
    financing = FinancingBreakdown(
        od_cost_bidding_window=od_cost_bid,
        opportunity_cost_bidding_window=opp_cost_bid,
        expected_od_cost_holding_window=exp_hold_od,
        expected_opportunity_cost_holding_window=exp_hold_opp,
        processing_fee=fin.processing_fee,
        other_charges=fin.other_financing_charges,
        fd_interest_earned=fd_interest,
        fd_interest_counted=fin.count_fd_interest_as_income,
    )

    expected_financing_cost = financing.expected_borrowing_cost
    expected_opportunity_cost = financing.expected_opportunity_cost
    fd_credit = financing.fd_interest_credit

    net_cash = (
        exp_gross - exp_txn - exp_tax - expected_financing_cost + fd_credit
    )
    net_economic = net_cash - expected_opportunity_cost

    unconditional_cost = (
        od_cost_bid + opp_cost_bid + financing.fixed_fees - fd_credit
    )
    net_if_all = all_allotted_net - unconditional_cost
    net_if_none = -unconditional_cost

    # -- capital efficiency ----------------------------------------------
    capital_weighted_days = _safe_div(
        total_application * bid_days + exp_investment * hold_days,
        total_application,
        bid_days,
    )
    economic_capital = plan.own_capital_deployed + plan.fd_collateral_locked
    headline_net = net_economic if fin.include_opportunity_cost else net_cash
    capital = CapitalEfficiency(
        total_application_amount=total_application,
        own_capital_deployed=plan.own_capital_deployed,
        borrowed_capital=plan.od_drawn,
        economic_capital_at_risk=economic_capital,
        expected_net_profit=headline_net,
        financing_cost=expected_financing_cost,
        expected_gross_profit=exp_gross,
        capital_weighted_days=capital_weighted_days,
        cycle_days=int(bid_days + hold_days),
    )

    allotment = allotment_distribution(
        [a.allotment_probability for a in inputs.accounts],
        independent=inputs.assume_independent_allotments,
    )

    break_even = _break_even_bundle(inputs)

    return AnalysisResult(
        inputs=inputs,
        funding=plan,
        financing=financing,
        accounts=tuple(outcomes),
        allotment=allotment,
        expected_gross_profit=exp_gross,
        expected_transaction_costs=exp_txn,
        expected_taxes=exp_tax,
        expected_financing_cost=expected_financing_cost,
        expected_opportunity_cost=expected_opportunity_cost,
        expected_net_profit_cash=net_cash,
        expected_net_profit_economic=net_economic,
        net_profit_if_all_allotted=net_if_all,
        net_profit_if_no_allotment=net_if_none,
        capital=capital,
        break_even=break_even,
    )


def expected_net_profit_for_exit_price(
    inputs: AnalysisInputs, exit_price: float
) -> float:
    """Headline expected net profit as a function of the exit price."""
    ipo = replace(inputs.ipo, expected_exit_price_override=exit_price)
    result = _analyze_light(replace(inputs, ipo=ipo))
    return result


def _analyze_light(inputs: AnalysisInputs) -> float:
    """Expected net profit only - avoids the recursive break-even solve."""
    ipo = inputs.ipo
    fin = inputs.financing
    basis = fin.day_count_basis or DAY_COUNT_DEFAULT
    price = ipo.expected_exit_price
    total_application = inputs.total_application_amount
    plan = build_funding_plan(total_application, fin)
    od_share_bid, own_share_bid = plan.od_share, plan.own_share
    if fin.finance_holding_period:
        od_share_hold, own_share_hold = od_share_bid, own_share_bid
    else:
        od_share_hold, own_share_hold = 0.0, 1.0
    hold_days = max(float(ipo.holding_period_days), 0.0)
    bid_days = max(float(fin.days_blocked), 0.0)
    opp_rate = fin.opportunity_cost_rate_pct if fin.include_opportunity_cost else 0.0

    total = 0.0
    for acct in inputs.accounts:
        shares = acct.shares_if_allotted(ipo.lot_size)
        investment = ipo.issue_price * shares
        gross = price * shares - investment
        txn = compute_transaction_costs(investment, price * shares, inputs.costs)
        deductible = txn.total if inputs.taxes.stt_deductible else txn.deductible_from_gain
        tax = compute_capital_gains_tax(gross, hold_days, inputs.taxes, deductible)
        hold_cost = simple_interest(
            investment * od_share_hold, fin.od_rate_pct, hold_days, basis
        ) + simple_interest(investment * own_share_hold, opp_rate, hold_days, basis)
        p = min(max(acct.allotment_probability, 0.0), 1.0)
        total += p * (gross - txn.total - tax - hold_cost)

    total -= simple_interest(plan.od_drawn, fin.od_rate_pct, bid_days, basis)
    total -= simple_interest(plan.own_capital_deployed, opp_rate, bid_days, basis)
    total -= fin.processing_fee + fin.other_financing_charges
    if fin.count_fd_interest_as_income:
        total += simple_interest(
            fin.fd_amount, fin.fd_rate_pct, bid_days + hold_days, basis
        )
    return total


def _conditional_net_profit(inputs: AnalysisInputs, exit_price: float) -> float:
    """Net profit assuming every account is allotted (the if-allotted view)."""
    ipo = inputs.ipo
    fin = inputs.financing
    basis = fin.day_count_basis or DAY_COUNT_DEFAULT
    plan = build_funding_plan(inputs.total_application_amount, fin)
    od_share_bid, own_share_bid = plan.od_share, plan.own_share
    if fin.finance_holding_period:
        od_share_hold, own_share_hold = od_share_bid, own_share_bid
    else:
        od_share_hold, own_share_hold = 0.0, 1.0
    hold_days = max(float(ipo.holding_period_days), 0.0)
    bid_days = max(float(fin.days_blocked), 0.0)
    opp_rate = fin.opportunity_cost_rate_pct if fin.include_opportunity_cost else 0.0

    total = 0.0
    for acct in inputs.accounts:
        shares = acct.shares_if_allotted(ipo.lot_size)
        investment = ipo.issue_price * shares
        exit_value = exit_price * shares
        gross = exit_value - investment
        txn = compute_transaction_costs(investment, exit_value, inputs.costs)
        deductible = txn.total if inputs.taxes.stt_deductible else txn.deductible_from_gain
        tax = compute_capital_gains_tax(gross, hold_days, inputs.taxes, deductible)
        carry = (
            simple_interest(investment * od_share_hold, fin.od_rate_pct, hold_days, basis)
            + simple_interest(investment * own_share_hold, opp_rate, hold_days, basis)
            + simple_interest(investment * od_share_bid, fin.od_rate_pct, bid_days, basis)
            + simple_interest(investment * own_share_bid, opp_rate, bid_days, basis)
        )
        total += gross - txn.total - tax - carry
    return total


def break_even_exit_price(
    inputs: AnalysisInputs, mode: str = "expected_value"
) -> Optional[float]:
    """Exit price at which net profit is exactly zero.

    ``mode='if_allotted'`` prices the allotted shares alone (their transaction
    costs, taxes and full cost of carry). ``mode='expected_value'`` also has to
    recover the financing cost of the capital blocked on applications that were
    *not* allotted, so it is always the higher of the two.
    """
    issue = inputs.ipo.issue_price
    if issue <= 0:
        return None
    if mode == "if_allotted":
        func = lambda p: _conditional_net_profit(inputs, p)  # noqa: E731
    elif mode == "expected_value":
        func = lambda p: expected_net_profit_for_exit_price(inputs, p)  # noqa: E731
    else:
        raise ValueError(f"Unknown break-even mode: {mode!r}")
    lo, hi = 0.0, issue * 5.0
    for _ in range(8):  # expand the bracket if the root is far out of the money
        if func(lo) * func(hi) <= 0:
            break
        hi *= 2.0
    return _solve_monotonic(func, lo, hi)


def max_sustainable_od_rate(
    inputs: AnalysisInputs, upper_bound_pct: float = 500.0
) -> Optional[float]:
    """Highest annualised OD rate at which expected net profit stays >= 0.

    Returns ``None`` when the strategy loses money even at a zero financing rate
    (i.e. the problem is the trade, not the cost of money).
    """
    def profit_at(rate_pct: float) -> float:
        fin = replace(inputs.financing, od_rate_pct=rate_pct)
        return _analyze_light(replace(inputs, financing=fin))

    if profit_at(0.0) <= 0:
        return None
    if profit_at(upper_bound_pct) > 0:
        return upper_bound_pct
    return _solve_monotonic(profit_at, 0.0, upper_bound_pct)


def min_allotment_probability(inputs: AnalysisInputs) -> Optional[float]:
    """Uniform allotment probability at which expected net profit is zero.

    Every account's probability is replaced by the same candidate value, so the
    answer reads as "you need at least this hit-rate for the strategy to wash
    its face". Returns ``None`` if the strategy never breaks even (or is already
    profitable at a zero hit-rate, which only happens with negative fees).
    """
    def profit_at(p: float) -> float:
        accounts = tuple(replace(a, allotment_probability=p) for a in inputs.accounts)
        return _analyze_light(replace(inputs, accounts=accounts))

    return _solve_monotonic(profit_at, 0.0, 1.0)


def _break_even_bundle(inputs: AnalysisInputs) -> BreakEvenResults:
    """Assemble every break-even metric for a set of inputs."""
    issue = inputs.ipo.issue_price
    cond = break_even_exit_price(inputs, "if_allotted")
    ev = break_even_exit_price(inputs, "expected_value")
    return BreakEvenResults(
        exit_price_if_allotted=cond,
        gmp_if_allotted=None if cond is None else gmp_from_listing_price(issue, cond),
        listing_gain_pct_if_allotted=None if cond is None else listing_gain_pct(issue, cond),
        exit_price_expected_value=ev,
        gmp_expected_value=None if ev is None else gmp_from_listing_price(issue, ev),
        listing_gain_pct_expected_value=None if ev is None else listing_gain_pct(issue, ev),
        min_allotment_probability=min_allotment_probability(inputs),
        max_od_rate_pct=max_sustainable_od_rate(inputs),
    )


# ---------------------------------------------------------------------------
# Data-integrity ledger
# ---------------------------------------------------------------------------
_DEFAULT_COSTS = TransactionCostAssumptions()
_DEFAULT_TAXES = TaxAssumptions()


def _prov(value: Any, default: Any) -> Provenance:
    """A field still holding its shipped default is an *assumption*, not input."""
    return Provenance.ASSUMED if value == default else Provenance.USER


def assumption_ledger(inputs: AnalysisInputs) -> List[AssumptionRecord]:
    """Every materially relevant number with its provenance.

    Rendered in the UI so that no assumption is ever applied silently: each row
    says whether the value was typed by the user, derived by the engine, or is a
    shipped default the user has not touched.
    """
    ipo, fin, costs, taxes = inputs.ipo, inputs.financing, inputs.costs, inputs.taxes
    plan = build_funding_plan(inputs.total_application_amount, fin)
    rows: List[AssumptionRecord] = [
        AssumptionRecord("IPO", "Issue price", ipo.issue_price, Provenance.USER),
        AssumptionRecord("IPO", "Lot size", ipo.lot_size, Provenance.USER),
        AssumptionRecord(
            "IPO",
            "GMP (Rs/share)",
            ipo.gmp_absolute,
            Provenance.USER if ipo.gmp_mode is GMPMode.ABSOLUTE else Provenance.CALCULATED,
            "Grey market premium is an unregulated, unverifiable quote. It is a "
            "sentiment proxy, never a guarantee of listing performance.",
        ),
        AssumptionRecord(
            "IPO",
            "Expected listing price",
            ipo.expected_listing_price,
            Provenance.CALCULATED if ipo.use_gmp_for_listing else Provenance.USER,
            "Issue price + GMP" if ipo.use_gmp_for_listing else "Entered directly",
        ),
        AssumptionRecord(
            "IPO",
            "Expected exit price",
            ipo.expected_exit_price,
            Provenance.ASSUMED
            if ipo.expected_exit_price_override is None
            else Provenance.USER,
            "Assumed equal to the expected listing price (sell on listing day)"
            if ipo.expected_exit_price_override is None
            else "Entered directly",
        ),
        AssumptionRecord(
            "IPO",
            "Expected listing gain %",
            ipo.expected_listing_gain_pct,
            Provenance.CALCULATED,
            "(listing / issue - 1) x 100",
        ),
        AssumptionRecord(
            "IPO", "Holding period (days)", ipo.holding_period_days, Provenance.USER
        ),
        AssumptionRecord(
            "Accounts",
            "Number of accounts (PANs)",
            len(inputs.accounts),
            Provenance.USER,
        ),
        AssumptionRecord(
            "Accounts",
            "Total application amount",
            inputs.total_application_amount,
            Provenance.CALCULATED,
            "Sum of issue price x lot size x lots applied",
        ),
        AssumptionRecord(
            "Accounts",
            "Allotment probabilities",
            [a.allotment_probability for a in inputs.accounts],
            Provenance.USER,
            "Historic subscription-based odds are not a guarantee of future odds.",
        ),
        AssumptionRecord(
            "Accounts",
            "Independent allotment draws",
            inputs.assume_independent_allotments,
            Provenance.ASSUMED,
            "P(no allotment) = prod(1 - p_i) requires independence across PANs.",
        ),
        AssumptionRecord(
            "Financing", "Funding mode", fin.funding_mode.value, Provenance.USER
        ),
        AssumptionRecord(
            "Financing", "Own capital deployed", plan.own_capital_deployed, Provenance.CALCULATED
        ),
        AssumptionRecord("Financing", "OD drawn", plan.od_drawn, Provenance.CALCULATED),
        AssumptionRecord(
            "Financing", "OD limit (FD x LTV)", plan.od_limit, Provenance.CALCULATED
        ),
        AssumptionRecord("Financing", "OD rate (% p.a.)", fin.od_rate_pct, Provenance.USER),
        AssumptionRecord("Financing", "FD rate (% p.a.)", fin.fd_rate_pct, Provenance.USER),
        AssumptionRecord(
            "Financing", "Days capital blocked", fin.days_blocked, Provenance.USER
        ),
        AssumptionRecord(
            "Financing",
            "Day-count basis",
            fin.day_count_basis,
            _prov(fin.day_count_basis, DAY_COUNT_DEFAULT),
            "Interest = principal x rate x days / basis",
        ),
        AssumptionRecord(
            "Financing",
            "Opportunity cost applied",
            fin.include_opportunity_cost,
            Provenance.ASSUMED,
            f"Own capital charged at {fin.opportunity_cost_rate_pct}% p.a. when enabled.",
        ),
        AssumptionRecord(
            "Financing",
            "FD interest counted as income",
            fin.count_fd_interest_as_income,
            Provenance.ASSUMED,
            "Off by default: a pledged FD keeps earning interest whether or not "
            "you bid, so it is not a benefit of this strategy.",
        ),
        AssumptionRecord(
            "Financing",
            "OD carried through holding period",
            fin.finance_holding_period,
            Provenance.ASSUMED,
        ),
    ]

    for label, value, default in (
        ("Brokerage % (sell)", costs.brokerage_pct_sell, _DEFAULT_COSTS.brokerage_pct_sell),
        ("Brokerage flat (sell)", costs.brokerage_flat_sell, _DEFAULT_COSTS.brokerage_flat_sell),
        ("STT % (sell)", costs.stt_pct_sell, _DEFAULT_COSTS.stt_pct_sell),
        ("STT % (buy/allotment)", costs.stt_pct_buy, _DEFAULT_COSTS.stt_pct_buy),
        ("Exchange txn %", costs.exchange_txn_pct, _DEFAULT_COSTS.exchange_txn_pct),
        ("SEBI turnover %", costs.sebi_turnover_pct, _DEFAULT_COSTS.sebi_turnover_pct),
        ("Stamp duty % (buy)", costs.stamp_duty_pct_buy, _DEFAULT_COSTS.stamp_duty_pct_buy),
        ("GST %", costs.gst_pct, _DEFAULT_COSTS.gst_pct),
        ("DP charges (sell)", costs.dp_charges_flat_sell, _DEFAULT_COSTS.dp_charges_flat_sell),
    ):
        rows.append(
            AssumptionRecord(
                "Transaction costs",
                label,
                value,
                _prov(value, default),
                "Broker/statutory rates change over time - verify against your contract note.",
            )
        )

    for label, value, default in (
        ("STCG rate %", taxes.stcg_rate_pct, _DEFAULT_TAXES.stcg_rate_pct),
        ("LTCG rate %", taxes.ltcg_rate_pct, _DEFAULT_TAXES.ltcg_rate_pct),
        ("LTCG threshold (days)", taxes.ltcg_threshold_days, _DEFAULT_TAXES.ltcg_threshold_days),
        ("Cess & surcharge %", taxes.cess_and_surcharge_pct, _DEFAULT_TAXES.cess_and_surcharge_pct),
        ("LTCG exemption applied", taxes.apply_ltcg_exemption, _DEFAULT_TAXES.apply_ltcg_exemption),
        ("Tax shield on losses", taxes.recognise_tax_shield_on_loss, _DEFAULT_TAXES.recognise_tax_shield_on_loss),
    ):
        rows.append(
            AssumptionRecord(
                "Taxes",
                label,
                value,
                _prov(value, default),
                "Tax rates are configurable assumptions, not embedded law.",
            )
        )
    return rows


def expected_net_profit(inputs: AnalysisInputs) -> float:
    """Headline expected net profit for a set of inputs (fast path).

    Identical to ``analyze(inputs).expected_net_profit`` but skips the
    break-even solves, so it is safe to call inside sensitivity grids and
    optimisation loops.
    """
    return _analyze_light(inputs)


def conditional_net_profit(inputs: AnalysisInputs, exit_price: Optional[float] = None) -> float:
    """Net profit assuming every application is allotted."""
    price = inputs.ipo.expected_exit_price if exit_price is None else float(exit_price)
    return _conditional_net_profit(inputs, price)
