"""Risk metrics, the exact outcome distribution, and the GO / NO-GO framework.

Nothing in this module is investment advice. The decision engine is a
*rules-based scoring of the user's own assumptions*: change the assumptions and
the verdict changes. Its job is to be sceptical - in particular to refuse a
strategy whose positive expected value rests entirely on an optimistic grey
market premium, an optimistic hit-rate, or cheap borrowed money.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import Enum

import numpy as np
import pandas as pd

from calculations import (
    AnalysisInputs,
    AnalysisResult,
    GMPMode,
    analyze,
    expected_net_profit,
)
from scenarios import DEFAULT_BEAR, ScenarioDefinition

__all__ = [
    "OutcomeDistribution",
    "RiskMetrics",
    "Verdict",
    "DecisionThresholds",
    "DecisionCheck",
    "DecisionOutcome",
    "outcome_distribution",
    "compute_risk_metrics",
    "evaluate_decision",
    "compare_opportunities",
]

_MAX_DP_STATES = 200_000


def _fmt_money(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"Rs {value:,.0f}"


def _fmt_pct(value: float | None) -> str:
    if value is None or (isinstance(value, float) and math.isinf(value)):
        return "n/a"
    return f"{value:.1%}"


# ---------------------------------------------------------------------------
# Exact outcome distribution
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OutcomeDistribution:
    """Exact discrete distribution of net profit across allotment outcomes.

    Only the allotment draw is treated as random here (prices, rates and the
    holding period are held at their base assumptions), which is precisely the
    risk that a positive expected value hides: most of the time nothing is
    allotted and the financing cost is a pure loss.
    """

    profits: tuple[float, ...]
    probabilities: tuple[float, ...]
    exact: bool

    @property
    def expected_profit(self) -> float:
        return float(np.dot(self.profits, self.probabilities))

    @property
    def probability_of_loss(self) -> float:
        return float(
            sum(
                p
                for v, p in zip(self.profits, self.probabilities, strict=True)
                if v < 0
            )
        )

    @property
    def probability_of_profit(self) -> float:
        return float(
            sum(
                p
                for v, p in zip(self.profits, self.probabilities, strict=True)
                if v > 0
            )
        )

    @property
    def expected_loss(self) -> float:
        """Probability-weighted loss (a negative number, 0 if losses impossible)."""
        return float(
            sum(
                v * p
                for v, p in zip(self.profits, self.probabilities, strict=True)
                if v < 0
            )
        )

    @property
    def expected_gain(self) -> float:
        return float(
            sum(
                v * p
                for v, p in zip(self.profits, self.probabilities, strict=True)
                if v > 0
            )
        )

    @property
    def worst_case(self) -> float:
        return float(min(self.profits)) if self.profits else 0.0

    @property
    def best_case(self) -> float:
        return float(max(self.profits)) if self.profits else 0.0

    def to_frame(self) -> pd.DataFrame:
        frame = pd.DataFrame(
            {"Net profit": self.profits, "Probability": self.probabilities}
        )
        return frame.sort_values("Net profit").reset_index(drop=True)


def outcome_distribution(result: AnalysisResult) -> OutcomeDistribution:
    """Convolve the per-account allotment outcomes into a profit distribution.

    Identical accounts collapse onto the same profit value, so the state space
    stays small in practice. If it does not, the calculation is abandoned and
    ``exact=False`` is returned with a two-point approximation.
    """
    fixed_cost = -result.net_profit_if_no_allotment  # positive: cost incurred anyway
    states: dict[int, float] = {0: 1.0}
    scale = 1_000_000  # merge values that agree to within 1e-6 rupees

    for account in result.accounts:
        # net gain contributed by this account if (and only if) it is allotted
        unit = (
            account.gross_profit_if_allotted
            - account.transaction_costs_if_allotted.total
            - account.tax_if_allotted
            - (
                account.expected_carry_cost / account.allotment_probability
                if account.allotment_probability > 0
                else 0.0
            )
        )
        p = account.allotment_probability
        nxt: dict[int, float] = {}
        for key, prob in states.items():
            value = key / scale
            nxt[key] = nxt.get(key, 0.0) + prob * (1.0 - p)
            hit = round((value + unit) * scale)
            nxt[hit] = nxt.get(hit, 0.0) + prob * p
        states = nxt
        if len(states) > _MAX_DP_STATES:
            return OutcomeDistribution(
                profits=(
                    result.net_profit_if_no_allotment,
                    result.net_profit_if_all_allotted,
                ),
                probabilities=(result.allotment.p_zero, 1.0 - result.allotment.p_zero),
                exact=False,
            )

    profits = tuple(key / scale - fixed_cost for key in states)
    probabilities = tuple(states.values())
    return OutcomeDistribution(profits=profits, probabilities=probabilities, exact=True)


# ---------------------------------------------------------------------------
# Risk metrics
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RiskMetrics:
    """Everything needed to judge whether a positive expected value is real."""

    probability_of_loss: float
    probability_of_profit: float
    expected_loss: float
    expected_gain: float
    maximum_loss: float
    maximum_gain: float
    profit_to_loss_ratio: float | None
    financing_cost_share_of_gross_profit: float | None
    financing_cost_share_of_net_profit: float | None
    gmp_margin_of_safety: float | None
    probability_margin_of_safety: float | None
    od_rate_headroom_pct: float | None
    gmp_elasticity: float | None
    probability_elasticity: float | None
    profit_if_lists_flat: float
    profit_if_lists_10pct_below: float
    bear_case_profit: float | None
    bear_loss_share_of_equity: float | None
    distribution: OutcomeDistribution
    flags: tuple[str, ...]

    def to_frame(self) -> pd.DataFrame:
        rows = [
            ("Probability of losing money", self.probability_of_loss),
            ("Probability of making money", self.probability_of_profit),
            ("Expected loss (probability weighted)", self.expected_loss),
            ("Expected gain (probability weighted)", self.expected_gain),
            ("Maximum modelled loss", self.maximum_loss),
            ("Maximum modelled gain", self.maximum_gain),
            ("Profit / loss ratio", self.profit_to_loss_ratio),
            (
                "Financing cost / expected gross profit",
                self.financing_cost_share_of_gross_profit,
            ),
            (
                "Financing cost / expected net profit",
                self.financing_cost_share_of_net_profit,
            ),
            ("GMP margin of safety", self.gmp_margin_of_safety),
            (
                "Allotment probability margin of safety",
                self.probability_margin_of_safety,
            ),
            ("OD rate headroom (pp)", self.od_rate_headroom_pct),
            ("Elasticity of profit to GMP", self.gmp_elasticity),
            (
                "Elasticity of profit to allotment probability",
                self.probability_elasticity,
            ),
            ("Profit if the stock lists flat", self.profit_if_lists_flat),
            (
                "Profit if the stock lists 10% below issue",
                self.profit_if_lists_10pct_below,
            ),
            ("Bear-case profit", self.bear_case_profit),
            ("Bear-case loss / own equity", self.bear_loss_share_of_equity),
        ]
        return pd.DataFrame(rows, columns=["Metric", "Value"])


def _elasticity(
    base_inputs: AnalysisInputs,
    bumped_inputs: AnalysisInputs,
    driver_base: float,
    bump: float,
) -> float | None:
    """Percentage change in expected profit per percentage change in a driver."""
    base_profit = expected_net_profit(base_inputs)
    if abs(base_profit) < 1e-9 or abs(driver_base) < 1e-12 or abs(bump) < 1e-12:
        return None
    bumped_profit = expected_net_profit(bumped_inputs)
    d_profit = (bumped_profit - base_profit) / abs(base_profit)
    d_driver = bump / driver_base
    if abs(d_driver) < 1e-12:
        return None
    return d_profit / d_driver


def compute_risk_metrics(
    result: AnalysisResult, bear: ScenarioDefinition | None = None
) -> RiskMetrics:
    """Derive the full risk picture from a completed analysis."""
    inputs = result.inputs
    ipo = inputs.ipo
    dist = outcome_distribution(result)

    expected_loss = dist.expected_loss
    expected_gain = dist.expected_gain
    pl_ratio = (
        None if abs(expected_loss) < 1e-12 else expected_gain / abs(expected_loss)
    )

    gross = result.expected_gross_profit
    financing = result.expected_financing_cost + result.expected_opportunity_cost
    fin_share_gross = None if abs(gross) < 1e-12 else financing / gross
    fin_share_net = (
        None
        if abs(result.expected_net_profit) < 1e-12
        else financing / result.expected_net_profit
    )

    # Margins of safety: how far a driver can fall before profit disappears.
    # Expressed as a fraction of the assumed premium / hit-rate, so a value of
    # 0.25 means "a quarter of the assumption can evaporate before break-even"
    # and a negative value means the assumption is already insufficient.
    net = result.expected_net_profit
    premium = ipo.expected_exit_price - ipo.issue_price
    be_price = result.break_even.exit_price_expected_value
    gmp_mos: float | None = None
    if premium > 1e-9 and be_price is not None:
        gmp_mos = (ipo.expected_exit_price - be_price) / premium
    elif premium <= 1e-9:
        # No assumed premium at all: there is nothing to give back.
        gmp_mos = None if net > 0 else -1.0

    min_p = result.break_even.min_allotment_probability
    base_p = (
        float(np.mean([a.allotment_probability for a in inputs.accounts]))
        if inputs.accounts
        else 0.0
    )
    prob_mos: float | None = None
    if min_p is not None and base_p > 1e-12:
        prob_mos = (base_p - min_p) / base_p
    elif base_p > 1e-12:
        # Unsolvable: either profitable even at a zero hit-rate (impossible once
        # any cost is charged) or never profitable at any hit-rate.
        prob_mos = 1.0 if _profit_at_probability(inputs, 0.0) > 0 else -1.0

    od_headroom = None
    if result.break_even.max_od_rate_pct is not None:
        od_headroom = result.break_even.max_od_rate_pct - inputs.financing.od_rate_pct

    # Local elasticities (1% relative bumps).
    gmp_bump = 0.01 * ipo.gmp_absolute if abs(ipo.gmp_absolute) > 1e-9 else 0.0
    gmp_elasticity = None
    if gmp_bump:
        bumped_ipo = replace(
            ipo,
            gmp_value=ipo.gmp_absolute + gmp_bump,
            gmp_mode=GMPMode.ABSOLUTE,
            use_gmp_for_listing=True,
            expected_listing_price_override=None,
            expected_exit_price_override=None,
        )
        gmp_elasticity = _elasticity(
            replace(inputs, ipo=replace(ipo, expected_exit_price_override=None)),
            replace(inputs, ipo=bumped_ipo),
            ipo.gmp_absolute,
            gmp_bump,
        )

    prob_elasticity = None
    if base_p > 1e-9:
        bump = 0.01 * base_p
        bumped_accounts = tuple(
            replace(a, allotment_probability=min(a.allotment_probability * 1.01, 1.0))
            for a in inputs.accounts
        )
        prob_elasticity = _elasticity(
            inputs, replace(inputs, accounts=bumped_accounts), base_p, bump
        )

    flat = _profit_at_exit_price(inputs, ipo.issue_price)
    down10 = _profit_at_exit_price(inputs, ipo.issue_price * 0.9)

    bear_definition = bear or DEFAULT_BEAR
    bear_result = analyze(bear_definition.apply(inputs))
    bear_profit = bear_result.expected_net_profit
    equity = result.capital.economic_capital_at_risk
    bear_share = None if equity <= 1e-12 else -min(bear_profit, 0.0) / equity

    flags: list[str] = []
    if abs(ipo.gmp_absolute) > 1e-9 and ipo.use_gmp_for_listing:
        flags.append(
            "The entire expected gain is derived from GMP, an unregulated grey "
            "market quote with no settlement guarantee."
        )
    if gmp_mos is not None and 0 <= gmp_mos < 0.30:
        flags.append(
            f"Thin GMP margin of safety: only {gmp_mos:.0%} of the assumed premium "
            "can evaporate before the expected profit is gone."
        )
    elif gmp_mos is not None and gmp_mos < 0:
        needed = result.break_even.gmp_expected_value
        flags.append(
            "The assumed premium is already too small: breaking even needs a GMP of "
            f"{_fmt_money(needed)} against the {_fmt_money(ipo.gmp_absolute)} you "
            f"assumed."
        )
    if prob_mos is not None and 0 <= prob_mos < 0.30:
        flags.append(
            f"Highly dependent on the allotment hit-rate: only {prob_mos:.0%} of the "
            "assumed probability can be lost before breaking even."
        )
    elif prob_mos is not None and prob_mos < 0:
        needed = result.break_even.min_allotment_probability
        flags.append(
            "The assumed hit-rate is already too low: breaking even needs "
            + (
                f"{needed:.1%} against the {base_p:.1%} you assumed."
                if needed is not None
                else "a hit-rate no allotment probability can reach."
            )
        )
    if fin_share_gross is not None and fin_share_gross > 0.5:
        flags.append(
            f"Financing and opportunity cost eats {fin_share_gross:.0%} of the "
            "expected gross profit."
        )
    if od_headroom is not None and od_headroom < 2.0:
        flags.append(
            "Less than 2 percentage points of headroom on the OD rate - a repricing "
            "of the overdraft would make this loss-making."
        )
    if od_headroom is None:
        flags.append(
            "The strategy loses money even at a zero financing rate: the problem is "
            "the trade, not the cost of money."
        )
    if dist.probability_of_loss > 0.5:
        flags.append(
            f"A loss is the most likely outcome ({dist.probability_of_loss:.0%} of the "
            "time)"
            + (
                ", even though the average outcome is positive: the mean is carried by "
                "rare allotments, so expect a long run of small losses between them."
                if net > 0
                else " and the average outcome is a loss too."
            )
        )
    if ipo.holding_period_days <= 1 and inputs.financing.days_blocked <= 3:
        flags.append(
            "Returns rely on a very short capital cycle; any delay in listing or "
            "refund materially reduces the annualised return."
        )
    if len(inputs.accounts) > 1:
        flags.append(
            "Multiple-PAN modelling assumes independent allotment draws and that "
            "every account is separately funded - the financing cost scales with "
            "the number of applications, the expected profit only with the hit-rate."
        )
    return RiskMetrics(
        probability_of_loss=dist.probability_of_loss,
        probability_of_profit=dist.probability_of_profit,
        expected_loss=expected_loss,
        expected_gain=expected_gain,
        maximum_loss=dist.worst_case,
        maximum_gain=dist.best_case,
        profit_to_loss_ratio=pl_ratio,
        financing_cost_share_of_gross_profit=fin_share_gross,
        financing_cost_share_of_net_profit=fin_share_net,
        gmp_margin_of_safety=gmp_mos,
        probability_margin_of_safety=prob_mos,
        od_rate_headroom_pct=od_headroom,
        gmp_elasticity=gmp_elasticity,
        probability_elasticity=prob_elasticity,
        profit_if_lists_flat=flat,
        profit_if_lists_10pct_below=down10,
        bear_case_profit=bear_profit,
        bear_loss_share_of_equity=bear_share,
        distribution=dist,
        flags=tuple(flags),
    )


def _profit_at_exit_price(inputs: AnalysisInputs, exit_price: float) -> float:
    """Expected net profit if the shares are sold at ``exit_price``."""
    ipo = replace(inputs.ipo, expected_exit_price_override=exit_price)
    return expected_net_profit(replace(inputs, ipo=ipo))


def _profit_at_probability(inputs: AnalysisInputs, probability: float) -> float:
    """Expected net profit with a uniform allotment probability."""
    accounts = tuple(
        replace(a, allotment_probability=probability) for a in inputs.accounts
    )
    return expected_net_profit(replace(inputs, accounts=accounts))


# ---------------------------------------------------------------------------
# GO / NO-GO framework
# ---------------------------------------------------------------------------
class Verdict(str, Enum):
    GO = "GO"
    BORDERLINE = "BORDERLINE"
    NO_GO = "NO-GO"


@dataclass(frozen=True)
class DecisionThresholds:
    """Tunable rule thresholds. Every one of them is a policy choice, not a law."""

    #: Minimum annualised return on own equity, expressed as a spread over the
    #: OD rate. Borrowing at 10.5% to earn 11% annualised is not worth the risk.
    min_annualised_spread_over_od_pct: float = 5.0
    #: Financing + opportunity cost may not exceed this share of gross profit.
    max_financing_share_of_gross: float = 0.60
    #: Above this probability of losing money the trade is at best borderline.
    max_probability_of_loss: float = 0.60
    #: Fraction of the assumed GMP that may evaporate before breaking even.
    min_gmp_margin_of_safety: float = 0.30
    #: Fraction of the assumed hit-rate that may be lost before breaking even.
    min_probability_margin_of_safety: float = 0.30
    #: Percentage points of OD-rate headroom required.
    min_od_rate_headroom_pct: float = 2.0
    #: Bear-case loss tolerated, as a share of own equity at risk. A leveraged
    #: applicant repeating this bet cannot absorb a large drawdown per cycle.
    max_bear_loss_share_of_equity: float = 0.05
    #: Break-even listing gain above this is treated as an unrealistic ask.
    max_plausible_breakeven_listing_gain_pct: float = 15.0


@dataclass(frozen=True)
class DecisionCheck:
    name: str
    passed: bool
    hard: bool
    detail: str

    @property
    def status(self) -> str:
        return "PASS" if self.passed else ("FAIL" if self.hard else "WARN")


@dataclass(frozen=True)
class DecisionOutcome:
    verdict: Verdict
    checks: tuple[DecisionCheck, ...]
    headline: str
    rationale: tuple[str, ...]
    thresholds: DecisionThresholds

    @property
    def passed_count(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "Check": c.name,
                    "Result": c.status,
                    "Severity": "Hard rule" if c.hard else "Soft rule",
                    "Detail": c.detail,
                }
                for c in self.checks
            ]
        )


DISCLAIMER = (
    "This is a quantitative decision framework applied to assumptions you "
    "supplied. It is not investment advice, and it cannot validate whether your "
    "GMP, allotment-probability or listing assumptions are realistic."
)


def evaluate_decision(
    result: AnalysisResult,
    risk: RiskMetrics,
    thresholds: DecisionThresholds | None = None,
) -> DecisionOutcome:
    """Apply the rules-based GO / BORDERLINE / NO-GO framework.

    A hard-rule failure forces NO-GO. Any soft-rule failure downgrades a GO to
    BORDERLINE. A positive expected value on its own is never sufficient.
    """
    th = thresholds or DecisionThresholds()
    net = result.expected_net_profit
    od_rate = result.inputs.financing.od_rate_pct
    annualised = result.capital.annualized_return_on_economic_capital
    checks: list[DecisionCheck] = []

    checks.append(
        DecisionCheck(
            "Expected net profit is positive",
            net > 0,
            True,
            f"Expected net profit after financing, costs and taxes: {_fmt_money(net)}.",
        )
    )

    if annualised is None or math.isinf(annualised):
        spread_ok = net > 0
        spread_detail = (
            "Annualised return could not be computed (no equity at risk or zero-day "
            "cycle); judged on absolute profit instead."
        )
    else:
        spread = annualised * 100.0 - od_rate
        spread_ok = spread >= th.min_annualised_spread_over_od_pct
        spread_detail = (
            f"Annualised return on own equity {annualised:.1%} vs OD rate "
            f"{od_rate:.2f}% "
            f"-> spread of {spread:.1f} pp (required: "
            f"{th.min_annualised_spread_over_od_pct:.1f} pp)."
        )
    checks.append(
        DecisionCheck(
            "Return clears the cost of borrowed money", spread_ok, True, spread_detail
        )
    )

    share = risk.financing_cost_share_of_gross_profit
    share_ok = share is not None and 0 <= share <= th.max_financing_share_of_gross
    checks.append(
        DecisionCheck(
            "Financing cost does not consume the profit",
            bool(share_ok),
            True,
            f"Financing + opportunity cost is {_fmt_pct(share)} of expected gross "
            f"profit "
            f"(limit {th.max_financing_share_of_gross:.0%}).",
        )
    )

    be_gain = result.break_even.listing_gain_pct_expected_value
    be_ok = (
        be_gain is not None and be_gain <= th.max_plausible_breakeven_listing_gain_pct
    )
    checks.append(
        DecisionCheck(
            "Break-even listing gain is realistic",
            bool(be_ok),
            True,
            f"The issue must list {be_gain:.2f}% above the issue price just to break "
            f"even (plausibility limit "
            f"{th.max_plausible_breakeven_listing_gain_pct:.1f}%)."
            if be_gain is not None
            else "Break-even listing price could not be solved.",
        )
    )

    checks.append(
        DecisionCheck(
            "Probability of loss is tolerable",
            risk.probability_of_loss <= th.max_probability_of_loss,
            False,
            f"{risk.probability_of_loss:.0%} of modelled outcomes lose money "
            f"(limit {th.max_probability_of_loss:.0%}). With a low hit-rate this is "
            "normal for IPO applications - the average is carried by rare allotments.",
        )
    )

    checks.append(
        DecisionCheck(
            "Bear case is survivable",
            risk.bear_loss_share_of_equity is None
            or risk.bear_loss_share_of_equity <= th.max_bear_loss_share_of_equity,
            False,
            f"Bear case profit {_fmt_money(risk.bear_case_profit)}; loss is "
            f"{_fmt_pct(risk.bear_loss_share_of_equity)} of own equity at risk "
            f"(limit {th.max_bear_loss_share_of_equity:.0%}).",
        )
    )

    checks.append(
        DecisionCheck(
            "Not over-dependent on the GMP assumption",
            risk.gmp_margin_of_safety is None
            or risk.gmp_margin_of_safety >= th.min_gmp_margin_of_safety,
            False,
            f"GMP margin of safety {_fmt_pct(risk.gmp_margin_of_safety)} "
            f"(required {th.min_gmp_margin_of_safety:.0%}).",
        )
    )

    checks.append(
        DecisionCheck(
            "Not over-dependent on the allotment hit-rate",
            risk.probability_margin_of_safety is None
            or risk.probability_margin_of_safety >= th.min_probability_margin_of_safety,
            False,
            f"Allotment-probability margin of safety "
            f"{_fmt_pct(risk.probability_margin_of_safety)} "
            f"(required {th.min_probability_margin_of_safety:.0%}).",
        )
    )

    checks.append(
        DecisionCheck(
            "Financing rate has headroom",
            risk.od_rate_headroom_pct is not None
            and risk.od_rate_headroom_pct >= th.min_od_rate_headroom_pct,
            False,
            f"Maximum sustainable OD rate "
            f"{result.break_even.max_od_rate_pct:.2f}% vs actual {od_rate:.2f}%."
            if result.break_even.max_od_rate_pct is not None
            else "Loss-making even at a 0% financing rate.",
        )
    )

    hard_failures = [c for c in checks if c.hard and not c.passed]
    soft_failures = [c for c in checks if not c.hard and not c.passed]

    # The headline sits next to the verdict badge in the UI, so it must not
    # repeat the verdict word.
    if hard_failures:
        verdict = Verdict.NO_GO
        headline = f"{len(hard_failures)} hard rule(s) failed on your assumptions"
    elif soft_failures:
        verdict = Verdict.BORDERLINE
        headline = "Positive expected value, but fragile"
    else:
        verdict = Verdict.GO
        headline = "Every hard and soft rule passes on your assumptions"

    rationale: list[str] = []
    if hard_failures:
        rationale.append(
            "Hard rules failed: "
            + "; ".join(c.name.lower() for c in hard_failures)
            + "."
        )
    if soft_failures:
        rationale.append(
            "Soft rules flagged: "
            + "; ".join(c.name.lower() for c in soft_failures)
            + "."
        )
    if not hard_failures and not soft_failures:
        rationale.append(
            "Every hard and soft rule passed on the stated assumptions; the residual "
            "risk is that the assumptions themselves are wrong."
        )
    rationale.extend(risk.flags)
    rationale.append(DISCLAIMER)

    return DecisionOutcome(
        verdict=verdict,
        checks=tuple(checks),
        headline=headline,
        rationale=tuple(rationale),
        thresholds=th,
    )


# ---------------------------------------------------------------------------
# Portfolio-level comparison
# ---------------------------------------------------------------------------
def compare_opportunities(
    opportunities: Sequence[tuple[str, AnalysisInputs]],
    thresholds: DecisionThresholds | None = None,
) -> pd.DataFrame:
    """Rank several IPO opportunities side by side.

    Returns one row per opportunity with capital usage, expected outcome, risk
    and the rules-based verdict. Sort the frame on any column to rank by
    expected profit, ROI, risk or capital efficiency.
    """
    rows = []
    for label, inputs in opportunities:
        result = analyze(inputs)
        risk = compute_risk_metrics(result)
        decision = evaluate_decision(result, risk, thresholds)
        cap = result.capital
        rows.append(
            {
                "IPO": label or inputs.ipo.name,
                "Application": cap.total_application_amount,
                "Own capital": cap.own_capital_deployed,
                "OD used": cap.borrowed_capital,
                "Expected allotments": result.expected_allotments,
                "Expected gross profit": result.expected_gross_profit,
                "Financing cost": result.expected_financing_cost,
                "Expected net profit": result.expected_net_profit,
                "ROI on own equity": cap.return_on_economic_capital,
                "ROI on application": cap.return_on_application_capital,
                "Annualized ROI": cap.annualized_return_on_economic_capital,
                "Profit per rupee of financing cost": cap.profit_to_financing_cost,
                "Probability of loss": risk.probability_of_loss,
                "Max loss": risk.maximum_loss,
                "Break-even GMP": result.break_even.gmp_expected_value,
                "Max sustainable OD rate %": result.break_even.max_od_rate_pct,
                "Decision": decision.verdict.value,
            }
        )
    return pd.DataFrame(rows)
