"""Scenario analysis, sensitivity grids and Monte Carlo simulation.

Everything here builds on :mod:`calculations`; nothing imports Streamlit, so the
whole module is usable from a notebook or a batch job.

Scenario assumptions are never invented silently: each scenario carries a
human-readable ``description`` of exactly what was changed relative to the
user's base case, and every default factor is exposed as a named constant that
the caller can override.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from calculations import (
    AnalysisInputs,
    AnalysisResult,
    GMPMode,
    analyze,
    build_funding_plan,
    expected_net_profit,
)

__all__ = [
    "ScenarioDefinition",
    "ScenarioResult",
    "DEFAULT_BEAR",
    "DEFAULT_BASE",
    "DEFAULT_BULL",
    "default_scenarios",
    "run_scenarios",
    "scenarios_to_frame",
    "sensitivity_gmp_vs_probability",
    "sensitivity_od_rate_vs_listing_gain",
    "sensitivity_grid",
    "MonteCarloConfig",
    "MonteCarloResult",
    "run_monte_carlo",
]


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ScenarioDefinition:
    """A named, fully explicit deviation from the user's base assumptions.

    Any field left as ``None`` (or 1.0 for the multipliers) inherits the base
    case, so a scenario always states exactly what it changes.
    """

    name: str
    gmp_absolute: float | None = None
    gmp_multiplier: float = 1.0
    allotment_probability_multiplier: float = 1.0
    allotment_probability_override: float | None = None
    od_rate_pct_override: float | None = None
    holding_period_days_override: int | None = None
    exit_price_override: float | None = None
    description: str = ""

    def apply(self, base: AnalysisInputs) -> AnalysisInputs:
        """Return a new :class:`AnalysisInputs` with this scenario applied."""
        ipo = base.ipo
        gmp = (
            self.gmp_absolute
            if self.gmp_absolute is not None
            else ipo.gmp_absolute * self.gmp_multiplier
        )
        ipo = replace(
            ipo,
            gmp_value=gmp,
            gmp_mode=GMPMode.ABSOLUTE,
            use_gmp_for_listing=True,
            expected_listing_price_override=None,
        )
        if self.holding_period_days_override is not None:
            ipo = replace(ipo, holding_period_days=self.holding_period_days_override)
        if self.exit_price_override is not None:
            ipo = replace(ipo, expected_exit_price_override=self.exit_price_override)
        elif base.ipo.expected_exit_price_override is not None:
            # The base case sold away from the listing price; preserve that
            # spread rather than silently forcing a listing-day exit.
            spread = base.ipo.expected_exit_price - base.ipo.expected_listing_price
            ipo = replace(
                ipo, expected_exit_price_override=ipo.expected_listing_price + spread
            )

        accounts = []
        for acct in base.accounts:
            if self.allotment_probability_override is not None:
                p = self.allotment_probability_override
            else:
                p = acct.allotment_probability * self.allotment_probability_multiplier
            accounts.append(replace(acct, allotment_probability=min(max(p, 0.0), 1.0)))

        financing = base.financing
        if self.od_rate_pct_override is not None:
            financing = replace(financing, od_rate_pct=self.od_rate_pct_override)

        return replace(base, ipo=ipo, accounts=tuple(accounts), financing=financing)


#: Default scenario factors. These are *editable defaults*, not forecasts: the
#: bear case assumes the grey-market premium halves and reverses (a listing at a
#: modest discount) and that the hit-rate falls by half; the bull case assumes
#: the premium expands by 50% and the hit-rate improves by a quarter.
DEFAULT_BEAR = ScenarioDefinition(
    name="Bear",
    gmp_multiplier=-0.5,
    allotment_probability_multiplier=0.5,
    description=(
        "GMP reverses to half its size as a discount (listing below issue price) "
        "and the allotment hit-rate halves."
    ),
)
DEFAULT_BASE = ScenarioDefinition(
    name="Base", description="The user's own assumptions, unchanged."
)
DEFAULT_BULL = ScenarioDefinition(
    name="Bull",
    gmp_multiplier=1.5,
    allotment_probability_multiplier=1.25,
    description="GMP expands by 50% and the allotment hit-rate improves by 25%.",
)


@dataclass(frozen=True)
class ScenarioResult:
    definition: ScenarioDefinition
    inputs: AnalysisInputs
    result: AnalysisResult

    @property
    def name(self) -> str:
        return self.definition.name


def default_scenarios() -> list[ScenarioDefinition]:
    """The shipped bear / base / bull triple (all factors user-editable)."""
    return [DEFAULT_BEAR, DEFAULT_BASE, DEFAULT_BULL]


def run_scenarios(
    base: AnalysisInputs, definitions: Sequence[ScenarioDefinition] | None = None
) -> list[ScenarioResult]:
    """Evaluate each scenario against the base inputs."""
    definitions = list(definitions) if definitions is not None else default_scenarios()
    out: list[ScenarioResult] = []
    for definition in definitions:
        scenario_inputs = definition.apply(base)
        out.append(
            ScenarioResult(
                definition=definition,
                inputs=scenario_inputs,
                result=analyze(scenario_inputs),
            )
        )
    return out


def scenarios_to_frame(scenarios: Sequence[ScenarioResult]) -> pd.DataFrame:
    """Tabulate scenarios for display/export (full precision, no rounding)."""
    rows = []
    for sc in scenarios:
        res = sc.result
        ipo = sc.inputs.ipo
        probs = [a.allotment_probability for a in sc.inputs.accounts]
        rows.append(
            {
                "Scenario": sc.name,
                "GMP (Rs)": ipo.gmp_absolute,
                "Listing price": ipo.expected_listing_price,
                "Exit price": ipo.expected_exit_price,
                "Listing gain %": ipo.expected_listing_gain_pct,
                "Allotment probability": float(np.mean(probs)) if probs else 0.0,
                "Expected allotments": res.expected_allotments,
                "Gross profit": res.expected_gross_profit,
                "Transaction costs": res.expected_transaction_costs,
                "Taxes": res.expected_taxes,
                "Financing cost": res.expected_financing_cost,
                "Opportunity cost": res.expected_opportunity_cost,
                "Net profit": res.expected_net_profit,
                "ROI on own equity": res.capital.return_on_economic_capital,
                "ROI on application capital": res.capital.return_on_application_capital,
                "Annualized ROI": res.capital.annualized_return_on_economic_capital,
                "Description": sc.definition.description,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------
def sensitivity_grid(
    base: AnalysisInputs,
    row_values: Sequence[float],
    col_values: Sequence[float],
    mutate: Callable[[AnalysisInputs, float, float], AnalysisInputs],
    metric: Callable[[AnalysisInputs], float] = expected_net_profit,
    row_label: str = "row",
    col_label: str = "col",
) -> pd.DataFrame:
    """Generic two-way sensitivity table.

    ``mutate(base, row_value, col_value)`` returns the modified inputs; ``metric``
    reduces those inputs to a single number (expected net profit by default).
    """
    data = [[float(metric(mutate(base, r, c))) for c in col_values] for r in row_values]
    frame = pd.DataFrame(data, index=list(row_values), columns=list(col_values))
    frame.index.name = row_label
    frame.columns.name = col_label
    return frame


def sensitivity_gmp_vs_probability(
    base: AnalysisInputs,
    gmp_values: Sequence[float],
    probability_values: Sequence[float],
    metric: Callable[[AnalysisInputs], float] = expected_net_profit,
) -> pd.DataFrame:
    """Expected net profit for each (GMP, allotment probability) pair.

    ``probability_values`` are fractions (0.05 = 5%) applied uniformly to every
    account.
    """

    def mutate(inputs: AnalysisInputs, gmp: float, prob: float) -> AnalysisInputs:
        ipo = replace(
            inputs.ipo,
            gmp_value=gmp,
            gmp_mode=GMPMode.ABSOLUTE,
            use_gmp_for_listing=True,
            expected_listing_price_override=None,
            expected_exit_price_override=None,
        )
        accounts = tuple(
            replace(a, allotment_probability=min(max(prob, 0.0), 1.0))
            for a in inputs.accounts
        )
        return replace(inputs, ipo=ipo, accounts=accounts)

    return sensitivity_grid(
        base,
        gmp_values,
        probability_values,
        mutate,
        metric,
        row_label="GMP (Rs)",
        col_label="Allotment probability",
    )


def sensitivity_od_rate_vs_listing_gain(
    base: AnalysisInputs,
    od_rate_values: Sequence[float],
    listing_gain_values: Sequence[float],
    metric: Callable[[AnalysisInputs], float] = expected_net_profit,
) -> pd.DataFrame:
    """Expected net profit for each (OD rate %, listing gain %) pair."""

    def mutate(
        inputs: AnalysisInputs, od_rate: float, gain_pct: float
    ) -> AnalysisInputs:
        issue = inputs.ipo.issue_price
        ipo = replace(
            inputs.ipo,
            gmp_value=issue * gain_pct / 100.0,
            gmp_mode=GMPMode.ABSOLUTE,
            use_gmp_for_listing=True,
            expected_listing_price_override=None,
            expected_exit_price_override=None,
        )
        financing = replace(inputs.financing, od_rate_pct=od_rate)
        return replace(inputs, ipo=ipo, financing=financing)

    return sensitivity_grid(
        base,
        od_rate_values,
        listing_gain_values,
        mutate,
        metric,
        row_label="OD rate (% p.a.)",
        col_label="Listing gain (%)",
    )


# ---------------------------------------------------------------------------
# Monte Carlo simulation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MonteCarloConfig:
    """Distribution assumptions for the stochastic view.

    The simulation randomises the four inputs that actually drive the outcome:
    the realised exit gain, the allotment hit-rate, the holding period and the
    financing rate. Every distribution is explicit and user-set - the defaults
    below are wide on purpose, because IPO listing outcomes are fat-tailed.
    """

    n_simulations: int = 10_000
    seed: int | None = 42

    # Exit gain (% over issue price)
    gain_distribution: str = "normal"  # normal | triangular | uniform | fixed
    gain_mean_pct: float | None = None  # None -> the base-case exit gain
    gain_std_pct: float = 15.0
    gain_low_pct: float = -20.0
    gain_high_pct: float = 40.0

    # Allotment probability
    probability_distribution: str = "fixed"  # fixed | beta
    probability_concentration: float = 20.0  # Beta concentration (alpha + beta)

    # Holding period (days)
    holding_distribution: str = "fixed"  # fixed | uniform_int
    holding_low_days: int = 1
    holding_high_days: int = 5

    # OD rate (% p.a.)
    od_rate_distribution: str = "fixed"  # fixed | normal | uniform
    od_rate_std_pct: float = 1.0
    od_rate_low_pct: float = 9.0
    od_rate_high_pct: float = 13.0


@dataclass(frozen=True)
class MonteCarloResult:
    """Simulated distribution of net profit."""

    profits: np.ndarray
    config: MonteCarloConfig
    expected_profit: float
    median_profit: float
    percentiles: dict[int, float]
    probability_of_profit: float
    probability_of_loss: float
    worst_case: float
    best_case: float
    std_dev: float
    expected_shortfall_5pct: float
    allotment_counts: np.ndarray

    def summary_frame(self) -> pd.DataFrame:
        rows = [
            ("Simulations", float(len(self.profits))),
            ("Expected (mean) profit", self.expected_profit),
            ("Median profit", self.median_profit),
            ("5th percentile", self.percentiles[5]),
            ("25th percentile", self.percentiles[25]),
            ("75th percentile", self.percentiles[75]),
            ("95th percentile", self.percentiles[95]),
            ("Standard deviation", self.std_dev),
            ("Worst simulated outcome", self.worst_case),
            ("Best simulated outcome", self.best_case),
            ("Expected shortfall (worst 5%)", self.expected_shortfall_5pct),
            ("Probability of profit", self.probability_of_profit),
            ("Probability of loss", self.probability_of_loss),
        ]
        return pd.DataFrame(rows, columns=["Metric", "Value"])


def _vector_transaction_costs(
    buy_value: np.ndarray, sell_value: np.ndarray, costs
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised twin of :func:`calculations.compute_transaction_costs`.

    Returns ``(total_costs, costs_deductible_against_capital_gains)``.
    """
    buy_value = np.maximum(buy_value, 0.0)
    sell_value = np.maximum(sell_value, 0.0)
    traded = buy_value + sell_value

    brokerage = (
        buy_value * costs.brokerage_pct_buy / 100.0
        + sell_value * costs.brokerage_pct_sell / 100.0
        + np.where(buy_value > 0, costs.brokerage_flat_buy, 0.0)
        + np.where(sell_value > 0, costs.brokerage_flat_sell, 0.0)
    )
    stt = (
        buy_value * costs.stt_pct_buy / 100.0 + sell_value * costs.stt_pct_sell / 100.0
    )
    exchange = traded * costs.exchange_txn_pct / 100.0
    sebi = traded * costs.sebi_turnover_pct / 100.0
    stamp = buy_value * costs.stamp_duty_pct_buy / 100.0
    gst = (brokerage + exchange + sebi) * costs.gst_pct / 100.0
    dp = np.where(sell_value > 0, costs.dp_charges_flat_sell, 0.0)
    other = np.where(traded > 0, costs.other_charges_flat, 0.0)
    total = brokerage + stt + exchange + sebi + stamp + gst + dp + other
    return total, total - stt


def _vector_tax(
    gross_gain: np.ndarray,
    holding_days: np.ndarray,
    taxes,
    deductible: np.ndarray,
) -> np.ndarray:
    """Vectorised twin of :func:`calculations.compute_capital_gains_tax`."""
    deduction = deductible if taxes.deduct_transaction_costs_from_gain else 0.0
    taxable = gross_gain - deduction
    is_long = holding_days > taxes.ltcg_threshold_days
    rate = np.where(is_long, taxes.ltcg_rate_pct, taxes.stcg_rate_pct) / 100.0
    multiplier = rate * (1.0 + taxes.cess_and_surcharge_pct / 100.0)
    if taxes.apply_ltcg_exemption:
        taxable = np.where(
            is_long & (taxable > 0),
            np.maximum(taxable - taxes.ltcg_exemption_amount, 0.0),
            taxable,
        )
    gains_tax = np.where(taxable > 0, taxable * multiplier, 0.0)
    if taxes.recognise_tax_shield_on_loss:
        gains_tax = np.where(taxable <= 0, taxable * multiplier, gains_tax)
    return gains_tax


def _draw_gain_pct(
    cfg: MonteCarloConfig, base_gain: float, rng: np.random.Generator
) -> np.ndarray:
    n = cfg.n_simulations
    mean = cfg.gain_mean_pct if cfg.gain_mean_pct is not None else base_gain
    kind = cfg.gain_distribution.lower()
    if kind == "fixed":
        return np.full(n, mean, dtype=float)
    if kind == "normal":
        return rng.normal(mean, max(cfg.gain_std_pct, 0.0), n)
    if kind == "uniform":
        return rng.uniform(cfg.gain_low_pct, cfg.gain_high_pct, n)
    if kind == "triangular":
        low, high = cfg.gain_low_pct, cfg.gain_high_pct
        mode = min(max(mean, low), high)
        return rng.triangular(low, mode, high, n)
    raise ValueError(f"Unknown gain distribution: {cfg.gain_distribution!r}")


def _draw_probability(
    cfg: MonteCarloConfig, base_p: float, rng: np.random.Generator
) -> np.ndarray:
    n = cfg.n_simulations
    kind = cfg.probability_distribution.lower()
    if kind == "fixed" or base_p <= 0.0 or base_p >= 1.0:
        return np.full(n, base_p, dtype=float)
    if kind == "beta":
        concentration = max(cfg.probability_concentration, 1e-6)
        alpha = base_p * concentration
        beta = (1.0 - base_p) * concentration
        return rng.beta(max(alpha, 1e-6), max(beta, 1e-6), n)
    raise ValueError(
        f"Unknown probability distribution: {cfg.probability_distribution!r}"
    )


def _draw_holding_days(
    cfg: MonteCarloConfig, base_days: int, rng: np.random.Generator
) -> np.ndarray:
    n = cfg.n_simulations
    kind = cfg.holding_distribution.lower()
    if kind == "fixed":
        return np.full(n, float(base_days))
    if kind == "uniform_int":
        low, high = int(cfg.holding_low_days), int(cfg.holding_high_days)
        if high < low:
            low, high = high, low
        return rng.integers(low, high + 1, n).astype(float)
    raise ValueError(f"Unknown holding distribution: {cfg.holding_distribution!r}")


def _draw_od_rate(
    cfg: MonteCarloConfig, base_rate: float, rng: np.random.Generator
) -> np.ndarray:
    n = cfg.n_simulations
    kind = cfg.od_rate_distribution.lower()
    if kind == "fixed":
        return np.full(n, base_rate, dtype=float)
    if kind == "normal":
        return np.maximum(rng.normal(base_rate, max(cfg.od_rate_std_pct, 0.0), n), 0.0)
    if kind == "uniform":
        return np.maximum(
            rng.uniform(cfg.od_rate_low_pct, cfg.od_rate_high_pct, n), 0.0
        )
    raise ValueError(f"Unknown OD rate distribution: {cfg.od_rate_distribution!r}")


def run_monte_carlo(
    inputs: AnalysisInputs, config: MonteCarloConfig | None = None
) -> MonteCarloResult:
    """Simulate the joint effect of listing, allotment, holding and rate risk.

    The per-simulation cash-flow logic is an exact vectorised replica of the
    deterministic engine, so a simulation with every distribution set to
    ``fixed`` reproduces the deterministic expected profit (up to the Bernoulli
    sampling error on allotment).
    """
    cfg = config or MonteCarloConfig()
    if cfg.n_simulations <= 0:
        raise ValueError("n_simulations must be positive")
    rng = np.random.default_rng(cfg.seed)

    ipo, fin = inputs.ipo, inputs.financing
    basis = fin.day_count_basis or 365
    bid_days = max(float(fin.days_blocked), 0.0)
    opp_rate = fin.opportunity_cost_rate_pct if fin.include_opportunity_cost else 0.0
    plan = build_funding_plan(inputs.total_application_amount, fin)
    od_share_bid, own_share_bid = plan.od_share, plan.own_share
    if fin.finance_holding_period:
        od_share_hold, own_share_hold = od_share_bid, own_share_bid
    else:
        od_share_hold, own_share_hold = 0.0, 1.0

    gain_pct = _draw_gain_pct(cfg, ipo.expected_exit_gain_pct, rng)
    exit_price = np.maximum(ipo.issue_price * (1.0 + gain_pct / 100.0), 0.0)
    holding_days = np.maximum(
        _draw_holding_days(cfg, ipo.holding_period_days, rng), 0.0
    )
    od_rate = _draw_od_rate(cfg, fin.od_rate_pct, rng)

    profits = np.zeros(cfg.n_simulations)
    allotment_counts = np.zeros(cfg.n_simulations)

    for acct in inputs.accounts:
        shares = acct.shares_if_allotted(ipo.lot_size)
        investment = ipo.issue_price * shares
        sell_value = exit_price * shares
        gross = sell_value - investment
        buy_value = np.full(cfg.n_simulations, investment)
        txn_total, txn_deductible = _vector_transaction_costs(
            buy_value, sell_value, inputs.costs
        )
        tax = _vector_tax(
            gross,
            holding_days,
            inputs.taxes,
            txn_total if inputs.taxes.stt_deductible else txn_deductible,
        )
        carry = (
            investment * od_share_hold * (od_rate / 100.0) * holding_days / basis
            + investment * own_share_hold * (opp_rate / 100.0) * holding_days / basis
        )
        p = _draw_probability(cfg, acct.allotment_probability, rng)
        allotted = (rng.random(cfg.n_simulations) < p).astype(float)
        allotment_counts += allotted
        profits += allotted * (gross - txn_total - tax - carry)

    profits -= plan.od_drawn * (od_rate / 100.0) * bid_days / basis
    profits -= plan.own_capital_deployed * (opp_rate / 100.0) * bid_days / basis
    profits -= fin.processing_fee + fin.other_financing_charges
    if fin.count_fd_interest_as_income:
        profits += (
            fin.fd_amount
            * (fin.fd_rate_pct / 100.0)
            * (bid_days + holding_days)
            / basis
        )

    pct = {
        int(q): float(np.percentile(profits, q))
        for q in (1, 5, 10, 25, 50, 75, 90, 95, 99)
    }
    tail = profits[profits <= pct[5]]
    return MonteCarloResult(
        profits=profits,
        config=cfg,
        expected_profit=float(np.mean(profits)),
        median_profit=float(np.median(profits)),
        percentiles=pct,
        probability_of_profit=float(np.mean(profits > 0)),
        probability_of_loss=float(np.mean(profits < 0)),
        worst_case=float(np.min(profits)),
        best_case=float(np.max(profits)),
        std_dev=float(np.std(profits, ddof=1)) if len(profits) > 1 else 0.0,
        expected_shortfall_5pct=float(np.mean(tail))
        if tail.size
        else float(np.min(profits)),
        allotment_counts=allotment_counts,
    )
