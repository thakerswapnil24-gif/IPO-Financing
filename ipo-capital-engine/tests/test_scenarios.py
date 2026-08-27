"""Tests for scenario analysis, sensitivity grids and the Monte Carlo engine."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from calculations import analyze, expected_net_profit
from scenarios import (
    DEFAULT_BEAR,
    DEFAULT_BULL,
    MonteCarloConfig,
    ScenarioDefinition,
    default_scenarios,
    run_monte_carlo,
    run_scenarios,
    scenarios_to_frame,
    sensitivity_gmp_vs_probability,
    sensitivity_od_rate_vs_listing_gain,
)
from tests.test_calculations import frictionless


def test_scenarios_run_in_bear_base_bull_order_and_are_monotonic():
    base = frictionless()
    results = run_scenarios(base)
    assert [r.name for r in results] == ["Bear", "Base", "Bull"]
    bear, mid, bull = (r.result.expected_net_profit for r in results)
    assert bear < mid < bull


def test_base_scenario_reproduces_the_users_own_assumptions():
    base = frictionless()
    results = run_scenarios(base)
    base_result = next(r for r in results if r.name == "Base")
    assert base_result.result.expected_net_profit == pytest.approx(
        analyze(base).expected_net_profit
    )
    assert base_result.inputs.ipo.gmp_absolute == pytest.approx(base.ipo.gmp_absolute)


def test_bear_scenario_applies_the_documented_factors():
    base = frictionless(gmp=20.0, probability=0.25)
    bear = DEFAULT_BEAR.apply(base)
    assert bear.ipo.gmp_absolute == pytest.approx(-10.0)  # -0.5 x 20
    assert bear.accounts[0].allotment_probability == pytest.approx(0.125)
    assert DEFAULT_BEAR.description  # never an unlabelled assumption


def test_bull_scenario_caps_probability_at_one():
    base = frictionless(probability=0.9)
    bull = ScenarioDefinition(name="Bull", allotment_probability_multiplier=2.0).apply(
        base
    )
    assert bull.accounts[0].allotment_probability == 1.0


def test_scenario_can_override_the_od_rate_and_holding_period():
    base = frictionless()
    definition = ScenarioDefinition(
        name="Stress", od_rate_pct_override=20.0, holding_period_days_override=30
    )
    stressed = definition.apply(base)
    assert stressed.financing.od_rate_pct == 20.0
    assert stressed.ipo.holding_period_days == 30
    assert expected_net_profit(stressed) < expected_net_profit(base)


def test_scenario_preserves_an_exit_price_that_differs_from_listing():
    base = frictionless(gmp=20.0)
    base = replace(base, ipo=replace(base.ipo, expected_exit_price_override=130.0))
    bull = DEFAULT_BULL.apply(base)
    # Base sells 10 above the listing price; the scenario keeps that spread.
    assert bull.ipo.expected_listing_price == pytest.approx(130.0)  # 20 x 1.5 + 100
    assert bull.ipo.expected_exit_price == pytest.approx(140.0)


def test_scenario_frame_has_the_required_columns():
    frame = scenarios_to_frame(run_scenarios(frictionless()))
    for column in (
        "Scenario",
        "Listing price",
        "Allotment probability",
        "Gross profit",
        "Financing cost",
        "Net profit",
        "ROI on own equity",
    ):
        assert column in frame.columns
    assert len(frame) == len(default_scenarios())


# ---------------------------------------------------------------------------
# Sensitivity
# ---------------------------------------------------------------------------
def test_gmp_versus_probability_grid_shape_and_monotonicity():
    base = frictionless()
    gmps = [0.0, 25.0, 50.0, 75.0]
    probabilities = [0.05, 0.10, 0.20, 0.30]
    grid = sensitivity_gmp_vs_probability(base, gmps, probabilities)
    assert grid.shape == (4, 4)
    assert list(grid.index) == gmps
    assert list(grid.columns) == probabilities
    # Profit rises with GMP down every column and with probability along each row
    for column in grid.columns:
        assert grid[column].is_monotonic_increasing
    for _, row in grid.iterrows():
        assert row.is_monotonic_increasing or row.is_monotonic_decreasing


def test_zero_gmp_with_any_hit_rate_still_loses_money():
    grid = sensitivity_gmp_vs_probability(frictionless(), [0.0], [0.05, 0.5, 1.0])
    assert (grid.loc[0.0] < 0).all()


def test_od_rate_versus_listing_gain_grid_declines_with_the_rate():
    grid = sensitivity_od_rate_vs_listing_gain(
        frictionless(), [8.0, 10.0, 12.0, 14.0], [0.0, 5.0, 10.0, 20.0]
    )
    assert grid.shape == (4, 4)
    for column in grid.columns:
        assert grid[column].is_monotonic_decreasing


def test_sensitivity_cell_matches_a_direct_calculation():
    base = frictionless()
    grid = sensitivity_gmp_vs_probability(base, [20.0], [0.25])
    assert grid.loc[20.0, 0.25] == pytest.approx(expected_net_profit(base))


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------
def test_monte_carlo_with_fixed_distributions_converges_on_the_expected_value():
    base = frictionless(probability=0.25)
    config = MonteCarloConfig(
        n_simulations=200_000,
        seed=7,
        gain_distribution="fixed",
        probability_distribution="fixed",
    )
    simulation = run_monte_carlo(base, config)
    analytic = analyze(base).expected_net_profit
    # Only the Bernoulli allotment draw is random, so the mean must converge.
    assert simulation.expected_profit == pytest.approx(analytic, rel=0.02)


def test_monte_carlo_is_reproducible_with_a_seed():
    base = frictionless()
    config = MonteCarloConfig(n_simulations=5_000, seed=123)
    first = run_monte_carlo(base, config)
    second = run_monte_carlo(base, config)
    assert np.array_equal(first.profits, second.profits)


def test_monte_carlo_reports_the_required_percentiles():
    simulation = run_monte_carlo(frictionless(), MonteCarloConfig(n_simulations=10_000))
    for quantile in (5, 25, 75, 95):
        assert quantile in simulation.percentiles
    assert simulation.percentiles[5] <= simulation.percentiles[25]
    assert simulation.percentiles[25] <= simulation.percentiles[75]
    assert simulation.percentiles[75] <= simulation.percentiles[95]
    assert (
        simulation.probability_of_profit + simulation.probability_of_loss <= 1.0 + 1e-9
    )
    assert len(simulation.profits) == 10_000


def test_monte_carlo_runs_at_least_ten_thousand_paths_by_default():
    assert MonteCarloConfig().n_simulations >= 10_000


def test_monte_carlo_allotment_counts_track_the_hit_rate():
    base = frictionless(n_accounts=4, probability=0.25)
    simulation = run_monte_carlo(base, MonteCarloConfig(n_simulations=50_000, seed=3))
    assert simulation.allotment_counts.mean() == pytest.approx(1.0, rel=0.05)
    assert simulation.allotment_counts.max() <= 4


def test_wider_listing_uncertainty_widens_the_profit_distribution():
    base = frictionless()
    narrow = run_monte_carlo(
        base, MonteCarloConfig(n_simulations=20_000, seed=5, gain_std_pct=5.0)
    )
    wide = run_monte_carlo(
        base, MonteCarloConfig(n_simulations=20_000, seed=5, gain_std_pct=40.0)
    )
    assert wide.std_dev > narrow.std_dev


def test_monte_carlo_rejects_an_unknown_distribution():
    with pytest.raises(ValueError):
        run_monte_carlo(frictionless(), MonteCarloConfig(gain_distribution="cauchy"))


def test_monte_carlo_rejects_a_non_positive_simulation_count():
    with pytest.raises(ValueError):
        run_monte_carlo(frictionless(), MonteCarloConfig(n_simulations=0))
