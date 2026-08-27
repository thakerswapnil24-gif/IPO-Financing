"""Worked example: run the full analysis from the command line.

    python example_analysis.py                 # analyse every bundled example
    python example_analysis.py --index 1       # analyse one example
    python example_analysis.py --export out/   # also write CSV, Excel and PDF

The point of this script is to show that the engine is completely independent of
Streamlit: everything the dashboard displays is produced here in plain Python.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from calculations import analyze, assumption_ledger
from example_data import ExamplePreset, load_examples
from export import build_report, bundle_to_csv, bundle_to_excel, bundle_to_pdf
from risk import compare_opportunities, compute_risk_metrics, evaluate_decision
from scenarios import (
    MonteCarloConfig,
    run_monte_carlo,
    run_scenarios,
    scenarios_to_frame,
    sensitivity_gmp_vs_probability,
    sensitivity_od_rate_vs_listing_gain,
)
from validation import validate_inputs


def rupees(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"Rs {value:>12,.2f}"


def percent(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2%}"


def rule(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def analyse_one(preset: ExamplePreset, simulations: int = 20_000) -> None:
    """Print a complete analysis of one set of assumptions."""
    inputs = preset.inputs
    print("=" * 78)
    print(preset.name)
    print(preset.notes)
    print("=" * 78)

    report = validate_inputs(inputs)
    if report.issues:
        rule("Validation")
        for issue in report.issues:
            print(f"  {issue}")
    if not report.is_valid:
        print("  Analysis skipped: inputs are invalid.")
        return

    result = analyze(inputs)
    risk = compute_risk_metrics(result)
    decision = evaluate_decision(result, risk)
    ipo, capital = inputs.ipo, result.capital

    rule("Assumptions that drive the answer")
    print(f"  Issue price               {rupees(ipo.issue_price)}")
    print(
        f"  GMP                       {rupees(ipo.gmp_absolute)}  "
        f"({ipo.gmp_percent:.2f}% of issue)"
    )
    print(
        f"  Expected listing price    {rupees(ipo.expected_listing_price)}  "
        f"({ipo.expected_listing_gain_pct:+.2f}%)"
    )
    print(f"  Expected exit price       {rupees(ipo.expected_exit_price)}")
    print(
        f"  Applications              {len(inputs.accounts)} account(s), "
        f"{result.expected_allotments:.2f} expected allotments"
    )
    print(
        f"  Days blocked / held       {inputs.financing.days_blocked} / "
        f"{ipo.holding_period_days}"
    )
    print(f"  OD rate                   {inputs.financing.od_rate_pct:.2f}% p.a.")

    rule("Capital")
    print(f"  Applied for (blocked)     {rupees(capital.total_application_amount)}")
    print(f"  Borrowed (OD drawn)       {rupees(capital.borrowed_capital)}")
    print(f"  Own cash deployed         {rupees(capital.own_capital_deployed)}")
    print(f"  FD collateral pledged     {rupees(result.funding.fd_collateral_locked)}")
    print(f"  Own economic capital      {rupees(capital.economic_capital_at_risk)}")
    print(
        f"  Expected amount invested  "
        f"{rupees(sum(a.expected_investment for a in result.accounts))}"
    )

    rule("Expected outcome")
    print(f"  Expected gross profit     {rupees(result.expected_gross_profit)}")
    print(f"  Transaction costs         {rupees(-result.expected_transaction_costs)}")
    print(f"  Taxes                     {rupees(-result.expected_taxes)}")
    print(f"  Financing cost            {rupees(-result.expected_financing_cost)}")
    print(f"  Opportunity cost          {rupees(-result.expected_opportunity_cost)}")
    print(f"  Expected NET profit       {rupees(result.expected_net_profit)}")
    print(
        f"  Return on application     {percent(capital.return_on_application_capital)}"
    )
    print(
        f"  Return on own equity      {percent(capital.return_on_economic_capital)}"
        f"   (annualised {percent(capital.annualized_return_on_economic_capital)})"
    )
    print(
        f"  Financing / gross profit  {percent(capital.financing_cost_to_gross_profit)}"
    )

    rule("Break-even")
    break_even = result.break_even
    print(
        f"  Exit price (expected value) {rupees(break_even.exit_price_expected_value)}"
        f"   GMP {rupees(break_even.gmp_expected_value)}"
    )
    print(
        f"  Exit price (if allotted)    {rupees(break_even.exit_price_if_allotted)}"
        f"   GMP {rupees(break_even.gmp_if_allotted)}"
    )
    print(
        f"  Minimum hit-rate            {percent(break_even.min_allotment_probability)}"
    )
    max_od_rate = break_even.max_od_rate_pct
    max_od_rate_text = "n/a" if max_od_rate is None else f"{max_od_rate:.2f}%"
    print(f"  Max sustainable OD rate     {max_od_rate_text}")

    rule("Scenarios")
    frame = scenarios_to_frame(run_scenarios(inputs))
    for _, row in frame.iterrows():
        print(
            f"  {row['Scenario']:<6} listing {row['Listing price']:>9,.2f}  "
            f"p {row['Allotment probability']:>6.1%}  "
            f"gross {row['Gross profit']:>10,.0f}  "
            f"financing {row['Financing cost']:>8,.0f}  "
            f"net {row['Net profit']:>10,.0f}  "
            f"ROI {row['ROI on own equity']:>8.2%}"
        )

    rule("Sensitivity: expected net profit by GMP (rows) and hit-rate (columns)")
    issue = ipo.issue_price
    gmps = [0.0, issue * 0.05, issue * 0.10, issue * 0.20, issue * 0.35]
    probabilities = [0.05, 0.10, 0.20, 0.30]
    grid = sensitivity_gmp_vs_probability(inputs, gmps, probabilities)
    header = "  GMP".ljust(14) + "".join(f"{p:>13.0%}" for p in probabilities)
    print(header)
    for gmp, row in grid.iterrows():
        print(f"  Rs {gmp:>9,.1f}" + "".join(f"{value:>13,.0f}" for value in row))

    rule(
        "Sensitivity: expected net profit by OD rate (rows) and listing gain (columns)"
    )
    rates = [0.0, 8.0, 11.0, 14.0, 18.0]
    gains = [-10.0, 0.0, 10.0, 25.0]
    rate_grid = sensitivity_od_rate_vs_listing_gain(inputs, rates, gains)
    print("  OD rate".ljust(14) + "".join(f"{g:>12.0f}%" for g in gains))
    for rate, row in rate_grid.iterrows():
        print(f"  {rate:>9.2f}%  " + "".join(f"{value:>13,.0f}" for value in row))

    rule("Risk")
    print(f"  Probability of loss       {percent(risk.probability_of_loss)}")
    print(f"  Maximum modelled loss     {rupees(risk.maximum_loss)}")
    print(
        f"  Expected loss / gain      {rupees(risk.expected_loss)} / "
        f"{rupees(risk.expected_gain)}"
    )
    print(f"  GMP margin of safety      {percent(risk.gmp_margin_of_safety)}")
    print(f"  Hit-rate margin of safety {percent(risk.probability_margin_of_safety)}")
    print(f"  If it lists flat          {rupees(risk.profit_if_lists_flat)}")
    print(f"  If it lists 10% down      {rupees(risk.profit_if_lists_10pct_below)}")

    rule("Monte Carlo")
    simulation = run_monte_carlo(
        inputs, MonteCarloConfig(n_simulations=simulations, seed=42)
    )
    print(
        f"  Mean {rupees(simulation.expected_profit)}   median "
        f"{rupees(simulation.median_profit)}"
    )
    print(
        f"  P5   {rupees(simulation.percentiles[5])}   P25 "
        f"{rupees(simulation.percentiles[25])}"
    )
    print(
        f"  P75  {rupees(simulation.percentiles[75])}   P95 "
        f"{rupees(simulation.percentiles[95])}"
    )
    print(f"  Probability of profit     {percent(simulation.probability_of_profit)}")

    rule(f"Decision: {decision.verdict.value}")
    for check in decision.checks:
        print(f"  [{check.status:<4}] {check.name}")
        print(f"           {check.detail}")
    print()
    for line in decision.rationale:
        print(f"  * {line}")

    untouched = sum(
        1
        for r in assumption_ledger(inputs)
        if r.provenance.value == "Default assumption"
    )
    print(
        f"\n  {untouched} inputs are still at their shipped defaults - review them "
        "before relying on this verdict."
    )


def write_exports(preset: ExamplePreset, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    inputs = preset.inputs
    result = analyze(inputs)
    risk = compute_risk_metrics(result)
    decision = evaluate_decision(result, risk)
    bundle = build_report(
        result,
        risk,
        decision,
        scenarios=run_scenarios(inputs),
        sensitivities={
            "GMP vs probability": sensitivity_gmp_vs_probability(
                inputs,
                [0.0, inputs.ipo.issue_price * 0.1, inputs.ipo.issue_price * 0.25],
                [0.05, 0.10, 0.25],
            )
        },
        monte_carlo=run_monte_carlo(inputs, MonteCarloConfig(n_simulations=20_000)),
    )
    stem = (
        "".join(c if c.isalnum() else "_" for c in inputs.ipo.name).strip("_")
        or "analysis"
    )
    (directory / f"{stem}.csv").write_text(bundle_to_csv(bundle), encoding="utf-8")
    (directory / f"{stem}.xlsx").write_bytes(bundle_to_excel(bundle))
    (directory / f"{stem}.pdf").write_bytes(bundle_to_pdf(bundle))
    print(f"  Wrote {stem}.csv, {stem}.xlsx and {stem}.pdf to {directory}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index", type=int, help="analyse a single bundled example (0-based)"
    )
    parser.add_argument(
        "--export", type=Path, help="directory to write CSV/Excel/PDF exports into"
    )
    parser.add_argument("--simulations", type=int, default=20_000)
    arguments = parser.parse_args(argv)

    presets: list[ExamplePreset] = load_examples()
    selected = [presets[arguments.index]] if arguments.index is not None else presets

    for preset in selected:
        analyse_one(preset, arguments.simulations)
        if arguments.export:
            write_exports(preset, arguments.export)

    if len(selected) > 1:
        print("\n" + "=" * 78)
        print("Portfolio comparison")
        print("=" * 78)
        frame = compare_opportunities(
            [(p.name.split(".")[0], p.inputs) for p in selected]
        )
        frame = frame.sort_values("Expected net profit", ascending=False)
        print(
            f"  {'IPO':<5}{'Application':>14}{'Exp. profit':>14}{'Financing':>12}"
            f"{'ROI':>10}{'P(loss)':>10}  Decision"
        )
        for _, row in frame.iterrows():
            roi = row["ROI on own equity"]
            print(
                f"  "
                f"{row['IPO']:<5}"
                f"{row['Application']:>14,.0f}"
                f"{row['Expected net profit']:>14,.0f}"
                f"{row['Financing cost']:>12,.0f}"
                f"{('n/a' if roi is None else f'{roi:.2%}'):>10}"
                f"{row['Probability of loss']:>10.0%}  {row['Decision']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
