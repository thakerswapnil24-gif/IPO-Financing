"""Plain-language documentation of every formula the engine uses.

Kept as data (not as Streamlit markdown) so that the same text can be rendered
in the UI, written into the exported report, or asserted against in tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pandas as pd

__all__ = ["MetricExplanation", "EXPLANATIONS", "explanations_frame", "GLOSSARY"]


@dataclass(frozen=True)
class MetricExplanation:
    metric: str
    formula: str
    inputs_used: str
    interpretation: str
    limitations: str


EXPLANATIONS: Tuple[MetricExplanation, ...] = (
    MetricExplanation(
        metric="Expected listing price",
        formula="Expected listing price = Issue price + GMP",
        inputs_used="Issue price; GMP (absolute Rs, or % of issue price).",
        interpretation=(
            "The price the grey market is currently implying for listing day. "
            "Listing gain % = (listing price / issue price) - 1."
        ),
        limitations=(
            "GMP is an unregulated, unsettled, thinly traded quote. It is a sentiment "
            "indicator, not a forecast, and it routinely collapses between the close "
            "of the bidding window and listing day. Treat it as an assumption you own, "
            "not as data."
        ),
    ),
    MetricExplanation(
        metric="Application (blocked) capital",
        formula="Application amount = Issue price x Lot size x Lots applied (summed over accounts)",
        inputs_used="Issue price; lot size; lots applied per account.",
        interpretation=(
            "The money that must be available and blocked (ASBA) or drawn (OD) during "
            "the bidding window. It is NOT the amount you end up invested in."
        ),
        limitations=(
            "The non-allotted portion is released at the end of the bidding window, so "
            "it costs you carry, not principal. Confusing this number with the invested "
            "amount is the single most common error in IPO financing arithmetic."
        ),
    ),
    MetricExplanation(
        metric="Own equity vs borrowed capital",
        formula=(
            "OD limit = FD amount x LTV%\n"
            "OD drawn = min(application - own capital deployed, OD limit)\n"
            "Economic capital at risk = own cash deployed + FD collateral pledged "
            "(= OD drawn / LTV)"
        ),
        inputs_used="Funding mode; own capital; FD amount; OD LTV%.",
        interpretation=(
            "Three different denominators, deliberately never merged: the application "
            "amount (what is blocked), your own cash (what you actually pay in), and "
            "your economic capital (your cash plus the deposit pledged as collateral)."
        ),
        limitations=(
            "A pledged FD is not consumed, but it is encumbered - you cannot break it "
            "or use it elsewhere while the OD is outstanding, so treating it as free is "
            "wrong. Where own cash deployed is zero, return on own cash is undefined and "
            "the economic-capital denominator is the honest one."
        ),
    ),
    MetricExplanation(
        metric="Financing (OD) cost",
        formula=(
            "Bidding window:  OD drawn x OD rate x days blocked / 365\n"
            "Holding window:  Allotted investment x OD share x OD rate x holding days / 365\n"
            "Total expected financing cost = bidding-window cost + "
            "P(allotment) x holding-window cost + processing fee + other charges"
        ),
        inputs_used="OD drawn; OD rate % p.a.; days blocked; holding period; fees.",
        interpretation=(
            "The bidding-window cost is unconditional - you pay it whether or not you "
            "are allotted. Only the allotted portion is carried through the holding "
            "window, so that leg is probability-weighted."
        ),
        limitations=(
            "Assumes simple interest on a 365-day basis and same-day drawdown/repayment. "
            "Banks compound monthly and may charge a minimum utilisation, and the OD rate "
            "can be repriced. Fees are modelled as one-time and unconditional."
        ),
    ),
    MetricExplanation(
        metric="Opportunity cost of own capital",
        formula="Own capital deployed x opportunity rate x days / 365",
        inputs_used="Own capital deployed; opportunity cost rate; days blocked and held.",
        interpretation=(
            "What your own money would have earned in its next-best use (typically the "
            "same FD rate). Profit measured before this is accounting profit; after it, "
            "economic profit."
        ),
        limitations=(
            "The opportunity rate is a judgement, not an observable. Switching it off "
            "flatters every return metric in the model."
        ),
    ),
    MetricExplanation(
        metric="Expected profit",
        formula=(
            "Expected profit = P(allotment) x (profit if allotted)\n"
            "Multi-account: Expected total profit = sum over accounts of "
            "p_i x profit_i, minus the unconditional financing cost of the whole "
            "application"
        ),
        inputs_used="Allotment probability per account; per-account profit if allotted.",
        interpretation=(
            "The average outcome over many repetitions of this exact bet. It is not the "
            "outcome of any single application: in a typical retail lottery the modal "
            "outcome is 'nothing allotted, financing cost paid'."
        ),
        limitations=(
            "Expected value says nothing about how long it takes to converge, and IPO "
            "outcomes are fat-tailed and correlated across issues (a cold primary market "
            "hits GMP and hit-rate at the same time). Sizing to the expected value while "
            "the modal outcome is a small loss is how leveraged strategies die."
        ),
    ),
    MetricExplanation(
        metric="Transaction costs and taxes",
        formula=(
            "Brokerage + STT + exchange transaction charges + SEBI turnover fees + "
            "stamp duty + GST on (brokerage + exchange + SEBI) + DP charges\n"
            "Tax = (gain - allowable transfer costs) x rate x (1 + cess), where the rate "
            "is short-term below the LTCG threshold and long-term above it"
        ),
        inputs_used="Every rate in the Costs & taxes panel (all user-editable).",
        interpretation=(
            "IPO allotment normally attracts no brokerage or STT on the buy leg; the exit "
            "leg carries the full delivery-sell cost stack."
        ),
        limitations=(
            "Rates are configurable assumptions, not embedded law, and they change with "
            "every finance act and broker tariff. STT is not deductible against capital "
            "gains by default. No set-off against other capital losses is modelled, and "
            "no surcharge slabs above the flat cess."
        ),
    ),
    MetricExplanation(
        metric="Break-even listing price and GMP",
        formula=(
            "Solve for the exit price P where net profit = 0.\n"
            "If-allotted view: gross profit(P) - transaction costs - tax - cost of carry "
            "on the allotted shares = 0.\n"
            "Expected-value view: expected net profit(P) = 0, which must also recover the "
            "financing cost of every application that was not allotted.\n"
            "Break-even GMP = break-even price - issue price"
        ),
        inputs_used="Issue price; costs; taxes; financing; allotment probabilities.",
        interpretation=(
            "The expected-value break-even is the number that matters when you are "
            "financing many applications for a few allotments: the winners must pay for "
            "the losers' carry. It is always the higher of the two."
        ),
        limitations=(
            "Solved numerically on the assumption that everything else is held constant. "
            "A break-even gain that looks small in percent terms can still be far outside "
            "the realistic range for a weak issue."
        ),
    ),
    MetricExplanation(
        metric="Maximum sustainable OD rate",
        formula="Largest OD rate r such that expected net profit(r) >= 0",
        inputs_used="All base assumptions, with the OD rate varied.",
        interpretation=(
            "How much the cost of money can rise before the strategy stops working. The "
            "gap between this and your actual OD rate is your financing headroom."
        ),
        limitations=(
            "Returns 'not applicable' when the strategy loses money even at a zero rate - "
            "in that case cheaper funding cannot rescue it."
        ),
    ),
    MetricExplanation(
        metric="Return on capital and annualisation",
        formula=(
            "Return on application capital = expected net profit / total application amount\n"
            "Return on own equity = expected net profit / economic capital at risk\n"
            "Annualised = (1 + return) ^ (365 / capital-weighted days) - 1"
        ),
        inputs_used="Expected net profit; capital denominators; days blocked and held.",
        interpretation=(
            "Capital-weighted days weights the full application by the bidding window and "
            "the expected allotted amount by the holding period, so a 7-day cycle is not "
            "annualised as if the money were tied up all year."
        ),
        limitations=(
            "Compounded annualisation assumes you can redeploy into an identical "
            "opportunity immediately and repeatedly. In practice IPO supply is lumpy, so "
            "the annualised figure is an upper bound on what is achievable."
        ),
    ),
    MetricExplanation(
        metric="Multiple accounts (PANs)",
        formula=(
            "P(no allotment) = product over accounts of (1 - p_i)\n"
            "P(at least one) = 1 - P(no allotment)\n"
            "Expected allotments = sum of p_i"
        ),
        inputs_used="Per-account allotment probability.",
        interpretation=(
            "More applications raise the chance of at least one allotment, but the "
            "financing cost scales with every application while profit scales only with "
            "the ones that hit."
        ),
        limitations=(
            "Requires independence across accounts, which is a modelling assumption: the "
            "registrar's lottery is run per application, but a change in subscription "
            "levels moves every account's odds together. Applying from multiple PANs that "
            "are not genuinely separate investors is a regulatory matter, not a modelling "
            "one."
        ),
    ),
    MetricExplanation(
        metric="Monte Carlo simulation",
        formula=(
            "For each of N draws: sample the exit gain, the allotment outcome, the "
            "holding period and the OD rate from the chosen distributions, then run the "
            "identical deterministic cash-flow calculation."
        ),
        inputs_used="The distribution parameters in the Monte Carlo panel; a fixed seed "
        "makes results reproducible.",
        interpretation=(
            "Shows the whole distribution rather than the average alone: the median, the "
            "5th percentile and the probability of loss say more about survivability than "
            "the mean does."
        ),
        limitations=(
            "The output is only as good as the assumed distributions. A normal "
            "distribution on listing gains understates crash risk; real listings gap. "
            "Draws are independent across accounts and across risk factors, so it does "
            "not model the correlation between a weak market, a weak GMP and a weak "
            "listing."
        ),
    ),
    MetricExplanation(
        metric="GO / NO-GO framework",
        formula=(
            "Hard rules (any failure -> NO-GO): positive expected net profit; annualised "
            "return clears the OD rate by the required spread; financing cost below the "
            "allowed share of gross profit; break-even listing gain within the "
            "plausibility limit.\n"
            "Soft rules (any failure -> BORDERLINE): probability of loss, bear-case "
            "survivability, GMP margin of safety, hit-rate margin of safety, OD-rate "
            "headroom."
        ),
        inputs_used="The computed metrics plus the editable decision thresholds.",
        interpretation=(
            "A structured way to fail fast. The framework is deliberately hard to satisfy "
            "because leveraged IPO applications lose money quietly through carry."
        ),
        limitations=(
            "It scores your assumptions, and it cannot tell you whether those assumptions "
            "are realistic. It is not investment advice."
        ),
    ),
)


GLOSSARY: Tuple[Tuple[str, str], ...] = (
    ("ASBA", "Application Supported by Blocked Amount - funds are blocked in the bank account, not debited, until allotment."),
    ("GMP", "Grey Market Premium - an unofficial, unregulated quote for the shares before listing."),
    ("LTV", "Loan-to-value: the percentage of the fixed deposit a bank will lend against."),
    ("OD", "Overdraft - a revolving credit line, here secured against a fixed deposit."),
    ("sNII / bNII", "Small (Rs 2-10 lakh) and big (above Rs 10 lakh) non-institutional investor categories."),
    ("STCG / LTCG", "Short- and long-term capital gains on listed equity."),
    ("Cut-off price", "Retail bids at the eventual issue price, which is what this model assumes."),
)


def explanations_frame() -> pd.DataFrame:
    """Tabular form of :data:`EXPLANATIONS` for export."""
    return pd.DataFrame(
        [
            {
                "Metric": e.metric,
                "Formula": e.formula,
                "Inputs used": e.inputs_used,
                "Interpretation": e.interpretation,
                "Limitations": e.limitations,
            }
            for e in EXPLANATIONS
        ]
    )
