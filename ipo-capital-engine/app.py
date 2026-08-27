"""Streamlit dashboard for the IPO capital allocation and financing engine.

This module is presentation only. Every number shown here is produced by
:mod:`calculations`, :mod:`scenarios` and :mod:`risk`, all of which run without
Streamlit, so the analysis can be reproduced in a notebook or a batch job.

Run with::

    streamlit run app.py
"""

from __future__ import annotations

import math
import traceback
from collections.abc import Sequence
from dataclasses import replace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from calculations import (
    AnalysisInputs,
    AnalysisResult,
    ApplicationAccount,
    FinancingAssumptions,
    FundingMode,
    GMPMode,
    IPOAssumptions,
    IPOCategory,
    Provenance,
    TaxAssumptions,
    TransactionCostAssumptions,
    analyze,
    annualize_simple,
    assumption_ledger,
    expected_net_profit,
)
from example_data import load_examples
from explanations import EXPLANATIONS, GLOSSARY
from export import build_report, bundle_to_csv, bundle_to_excel, bundle_to_pdf
from risk import (
    DISCLAIMER,
    DecisionOutcome,
    DecisionThresholds,
    RiskMetrics,
    Verdict,
    compare_opportunities,
    compute_risk_metrics,
    evaluate_decision,
)
from scenarios import (
    DEFAULT_BEAR,
    DEFAULT_BULL,
    MonteCarloConfig,
    ScenarioDefinition,
    ScenarioResult,
    run_monte_carlo,
    run_scenarios,
    scenarios_to_frame,
    sensitivity_gmp_vs_probability,
    sensitivity_od_rate_vs_listing_gain,
)
from validation import ValidationReport, validate_inputs
from version import BETA_NOTICE, IS_PRERELEASE, RELEASE_NAME

ISSUE_URL = "https://github.com/thakerswapnil24-gif/IPO-Financing/issues/new/choose"

st.set_page_config(
    page_title=f"IPO Capital Allocation & Financing Engine {RELEASE_NAME}",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="auto",
)

#: Plotly's floating mode bar overlaps an in-figure title on a narrow screen, so
#: titles live in the page (see render_chart) and the bar is trimmed to the
#: controls that are actually useful here.
PLOTLY_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "modeBarButtonsToRemove": [
        "select2d",
        "lasso2d",
        "zoomIn2d",
        "zoomOut2d",
        "autoScale2d",
    ],
}

POSITIVE = "#1a7f5a"
NEGATIVE = "#b3261e"
NEUTRAL = "#5b6b7b"
ACCENT = "#1f4e79"
VERDICT_STYLE = {
    Verdict.GO: ("✅", POSITIVE),
    Verdict.BORDERLINE: ("⚠️", "#b26a00"),
    Verdict.NO_GO: ("⛔", NEGATIVE),
}


# ---------------------------------------------------------------------------
# Formatting helpers (rounding happens here and nowhere else)
# ---------------------------------------------------------------------------
def inr(value: float | None, decimals: int = 0) -> str:
    """Format a number in the Indian digit grouping with a rupee sign."""
    if value is None or (
        isinstance(value, float) and (math.isnan(value) or math.isinf(value))
    ):
        return "n/a"
    sign = "-" if value < 0 else ""
    magnitude = abs(float(value))
    whole = int(magnitude)
    fraction = magnitude - whole
    digits = str(whole)
    if len(digits) > 3:
        head, tail = digits[:-3], digits[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        digits = ",".join([*groups, tail])
    if decimals:
        digits += f"{fraction:.{decimals}f}"[1:]
    return f"{sign}Rs {digits}"


def compact_inr(value: float | None) -> str:
    """Rupees in lakh/crore units for KPI cards."""
    if value is None or (
        isinstance(value, float) and (math.isnan(value) or math.isinf(value))
    ):
        return "n/a"
    magnitude = abs(float(value))
    sign = "-" if value < 0 else ""
    if magnitude >= 1e7:
        return f"{sign}Rs {magnitude / 1e7:,.2f} Cr"
    if magnitude >= 1e5:
        return f"{sign}Rs {magnitude / 1e5:,.2f} L"
    return inr(value)


def pct(value: float | None, decimals: int = 2) -> str:
    """Format a fraction (0.05 -> '5.00%')."""
    if value is None or (
        isinstance(value, float) and (math.isnan(value) or math.isinf(value))
    ):
        return "n/a"
    return f"{value * 100:.{decimals}f}%"


def pct_points(value: float | None, decimals: int = 2) -> str:
    """Format a number already expressed in percent (10.5 -> '10.50%')."""
    if value is None or (
        isinstance(value, float) and (math.isnan(value) or math.isinf(value))
    ):
        return "n/a"
    return f"{value:.{decimals}f}%"


# ---------------------------------------------------------------------------
# Session state and presets
# ---------------------------------------------------------------------------
ACCOUNT_COLUMNS = [
    "Account",
    "Category",
    "Lots applied",
    "Allotment probability %",
    "Lots if allotted",
]


def accounts_to_frame(accounts: Sequence[ApplicationAccount]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Account": a.label,
                "Category": a.category.value,
                "Lots applied": int(a.lots_applied),
                "Allotment probability %": float(a.allotment_probability * 100.0),
                "Lots if allotted": float(a.lots_allotted_if_successful),
            }
            for a in accounts
        ],
        columns=ACCOUNT_COLUMNS,
    )


def frame_to_accounts(frame: pd.DataFrame) -> list[ApplicationAccount]:
    accounts: list[ApplicationAccount] = []
    for index, row in frame.iterrows():
        label = str(row.get("Account") or f"Account {index + 1}").strip()
        try:
            category = IPOCategory(str(row.get("Category", IPOCategory.RETAIL.value)))
        except ValueError:
            category = IPOCategory.RETAIL
        accounts.append(
            ApplicationAccount(
                label=label or f"Account {index + 1}",
                category=category,
                lots_applied=int(row.get("Lots applied") or 0),
                allotment_probability=float(row.get("Allotment probability %") or 0.0)
                / 100.0,
                lots_allotted_if_successful=float(row.get("Lots if allotted") or 0.0),
            )
        )
    return accounts


def apply_preset(inputs: AnalysisInputs) -> None:
    """Push a preset into every widget's session state, then force a rerun."""
    ipo, fin, costs, taxes = inputs.ipo, inputs.financing, inputs.costs, inputs.taxes
    state = st.session_state
    state.update(
        {
            "ipo_name": ipo.name,
            "issue_price": float(ipo.issue_price),
            "lot_size": int(ipo.lot_size),
            "gmp_mode": ipo.gmp_mode.value,
            "gmp_value": float(ipo.gmp_value),
            "use_gmp": bool(ipo.use_gmp_for_listing),
            "listing_override": float(
                ipo.expected_listing_price_override
                if ipo.expected_listing_price_override is not None
                else ipo.expected_listing_price
            ),
            "exit_differs": ipo.expected_exit_price_override is not None,
            "exit_override": float(ipo.expected_exit_price),
            "holding_days": int(ipo.holding_period_days),
            "funding_mode": fin.funding_mode.value,
            "own_available": float(fin.own_capital_available),
            "own_deployed": float(fin.own_capital_deployed),
            "fd_amount": float(fin.fd_amount),
            "fd_rate": float(fin.fd_rate_pct),
            "od_ltv": float(fin.od_ltv_pct),
            "od_rate": float(fin.od_rate_pct),
            "processing_fee": float(fin.processing_fee),
            "other_financing": float(fin.other_financing_charges),
            "days_blocked": int(fin.days_blocked),
            "opp_rate": float(fin.opportunity_cost_rate_pct),
            "include_opp": bool(fin.include_opportunity_cost),
            "count_fd": bool(fin.count_fd_interest_as_income),
            "finance_hold": bool(fin.finance_holding_period),
            "day_basis": int(fin.day_count_basis),
            "brokerage_pct_sell": float(costs.brokerage_pct_sell),
            "brokerage_flat_sell": float(costs.brokerage_flat_sell),
            "brokerage_flat_buy": float(costs.brokerage_flat_buy),
            "stt_sell": float(costs.stt_pct_sell),
            "stt_buy": float(costs.stt_pct_buy),
            "exchange_pct": float(costs.exchange_txn_pct),
            "sebi_pct": float(costs.sebi_turnover_pct),
            "stamp_pct": float(costs.stamp_duty_pct_buy),
            "gst_pct": float(costs.gst_pct),
            "dp_charges": float(costs.dp_charges_flat_sell),
            "other_charges": float(costs.other_charges_flat),
            "stcg": float(taxes.stcg_rate_pct),
            "ltcg": float(taxes.ltcg_rate_pct),
            "ltcg_days": int(taxes.ltcg_threshold_days),
            "cess": float(taxes.cess_and_surcharge_pct),
            "apply_exemption": bool(taxes.apply_ltcg_exemption),
            "exemption_amount": float(taxes.ltcg_exemption_amount),
            "deduct_costs": bool(taxes.deduct_transaction_costs_from_gain),
            "tax_shield": bool(taxes.recognise_tax_shield_on_loss),
            "independent": bool(inputs.assume_independent_allotments),
            "accounts_df": accounts_to_frame(inputs.accounts),
        }
    )
    state["editor_version"] = state.get("editor_version", 0) + 1


def init_state() -> None:
    if "initialised" not in st.session_state:
        st.session_state["portfolio"] = []
        st.session_state["editor_version"] = 0
        apply_preset(load_examples()[0].inputs)
        st.session_state["initialised"] = True


# ---------------------------------------------------------------------------
# Sidebar: every assumption the model uses
# ---------------------------------------------------------------------------
def sidebar() -> tuple[
    AnalysisInputs, DecisionThresholds, MonteCarloConfig, dict[str, ScenarioDefinition]
]:
    """Collect every input. Nothing is assumed that is not shown here."""
    st.sidebar.title("Assumptions")
    st.sidebar.caption(
        "Everything below is an input you own. Defaults are starting points, "
        "not research."
    )

    presets = load_examples()
    with st.sidebar.expander("Load an example", expanded=False):
        names = [p.name for p in presets]
        choice = st.selectbox("Example dataset", names, key="preset_choice")
        preset = presets[names.index(choice)]
        st.caption(preset.notes)
        if st.button("Load this example", width="stretch"):
            apply_preset(preset.inputs)
            st.rerun()

    # -- IPO ---------------------------------------------------------------
    with st.sidebar.expander("1. IPO assumptions", expanded=True):
        name = st.text_input("IPO name", key="ipo_name")
        issue_price = st.number_input(
            "Issue price (Rs/share)", min_value=0.0, step=1.0, key="issue_price"
        )
        lot_size = st.number_input(
            "Lot size (shares)", min_value=1, step=1, key="lot_size"
        )
        st.caption(f"One lot = {inr(issue_price * lot_size)} at the cut-off price.")

        gmp_mode = st.radio(
            "Express GMP as",
            [GMPMode.ABSOLUTE.value, GMPMode.PERCENT.value],
            key="gmp_mode",
            horizontal=True,
        )
        gmp_value = st.number_input(
            "GMP (Rs/share)"
            if gmp_mode == GMPMode.ABSOLUTE.value
            else "GMP (% of issue price)",
            step=1.0,
            key="gmp_value",
        )
        use_gmp = st.checkbox(
            "Derive the expected listing price from GMP", key="use_gmp"
        )
        listing_override = st.number_input(
            "Expected listing price (Rs)",
            min_value=0.0,
            step=1.0,
            key="listing_override",
            disabled=use_gmp,
            help="Used only when the GMP-derived price is switched off.",
        )
        st.caption(
            "GMP is an unregulated grey-market quote. It is an assumption about "
            "sentiment, never a guaranteed listing price."
        )
        exit_differs = st.checkbox(
            "Exit at a price other than the listing price", key="exit_differs"
        )
        exit_override = st.number_input(
            "Expected exit price (Rs)",
            min_value=0.0,
            step=1.0,
            key="exit_override",
            disabled=not exit_differs,
        )
        holding_days = st.number_input(
            "Holding period after allotment (days)",
            min_value=0,
            step=1,
            key="holding_days",
            help=(
                "0 = sell on listing day. Drives both the carry cost and the tax rate."
            ),
        )

    ipo = IPOAssumptions(
        name=name,
        issue_price=float(issue_price),
        lot_size=int(lot_size),
        gmp_value=float(gmp_value),
        gmp_mode=GMPMode(gmp_mode),
        use_gmp_for_listing=bool(use_gmp),
        expected_listing_price_override=None if use_gmp else float(listing_override),
        expected_exit_price_override=float(exit_override) if exit_differs else None,
        holding_period_days=int(holding_days),
    )
    st.sidebar.metric(
        "Expected listing price",
        inr(ipo.expected_listing_price, 2),
        f"{ipo.expected_listing_gain_pct:.2f}% vs issue price",
    )

    # -- Accounts ----------------------------------------------------------
    with st.sidebar.expander("2. Accounts / PAN structure", expanded=True):
        st.caption(
            "One row per eligible application. Probabilities are per account and "
            "are treated as independent draws."
        )
        edited = st.data_editor(
            st.session_state["accounts_df"],
            num_rows="dynamic",
            width="stretch",
            key=f"accounts_editor_{st.session_state['editor_version']}",
            column_config={
                "Account": st.column_config.TextColumn("Account", required=True),
                "Category": st.column_config.SelectboxColumn(
                    "Category", options=[c.value for c in IPOCategory], required=True
                ),
                "Lots applied": st.column_config.NumberColumn(
                    "Lots applied", min_value=1, step=1, format="%d"
                ),
                "Allotment probability %": st.column_config.NumberColumn(
                    "Allotment prob %",
                    min_value=0.0,
                    max_value=100.0,
                    step=1.0,
                    format="%.1f",
                ),
                "Lots if allotted": st.column_config.NumberColumn(
                    "Lots if allotted",
                    min_value=0.0,
                    step=0.5,
                    format="%.2f",
                    help="1.0 for an oversubscribed retail lottery; fractional values "
                    "represent an expected proportionate NII allotment.",
                ),
            },
        )
        st.session_state["accounts_df"] = edited
        independent = st.checkbox(
            "Assume independent allotment draws across accounts",
            key="independent",
            help="Required for P(no allotment) = product of (1 - p). Switch off to "
            "flag the distribution as indicative only.",
        )

    accounts = frame_to_accounts(edited)
    if not accounts:
        accounts = [ApplicationAccount()]
    total_application = sum(
        a.application_amount(ipo.issue_price, ipo.lot_size) for a in accounts
    )
    st.sidebar.metric("Total application amount", compact_inr(total_application))

    # -- Financing ---------------------------------------------------------
    with st.sidebar.expander("3. Financing assumptions", expanded=True):
        funding_mode = st.selectbox(
            "Funding mode", [m.value for m in FundingMode], key="funding_mode"
        )
        own_available = st.number_input(
            "Own capital available (Rs)",
            min_value=0.0,
            step=10_000.0,
            key="own_available",
        )
        own_deployed = st.number_input(
            "Own capital to deploy (Rs)",
            min_value=0.0,
            step=10_000.0,
            key="own_deployed",
            disabled=FundingMode(funding_mode) is not FundingMode.MIXED,
            help="Used in mixed mode only; the other modes derive the split.",
        )
        fd_amount = st.number_input(
            "FD amount pledged (Rs)", min_value=0.0, step=10_000.0, key="fd_amount"
        )
        fd_rate = st.number_input(
            "FD interest rate (% p.a.)", min_value=0.0, step=0.25, key="fd_rate"
        )
        od_ltv = st.number_input(
            "OD limit against FD (% LTV)",
            min_value=0.0,
            max_value=100.0,
            step=5.0,
            key="od_ltv",
        )
        od_rate = st.number_input(
            "OD interest rate (% p.a.)", min_value=0.0, step=0.25, key="od_rate"
        )
        st.caption(f"Sanctioned OD limit: {compact_inr(fd_amount * od_ltv / 100.0)}")
        processing_fee = st.number_input(
            "Processing fee (Rs, one-time)",
            min_value=0.0,
            step=100.0,
            key="processing_fee",
        )
        other_financing = st.number_input(
            "Other financing charges (Rs)",
            min_value=0.0,
            step=100.0,
            key="other_financing",
        )
        days_blocked = st.number_input(
            "Days capital is blocked",
            min_value=0,
            step=1,
            key="days_blocked",
            help="Application date to refund/allotment. Typically 5-7 days on the "
            "T+3 timetable.",
        )
        opp_rate = st.number_input(
            "Opportunity cost of own capital (% p.a.)",
            min_value=0.0,
            step=0.25,
            key="opp_rate",
        )
        include_opp = st.checkbox(
            "Charge opportunity cost on own capital", key="include_opp"
        )
        finance_hold = st.checkbox(
            "Keep the OD outstanding through the holding period", key="finance_hold"
        )
        count_fd = st.checkbox(
            "Count FD interest as income of this strategy",
            key="count_fd",
            help="Off by default: a pledged FD earns its interest whether or not you "
            "bid, so it is not a benefit of the strategy.",
        )
        day_basis = st.selectbox("Day-count basis", [365, 360, 366], key="day_basis")

    financing = FinancingAssumptions(
        funding_mode=FundingMode(funding_mode),
        own_capital_available=float(own_available),
        own_capital_deployed=float(own_deployed),
        fd_amount=float(fd_amount),
        fd_rate_pct=float(fd_rate),
        od_ltv_pct=float(od_ltv),
        od_rate_pct=float(od_rate),
        processing_fee=float(processing_fee),
        other_financing_charges=float(other_financing),
        days_blocked=int(days_blocked),
        opportunity_cost_rate_pct=float(opp_rate),
        include_opportunity_cost=bool(include_opp),
        count_fd_interest_as_income=bool(count_fd),
        finance_holding_period=bool(finance_hold),
        day_count_basis=int(day_basis),
    )

    # -- Costs and taxes ---------------------------------------------------
    with st.sidebar.expander("4. Transaction costs", expanded=False):
        st.caption("Broker and statutory charges. Verify against your contract note.")
        brokerage_pct_sell = st.number_input(
            "Brokerage % of sell value",
            min_value=0.0,
            step=0.01,
            format="%.4f",
            key="brokerage_pct_sell",
        )
        brokerage_flat_sell = st.number_input(
            "Brokerage flat per sell order (Rs)",
            min_value=0.0,
            step=1.0,
            key="brokerage_flat_sell",
        )
        brokerage_flat_buy = st.number_input(
            "Brokerage flat on allotment (Rs)",
            min_value=0.0,
            step=1.0,
            key="brokerage_flat_buy",
        )
        stt_sell = st.number_input(
            "STT % on sell", min_value=0.0, step=0.01, format="%.4f", key="stt_sell"
        )
        stt_buy = st.number_input(
            "STT % on allotment",
            min_value=0.0,
            step=0.01,
            format="%.4f",
            key="stt_buy",
            help="Primary-market allotment normally attracts no STT.",
        )
        exchange_pct = st.number_input(
            "Exchange transaction charges %",
            min_value=0.0,
            step=0.0001,
            format="%.5f",
            key="exchange_pct",
        )
        sebi_pct = st.number_input(
            "SEBI turnover fees %",
            min_value=0.0,
            step=0.0001,
            format="%.5f",
            key="sebi_pct",
        )
        stamp_pct = st.number_input(
            "Stamp duty % on allotment",
            min_value=0.0,
            step=0.001,
            format="%.4f",
            key="stamp_pct",
        )
        gst_pct_value = st.number_input(
            "GST % on brokerage and charges", min_value=0.0, step=1.0, key="gst_pct"
        )
        dp_charges = st.number_input(
            "DP charges per sell (Rs)", min_value=0.0, step=1.0, key="dp_charges"
        )
        other_charges = st.number_input(
            "Other charges (Rs)", min_value=0.0, step=1.0, key="other_charges"
        )

    costs = TransactionCostAssumptions(
        brokerage_pct_sell=float(brokerage_pct_sell),
        brokerage_flat_sell=float(brokerage_flat_sell),
        brokerage_flat_buy=float(brokerage_flat_buy),
        stt_pct_sell=float(stt_sell),
        stt_pct_buy=float(stt_buy),
        exchange_txn_pct=float(exchange_pct),
        sebi_turnover_pct=float(sebi_pct),
        stamp_duty_pct_buy=float(stamp_pct),
        gst_pct=float(gst_pct_value),
        dp_charges_flat_sell=float(dp_charges),
        other_charges_flat=float(other_charges),
    )

    with st.sidebar.expander("5. Taxes", expanded=False):
        st.caption("Rates are configurable assumptions, not embedded law.")
        stcg = st.number_input(
            "Short-term capital gains rate %", min_value=0.0, step=0.5, key="stcg"
        )
        ltcg = st.number_input(
            "Long-term capital gains rate %", min_value=0.0, step=0.5, key="ltcg"
        )
        ltcg_days = st.number_input(
            "Long-term threshold (days)", min_value=1, step=1, key="ltcg_days"
        )
        cess = st.number_input(
            "Cess and surcharge %", min_value=0.0, step=0.5, key="cess"
        )
        deduct_costs = st.checkbox(
            "Deduct transfer costs from the taxable gain", key="deduct_costs"
        )
        apply_exemption = st.checkbox(
            "Apply the annual LTCG exemption", key="apply_exemption"
        )
        exemption_amount = st.number_input(
            "LTCG exemption (Rs)",
            min_value=0.0,
            step=25_000.0,
            key="exemption_amount",
            disabled=not apply_exemption,
        )
        tax_shield = st.checkbox(
            "Recognise a tax shield on losses",
            key="tax_shield",
            help="Assumes you have other realised gains to set the loss against.",
        )

    taxes = TaxAssumptions(
        stcg_rate_pct=float(stcg),
        ltcg_rate_pct=float(ltcg),
        ltcg_threshold_days=int(ltcg_days),
        cess_and_surcharge_pct=float(cess),
        apply_ltcg_exemption=bool(apply_exemption),
        ltcg_exemption_amount=float(exemption_amount),
        deduct_transaction_costs_from_gain=bool(deduct_costs),
        recognise_tax_shield_on_loss=bool(tax_shield),
    )

    # -- Scenario, threshold and simulation settings -----------------------
    with st.sidebar.expander("6. Scenario definitions", expanded=False):
        st.caption("Scenario factors are editable; nothing here is a forecast.")
        bear_gmp = st.slider(
            "Bear: GMP multiplier",
            -2.0,
            1.0,
            float(DEFAULT_BEAR.gmp_multiplier),
            0.1,
            key="bear_gmp",
        )
        bear_prob = st.slider(
            "Bear: hit-rate multiplier",
            0.0,
            1.0,
            float(DEFAULT_BEAR.allotment_probability_multiplier),
            0.05,
            key="bear_prob",
        )
        bull_gmp = st.slider(
            "Bull: GMP multiplier",
            1.0,
            3.0,
            float(DEFAULT_BULL.gmp_multiplier),
            0.1,
            key="bull_gmp",
        )
        bull_prob = st.slider(
            "Bull: hit-rate multiplier",
            1.0,
            3.0,
            float(DEFAULT_BULL.allotment_probability_multiplier),
            0.05,
            key="bull_prob",
        )

    scenario_definitions = {
        "bear": ScenarioDefinition(
            name="Bear",
            gmp_multiplier=bear_gmp,
            allotment_probability_multiplier=bear_prob,
            description=(
                f"GMP x {bear_gmp:g} and allotment hit-rate x {bear_prob:g} "
                "relative to your base case."
            ),
        ),
        "base": ScenarioDefinition(
            name="Base", description="Your own assumptions, unchanged."
        ),
        "bull": ScenarioDefinition(
            name="Bull",
            gmp_multiplier=bull_gmp,
            allotment_probability_multiplier=bull_prob,
            description=(
                f"GMP x {bull_gmp:g} and allotment hit-rate x {bull_prob:g} "
                "relative to your base case."
            ),
        ),
    }

    with st.sidebar.expander("7. Decision thresholds", expanded=False):
        st.caption("Policy choices that turn metrics into a verdict.")
        spread = st.number_input(
            "Required annualised spread over the OD rate (pp)",
            min_value=0.0,
            step=1.0,
            value=5.0,
        )
        max_fin_share = st.slider(
            "Max financing cost / gross profit", 0.0, 1.0, 0.60, 0.05
        )
        max_p_loss = st.slider("Max probability of loss", 0.0, 1.0, 0.60, 0.05)
        min_gmp_mos = st.slider("Min GMP margin of safety", 0.0, 1.0, 0.30, 0.05)
        min_prob_mos = st.slider("Min hit-rate margin of safety", 0.0, 1.0, 0.30, 0.05)
        min_headroom = st.number_input(
            "Min OD-rate headroom (pp)", min_value=0.0, step=0.5, value=2.0
        )
        max_bear = st.slider("Max bear-case loss / own equity", 0.0, 0.5, 0.05, 0.01)
        max_be_gain = st.number_input(
            "Max plausible break-even listing gain %",
            min_value=0.0,
            step=1.0,
            value=15.0,
        )

    thresholds = DecisionThresholds(
        min_annualised_spread_over_od_pct=float(spread),
        max_financing_share_of_gross=float(max_fin_share),
        max_probability_of_loss=float(max_p_loss),
        min_gmp_margin_of_safety=float(min_gmp_mos),
        min_probability_margin_of_safety=float(min_prob_mos),
        min_od_rate_headroom_pct=float(min_headroom),
        max_bear_loss_share_of_equity=float(max_bear),
        max_plausible_breakeven_listing_gain_pct=float(max_be_gain),
    )

    with st.sidebar.expander("8. Monte Carlo settings", expanded=False):
        n_simulations = st.select_slider(
            "Simulations", options=[10_000, 25_000, 50_000, 100_000], value=10_000
        )
        seed = st.number_input("Random seed", min_value=0, step=1, value=42)
        gain_distribution = st.selectbox(
            "Listing gain distribution", ["normal", "triangular", "uniform", "fixed"]
        )
        gain_std = st.number_input(
            "Listing gain standard deviation (pp)", min_value=0.0, step=1.0, value=15.0
        )
        gain_low, gain_high = st.slider(
            "Listing gain range for triangular/uniform (%)",
            -60.0,
            120.0,
            (-20.0, 40.0),
            5.0,
        )
        probability_distribution = st.selectbox(
            "Allotment probability distribution", ["fixed", "beta"]
        )
        concentration = st.number_input(
            "Beta concentration (higher = more certain)",
            min_value=1.0,
            step=1.0,
            value=20.0,
        )
        holding_distribution = st.selectbox(
            "Holding period distribution", ["fixed", "uniform_int"]
        )
        holding_low, holding_high = st.slider(
            "Holding period range (days)", 0, 60, (1, 5)
        )
        od_distribution = st.selectbox(
            "OD rate distribution", ["fixed", "normal", "uniform"]
        )
        od_std = st.number_input(
            "OD rate standard deviation (pp)", min_value=0.0, step=0.25, value=1.0
        )
        od_low, od_high = st.slider("OD rate range (%)", 0.0, 30.0, (9.0, 13.0), 0.5)

    monte_carlo_config = MonteCarloConfig(
        n_simulations=int(n_simulations),
        seed=int(seed),
        gain_distribution=gain_distribution,
        gain_std_pct=float(gain_std),
        gain_low_pct=float(gain_low),
        gain_high_pct=float(gain_high),
        probability_distribution=probability_distribution,
        probability_concentration=float(concentration),
        holding_distribution=holding_distribution,
        holding_low_days=int(holding_low),
        holding_high_days=int(holding_high),
        od_rate_distribution=od_distribution,
        od_rate_std_pct=float(od_std),
        od_rate_low_pct=float(od_low),
        od_rate_high_pct=float(od_high),
    )

    inputs = AnalysisInputs(
        ipo=ipo,
        accounts=tuple(accounts),
        financing=financing,
        costs=costs,
        taxes=taxes,
        assume_independent_allotments=bool(independent),
    )
    return inputs, thresholds, monte_carlo_config, scenario_definitions


# ---------------------------------------------------------------------------
# Cached computation wrappers
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def cached_analysis(inputs: AnalysisInputs) -> AnalysisResult:
    return analyze(inputs)


@st.cache_data(show_spinner=False)
def cached_risk(result: AnalysisResult, bear: ScenarioDefinition) -> RiskMetrics:
    return compute_risk_metrics(result, bear)


@st.cache_data(show_spinner=False)
def cached_scenarios(
    inputs: AnalysisInputs, definitions: tuple[ScenarioDefinition, ...]
) -> list[ScenarioResult]:
    return run_scenarios(inputs, definitions)


@st.cache_data(show_spinner="Building sensitivity grid...")
def cached_gmp_grid(
    inputs: AnalysisInputs, gmps: tuple[float, ...], probabilities: tuple[float, ...]
) -> pd.DataFrame:
    return sensitivity_gmp_vs_probability(inputs, gmps, probabilities)


@st.cache_data(show_spinner="Building sensitivity grid...")
def cached_rate_grid(
    inputs: AnalysisInputs, rates: tuple[float, ...], gains: tuple[float, ...]
) -> pd.DataFrame:
    return sensitivity_od_rate_vs_listing_gain(inputs, rates, gains)


@st.cache_data(show_spinner="Running simulations...")
def cached_monte_carlo(inputs: AnalysisInputs, config: MonteCarloConfig):
    return run_monte_carlo(inputs, config)


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def profit_waterfall(result: AnalysisResult) -> go.Figure:
    """Gross profit down to net profit, one bar per drag on the return."""
    # Short axis labels: the full wording is on hover, because a phone-width
    # axis turns long category names into unreadable diagonal text.
    labels = ["Gross profit", "Costs", "Taxes", "Financing"]
    full_labels = [
        "Expected gross profit",
        "Transaction costs",
        "Taxes",
        "Financing cost",
    ]
    values = [
        result.expected_gross_profit,
        -result.expected_transaction_costs,
        -result.expected_taxes,
        -result.expected_financing_cost,
    ]
    measures = ["absolute", "relative", "relative", "relative"]
    if result.financing.fd_interest_credit:
        labels.append("FD interest")
        full_labels.append("FD interest counted")
        values.append(result.financing.fd_interest_credit)
        measures.append("relative")
    if result.inputs.financing.include_opportunity_cost:
        labels.append("Opportunity")
        full_labels.append("Opportunity cost")
        values.append(-result.expected_opportunity_cost)
        measures.append("relative")
    labels.append("Net profit")
    full_labels.append("Expected net profit")
    values.append(0.0)
    measures.append("total")

    figure = go.Figure(
        go.Waterfall(
            orientation="v",
            measure=measures,
            x=labels,
            y=values,
            text=[
                inr(v) if m != "total" else inr(result.expected_net_profit)
                for v, m in zip(values, measures, strict=True)
            ],
            textposition="outside",
            cliponaxis=False,
            customdata=full_labels,
            hovertemplate="%{customdata}: %{y:,.0f}<extra></extra>",
            connector={"line": {"color": NEUTRAL}},
            increasing={"marker": {"color": POSITIVE}},
            decreasing={"marker": {"color": NEGATIVE}},
            totals={"marker": {"color": ACCENT}},
        )
    )
    figure.update_layout(
        yaxis_title="Rupees",
        margin={"t": 30, "b": 80, "l": 10, "r": 10},
        height=420,
    )
    return figure


def break_even_curve(inputs: AnalysisInputs, result: AnalysisResult) -> go.Figure:
    """Expected net profit as a function of the realised exit price."""
    issue = inputs.ipo.issue_price
    expected_exit = inputs.ipo.expected_exit_price
    high = max(expected_exit * 1.4, issue * 1.4)
    prices = np.linspace(max(issue * 0.6, 0.0), high, 60)
    profits = [
        expected_net_profit(
            replace(
                inputs,
                ipo=replace(inputs.ipo, expected_exit_price_override=float(price)),
            )
        )
        for price in prices
    ]
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=prices,
            y=profits,
            mode="lines",
            name="Expected net profit",
            line={"color": ACCENT, "width": 3},
        )
    )
    figure.add_hline(y=0, line_dash="dot", line_color=NEUTRAL)
    figure.add_vline(
        x=issue,
        line_dash="dot",
        line_color=NEUTRAL,
        annotation_text="Issue price",
        annotation_position="top left",
    )
    break_even = result.break_even.exit_price_expected_value
    if break_even is not None:
        figure.add_vline(
            x=break_even,
            line_dash="dash",
            line_color=NEGATIVE,
            annotation_text=f"Break-even {inr(break_even, 2)}",
            annotation_position="top right",
        )
    figure.add_trace(
        go.Scatter(
            x=[expected_exit],
            y=[result.expected_net_profit],
            mode="markers",
            name="Your assumption",
            marker={
                "size": 12,
                "color": POSITIVE if result.expected_net_profit > 0 else NEGATIVE,
            },
        )
    )
    figure.update_layout(
        xaxis_title="Exit price (Rs)",
        yaxis_title="Expected net profit (Rs)",
        height=420,
        margin={"t": 30, "b": 60, "l": 10, "r": 10},
    )
    return figure


def heatmap(
    frame: pd.DataFrame,
    x_title: str,
    y_title: str,
    x_format: str = "{:.0%}",
) -> go.Figure:
    """Conditional-formatted sensitivity grid centred on break-even."""
    z = frame.to_numpy(dtype=float)
    limit = float(np.nanmax(np.abs(z))) or 1.0
    figure = go.Figure(
        go.Heatmap(
            z=z,
            x=[x_format.format(c) for c in frame.columns],
            y=[f"{r:,.2f}" for r in frame.index],
            colorscale="RdYlGn",
            zmid=0.0,
            zmin=-limit,
            zmax=limit,
            text=[[inr(value) for value in row] for row in z],
            texttemplate="%{text}",
            textfont={"size": 11},
            colorbar={"title": "Net profit"},
            hovertemplate=(
                f"{y_title}: %{{y}}<br>{x_title}: %{{x}}<br>Expected net profit: "
                f"%{{text}}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        xaxis_title=x_title,
        yaxis_title=y_title,
        height=460,
        margin={"t": 30, "b": 60, "l": 10, "r": 10},
    )
    return figure


def monte_carlo_chart(simulation) -> go.Figure:
    """Profit distribution with the loss region shaded and percentiles marked."""
    profits = simulation.profits
    figure = go.Figure()
    figure.add_trace(
        go.Histogram(
            x=profits,
            nbinsx=80,
            marker={"color": ACCENT},
            name="Simulated outcomes",
            hovertemplate="Net profit: %{x}<br>Paths: %{y}<extra></extra>",
        )
    )
    figure.add_vline(
        x=0, line_color=NEGATIVE, line_dash="dot", annotation_text="Break-even"
    )
    for quantile, colour in ((5, NEGATIVE), (50, "#b26a00"), (95, POSITIVE)):
        figure.add_vline(
            x=simulation.percentiles[quantile],
            line_color=colour,
            line_dash="dash",
            annotation_text=f"P{quantile}",
            annotation_position="top",
        )
    figure.update_layout(
        xaxis_title="Net profit (Rs)",
        yaxis_title="Number of simulated paths",
        height=440,
        margin={"t": 30, "b": 60, "l": 10, "r": 10},
        showlegend=False,
    )
    return figure


def outcome_distribution_chart(risk: RiskMetrics) -> go.Figure:
    """Exact discrete distribution of profit across allotment outcomes."""
    frame = risk.distribution.to_frame()
    colours = [POSITIVE if value > 0 else NEGATIVE for value in frame["Net profit"]]
    figure = go.Figure(
        go.Bar(
            x=[inr(value) for value in frame["Net profit"]],
            y=frame["Probability"],
            marker_color=colours,
            text=[f"{p:.1%}" for p in frame["Probability"]],
            textposition="outside",
        )
    )
    figure.update_layout(
        xaxis_title="Net profit",
        yaxis_title="Probability",
        yaxis_tickformat=".0%",
        height=400,
        margin={"t": 30, "b": 60, "l": 10, "r": 10},
    )
    return figure


def scenario_chart(scenarios: Sequence[ScenarioResult]) -> go.Figure:
    """Net profit by scenario, split into its gross and cost components."""
    names = [s.name for s in scenarios]
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            name="Expected gross profit",
            x=names,
            y=[s.result.expected_gross_profit for s in scenarios],
            marker_color="#9dc3e6",
        )
    )
    figure.add_trace(
        go.Bar(
            name="Costs, taxes and financing",
            x=names,
            y=[
                -(
                    s.result.expected_transaction_costs
                    + s.result.expected_taxes
                    + s.result.expected_financing_cost
                    + s.result.expected_opportunity_cost
                )
                for s in scenarios
            ],
            marker_color="#f2a6a2",
        )
    )
    figure.add_trace(
        go.Scatter(
            name="Expected net profit",
            x=names,
            y=[s.result.expected_net_profit for s in scenarios],
            mode="markers+text",
            marker={"size": 14, "color": ACCENT, "symbol": "diamond"},
            text=[inr(s.result.expected_net_profit) for s in scenarios],
            textposition="top center",
        )
    )
    figure.update_layout(
        barmode="relative",
        yaxis_title="Rupees",
        height=420,
        margin={"t": 30, "b": 60, "l": 10, "r": 10},
    )
    return figure


#: Responsive rules. Streamlit stacks its columns below roughly 768px, but a
#: phone whose browser is in desktop-site mode reports a layout viewport near
#: 980px, so the dashboard arrives as a squeezed desktop layout instead. These
#: rules stack it on any narrow-ish viewport, and they are keyed to Streamlit's
#: test ids: if a future release renames one, the rule stops matching and the
#: layout falls back to Streamlit's own behaviour rather than breaking.
RESPONSIVE_CSS = """
<style>
/* Streamlit floats each element's hover toolbar about 42px *above* the element,
   where it lands on the heading before it. Pull it inside its own element. */
[data-testid="stElementToolbar"] { top: 0.25rem !important; }

@media (max-width: 992px) {
    /* One column per row: a two-up table and chart are both unreadable once
       the viewport is this narrow. */
    [data-testid="stHorizontalBlock"] { flex-wrap: wrap !important; }
    [data-testid="stColumn"] {
        flex: 1 1 100% !important;
        min-width: 100% !important;
    }
    /* KPI cards are the exception - short values stay scannable two-up, and it
       halves the scrolling needed to see all eight. */
    .st-key-kpi_cards [data-testid="stColumn"] {
        flex: 1 1 47% !important;
        min-width: 47% !important;
    }
    /* Reclaim the wide-screen page gutters. */
    .block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        padding-top: 2.5rem !important;
    }
    /* The tab strip scrolls horizontally; make that obvious and thumb-friendly. */
    [data-testid="stTabs"] [role="tablist"] { overflow-x: auto; scrollbar-width: thin; }
    [data-testid="stTabs"] [role="tab"] { white-space: nowrap; }
}

@media (max-width: 480px) {
    /* Long money values wrap badly at the default metric size. */
    [data-testid="stMetricValue"] { font-size: 1.6rem !important; }
    [data-testid="stMetricLabel"] { font-size: 0.8rem !important; }
}
</style>
"""


def render_chart(figure: go.Figure, title: str, key: str) -> None:
    """Draw a chart with its title as ordinary page text.

    Plotly paints its title inside the figure, where the floating mode bar sits
    on top of it once the viewport is narrow. Keeping the title in the page lets
    it wrap like any other text and puts it out of the mode bar's reach.
    """
    st.markdown(f"**{title}**")
    st.plotly_chart(figure, width="stretch", key=key, config=PLOTLY_CONFIG)


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------
def render_validation(report: ValidationReport) -> None:
    if report.errors:
        st.error(
            "**The analysis below cannot be trusted until these are fixed:**\n\n"
            + "\n".join(f"- **{i.field}** - {i.message}" for i in report.errors)
        )
    if report.warnings:
        with st.expander(
            f"{len(report.warnings)} warning(s) about your inputs",
            expanded=bool(report.errors),
        ):
            for issue in report.warnings:
                st.warning(f"**{issue.field}** - {issue.message}")


def render_kpis(
    result: AnalysisResult, risk: RiskMetrics, decision: DecisionOutcome
) -> None:
    capital = result.capital
    icon, colour = VERDICT_STYLE[decision.verdict]

    # Keyed so the responsive stylesheet can lay these out two-up on a phone
    # while every other column layout stacks to one.
    kpi_cards = st.container(key="kpi_cards")
    row1 = kpi_cards.columns(4)
    row1[0].metric(
        "Total application capital", compact_inr(capital.total_application_amount)
    )
    row1[1].metric(
        "OD drawn",
        compact_inr(capital.borrowed_capital),
        f"{result.funding.od_utilisation_pct:.0f}% of limit"
        if result.funding.od_limit
        else "no OD limit set",
        delta_color="off",
    )
    row1[2].metric(
        "Own equity at risk",
        compact_inr(capital.economic_capital_at_risk),
        f"own cash {compact_inr(capital.own_capital_deployed)}",
        delta_color="off",
    )
    row1[3].metric(
        "Expected allotments",
        f"{result.expected_allotments:.2f}",
        f"P(at least one) {result.allotment.p_at_least_one:.1%}",
        delta_color="off",
    )

    row2 = kpi_cards.columns(4)
    net = result.expected_net_profit
    row2[0].metric(
        "Expected net profit",
        inr(net),
        "after financing, costs and taxes",
        delta_color="normal" if net > 0 else "inverse",
    )
    row2[1].metric(
        "Return on own equity",
        pct(capital.return_on_economic_capital),
        f"annualised {pct(capital.annualized_return_on_economic_capital)}",
        delta_color="off",
        help=(
            f"The headline figure is the return over one "
            f"{capital.capital_weighted_days:.1f}-day "
            "capital cycle. The annualised number compounds that cycle across a full "
            "year, which assumes you can find and fund an identical opportunity "
            "immediately and repeatedly. IPO supply is lumpy, so treat it as an "
            "upper bound rather than an expected yearly return."
        ),
    )
    row2[2].metric(
        "Break-even GMP",
        inr(result.break_even.gmp_expected_value, 2),
        f"you assumed {inr(result.inputs.ipo.gmp_absolute, 2)}",
        delta_color="off",
    )
    row2[3].metric(
        "Probability of profit",
        pct(risk.probability_of_profit, 1),
        f"loss {pct(risk.probability_of_loss, 1)}",
        delta_color="off",
    )

    st.markdown(
        f"<div style='padding:14px 18px;border-radius:8px;border-left:6px solid "
        f"{colour};"
        f"background:rgba(128,128,128,0.08);margin-top:8px'>"
        f"<span style='font-size:1.3rem;font-weight:700;color:{colour}'>{icon} "
        f"{decision.verdict.value}</span>"
        f"<span style='margin-left:12px;opacity:0.85'>{decision.headline}</span></div>",
        unsafe_allow_html=True,
    )


def _ledger_value(value: object) -> str:
    """Render a value as display text without losing precision to rounding."""
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, (list, tuple)):
        return ", ".join(_ledger_value(v) for v in value)
    if isinstance(value, float):
        return f"{value:,.4f}".rstrip("0").rstrip(".")
    return str(value)


def render_assumption_ledger(inputs: AnalysisInputs) -> None:
    st.subheader("Assumption ledger")
    st.caption(
        "Every number that materially moves the result, and where it came from. "
        "Anything marked *Default assumption* is a value you have not reviewed."
    )
    ledger = assumption_ledger(inputs)
    frame = pd.DataFrame(
        [
            {
                "Section": r.section,
                "Assumption": r.name,
                # Rendered as text so that mixed types (numbers, booleans, lists of
                # per-account probabilities) survive the trip to the table widget.
                "Value": _ledger_value(r.value),
                "Source": r.provenance.value,
                "Note": r.note,
            }
            for r in ledger
        ]
    )
    sections = st.multiselect(
        "Filter by section",
        sorted(frame["Section"].unique()),
        default=[],
        key="ledger_filter",
    )
    if sections:
        frame = frame[frame["Section"].isin(sections)]
    st.dataframe(frame, width="stretch", hide_index=True)
    untouched = sum(1 for r in ledger if r.provenance is Provenance.ASSUMED)
    st.info(
        f"{untouched} of {len(ledger)} inputs are still at their shipped defaults. "
        "Review them before relying on the verdict."
    )


def render_expected_return(result: AnalysisResult) -> None:
    st.subheader("Expected return")
    inputs = result.inputs
    left, right = st.columns([3, 2])
    with left:
        render_chart(
            profit_waterfall(result),
            "From expected gross profit to expected net profit",
            "waterfall_expected_return",
        )
    with right:
        st.markdown("**Capital, three different denominators**")
        capital = result.capital
        frame = pd.DataFrame(
            [
                {
                    "Denominator": "Application capital (blocked)",
                    "Amount": inr(capital.total_application_amount),
                    "Return": pct(capital.return_on_application_capital),
                    "Annualised": pct(capital.annualized_return_on_application_capital),
                },
                {
                    "Denominator": "Own cash deployed",
                    "Amount": inr(capital.own_capital_deployed),
                    "Return": pct(capital.return_on_own_capital),
                    "Annualised": pct(
                        None
                        if capital.return_on_own_capital is None
                        else annualize_simple(
                            capital.return_on_own_capital, capital.capital_weighted_days
                        )
                    ),
                },
                {
                    "Denominator": "Own economic capital at risk",
                    "Amount": inr(capital.economic_capital_at_risk),
                    "Return": pct(capital.return_on_economic_capital),
                    "Annualised": pct(capital.annualized_return_on_economic_capital),
                },
            ]
        )
        st.dataframe(frame, width="stretch", hide_index=True)
        st.caption(
            f"Capital-weighted holding period: {capital.capital_weighted_days:.2f} "
            f"days "
            f"(full cycle {capital.cycle_days} days). Annualisation compounds that "
            f"cycle, "
            "which assumes you can repeat this bet immediately and indefinitely."
        )
        st.markdown("**Capital efficiency**")
        efficiency = pd.DataFrame(
            [
                {
                    "Ratio": "Expected profit / own equity",
                    "Value": pct(capital.return_on_economic_capital),
                },
                {
                    "Ratio": "Expected profit / application capital",
                    "Value": pct(capital.return_on_application_capital),
                },
                {
                    "Ratio": "Financing cost / expected gross profit",
                    "Value": pct(capital.financing_cost_to_gross_profit),
                },
                {
                    "Ratio": "Expected profit per rupee of financing cost",
                    "Value": "n/a"
                    if capital.profit_to_financing_cost is None
                    else f"{capital.profit_to_financing_cost:,.2f}x",
                },
            ]
        )
        st.dataframe(efficiency, width="stretch", hide_index=True)

    st.markdown("**Per-account economics**")
    accounts = pd.DataFrame(
        [
            {
                "Account": a.label,
                "Category": a.category.value,
                "Lots": a.lots_applied,
                "Application": inr(a.application_amount),
                "P(allotment)": f"{a.allotment_probability:.1%}",
                "Shares if allotted": f"{a.shares_if_allotted:,.0f}",
                "Gross if allotted": inr(a.gross_profit_if_allotted),
                "Costs": inr(a.transaction_costs_if_allotted.total),
                "Tax": inr(a.tax_if_allotted),
                "Carry": inr(a.carry_cost_if_allotted),
                "Net if allotted": inr(a.net_profit_if_allotted),
                "Expected contribution": inr(a.expected_net_profit_contribution),
            }
            for a in result.accounts
        ]
    )
    st.dataframe(accounts, width="stretch", hide_index=True)
    st.caption(
        "'Expected contribution' is the account's probability-weighted profit less "
        "the unconditional cost of financing its own application, so the column sums "
        "to the headline expected profit (net of one-time fees)."
    )

    left, right = st.columns(2)
    with left:
        st.markdown("**Allotment outcomes**")
        distribution = pd.DataFrame(
            {
                "Allotments": range(len(result.allotment.probabilities)),
                "Probability": [f"{p:.2%}" for p in result.allotment.probabilities],
            }
        )
        st.dataframe(distribution, width="stretch", hide_index=True)
        st.caption(
            f"Expected allotments {result.allotment.expected_allotments:.3f}; "
            f"P(zero) {result.allotment.p_zero:.2%}; "
            f"P(at least one) {result.allotment.p_at_least_one:.2%}."
        )
        if not inputs.assume_independent_allotments:
            st.warning(
                "You have switched off the independence assumption, so this "
                "distribution is indicative only: P(no allotment) = product of "
                "(1 - p) requires independent draws."
            )
    with right:
        st.markdown("**Financing and cost of capital**")
        financing = result.financing
        frame = pd.DataFrame(
            [
                {
                    "Item": "OD interest, bidding window",
                    "Amount": inr(financing.od_cost_bidding_window, 2),
                },
                {
                    "Item": "OD interest, holding window (expected)",
                    "Amount": inr(financing.expected_od_cost_holding_window, 2),
                },
                {"Item": "Processing fee", "Amount": inr(financing.processing_fee, 2)},
                {
                    "Item": "Other financing charges",
                    "Amount": inr(financing.other_charges, 2),
                },
                {
                    "Item": "Total borrowing cost",
                    "Amount": inr(financing.expected_borrowing_cost, 2),
                },
                {
                    "Item": "Opportunity cost of own capital",
                    "Amount": inr(financing.expected_opportunity_cost, 2),
                },
                {
                    "Item": "FD interest earned (informational)",
                    "Amount": inr(financing.fd_interest_earned, 2),
                },
                {
                    "Item": "FD interest counted as income",
                    "Amount": inr(financing.fd_interest_credit, 2),
                },
            ]
        )
        st.dataframe(frame, width="stretch", hide_index=True)
        if result.funding.shortfall > 0:
            st.error(
                f"Funding shortfall of {inr(result.funding.shortfall)}: your own cash "
                "plus the sanctioned OD limit do not cover this application."
            )


def render_break_even(inputs: AnalysisInputs, result: AnalysisResult) -> None:
    st.subheader("Break-even analysis")
    break_even = result.break_even
    columns = st.columns(4)
    columns[0].metric(
        "Break-even exit price (expected value)",
        inr(break_even.exit_price_expected_value, 2),
        f"{break_even.listing_gain_pct_expected_value:.2f}% listing gain"
        if break_even.listing_gain_pct_expected_value is not None
        else "unsolvable",
        delta_color="off",
    )
    columns[1].metric(
        "Break-even GMP (expected value)",
        inr(break_even.gmp_expected_value, 2),
        delta_color="off",
    )
    columns[2].metric(
        "Break-even if allotted",
        inr(break_even.exit_price_if_allotted, 2),
        f"GMP {inr(break_even.gmp_if_allotted, 2)}",
        delta_color="off",
    )
    columns[3].metric(
        "Max sustainable OD rate",
        pct_points(break_even.max_od_rate_pct),
        f"you pay {inputs.financing.od_rate_pct:.2f}%",
        delta_color="off",
    )
    render_chart(
        break_even_curve(inputs, result),
        "Expected net profit against the realised exit price",
        "break_even_curve",
    )
    minimum_probability = break_even.min_allotment_probability
    st.caption(
        "The expected-value break-even is the higher hurdle because the allotted "
        "shares must also pay the carry on every application that was refused. "
        + (
            f"Minimum uniform allotment probability for break-even: "
            f"{minimum_probability:.2%}."
            if minimum_probability is not None
            else "No allotment probability makes this strategy break even."
        )
    )


def render_scenarios(scenarios: Sequence[ScenarioResult]) -> None:
    st.subheader("Scenario analysis")
    st.caption(
        "Bear and bull factors are set in the sidebar. They are stress tests of "
        "your own assumptions, not forecasts of the issue."
    )
    frame = scenarios_to_frame(scenarios)
    display = pd.DataFrame(
        {
            "Scenario": frame["Scenario"],
            "GMP": frame["GMP (Rs)"].map(lambda v: inr(v, 2)),
            "Listing price": frame["Listing price"].map(lambda v: inr(v, 2)),
            "Exit price": frame["Exit price"].map(lambda v: inr(v, 2)),
            "Allotment prob.": frame["Allotment probability"].map(lambda v: f"{v:.1%}"),
            "Gross profit": frame["Gross profit"].map(inr),
            "Financing cost": frame["Financing cost"].map(inr),
            "Net profit": frame["Net profit"].map(inr),
            "ROI on own equity": frame["ROI on own equity"].map(pct),
            "Annualised ROI": frame["Annualized ROI"].map(pct),
        }
    )
    st.dataframe(display, width="stretch", hide_index=True)
    render_chart(scenario_chart(scenarios), "Scenario outcomes", "scenario_chart")
    for scenario in scenarios:
        st.caption(f"**{scenario.name}** - {scenario.definition.description}")

    bear = next((s for s in scenarios if s.name == "Bear"), None)
    if bear is not None and bear.result.expected_net_profit < 0:
        bear_loss = abs(bear.result.expected_net_profit)
        equity_at_risk = bear.result.capital.economic_capital_at_risk
        st.warning(
            f"The bear case loses {inr(bear_loss)}, which is "
            f"{pct(bear_loss / equity_at_risk)} of your own equity at risk. Size the "
            "position so that this outcome is survivable."
        )


def render_sensitivity(inputs: AnalysisInputs) -> None:
    st.subheader("Sensitivity analysis")
    st.caption(
        "Each cell is the expected net profit for that combination, holding every "
        "other assumption constant. Red is a loss."
    )
    issue = inputs.ipo.issue_price
    base_gmp = inputs.ipo.gmp_absolute
    base_probability = float(
        np.mean([a.allotment_probability for a in inputs.accounts])
    )

    left, right = st.columns(2)
    with left:
        gmp_high = st.number_input(
            "Highest GMP to test (Rs)",
            min_value=0.0,
            value=float(max(base_gmp * 2.0, issue * 0.2, 1.0)),
            step=1.0,
            key="gmp_grid_high",
        )
    with right:
        probability_high = st.slider(
            "Highest allotment probability to test",
            0.05,
            1.0,
            float(min(max(base_probability * 2.0, 0.30), 1.0)),
            0.05,
            key="prob_grid_high",
        )

    gmps = tuple(float(v) for v in np.round(np.linspace(0.0, gmp_high, 6), 2))
    probabilities = tuple(
        float(v) for v in np.round(np.linspace(0.05, probability_high, 6), 3)
    )
    grid = cached_gmp_grid(inputs, gmps, probabilities)
    render_chart(
        heatmap(grid, "Allotment probability", "GMP (Rs)"),
        "Expected net profit: GMP vs allotment probability",
        "heatmap_gmp_probability",
    )

    rate_high = max(inputs.financing.od_rate_pct * 1.8, 18.0)
    rates = tuple(float(v) for v in np.round(np.linspace(0.0, rate_high, 6), 2))
    gains = tuple(
        float(v)
        for v in np.round(
            np.linspace(
                -20.0, max(inputs.ipo.expected_listing_gain_pct * 1.5, 30.0), 6
            ),
            2,
        )
    )
    rate_grid = cached_rate_grid(inputs, rates, gains)
    render_chart(
        heatmap(rate_grid, "Listing gain (%)", "OD rate (% p.a.)", x_format="{:.1f}%"),
        "Expected net profit: OD rate vs listing gain",
        "heatmap_rate_gain",
    )
    st.caption(
        "Read the left-hand column of the second grid first: it is what happens "
        "with free money. If the strategy is thin there, no financing rate saves it."
    )


def render_monte_carlo(inputs: AnalysisInputs, config: MonteCarloConfig) -> None:
    st.subheader("Monte Carlo simulation")
    st.caption(
        "Randomises the listing gain, the allotment draw, the holding period and "
        "the OD rate together. Distribution shapes are chosen by you in the sidebar."
    )
    simulation = cached_monte_carlo(inputs, config)
    columns = st.columns(4)
    columns[0].metric("Mean profit", inr(simulation.expected_profit))
    columns[1].metric("Median profit", inr(simulation.median_profit))
    columns[2].metric("Probability of profit", pct(simulation.probability_of_profit, 1))
    columns[3].metric(
        "5th percentile",
        inr(simulation.percentiles[5]),
        "worst 1-in-20 outcome",
        delta_color="off",
    )
    render_chart(
        monte_carlo_chart(simulation),
        f"Distribution of net profit across {len(simulation.profits):,} simulations",
        "monte_carlo_chart",
    )

    left, right = st.columns(2)
    with left:
        frame = simulation.summary_frame().copy()
        frame["Value"] = [
            f"{value:,.0f}" if abs(value) > 1 or value == 0 else f"{value:.2%}"
            for value in frame["Value"]
        ]
        st.dataframe(frame, width="stretch", hide_index=True)
    with right:
        st.markdown("**Percentiles of net profit**")
        percentiles = pd.DataFrame(
            [
                {"Percentile": f"P{q}", "Net profit": inr(simulation.percentiles[q])}
                for q in sorted(simulation.percentiles)
            ]
        )
        st.dataframe(percentiles, width="stretch", hide_index=True)
    if simulation.median_profit < 0 < simulation.expected_profit:
        st.warning(
            "The median simulated outcome is a loss while the mean is a profit: the "
            "average is being carried by a small number of large wins. Expect a long "
            "run of small losses between them."
        )
    st.caption(
        "A normal distribution on listing gains understates gap risk; real listings "
        "jump. The simulation also draws each risk factor independently, so it does "
        "not model a cold market hitting GMP, hit-rate and listing at once."
    )


def render_risk(result: AnalysisResult, risk: RiskMetrics) -> None:
    st.subheader("Risk analysis")
    columns = st.columns(4)
    columns[0].metric("Probability of losing money", pct(risk.probability_of_loss, 1))
    columns[1].metric("Maximum modelled loss", inr(risk.maximum_loss))
    columns[2].metric(
        "Expected loss / expected gain",
        f"{inr(risk.expected_loss)} / {inr(risk.expected_gain)}",
        delta_color="off",
    )
    columns[3].metric(
        "Profit-to-loss ratio",
        "n/a"
        if risk.profit_to_loss_ratio is None
        else f"{risk.profit_to_loss_ratio:,.2f}x",
        delta_color="off",
    )

    left, right = st.columns([3, 2])
    with left:
        render_chart(
            outcome_distribution_chart(risk),
            "Every possible outcome of the allotment lottery",
            "outcome_distribution",
        )
    with right:
        st.markdown("**Dependence on each assumption**")
        frame = pd.DataFrame(
            [
                {
                    "Driver": "GMP margin of safety",
                    "Value": pct(risk.gmp_margin_of_safety),
                },
                {
                    "Driver": "Hit-rate margin of safety",
                    "Value": pct(risk.probability_margin_of_safety),
                },
                {
                    "Driver": "OD rate headroom",
                    "Value": "n/a"
                    if risk.od_rate_headroom_pct is None
                    else f"{risk.od_rate_headroom_pct:,.2f} pp",
                },
                {
                    "Driver": "Profit sensitivity to GMP",
                    "Value": "n/a"
                    if risk.gmp_elasticity is None
                    else f"{risk.gmp_elasticity:,.2f}x",
                },
                {
                    "Driver": "Profit sensitivity to hit-rate",
                    "Value": "n/a"
                    if risk.probability_elasticity is None
                    else f"{risk.probability_elasticity:,.2f}x",
                },
                {
                    "Driver": "Financing cost / gross profit",
                    "Value": pct(risk.financing_cost_share_of_gross_profit),
                },
            ]
        )
        st.dataframe(frame, width="stretch", hide_index=True)
        st.caption(
            "A sensitivity of 2.0x means a 1% fall in that driver removes 2% of the "
            "expected profit. A margin of safety is the share of the assumption that "
            "can evaporate before the trade breaks even."
        )
        st.markdown("**Downside if the listing disappoints**")
        downside = pd.DataFrame(
            [
                {
                    "Listing outcome": "Lists flat at the issue price",
                    "Expected net profit": inr(risk.profit_if_lists_flat),
                },
                {
                    "Listing outcome": "Lists 10% below the issue price",
                    "Expected net profit": inr(risk.profit_if_lists_10pct_below),
                },
                {
                    "Listing outcome": "Bear scenario",
                    "Expected net profit": inr(risk.bear_case_profit),
                },
                {
                    "Listing outcome": "No allotment at all",
                    "Expected net profit": inr(result.net_profit_if_no_allotment),
                },
            ]
        )
        st.dataframe(downside, width="stretch", hide_index=True)

    if risk.flags:
        st.markdown("**What this strategy is leaning on**")
        for flag in risk.flags:
            st.warning(flag)


def render_decision(decision: DecisionOutcome) -> None:
    st.subheader("Final decision")
    icon, colour = VERDICT_STYLE[decision.verdict]
    st.markdown(
        f"<div style='padding:18px;border-radius:8px;border:2px solid {colour};'>"
        f"<div style='font-size:1.8rem;font-weight:700;color:{colour}'>{icon} "
        f"{decision.verdict.value}</div>"
        f"<div style='font-size:1.05rem;margin-top:4px'>"
        f"{decision.headline}</div></div>",
        unsafe_allow_html=True,
    )
    st.markdown("")
    frame = decision.to_frame()
    icons = {"PASS": "✅", "FAIL": "⛔", "WARN": "⚠️"}
    frame["Result"] = frame["Result"].map(lambda value: f"{icons[value]} {value}")
    st.dataframe(frame, width="stretch", hide_index=True)

    st.markdown("**Why**")
    for line in decision.rationale:
        st.markdown(f"- {line}")
    st.info(
        "Hard rules force a NO-GO on their own. Soft rules downgrade a GO to "
        "BORDERLINE. Thresholds are yours to set in the sidebar - change them and "
        "the verdict changes, which is exactly why this is a framework and not advice."
    )


def render_portfolio(inputs: AnalysisInputs) -> None:
    st.subheader("Portfolio comparison")
    st.caption("Add several IPOs and rank them on whichever measure you care about.")
    columns = st.columns([3, 1, 1])
    label = columns[0].text_input(
        "Label for the current analysis", value=inputs.ipo.name, key="portfolio_label"
    )
    if columns[1].button("Add current analysis", width="stretch"):
        st.session_state["portfolio"] = [
            entry for entry in st.session_state["portfolio"] if entry[0] != label
        ] + [(label, inputs)]
    if columns[2].button("Clear list", width="stretch"):
        st.session_state["portfolio"] = []

    portfolio = st.session_state["portfolio"]
    if not portfolio:
        st.info(
            "No opportunities added yet. Add the current analysis to start comparing."
        )
        return

    frame = compare_opportunities(portfolio)
    sort_options = [
        "Expected net profit",
        "ROI on own equity",
        "Annualized ROI",
        "Probability of loss",
        "Profit per rupee of financing cost",
        "Application",
    ]
    left, right = st.columns([2, 1])
    sort_by = left.selectbox("Sort by", sort_options, key="portfolio_sort")
    ascending = right.checkbox(
        "Ascending", value=sort_by == "Probability of loss", key="portfolio_asc"
    )
    frame = frame.sort_values(sort_by, ascending=ascending)

    display = pd.DataFrame(
        {
            "IPO": frame["IPO"],
            "Application": frame["Application"].map(compact_inr),
            "Own capital": frame["Own capital"].map(compact_inr),
            "OD used": frame["OD used"].map(compact_inr),
            "Expected allotments": frame["Expected allotments"].map(
                lambda v: f"{v:.2f}"
            ),
            "Expected profit": frame["Expected net profit"].map(inr),
            "Financing cost": frame["Financing cost"].map(inr),
            "ROI on equity": frame["ROI on own equity"].map(pct),
            "Annualised": frame["Annualized ROI"].map(pct),
            "P(loss)": frame["Probability of loss"].map(lambda v: f"{v:.0%}"),
            "Max loss": frame["Max loss"].map(inr),
            "Decision": frame["Decision"],
        }
    )
    st.dataframe(display, width="stretch", hide_index=True)
    st.caption(
        "Capital is not free: two IPOs cannot both be funded from the same overdraft "
        "limit at the same time. Rank by profit per rupee of financing cost when the "
        "limit, not the idea, is the binding constraint."
    )


def render_methodology() -> None:
    st.subheader("How is this calculated?")
    for entry in EXPLANATIONS:
        with st.expander(entry.metric, expanded=False):
            st.markdown("**Formula**")
            st.code(entry.formula, language="text")
            st.markdown(f"**Inputs used** - {entry.inputs_used}")
            st.markdown(f"**Interpretation** - {entry.interpretation}")
            st.markdown(f"**Limitations** - {entry.limitations}")
    with st.expander("Glossary", expanded=False):
        for term, meaning in GLOSSARY:
            st.markdown(f"- **{term}**: {meaning}")


def _display_safe(frame: pd.DataFrame) -> pd.DataFrame:
    """Render every cell as text so mixed-type report tables display cleanly.

    The exported CSV and Excel files keep the underlying numbers; only this
    on-screen preview is stringified.
    """
    display = frame.copy()
    display.columns = [str(column) for column in display.columns]
    for column in display.columns:
        if display[column].dtype == object:
            display[column] = display[column].map(_ledger_value)
    return display


def render_export(
    result: AnalysisResult,
    risk: RiskMetrics,
    decision: DecisionOutcome,
    scenarios: Sequence[ScenarioResult],
    sensitivities: dict[str, pd.DataFrame],
    simulation,
) -> None:
    st.subheader("Export the analysis")
    st.caption(
        "Every export carries the inputs, the assumption ledger, the calculations, "
        "the scenarios, the sensitivities, the risk metrics and the verdict."
    )
    bundle = build_report(result, risk, decision, scenarios, sensitivities, simulation)
    stem = (
        "".join(c if c.isalnum() else "_" for c in result.inputs.ipo.name).strip("_")
        or "ipo"
    )
    columns = st.columns(3)
    columns[0].download_button(
        "Download CSV",
        data=bundle_to_csv(bundle),
        file_name=f"{stem}_analysis.csv",
        mime="text/csv",
        width="stretch",
    )
    columns[1].download_button(
        "Download Excel",
        data=bundle_to_excel(bundle),
        file_name=f"{stem}_analysis.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        width="stretch",
    )
    columns[2].download_button(
        "Download PDF summary",
        data=bundle_to_pdf(bundle),
        file_name=f"{stem}_analysis.pdf",
        mime="application/pdf",
        width="stretch",
    )
    with st.expander("Preview the report tables", expanded=False):
        for name, frame in bundle.tables.items():
            st.markdown(f"**{name}**")
            st.dataframe(_display_safe(frame), width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    st.markdown(RESPONSIVE_CSS, unsafe_allow_html=True)
    init_state()
    inputs, thresholds, monte_carlo_config, scenario_definitions = sidebar()

    heading, badge = st.columns([5, 1])
    with heading:
        st.title("IPO Capital Allocation & Financing Decision Engine")
    with badge:
        badge_style = (
            f"background:{ACCENT};color:#fff;padding:4px 10px;border-radius:12px;"
            "font-size:0.8rem;font-weight:600;letter-spacing:0.04em"
        )
        st.markdown(
            f"<div style='text-align:right;padding-top:18px'>"
            f"<span style='{badge_style}'>{RELEASE_NAME}</span></div>",
            unsafe_allow_html=True,
        )
    st.caption(
        "Does the risk-adjusted return on your own equity justify the cost and risk "
        "of the money you borrowed to apply? " + DISCLAIMER
    )
    if IS_PRERELEASE:
        st.warning(f"{BETA_NOTICE}  \n[Report a problem or send feedback]({ISSUE_URL})")

    report = validate_inputs(inputs)
    render_validation(report)
    if not report.is_valid:
        st.stop()

    bear_definition = scenario_definitions["bear"]
    result = cached_analysis(inputs)
    risk = cached_risk(result, bear_definition)
    decision = evaluate_decision(result, risk, thresholds)
    scenarios = cached_scenarios(
        inputs,
        (
            scenario_definitions["bear"],
            scenario_definitions["base"],
            scenario_definitions["bull"],
        ),
    )

    render_kpis(result, risk, decision)

    tabs = st.tabs(
        [
            "Overview",
            "Expected return",
            "Break-even",
            "Scenarios",
            "Sensitivity",
            "Monte Carlo",
            "Risk",
            "Decision",
            "Portfolio",
            "Method & export",
        ]
    )

    with tabs[0]:
        left, right = st.columns([2, 3])
        with left:
            st.subheader("The three capital buckets")
            capital = result.capital
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Bucket": "Applied for (blocked)",
                            "Amount": inr(capital.total_application_amount),
                        },
                        {
                            "Bucket": "Borrowed (OD drawn)",
                            "Amount": inr(capital.borrowed_capital),
                        },
                        {
                            "Bucket": "Own cash deployed",
                            "Amount": inr(capital.own_capital_deployed),
                        },
                        {
                            "Bucket": "FD collateral pledged",
                            "Amount": inr(result.funding.fd_collateral_locked),
                        },
                        {
                            "Bucket": "Own economic capital at risk",
                            "Amount": inr(capital.economic_capital_at_risk),
                        },
                        {
                            "Bucket": "Expected amount actually invested",
                            "Amount": inr(
                                sum(a.expected_investment for a in result.accounts)
                            ),
                        },
                    ]
                ),
                width="stretch",
                hide_index=True,
            )
            st.caption(
                "The application amount is not the amount invested. Only the allotted "
                "portion becomes an investment; the rest is released and costs you "
                "carry, not principal."
            )
        with right:
            st.subheader("Where the money goes")
            render_chart(
                profit_waterfall(result),
                "From expected gross profit to expected net profit",
                "waterfall_overview",
            )
        render_assumption_ledger(inputs)

    with tabs[1]:
        render_expected_return(result)
    with tabs[2]:
        render_break_even(inputs, result)
    with tabs[3]:
        render_scenarios(scenarios)
    with tabs[4]:
        render_sensitivity(inputs)
    with tabs[5]:
        render_monte_carlo(inputs, monte_carlo_config)
    with tabs[6]:
        render_risk(result, risk)
    with tabs[7]:
        render_decision(decision)
    with tabs[8]:
        render_portfolio(inputs)
    with tabs[9]:
        render_methodology()
        st.divider()
        base_probability = float(
            np.mean([a.allotment_probability for a in inputs.accounts])
        )
        sensitivities = {
            "GMP vs probability": cached_gmp_grid(
                inputs,
                tuple(
                    float(v)
                    for v in np.round(
                        np.linspace(0.0, max(inputs.ipo.gmp_absolute * 2.0, 1.0), 6), 2
                    )
                ),
                tuple(
                    float(v)
                    for v in np.round(
                        np.linspace(
                            0.05, min(max(base_probability * 2.0, 0.3), 1.0), 6
                        ),
                        3,
                    )
                ),
            ),
            "OD rate vs listing gain": cached_rate_grid(
                inputs,
                tuple(
                    float(v)
                    for v in np.round(
                        np.linspace(
                            0.0, max(inputs.financing.od_rate_pct * 1.8, 18.0), 6
                        ),
                        2,
                    )
                ),
                tuple(
                    float(v)
                    for v in np.round(
                        np.linspace(
                            -20.0,
                            max(inputs.ipo.expected_listing_gain_pct * 1.5, 30.0),
                            6,
                        ),
                        2,
                    )
                ),
            ),
        }
        render_export(
            result,
            risk,
            decision,
            scenarios,
            sensitivities,
            cached_monte_carlo(inputs, monte_carlo_config),
        )

    st.divider()
    st.caption(
        "Built for analysing your own assumptions. GMP is not a forecast, historical "
        "allotment odds are not future odds, and a positive expected value is not a "
        "guaranteed profit. " + DISCLAIMER
    )
    st.caption(f"{RELEASE_NAME} - [report a problem or send feedback]({ISSUE_URL})")


def run() -> None:
    """Entry point with a crash guard.

    An unhandled exception would otherwise leave a half-rendered page and a raw
    traceback. During a beta the traceback is genuinely useful, so it is offered
    for copying into a bug report rather than hidden.
    """
    try:
        main()
    except Exception as error:
        st.error(
            f"**Something went wrong while building this analysis.**\n\n"
            f"`{type(error).__name__}: {error}`\n\n"
            f"This is a bug. Please [open an issue]({ISSUE_URL}) with the details "
            "below and the inputs you used."
        )
        with st.expander("Technical details to paste into the report", expanded=False):
            st.code("".join(traceback.format_exc()), language="text")
        st.caption(RELEASE_NAME)


if __name__ == "__main__":
    run()
