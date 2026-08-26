# IPO Capital Allocation & Financing Decision Engine

A quantitative tool for one question:

> **Does the risk-adjusted return on my own equity justify the cost and the risk of the
> money I borrowed to apply for this IPO?**

It is not an IPO return calculator. A calculator tells you what one lot pays if the
stock lists 30% up. This tells you whether a strategy of applying — usually from
several accounts, usually on an FD-backed overdraft, usually for a small chance of a
single lot — makes money after financing, transaction costs, taxes and the
probability of getting nothing at all.

The engine is deliberately sceptical. It is fully capable of concluding
**"do not pursue this strategy"**, and on realistic inputs it frequently does.

---

## Table of contents

- [Installation](#installation)
- [Running it](#running-it)
- [Project structure](#project-structure)
- [The financial model](#the-financial-model)
  - [Three kinds of capital](#three-kinds-of-capital)
  - [Two financing windows](#two-financing-windows)
  - [Expected value](#expected-value)
  - [Transaction costs and taxes](#transaction-costs-and-taxes)
  - [Break-even analysis](#break-even-analysis)
  - [Capital efficiency and annualisation](#capital-efficiency-and-annualisation)
  - [Multiple accounts](#multiple-accounts)
- [Scenario, sensitivity and Monte Carlo analysis](#scenario-sensitivity-and-monte-carlo-analysis)
- [Risk analysis](#risk-analysis)
- [The GO / NO-GO framework](#the-go--no-go-framework)
- [Data integrity](#data-integrity)
- [Example dataset and worked analysis](#example-dataset-and-worked-analysis)
- [Exports](#exports)
- [Using the engine without Streamlit](#using-the-engine-without-streamlit)
- [Testing](#testing)
- [Assumptions and limitations](#assumptions-and-limitations)
- [Disclaimer](#disclaimer)

---

## Installation

Requires **Python 3.11 or newer**.

```bash
git clone <this-repository>
cd ipo-capital-engine

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## Running it

```bash
streamlit run app.py
```

The dashboard opens at <http://localhost:8501>. Load one of the bundled examples from
the sidebar ("Load an example") to see a complete, internally consistent analysis
immediately, then edit any assumption.

Command-line worked example, no browser required:

```bash
python example_analysis.py                 # analyse all five bundled examples
python example_analysis.py --index 4       # analyse just the leveraged bNII example
python example_analysis.py --export ./out  # also write CSV, Excel and PDF reports
```

Run the tests:

```bash
pytest -q
```

## Project structure

```
ipo-capital-engine/
├── app.py                  Streamlit dashboard (presentation only)
├── calculations.py         Core engine: capital, financing, expected value, break-even
├── scenarios.py            Scenarios, sensitivity grids, Monte Carlo simulation
├── risk.py                 Risk metrics, outcome distribution, GO/NO-GO, portfolio table
├── validation.py           Input validation with clear, actionable messages
├── explanations.py         Formula documentation surfaced in the UI and the report
├── export.py               CSV / Excel / PDF report assembly
├── example_data.py         Preset loader and JSON (de)serialisation
├── example_analysis.py     Command-line worked example
├── data/
│   └── example_ipos.json   Illustrative example dataset
├── requirements.txt
├── README.md
└── tests/
    ├── test_calculations.py    engine arithmetic, hand-verified
    ├── test_scenarios.py       scenarios, sensitivity, Monte Carlo
    ├── test_risk.py            risk metrics and the decision framework
    ├── test_validation.py      validation rules and the assumption ledger
    ├── test_export.py          reports, exports, example dataset
    └── test_app.py             Streamlit dashboard smoke tests
```

`app.py` contains no financial logic. Every number it displays comes from the engine
modules, which import nothing from Streamlit.

---

## The financial model

### Three kinds of capital

The single most common error in IPO financing arithmetic is treating the application
amount as the amount invested. This model keeps them apart:

| Bucket | Meaning |
| --- | --- |
| **Application capital** | `issue price × lot size × lots applied`, summed over accounts. Blocked (ASBA) or drawn (OD) during the bidding window. |
| **Borrowed capital** | The overdraft actually drawn, capped at `FD amount × LTV%`. |
| **Own cash deployed** | Your own money committed to the application. |
| **Economic capital at risk** | `own cash deployed + FD collateral pledged`, where pledged collateral is `OD drawn / LTV`. A pledged deposit is not spent, but it is encumbered. |
| **Expected amount invested** | `Σ p_i × issue price × shares allotted`. Usually a small fraction of the application. |

Returns are reported against three different denominators and never merged into one
"ROI":

```
Return on application capital = expected net profit / total application amount
Return on own cash            = expected net profit / own cash deployed
Return on own equity          = expected net profit / economic capital at risk
```

When no own cash is deployed, return on own cash is reported as `n/a` rather than
infinity, and economic capital becomes the honest denominator.

### Two financing windows

Modelling financing as one lump on the whole application for the whole period
materially overstates the cost. The engine splits it:

```
Bidding window  (unconditional, `days_blocked`):
    OD drawn × OD rate × days_blocked / 365
  + own cash deployed × opportunity rate × days_blocked / 365

Holding window  (contingent on allotment, `holding_period_days`):
    Σ p_i × allotted investment_i × OD share × OD rate × holding_days / 365
  + Σ p_i × allotted investment_i × own share × opportunity rate × holding_days / 365

One-time:
    processing fee + other financing charges
```

The bidding-window cost is paid whether or not anything is allotted. Only the allotted
portion is carried into the holding window, so that leg is probability-weighted.

Interest is simple interest on a **365-day basis** (configurable to 360 or 366).

**FD interest is excluded from profit by default.** A pledged deposit earns its
interest whether or not you bid, so counting it as a benefit of the strategy flatters
the result. It is displayed for information and can be switched on explicitly.

**Opportunity cost** on own capital is charged by default, so the headline figure is
economic profit. Both the cash and economic figures are reported.

### Expected value

```
Profit if allotted   = gross profit − transaction costs − tax − cost of carry
Expected profit      = Σ p_i × (profit if allotted)_i − unconditional financing cost
```

Expected value is the average over many repetitions of the same bet. It is **not** the
outcome of any single application: with a 10% hit-rate the modal outcome is "nothing
allotted, financing cost paid". The dashboard shows the exact discrete distribution of
outcomes alongside the mean for exactly this reason.

### Transaction costs and taxes

Every rate is a configurable input, not a constant baked into the code:

```
brokerage + STT + exchange transaction charges + SEBI turnover fees
+ stamp duty + GST on (brokerage + exchange + SEBI) + DP charges + other
```

```
taxable gain = gross gain − allowable transfer expenses (STT excluded by default)
tax          = taxable gain × rate × (1 + cess),
               short-term at or below the LTCG threshold, long-term above it
```

Losses generate no tax unless you explicitly ask the model to recognise a tax shield,
which assumes you have other realised gains to set them off against. The shipped
defaults reflect common Indian equity-delivery charges at the time of writing; they
change with every finance act and broker tariff, and the assumption ledger marks any
default you have not reviewed.

### Break-even analysis

Two break-evens, both solved numerically (Brent's method, with a bisection fallback):

- **If allotted** — the exit price at which the allotted shares alone wash their face,
  after their transaction costs, tax and full cost of carry.
- **Expected value** — the exit price at which the *whole strategy* breaks even. The
  winners must also pay the carry on every application that was refused, so this is
  always the higher hurdle. It is the number that matters when financing many
  applications for a few allotments.

```
Break-even GMP = break-even exit price − issue price
```

Two further inversions of the same model:

- **Maximum sustainable OD rate** — the highest annualised financing rate at which
  expected net profit stays at or above zero. Returns `n/a` when the strategy loses
  money even at a zero rate, because then cheaper funding cannot rescue it.
- **Minimum allotment probability** — the uniform hit-rate at which expected net
  profit is zero.

### Capital efficiency and annualisation

```
capital-weighted days = (application × days_blocked + expected invested × holding_days)
                        / application
annualised return     = (1 + period return) ^ (365 / capital-weighted days) − 1
```

Weighting by capital stops a 7-day cycle being annualised as though the money were
tied up all year. Compounded annualisation still assumes you can redeploy into an
identical opportunity immediately and repeatedly, so treat it as an upper bound. The
engine also reports the simple (non-compounded) annualisation.

Also reported: `financing cost / expected gross profit` and
`expected profit / financing cost`, because a strategy can show a fine nominal IPO
return and a poor one after the cost of money.

### Multiple accounts

```
P(no allotment)        = Π (1 − p_i)
P(at least one)        = 1 − P(no allotment)
Expected allotments    = Σ p_i
```

The full distribution of the number of allotments is computed exactly
(Poisson-binomial, by dynamic programming), not approximated.

**Independence is a modelling assumption and is labelled as such.** The registrar runs
the lottery per application, which is close to independent, but subscription levels
move every account's odds at once. Switching the assumption off keeps the expectation
and flags the distribution as indicative only.

More accounts raise the chance of at least one allotment, but the financing cost
scales with every application while profit scales only with the ones that hit. The
engine never treats extra PANs as free upside.

---

## Scenario, sensitivity and Monte Carlo analysis

**Scenarios.** Bear / base / bull, where bear and bull are explicit multipliers on GMP
and hit-rate that you set in the sidebar. Each scenario carries a written description
of exactly what it changed. The default bear case reverses the premium to half its
size as a discount and halves the hit-rate; the default bull expands the premium by
50% and the hit-rate by 25%. These are editable stress tests, not forecasts.

**Sensitivity.** Two heatmaps of expected net profit: GMP against allotment
probability, and OD rate against listing gain. Read the zero-rate row of the second
grid first — if the strategy is thin with free money, no financing rate saves it.

**Monte Carlo.** At least 10,000 paths (configurable up to 100,000), randomising:

| Driver | Distributions |
| --- | --- |
| Listing/exit gain | normal, triangular, uniform, fixed |
| Allotment probability | fixed, or Beta with a concentration parameter |
| Holding period | fixed, or discrete uniform |
| OD rate | fixed, normal, uniform |

The per-path cash-flow logic is a vectorised replica of the deterministic engine, and
a test asserts that a simulation with fixed distributions converges on the analytic
expected value. Output includes the mean, median, P5/P25/P75/P95, standard deviation,
expected shortfall of the worst 5%, and probabilities of profit and loss, with a
profit-distribution chart.

## Risk analysis

Because a positive expected value is not a reason to trade on borrowed money, the tool
also reports:

- probability of losing money, and the exact discrete distribution of outcomes
- maximum modelled loss, expected loss, expected gain, profit-to-loss ratio
- financing cost as a share of expected gross and net profit
- **margins of safety**: the fraction of the assumed GMP, and of the assumed hit-rate,
  that can evaporate before break-even
- **elasticities**: the percentage of expected profit lost per 1% fall in GMP or hit-rate
- OD-rate headroom in percentage points
- profit if the stock lists flat, and if it lists 10% below the issue price

It then states in plain language what the strategy is leaning on — GMP, the hit-rate,
cheap money, or a short capital cycle.

## The GO / NO-GO framework

A rules-based scoring of *your* assumptions. Thresholds are editable in the sidebar.

**Hard rules — any failure forces NO-GO**

1. Expected net profit is positive.
2. Annualised return on own equity clears the OD rate by the required spread
   (default 5 percentage points).
3. Financing plus opportunity cost is at most 60% of expected gross profit.
4. Break-even listing gain is within the plausibility limit (default 15%).

**Soft rules — any failure downgrades GO to BORDERLINE**

5. Probability of loss is tolerable (default limit 60%).
6. The bear case is survivable (default limit: 5% of own equity).
7. GMP margin of safety is at least 30%.
8. Hit-rate margin of safety is at least 30%.
9. OD-rate headroom is at least 2 percentage points.

Rule 2 is the one that most often fails on real inputs: a strategy can be profitable
in absolute terms and still be a bad use of borrowed money.

This is a decision framework, not financial advice. It scores your assumptions and
cannot tell you whether those assumptions are realistic.

## Data integrity

- **The assumption ledger.** Every material input is listed with its provenance —
  *User entered*, *Calculated*, or *Default assumption* — plus a note on what it means
  and where it can mislead. The dashboard reports how many inputs are still at their
  shipped defaults.
- **Validation.** Errors block the analysis; warnings do not. Covered: non-positive
  issue prices, invalid lot sizes, probabilities outside 0–1, negative amounts and
  rates, OD above the sanctioned LTV, drawing an overdraft with no deposit, funding
  shortfalls, retail bids above the SEBI cap, category/size mismatches, zero or
  negative day counts, implausible rates, and missing listing-price inputs.
- **Precision.** No intermediate rounding anywhere. Values are rounded only for
  display, and the CSV/Excel exports carry full precision.

## Example dataset and worked analysis

`data/example_ipos.json` contains five **illustrative constructions** — not records of
real issues, and none of the GMP or probability values are observed market data. They
span the decision range on the shipped thresholds:

| Example | Setup | Verdict |
| --- | --- | --- |
| A | Strong mainboard IPO, one retail application, own funds | BORDERLINE |
| B | Same issue, five PANs on an FD-backed OD at 11% | GO |
| C | Three PANs, modest GMP, 14% OD, 12 days blocked | NO-GO |
| D | SME issue, thin GMP, long block, expensive money | NO-GO |
| E | ₹20 lakh bNII bid on an OD for a one-lot lottery | NO-GO |

Example C is the instructive one: expected profit is *positive* (about ₹194) and the
verdict is still NO-GO, because 12 days of 14% money on ₹45,000 returns 12.2%
annualised on equity — less than the 14% it costs to borrow.

Example E is the leverage trap: ₹20.1 lakh applied, ₹3,750 expected to be invested,
₹5,049 of financing cost against ₹938 of expected gross profit, and a break-even that
needs the stock to list 171% up.

## Exports

Every export contains the inputs, the assumption ledger, the calculations, the
scenarios, the sensitivities, the risk metrics, the Monte Carlo summary, the method
notes and the verdict.

- **CSV** — one document, section-delimited, full precision.
- **Excel** — one sheet per table (via `openpyxl`).
- **PDF** — a plain-text A4 summary written by a small dependency-free PDF writer, so
  the report works in a minimal environment.

## Using the engine without Streamlit

```python
from calculations import (
    AnalysisInputs, ApplicationAccount, FinancingAssumptions,
    FundingMode, IPOAssumptions, analyze,
)
from risk import compute_risk_metrics, evaluate_decision

inputs = AnalysisInputs(
    ipo=IPOAssumptions(
        name="Example",
        issue_price=300.0,
        lot_size=50,
        gmp_value=105.0,          # Rs per share; GMPMode.PERCENT for a percentage
        holding_period_days=0,    # sell on listing day
    ),
    accounts=(
        ApplicationAccount(label="PAN 1", lots_applied=1, allotment_probability=0.10),
        ApplicationAccount(label="PAN 2", lots_applied=1, allotment_probability=0.10),
    ),
    financing=FinancingAssumptions(
        funding_mode=FundingMode.OD,
        own_capital_available=0.0,
        fd_amount=100_000.0,
        od_ltv_pct=90.0,
        od_rate_pct=11.0,
        days_blocked=6,
        opportunity_cost_rate_pct=7.25,
    ),
)

result = analyze(inputs)
risk = compute_risk_metrics(result)
decision = evaluate_decision(result, risk)

print(result.expected_net_profit)                       # after everything
print(result.break_even.gmp_expected_value)             # GMP needed to break even
print(result.break_even.max_od_rate_pct)                # financing headroom
print(risk.probability_of_loss, decision.verdict.value)
```

## Testing

```bash
pytest -q          # 147 tests
```

The suite pins the arithmetic rather than the code's current output: expected values
are derived by hand from the formulas above. Coverage includes basic IPO profit,
financing cost across both windows, break-even price and GMP, maximum sustainable OD
rate, minimum hit-rate, multiple accounts, Poisson-binomial probabilities, negative
listing scenarios, zero allotment, 100% allotment, OD financing, mixed own/borrowed
funding, day-count conventions, tax treatment (short-term, long-term, losses,
exemptions), validation rules, scenario and sensitivity behaviour, Monte Carlo
convergence against the analytic expectation, the decision framework, the exports, and
a Streamlit smoke test that drives the real dashboard.

Three independent implementations of the expected value — the closed-form engine, the
exact Poisson-binomial outcome distribution, and the vectorised Monte Carlo — are
asserted to agree.

## Assumptions and limitations

Read this section before trusting any number the tool produces.

- **GMP is not a forecast.** It is an unregulated, unsettled, thinly traded quote that
  routinely collapses between the close of bidding and listing day. The tool treats it
  as an assumption you own and flags when the entire expected gain depends on it.
- **Historical allotment odds are not future odds.** Subscription levels are known only
  after the book closes.
- **Independence across accounts** is assumed for the allotment distribution.
- **Retail allotment** is modelled as one lot per successful application, which is the
  SEBI minimum-lot lottery in an oversubscribed book. NII proportionate allotment is
  modelled through a user-set expected number of lots.
- **Simple interest, same-day drawdown and repayment.** Banks compound monthly, may
  charge minimum utilisation, and can reprice an overdraft.
- **Tax rates are inputs, not law.** No set-off against other capital losses, no
  surcharge slabs beyond the flat cess, no securities-transaction-tax deduction.
- **Distributions are yours.** A normal distribution on listing gains understates gap
  risk; real listings jump. Monte Carlo draws risk factors independently, so it does
  not model a cold market hitting GMP, hit-rate and listing at the same time.
- **Applying from multiple PANs** that are not genuinely separate investors is a
  regulatory question, not a modelling one. This tool takes no view on it.
- **Nothing here validates your inputs against the market.** There is no data feed.

## Disclaimer

This software is a quantitative decision framework applied to assumptions supplied by
the user. It is **not** investment advice, not a recommendation to apply for or avoid
any issue, and not a substitute for professional judgement. Expected value is not
guaranteed profit. Leveraged applications can and do lose money. Use at your own risk.
