# Beta testing guide

Thank you for testing this. The build you are using is **v0.1.0b1 (beta)**.

**Open it here: <https://ipo-capital-engine.streamlit.app>**

Nothing to install. The version badge beside the title should read `v0.1.0b1
beta` — if it says something else, you are on a newer build than this guide.

The engine's arithmetic is covered by 147 automated tests, and every expected
value in those tests was derived by hand from the documented formulas. What has
**not** been validated is the judgement layer: the default cost and tax rates,
the default decision thresholds, and whether the model's picture of an IPO
application matches how issues actually behave for you. That is what this beta
is for.

---

## What this tool is, in one paragraph

It answers whether the risk-adjusted return on **your own equity** justifies the
cost and risk of the money you borrowed to apply for an IPO. It is not an IPO
return calculator, and it is not investment advice. It scores assumptions that
you supply, and it cannot tell you whether those assumptions are realistic.

## A quirk of the free hosting, not a bug

The beta runs on Streamlit Community Cloud, which puts an app to sleep after a
stretch with no visitors. If you are the first person back, the page can take
about half a minute to wake up, and may show a "getting the app back up" notice
first. That is the host, not the app. Please do not report it.

The same tier caps memory at roughly 1 GB. The heaviest thing here is the Monte
Carlo tab; the default 10,000 paths are comfortable, but if you push it to
100,000 and the app restarts itself, that is the ceiling rather than a crash
worth reporting.

## Before you start

1. Open the sidebar and load one of the five bundled examples. They are
   illustrative constructions, not real issues, and no GMP or allotment
   probability in them is observed market data.
2. Look at the **Overview** tab's assumption ledger. It tells you how many
   inputs are still at their shipped defaults. Review those before you trust a
   verdict.
3. Replace every number with your own.

## What would be most useful to test

**The three capital buckets.** The tool insists that "applied for", "borrowed",
"own cash" and "expected amount actually invested" are four different numbers.
Check the Overview tab against a real application you have made. Does the
expected invested amount match what actually got debited?

**The financing cost.** The model charges interest on the whole application for
the days it is blocked, then on the allotted portion only for the holding
period, simple interest on a 365-day basis. Compare it to a real overdraft
statement. Banks compound monthly and some charge minimum utilisation; if your
bank's arithmetic differs materially, that is a finding worth reporting.

**The cost and tax stack.** Open the sidebar's *Transaction costs* and *Taxes*
panels and compare every rate against a recent contract note and your own tax
position. The shipped defaults are common Indian equity-delivery values at the
time of writing, not law, and they are the single most likely thing to be stale.

**The decision thresholds.** The GO / BORDERLINE / NO-GO framework is
deliberately hard to satisfy. Its thresholds are policy choices, editable in the
sidebar. If it rejects something you would clearly take, or waves through
something you would clearly refuse, say so — that is the most valuable feedback
you can give, and please include the numbers.

**Break-even and sensitivity.** The expected-value break-even is always the
harder hurdle, because the allotted shares must also pay the carry on every
application that was refused. Does the number it produces look sane for issues
you know?

## Known limitations in this build

These are understood and documented, not bugs. Reporting them is fine, but they
will not surprise anyone:

- **No market data.** There is no feed of any kind. GMP, allotment probability
  and listing price are all inputs you type.
- **Independence across accounts** is assumed for the allotment distribution.
- **Retail allotment** is modelled as one lot per successful application. NII
  proportionate allotment is modelled with a user-set expected number of lots.
- **Annualised returns compound a short cycle**, which assumes you can redeploy
  into an identical opportunity immediately and repeatedly. Treat the annualised
  figure as an upper bound; the tooltip on that card says the same.
- **Monte Carlo draws risk factors independently**, so it does not model a cold
  market hitting GMP, hit-rate and listing at the same time. A normal
  distribution on listing gains also understates gap risk.
- **Taxes are simplified**: no set-off against other capital losses, no
  surcharge slabs beyond the flat cess.
- **Nothing is saved between sessions.** Close the tab and your inputs are gone.
  Use the CSV or Excel export to keep an analysis.

## What counts as a bug

- A number that disagrees with your own arithmetic. **Please include the inputs**
  — the CSV export contains all of them plus the assumption ledger, and it is by
  far the fastest way to get a numerical bug fixed.
- A crash. The app will show you a technical-details box; paste it in.
- A validation message that blocks something legitimate, or fails to block
  something impossible.
- Anything in the interface that reads as advice rather than as arithmetic on
  your own assumptions. That framing matters and I want to get it right.

## How to report

Open an issue: <https://github.com/thakerswapnil24-gif/IPO-Financing/issues/new/choose>

There are templates for a bug report and for general feedback. For anything
numerical, attach the CSV export from the **Method & export** tab — it carries
the inputs, the assumption ledger, the calculations and the verdict, so the
whole case can be reproduced exactly.

## A closing word on what the verdicts mean

A **GO** means every rule passed *on the assumptions you typed*. It does not
mean the trade is good. If your GMP assumption is optimistic, the verdict is
optimistic, and the tool says so in the risk flags.

A **NO-GO** is the more useful output. The rule that fires most often is that
the annualised return on equity must clear the overdraft rate by a margin — a
strategy can be profitable in absolute terms and still be a poor use of borrowed
money. The tool was built to be able to say "do not do this", and it frequently
does.
