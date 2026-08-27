"""Smoke tests for the Streamlit dashboard.

These drive the real app through Streamlit's AppTest harness, so a broken
widget, a duplicate element key or a serialisation failure fails the build
rather than the user's browser.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


@pytest.fixture(scope="module")
def app() -> AppTest:
    instance = AppTest.from_file(APP, default_timeout=300)
    instance.run()
    return instance


def test_the_app_renders_without_an_exception(app):
    assert not app.exception
    assert not app.error


def test_the_dashboard_shows_its_kpi_cards_and_sections(app):
    labels = {metric.label for metric in app.metric}
    for expected in (
        "Total application capital",
        "OD drawn",
        "Own equity at risk",
        "Expected allotments",
        "Expected net profit",
        "Return on own equity",
        "Break-even GMP",
        "Probability of profit",
    ):
        assert expected in labels
    assert len(app.tabs) == 10


def test_the_disclaimer_is_always_visible(app):
    captions = " ".join(element.value for element in app.caption)
    assert "not investment advice" in captions.lower()


def test_changing_the_gmp_changes_the_expected_profit():
    app = AppTest.from_file(APP, default_timeout=300)
    app.run()
    baseline = next(m for m in app.metric if m.label == "Expected net profit").value
    app.number_input(key="gmp_value").set_value(1.0).run()
    assert not app.exception
    reduced = next(m for m in app.metric if m.label == "Expected net profit").value
    assert reduced != baseline


def test_an_invalid_input_blocks_the_analysis_with_a_clear_message():
    app = AppTest.from_file(APP, default_timeout=300)
    app.run()
    app.number_input(key="issue_price").set_value(0.0).run()
    assert not app.exception
    assert app.error, "a zero issue price must raise a blocking validation error"
    assert "Issue price" in app.error[0].value


def test_a_funding_shortfall_is_reported_as_an_error():
    app = AppTest.from_file(APP, default_timeout=300)
    app.run()
    app.selectbox(key="funding_mode").set_value("Own capital only").run()
    app.number_input(key="own_available").set_value(0.0).run()
    assert not app.exception
    assert any("Shortfall" in element.value for element in app.error)


def test_loading_an_example_preset_updates_the_inputs():
    app = AppTest.from_file(APP, default_timeout=300)
    app.run()
    options = app.selectbox(key="preset_choice").options
    target = next(option for option in options if option.startswith("D."))
    app.selectbox(key="preset_choice").set_value(target).run()
    next(b for b in app.button if b.label == "Load this example").click().run()
    assert not app.exception
    assert app.number_input(key="issue_price").value == pytest.approx(100.0)
    assert app.number_input(key="lot_size").value == 1200
    assert app.number_input(key="od_rate").value == pytest.approx(14.0)
    assert len(app.session_state["accounts_df"]) == 1


def test_switching_off_gmp_derivation_enables_the_listing_price_input():
    app = AppTest.from_file(APP, default_timeout=300)
    app.run()
    assert app.number_input(key="listing_override").disabled
    app.checkbox(key="use_gmp").set_value(False).run()
    assert not app.exception
    assert not app.number_input(key="listing_override").disabled


def test_adding_an_opportunity_to_the_portfolio_builds_the_comparison_table():
    app = AppTest.from_file(APP, default_timeout=300)
    app.run()
    add_button = next(b for b in app.button if b.label == "Add current analysis")
    add_button.click().run()
    assert not app.exception
    assert len(app.session_state["portfolio"]) == 1
    headers = [column for frame in app.dataframe for column in frame.value.columns]
    assert "Decision" in headers


def test_the_app_exposes_a_crash_guarded_entry_point(monkeypatch):
    """An unhandled error must render a report path, not propagate."""
    import app as dashboard

    def explode() -> None:
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(dashboard, "main", explode)
    dashboard.run()  # must not raise


def test_the_beta_banner_and_version_badge_are_shown(app):
    from version import RELEASE_NAME, __version__

    warnings = " ".join(element.value for element in app.warning)
    assert "beta" in warnings.lower(), "testers must be told this is a beta build"
    assert "issues/new" in warnings, "the beta notice needs a feedback link"

    markdown = " ".join(element.value for element in app.markdown)
    assert RELEASE_NAME in markdown, "the running version must be visible in the UI"

    captions = " ".join(element.value for element in app.caption)
    assert __version__ in captions


# ---------------------------------------------------------------------------
# Narrow-screen layout
#
# Three separate elements used to land on top of page text at phone width:
# Plotly's in-figure title sat under its floating mode bar, Streamlit's element
# toolbar floats above its own element and covered the heading before it, and
# the sidebar was forced open over the whole viewport. These pin the fixes.
# ---------------------------------------------------------------------------
def test_no_chart_carries_an_in_figure_title():
    """Titles live in the page, where the mode bar cannot cover them."""
    import app as dashboard
    from calculations import analyze
    from risk import compute_risk_metrics
    from scenarios import MonteCarloConfig, run_monte_carlo, run_scenarios
    from tests.test_calculations import frictionless

    inputs = frictionless(gmp=25.0, probability=0.2)
    result = analyze(inputs)
    figures = {
        "waterfall": dashboard.profit_waterfall(result),
        "break_even": dashboard.break_even_curve(inputs, result),
        "scenarios": dashboard.scenario_chart(run_scenarios(inputs)),
        "outcomes": dashboard.outcome_distribution_chart(compute_risk_metrics(result)),
        "monte_carlo": dashboard.monte_carlo_chart(
            run_monte_carlo(inputs, MonteCarloConfig(n_simulations=500))
        ),
    }
    for name, figure in figures.items():
        title = figure.layout.title.text
        assert not title, f"{name} still sets an in-figure title: {title!r}"


def test_every_chart_is_drawn_through_the_shared_renderer():
    """One call site means the mode-bar config cannot be forgotten on a chart."""
    source = (Path(APP)).read_text(encoding="utf-8")
    assert source.count("st.plotly_chart(") == 1, (
        "charts must go through render_chart so they share the mode-bar config"
    )
    assert "PLOTLY_CONFIG" in source


def test_the_sidebar_is_allowed_to_collapse_on_a_narrow_screen():
    source = (Path(APP)).read_text(encoding="utf-8")
    assert 'initial_sidebar_state="auto"' in source, (
        "an expanded sidebar covers the whole viewport on a phone"
    )


def test_the_element_toolbar_is_pulled_inside_its_element():
    source = (Path(APP)).read_text(encoding="utf-8")
    assert "stElementToolbar" in source, (
        "without this override Streamlit's toolbar sits on the heading above it"
    )


def test_waterfall_axis_labels_stay_short_enough_for_a_narrow_axis():
    import app as dashboard
    from calculations import analyze
    from tests.test_calculations import frictionless

    figure = dashboard.profit_waterfall(analyze(frictionless()))
    labels = list(figure.data[0].x)
    assert labels, "the waterfall must label its bars"
    longest = max(labels, key=len)
    assert len(longest) <= 14, (
        f"{longest!r} is too long for a phone-width axis; the full wording "
        "belongs in the hover text"
    )
    # ... and the full wording must still be reachable.
    assert figure.data[0].customdata is not None
    assert "Expected gross profit" in list(figure.data[0].customdata)
