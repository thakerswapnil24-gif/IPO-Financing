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
