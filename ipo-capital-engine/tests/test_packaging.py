"""Tests for the release packaging and deployment configuration.

Unit tests prove the arithmetic; these prove the thing can actually be shipped.
They guard the parts that are easy to break silently: the deploy manifest, the
Streamlit config, the container definition and the version stamping.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

from version import (
    BETA_NOTICE,
    IS_PRERELEASE,
    RELEASE_NAME,
    RELEASE_STAGE,
    VERSION,
    __version__,
)

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parent

#: PEP 440 for the release forms this project uses: 1.2.3, 1.2.3b4, 1.2.3rc1.
PEP440 = re.compile(r"^\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?$")


# ---------------------------------------------------------------------------
# Version identity
# ---------------------------------------------------------------------------
def test_version_is_a_valid_pep440_string():
    assert PEP440.match(__version__), f"{__version__!r} is not a PEP 440 release"
    assert __version__ == VERSION


def test_release_stage_and_prerelease_flag_agree():
    assert RELEASE_STAGE in {"alpha", "beta", "rc", "stable"}
    assert (RELEASE_STAGE != "stable") == IS_PRERELEASE
    if IS_PRERELEASE:
        # A pre-release version string must carry a pre-release segment.
        assert re.search(r"(a|b|rc)\d+$", __version__), (
            f"{__version__!r} claims to be {RELEASE_STAGE} but reads as a final release"
        )


def test_release_name_carries_the_version():
    assert __version__ in RELEASE_NAME
    assert RELEASE_STAGE in RELEASE_NAME


def test_beta_notice_is_honest_about_what_is_unverified():
    lowered = BETA_NOTICE.lower()
    assert "beta" in lowered
    # It must not imply the defaults have been validated.
    assert "not been checked" in lowered or "not been validated" in lowered


def test_the_changelog_documents_the_current_version():
    changelog = (PROJECT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert f"[{__version__}]" in changelog, "the current version has no changelog entry"


def test_exported_reports_are_stamped_with_the_version():
    from calculations import analyze
    from export import build_report
    from risk import compute_risk_metrics, evaluate_decision
    from tests.test_calculations import frictionless

    result = analyze(frictionless())
    risk = compute_risk_metrics(result)
    bundle = build_report(result, risk, evaluate_decision(result, risk))
    assert __version__ in bundle.generated_at


# ---------------------------------------------------------------------------
# Deploy manifest
# ---------------------------------------------------------------------------
def _requirement_lines(name: str) -> list[str]:
    text = (PROJECT / name).read_text(encoding="utf-8")
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def test_runtime_requirements_exclude_test_tooling():
    runtime = _requirement_lines("requirements.txt")
    assert runtime, "requirements.txt must not be empty"
    for line in runtime:
        assert not line.lower().startswith("pytest"), (
            "a deployment should not install test tooling"
        )


def test_runtime_requirements_cover_every_imported_third_party_package():
    runtime = " ".join(_requirement_lines("requirements.txt")).lower()
    for package in ("streamlit", "pandas", "numpy", "plotly", "scipy", "openpyxl"):
        assert package in runtime, f"{package} is imported at runtime but not declared"


def test_every_runtime_requirement_is_bounded_below_and_above():
    for line in _requirement_lines("requirements.txt"):
        assert ">=" in line, f"{line!r} has no lower bound"
        assert "<" in line, (
            f"{line!r} has no upper bound; a breaking major release could reach testers"
        )


def test_dev_requirements_build_on_the_runtime_manifest():
    dev = _requirement_lines("requirements-dev.txt")
    assert "-r requirements.txt" in dev, "dev requirements must include the runtime set"
    assert any(line.startswith("pytest") for line in dev)


# ---------------------------------------------------------------------------
# Streamlit configuration
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def streamlit_config() -> dict:
    with open(PROJECT / ".streamlit" / "config.toml", "rb") as handle:
        return tomllib.load(handle)


def test_streamlit_config_parses_and_pins_the_theme(streamlit_config):
    assert streamlit_config["theme"]["base"] == "light"
    assert streamlit_config["browser"]["gatherUsageStats"] is False


def test_streamlit_config_does_not_weaken_security_defaults(streamlit_config):
    server = streamlit_config.get("server", {})
    # Disabling XSRF protection or opening CORS would be a real regression, and
    # neither is needed to deploy this app.
    assert "enableXsrfProtection" not in server
    assert "enableCORS" not in server


def test_error_details_stay_visible_for_beta_testers(streamlit_config):
    assert streamlit_config["client"]["showErrorDetails"] is True


# ---------------------------------------------------------------------------
# Container
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def dockerfile() -> str:
    return (PROJECT / "Dockerfile").read_text(encoding="utf-8")


def test_dockerfile_pins_a_python_minor_version(dockerfile):
    match = re.search(r"^FROM python:(\d+\.\d+)", dockerfile, re.MULTILINE)
    assert match, "the base image must pin a Python minor version"


def test_container_runs_as_an_unprivileged_user(dockerfile):
    assert re.search(r"^USER (?!root)", dockerfile, re.MULTILINE), (
        "the container must not run as root"
    )


def test_container_declares_a_health_check_and_port(dockerfile):
    assert "HEALTHCHECK" in dockerfile
    assert "_stcore/health" in dockerfile
    assert "EXPOSE 8501" in dockerfile


def test_container_installs_runtime_requirements_only(dockerfile):
    assert "requirements.txt" in dockerfile
    assert "requirements-dev.txt" not in dockerfile


def test_dockerignore_excludes_the_git_directory_and_caches():
    ignored = (PROJECT / ".dockerignore").read_text(encoding="utf-8").split()
    for entry in (".git", "__pycache__", "tests"):
        assert entry in ignored


# ---------------------------------------------------------------------------
# Repository-level release assets
# ---------------------------------------------------------------------------
def _workflow() -> str:
    return (REPO / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")


def test_ci_workflow_covers_lint_tests_boot_and_container():
    workflow = _workflow()
    for job in ("lint:", "pytest:", "smoke:", "docker:"):
        assert job in workflow, f"the workflow is missing the {job.strip(':')} job"
    assert "_stcore/health" in workflow, "CI must verify the app actually serves"


def test_ci_checks_formatting_without_rewriting_it():
    workflow = _workflow()
    assert "ruff format --check" in workflow, (
        "CI must fail on drifted formatting rather than silently reformatting"
    )
    assert "ruff check" in workflow


def test_the_formatter_configuration_is_committed():
    config = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    assert "[tool.ruff]" in config
    assert "line-length = 88" in config, (
        "the width must be pinned so every contributor formats identically"
    )


def test_ruff_is_pinned_exactly_for_reproducible_formatting():
    dev = (PROJECT / "requirements-dev.txt").read_text(encoding="utf-8")
    assert "ruff==" in dev, (
        "a floating ruff can reformat the tree on a minor release and redden CI"
    )


def test_no_source_line_exceeds_the_configured_width():
    """The point of the width is that GitHub's viewer never scrolls sideways."""
    offenders = []
    for path in sorted(PROJECT.glob("*.py")) + sorted((PROJECT / "tests").glob("*.py")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if len(line) > 88:
                offenders.append(f"{path.name}:{number} ({len(line)} chars)")
    assert not offenders, "lines over 88 characters: " + ", ".join(offenders[:10])


def test_issue_templates_exist_for_beta_feedback():
    templates = REPO / ".github" / "ISSUE_TEMPLATE"
    assert (templates / "bug_report.yml").exists()
    assert (templates / "feedback.yml").exists()


def test_the_beta_guide_documents_the_known_limitations():
    guide = (PROJECT / "BETA_TESTING.md").read_text(encoding="utf-8").lower()
    for topic in ("no market data", "independence", "annualised", "monte carlo"):
        assert topic in guide, f"the beta guide should mention {topic!r}"


def test_the_deployment_guide_covers_every_supported_target():
    guide = (PROJECT / "DEPLOYMENT.md").read_text(encoding="utf-8").lower()
    for target in ("streamlit community cloud", "docker", "local"):
        assert target in guide
    assert "app.py" in guide, "the deployment guide must name the entry point"


# ---------------------------------------------------------------------------
# The hosted beta
# ---------------------------------------------------------------------------
#: Where the beta is deployed. Update this alongside the docs if it moves.
LIVE_URL = "https://ipo-capital-engine.streamlit.app"


def test_the_live_url_is_documented_everywhere_a_tester_would_look():
    for name in ("README.md", "BETA_TESTING.md", "DEPLOYMENT.md"):
        assert LIVE_URL in (PROJECT / name).read_text(encoding="utf-8"), (
            f"{name} does not tell anyone where the beta actually runs"
        )
    config = (REPO / ".github" / "ISSUE_TEMPLATE" / "config.yml").read_text(
        encoding="utf-8"
    )
    assert LIVE_URL in config


def test_testers_are_warned_about_the_free_tier_sleep():
    guide = (PROJECT / "BETA_TESTING.md").read_text(encoding="utf-8").lower()
    assert "sleep" in guide, (
        "a cold start on Community Cloud will otherwise be reported as a bug"
    )
