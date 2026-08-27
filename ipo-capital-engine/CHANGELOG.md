# Changelog

All notable changes to this project are recorded here. Versions follow
[PEP 440](https://peps.python.org/pep-0440/); `bN` suffixes mark beta builds.

## Unreleased

### Fixed

- **Narrow-screen layout.** Three elements were landing on top of page text at
  phone width. Plotly's floating mode bar covered the in-figure chart title (a
  measured 194x19px overlap on a 412px viewport), so titles now sit in the page
  where they wrap like any other text and the bar cannot reach them. Streamlit's
  element toolbar floats 42px above its own element and was covering the heading
  before it, so it is pulled inside its element. The sidebar was forced open and
  covered the entire viewport on a phone; it is now allowed to collapse itself,
  while still opening by default on a wide screen.
- Waterfall axis labels were long enough to become unreadable diagonal text on a
  narrow axis; they are now short, with the full wording on hover, and the plot
  reserves room for the x-axis band.

### Changed

- **Formatted the whole codebase with Ruff** and pinned the configuration in
  `pyproject.toml` (88 columns, so GitHub's code viewer never scrolls sideways
  on a narrow screen). 68 over-long string literals were rewrapped, and the
  transformation was verified by asserting the parsed AST came out identical
  for every file, then by diffing the full worked-example output against the
  previous commit: 582 lines, byte for byte the same.
- `zip()` calls over parallel sequences now pass `strict=True`, so a length
  mismatch fails loudly instead of silently truncating.
- Extracted a few deeply nested f-string expressions into named locals, and
  lifted the version-badge CSS out of the markup string.

### Added

- A `lint` CI job running `ruff format --check` and `ruff check`, so drifted
  formatting blocks a merge rather than being silently rewritten.
- `.editorconfig` and `.gitattributes` to keep indentation and line endings
  consistent across editors and platforms.
- Four packaging tests covering the formatter configuration, the pinned Ruff
  version, the lint job, and a hard assertion that no source line exceeds 88
  characters.

## [0.1.0b1] - 2026-08-27

First beta, packaged for live testing.

### Added

- **Release identity.** `version.py` is the single source of truth; the version
  appears in the page title, as a badge beside the dashboard title, in the
  footer, and stamped into every exported report, so any bug report identifies
  its build.
- **Beta notice** in the app, stating plainly that the arithmetic is tested but
  the default cost, tax and threshold values have not been checked against a
  live issue, with a link to open an issue.
- **Crash guard.** An unhandled exception now renders a readable message and a
  copyable traceback for a bug report instead of a half-drawn page.
- **Deployment configuration**: `.streamlit/config.toml` (pinned light theme,
  usage stats off, error details kept for testers), a `Dockerfile` running as an
  unprivileged user with a health check, and `.dockerignore`.
- **Documentation**: `DEPLOYMENT.md` (Streamlit Community Cloud, Docker, local,
  plus how to cut the next build) and `BETA_TESTING.md` (what to test, known
  limitations, what counts as a bug).
- **Issue templates** for bug reports and general feedback.
- **CI**: a smoke job that boots the real server from `requirements.txt` alone
  and checks Streamlit's health endpoint and the root page, and a job that
  builds the container image and boots it. Both catch deployment breakage that
  unit tests cannot.

### Changed

- **Requirements split**: `requirements.txt` is now runtime-only, so a
  deployment no longer installs test tooling; `requirements-dev.txt` adds
  pytest. Major-version upper bounds were added so a breaking upstream release
  cannot disrupt testers mid-beta, while minor and patch upgrades still flow
  through and are exercised by CI.
- **SciPy is imported once at module load** rather than lazily inside the
  root-finder, moving a roughly half-second cost from a user's first analysis to
  server start-up.
- **Verdict headline** no longer repeats the verdict word next to the badge.
- **Annualised return** on the KPI card now carries a tooltip explaining that it
  compounds a short capital cycle and is an upper bound.

## [0.1.0] - 2026-08-26

Initial implementation.

### Added

- Calculation engine separating application capital, borrowed capital, own cash
  and economic capital at risk, with two-phase financing, expected value over an
  exact Poisson-binomial allotment distribution, break-even exit price and GMP
  (conditional and expected-value), maximum sustainable OD rate and minimum
  allotment probability.
- Scenario analysis, two-way sensitivity grids and a vectorised Monte Carlo.
- Risk metrics, margins of safety, elasticities and a rules-based
  GO / BORDERLINE / NO-GO framework with editable thresholds.
- Input validation and an assumption ledger marking every input as user-entered,
  calculated or an untouched default.
- CSV, Excel and dependency-free PDF export.
- Streamlit dashboard with ten sections.
- 147 tests with hand-derived expected values, and CI on Python 3.11 and 3.12.
