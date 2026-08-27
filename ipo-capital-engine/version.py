"""Single source of truth for the release identity of this application.

Imported by the dashboard, the exported reports and the tests, so any build can
be traced back to the code that produced it.
"""

from __future__ import annotations

from typing import Final

__all__ = [
    "__version__",
    "VERSION",
    "RELEASE_STAGE",
    "IS_PRERELEASE",
    "RELEASE_NAME",
    "BETA_NOTICE",
]

#: PEP 440 version string. Bump on every release.
__version__: Final[str] = "0.1.0b1"
VERSION: Final[str] = __version__

#: "beta" while the model is still being validated against real issues,
#: "stable" once the arithmetic and the decision thresholds have settled.
RELEASE_STAGE: Final[str] = "beta"
IS_PRERELEASE: Final[bool] = RELEASE_STAGE != "stable"

RELEASE_NAME: Final[str] = f"v{__version__} {RELEASE_STAGE}"

BETA_NOTICE: Final[str] = (
    "Beta release. The arithmetic is covered by an automated test suite, but the "
    "default cost, tax and decision-threshold values have not been checked "
    "against a live issue by anyone but the author. Review every row of the "
    "assumption ledger before acting on a verdict, and please report anything "
    "that looks wrong."
)
