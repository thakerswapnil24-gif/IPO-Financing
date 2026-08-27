"""Example dataset, presets and (de)serialisation helpers.

The presets are *illustrative constructions*, not records of real issues: they
exist so a new user can load a complete, internally consistent set of
assumptions and see how the engine behaves across the GO / BORDERLINE / NO-GO
range. Any resemblance to a specific listing is coincidental.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from calculations import (
    AnalysisInputs,
    ApplicationAccount,
    FinancingAssumptions,
    FundingMode,
    GMPMode,
    IPOAssumptions,
    IPOCategory,
    TaxAssumptions,
    TransactionCostAssumptions,
)

__all__ = [
    "ExamplePreset",
    "DATA_FILE",
    "load_examples",
    "inputs_from_dict",
    "inputs_to_dict",
]

DATA_FILE = Path(__file__).parent / "data" / "example_ipos.json"


@dataclass(frozen=True)
class ExamplePreset:
    name: str
    notes: str
    inputs: AnalysisInputs


def inputs_from_dict(payload: dict[str, Any]) -> AnalysisInputs:
    """Build :class:`AnalysisInputs` from a plain dictionary (JSON friendly)."""
    ipo_data = dict(payload.get("ipo", {}))
    if "gmp_mode" in ipo_data:
        ipo_data["gmp_mode"] = GMPMode(ipo_data["gmp_mode"])
    ipo = IPOAssumptions(**ipo_data)

    accounts: list[ApplicationAccount] = []
    for account in payload.get("accounts", []):
        data = dict(account)
        if "category" in data:
            data["category"] = IPOCategory(data["category"])
        accounts.append(ApplicationAccount(**data))
    if not accounts:
        accounts.append(ApplicationAccount())

    financing_data = dict(payload.get("financing", {}))
    if "funding_mode" in financing_data:
        financing_data["funding_mode"] = FundingMode(financing_data["funding_mode"])
    financing = FinancingAssumptions(**financing_data)

    costs = TransactionCostAssumptions(**payload.get("costs", {}))
    taxes = TaxAssumptions(**payload.get("taxes", {}))
    return AnalysisInputs(
        ipo=ipo,
        accounts=tuple(accounts),
        financing=financing,
        costs=costs,
        taxes=taxes,
        assume_independent_allotments=payload.get(
            "assume_independent_allotments", True
        ),
    )


def inputs_to_dict(inputs: AnalysisInputs) -> dict[str, Any]:
    """Inverse of :func:`inputs_from_dict` - JSON-serialisable."""

    def clean(mapping: dict[str, Any]) -> dict[str, Any]:
        return {
            key: (value.value if hasattr(value, "value") else value)
            for key, value in mapping.items()
        }

    return {
        "ipo": clean(asdict(inputs.ipo)),
        "accounts": [clean(asdict(a)) for a in inputs.accounts],
        "financing": clean(asdict(inputs.financing)),
        "costs": clean(asdict(inputs.costs)),
        "taxes": clean(asdict(inputs.taxes)),
        "assume_independent_allotments": inputs.assume_independent_allotments,
    }


def load_examples(path: Path | None = None) -> list[ExamplePreset]:
    """Load the bundled example dataset."""
    source = Path(path) if path is not None else DATA_FILE
    payload = json.loads(source.read_text(encoding="utf-8"))
    return [
        ExamplePreset(
            name=entry["name"],
            notes=entry.get("notes", ""),
            inputs=inputs_from_dict(entry),
        )
        for entry in payload["examples"]
    ]
