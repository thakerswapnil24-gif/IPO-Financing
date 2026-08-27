"""Report assembly and export (CSV, Excel, PDF).

The exporter takes a completed analysis and produces a single bundle of tables
that can be written to disk or streamed to a browser. The PDF writer is a small
dependency-free implementation so that the report works in a bare environment;
it lays out plain text only, which is what a numbers-first summary needs.
"""

from __future__ import annotations

import io
import textwrap
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from calculations import AnalysisResult, AssumptionRecord, assumption_ledger
from explanations import explanations_frame
from risk import DecisionOutcome, RiskMetrics
from version import RELEASE_NAME
from scenarios import MonteCarloResult, ScenarioResult, scenarios_to_frame

__all__ = [
    "ReportBundle",
    "build_report",
    "bundle_to_csv",
    "bundle_to_excel",
    "bundle_to_pdf",
]


@dataclass(frozen=True)
class ReportBundle:
    """A named collection of tables making up the full analysis report."""

    title: str
    generated_at: str
    tables: Dict[str, pd.DataFrame]

    def sheet_names(self) -> List[str]:
        return list(self.tables)


def _kv_frame(mapping: Dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"Item": k, "Value": v} for k, v in mapping.items()]
    )


def _ledger_frame(records: Sequence[AssumptionRecord]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Section": r.section,
                "Assumption": r.name,
                "Value": r.value,
                "Provenance": r.provenance.value,
                "Note": r.note,
            }
            for r in records
        ]
    )


def _accounts_frame(result: AnalysisResult) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Account": a.label,
                "Category": a.category.value,
                "Lots applied": a.lots_applied,
                "Application amount": a.application_amount,
                "Allotment probability": a.allotment_probability,
                "Shares if allotted": a.shares_if_allotted,
                "Investment if allotted": a.investment_if_allotted,
                "Exit value if allotted": a.exit_value_if_allotted,
                "Gross profit if allotted": a.gross_profit_if_allotted,
                "Transaction costs if allotted": a.transaction_costs_if_allotted.total,
                "Tax if allotted": a.tax_if_allotted,
                "Cost of carry if allotted": a.carry_cost_if_allotted,
                "Net profit if allotted": a.net_profit_if_allotted,
                "Expected net contribution": a.expected_net_profit_contribution,
            }
            for a in result.accounts
        ]
    )


def build_report(
    result: AnalysisResult,
    risk: RiskMetrics,
    decision: DecisionOutcome,
    scenarios: Optional[Sequence[ScenarioResult]] = None,
    sensitivities: Optional[Dict[str, pd.DataFrame]] = None,
    monte_carlo: Optional[MonteCarloResult] = None,
) -> ReportBundle:
    """Assemble every table that belongs in the exported analysis."""
    inputs = result.inputs
    tables: Dict[str, pd.DataFrame] = {}

    tables["Summary"] = _kv_frame(result.summary_dict())
    tables["Decision"] = pd.concat(
        [
            _kv_frame(
                {
                    "Verdict": decision.verdict.value,
                    "Headline": decision.headline,
                }
            ),
            pd.DataFrame({"Item": ["Rationale"] * len(decision.rationale), "Value": list(decision.rationale)}),
        ],
        ignore_index=True,
    )
    tables["Decision checks"] = decision.to_frame()
    tables["Assumptions"] = _ledger_frame(assumption_ledger(inputs))
    tables["Accounts"] = _accounts_frame(result)
    tables["Allotment distribution"] = pd.DataFrame(
        {
            "Number of allotments": range(len(result.allotment.probabilities)),
            "Probability": result.allotment.probabilities,
        }
    )
    tables["Capital & financing"] = _kv_frame(
        {
            "Total application amount": result.capital.total_application_amount,
            "Own capital deployed": result.funding.own_capital_deployed,
            "OD drawn": result.funding.od_drawn,
            "OD limit": result.funding.od_limit,
            "OD utilisation %": result.funding.od_utilisation_pct,
            "FD collateral locked": result.funding.fd_collateral_locked,
            "Economic capital at risk": result.capital.economic_capital_at_risk,
            "OD cost (bidding window)": result.financing.od_cost_bidding_window,
            "Expected OD cost (holding window)": result.financing.expected_od_cost_holding_window,
            "Processing fee": result.financing.processing_fee,
            "Other financing charges": result.financing.other_charges,
            "Opportunity cost of own capital": result.expected_opportunity_cost,
            "FD interest earned (informational)": result.financing.fd_interest_earned,
            "FD interest counted as income": result.financing.fd_interest_counted,
            "Capital-weighted days": result.capital.capital_weighted_days,
            "Cycle days": result.capital.cycle_days,
        }
    )
    tables["Risk metrics"] = risk.to_frame()
    tables["Profit distribution"] = risk.distribution.to_frame()
    tables["Risk flags"] = pd.DataFrame({"Flag": list(risk.flags)})

    if scenarios:
        tables["Scenarios"] = scenarios_to_frame(scenarios)
    if sensitivities:
        for name, frame in sensitivities.items():
            tables[f"Sensitivity - {name}"[:31]] = frame.reset_index()
    if monte_carlo is not None:
        tables["Monte Carlo"] = monte_carlo.summary_frame()
    tables["Method notes"] = explanations_frame()

    return ReportBundle(
        title=f"IPO financing analysis - {inputs.ipo.name}",
        generated_at=(
            f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} "
            f"by IPO Capital Engine {RELEASE_NAME}"
        ),
        tables=tables,
    )


def bundle_to_csv(bundle: ReportBundle) -> str:
    """Flatten every table into one CSV document with section headers."""
    buffer = io.StringIO()
    buffer.write(f"# {bundle.title}\n# Generated: {bundle.generated_at}\n")
    for name, frame in bundle.tables.items():
        buffer.write(f"\n## {name}\n")
        frame.to_csv(buffer, index=False)
    return buffer.getvalue()


def bundle_to_excel(bundle: ReportBundle) -> bytes:
    """Write the bundle to an Excel workbook, one sheet per table."""
    output = io.BytesIO()
    used: set[str] = set()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for name, frame in bundle.tables.items():
            sheet = name[:31] or "Sheet"
            counter = 1
            while sheet in used:
                counter += 1
                suffix = f"_{counter}"
                sheet = f"{name[:31 - len(suffix)]}{suffix}"
            used.add(sheet)
            frame.to_excel(writer, sheet_name=sheet, index=False)
    return output.getvalue()


# ---------------------------------------------------------------------------
# Minimal, dependency-free PDF writer
# ---------------------------------------------------------------------------
_PAGE_WIDTH, _PAGE_HEIGHT = 595.28, 841.89  # A4 in points
_MARGIN = 36.0
_BODY_SIZE = 8.0
_LEADING = 10.5
_MONO_CHARS_PER_LINE = int((_PAGE_WIDTH - 2 * _MARGIN) / (_BODY_SIZE * 0.6))
_LINES_PER_PAGE = int((_PAGE_HEIGHT - 2 * _MARGIN) / _LEADING) - 2


def _pdf_escape(text: str) -> str:
    """Escape a string for a PDF literal and drop characters the fonts lack."""
    safe = text.encode("latin-1", "replace").decode("latin-1")
    return safe.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def _format_value(value: object) -> str:
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, float):
        if value != value:  # NaN
            return "n/a"
        if abs(value) >= 1000:
            return f"{value:,.2f}"
        return f"{value:,.4f}".rstrip("0").rstrip(".")
    if value is None:
        return "n/a"
    return str(value)


def _frame_to_lines(frame: pd.DataFrame, max_rows: int = 60) -> List[str]:
    """Render a DataFrame as fixed-width text lines that fit the page."""
    if frame.empty:
        return ["(no rows)"]
    display = frame.head(max_rows).copy()
    for column in display.columns:
        display[column] = display[column].map(_format_value)
    display.columns = [str(c) for c in display.columns]

    n_cols = len(display.columns)
    budget = _MONO_CHARS_PER_LINE - (n_cols - 1)
    widths = []
    for column in display.columns:
        widths.append(max(len(column), *(len(v) for v in display[column])) if len(display) else len(column))
    total = sum(widths)
    if total > budget:  # shrink the widest columns first
        scale = budget / total
        widths = [max(6, int(w * scale)) for w in widths]

    def row_text(cells: Sequence[str]) -> str:
        parts = []
        for cell, width in zip(cells, widths):
            cell = cell if len(cell) <= width else cell[: width - 1] + "~"
            parts.append(cell.ljust(width))
        return " ".join(parts).rstrip()

    lines = [row_text(list(display.columns)), "-" * min(_MONO_CHARS_PER_LINE, sum(widths) + n_cols - 1)]
    for _, row in display.iterrows():
        lines.append(row_text([row[c] for c in display.columns]))
    if len(frame) > max_rows:
        lines.append(f"... {len(frame) - max_rows} more rows (see the CSV or Excel export)")
    return lines


def _build_pdf(pages: Sequence[Sequence[Tuple[str, str]]]) -> bytes:
    """Assemble PDF bytes from pages of ``(style, text)`` lines."""
    objects: List[bytes] = []

    def add(payload: bytes) -> int:
        objects.append(payload)
        return len(objects)

    font_regular = b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>"
    font_bold = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>"

    catalog_id = add(b"")  # 1 - patched below
    pages_id = add(b"")  # 2 - patched below
    regular_id = add(font_regular)
    bold_id = add(font_bold)

    page_ids: List[int] = []
    for lines in pages:
        stream = io.StringIO()
        stream.write("BT\n")
        y = _PAGE_HEIGHT - _MARGIN
        for style, text in lines:
            if style == "title":
                font, size = "F2", 13.0
            elif style == "heading":
                font, size = "F2", 10.0
            else:
                font, size = "F1", _BODY_SIZE
            y -= _LEADING
            stream.write(f"/{font} {size:.1f} Tf 1 0 0 1 {_MARGIN:.2f} {y:.2f} Tm ")
            stream.write(f"({_pdf_escape(text)}) Tj\n")
        stream.write("ET")
        content = stream.getvalue().encode("latin-1", "replace")
        content_id = add(
            b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream"
        )
        page_id = add(b"")
        page_ids.append(page_id)
        objects[page_id - 1] = (
            f"<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 {_PAGE_WIDTH:.2f} "
            f"{_PAGE_HEIGHT:.2f}] /Resources << /Font << /F1 {regular_id} 0 R /F2 "
            f"{bold_id} 0 R >> >> /Contents {content_id} 0 R >>"
        ).encode("latin-1")

    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    objects[pages_id - 1] = (
        f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>"
    ).encode("latin-1")
    objects[catalog_id - 1] = f"<< /Type /Catalog /Pages {pages_id} 0 R >>".encode("latin-1")

    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets = [0]
    for index, payload in enumerate(objects, start=1):
        offsets.append(out.tell())
        out.write(f"{index} 0 obj\n".encode("latin-1"))
        out.write(payload)
        out.write(b"\nendobj\n")
    xref_pos = out.tell()
    out.write(f"xref\n0 {len(objects) + 1}\n".encode("latin-1"))
    out.write(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        out.write(f"{offset:010d} 00000 n \n".encode("latin-1"))
    out.write(
        f"trailer\n<< /Size {len(objects) + 1} /Root {catalog_id} 0 R >>\nstartxref\n"
        f"{xref_pos}\n%%EOF\n".encode("latin-1")
    )
    return out.getvalue()


def bundle_to_pdf(
    bundle: ReportBundle,
    include_tables: Optional[Sequence[str]] = None,
    max_rows_per_table: int = 45,
) -> bytes:
    """Render the report bundle as a plain-text A4 PDF summary."""
    lines: List[Tuple[str, str]] = [
        ("title", bundle.title),
        ("body", f"Generated {bundle.generated_at}"),
        ("body", ""),
    ]
    names = list(include_tables) if include_tables else list(bundle.tables)
    for name in names:
        frame = bundle.tables.get(name)
        if frame is None:
            continue
        lines.append(("heading", name))
        for text in _frame_to_lines(frame, max_rows_per_table):
            for wrapped in textwrap.wrap(text, _MONO_CHARS_PER_LINE) or [""]:
                lines.append(("body", wrapped))
        lines.append(("body", ""))
    lines.append(
        (
            "body",
            "Quantitative decision framework based on user-supplied assumptions. Not investment advice.",
        )
    )

    pages: List[List[Tuple[str, str]]] = []
    current: List[Tuple[str, str]] = []
    for entry in lines:
        if len(current) >= _LINES_PER_PAGE:
            pages.append(current)
            current = []
        current.append(entry)
    if current:
        pages.append(current)
    return _build_pdf(pages)
