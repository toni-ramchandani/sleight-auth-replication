"""Deterministic Stage 1 representations for the Experiment 5 monitor."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from authorization_evidence import EvidenceLedger, canonical, digest

ABLATION_MODES = ("formatted_full", "neutral_context", "reduced_metadata", "evidence")
REPRESENTATION_POLICY = "stage1-representation-ablation-v1"
TOKEN_PROXY_POLICY = "unicode-alnum-runs-4-or-punctuation-v1"
LEDGER_SESSION_OPEN = '<session index="1" role="prior_context" representation="source_grounded_candidate_ledger">\n'
_PROXY_PARTS = re.compile(r"[^\W_]+|[^\w\s]|_")
_TOOL_PROTOCOL = {("assistant", "tool_use"), ("user", "tool_result")}
_EVIDENCE_ANCHOR_FIELDS = (
    "source_record_index",
    "source_block_index",
    "source_record_sha256",
    "source_block_sha256",
    "start",
    "end",
)


@dataclass(frozen=True)
class Representation:
    # text may contain sensitive excerpts. Only provenance goes in results.
    text: str
    provenance: dict[str, Any]


@dataclass(frozen=True)
class _Record:
    line: str
    value: dict[str, Any]
    sha256: str

    @property
    def blocks(self) -> list:
        content = self.value.get("content")
        if isinstance(content, dict):
            return [content]
        return content if isinstance(content, list) else []


def _token_proxy(text: str) -> int:
    return sum((len(part) + 3) // 4 for part in _PROXY_PARTS.findall(text))


def _ledger_json(payload: dict) -> str:
    return canonical(payload).replace("<", r"\u003c").replace(">", r"\u003e").replace("&", r"\u0026")


def _records(stripped_stage1: str) -> dict[int, _Record]:
    records = {}
    for index, line in enumerate(stripped_stage1.splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid stripped Stage 1 JSONL record {index}") from None
        if not isinstance(value, dict):
            raise ValueError(f"Invalid stripped Stage 1 record {index}")
        records[index] = _Record(line, value, digest(canonical(value)))
    if not records:
        raise ValueError("No stripped Stage 1 records")
    return records


def _full_history(records: dict[int, _Record]) -> tuple[str, list[dict]]:
    anchors = []
    entries = []
    for index, record in records.items():
        anchor = {"source_record_index": index, "source_record_sha256": record.sha256}
        anchors.append(anchor)
        entries.append({
            **anchor,
            "source_block_index": None,
            "source_block_sha256": None,
            "source_block_length": None,
            "source_class": "transcript_record",
            "selection_reason": "complete_source_record",
            "start": 0,
            "end": len(record.line),
            "verbatim_excerpt": record.line,
        })
    return _ledger_json({
        "kind": "stage1_full_history",
        "extraction_policy": REPRESENTATION_POLICY,
        "selection_mode": "all_stripped_records",
        "explanation": "Entries contain complete earlier-session transcript records in source order.",
        "entries": entries,
    }), anchors


def _neutral_context(records: dict[int, _Record], target: int) -> tuple[str, list[dict], int]:
    anchors = []
    entries = []
    for record_index, record in records.items():
        role = record.value.get("role")
        if role not in ("assistant", "user"):
            continue
        for block_index, block in enumerate(record.blocks):
            if not isinstance(block, dict):
                continue
            kind = block.get("type")
            if kind not in ("tool_use", "tool_result") or (role, kind) not in _TOOL_PROTOCOL:
                continue
            location = {"source_record_index": record_index, "source_block_index": block_index}
            entries.append({**location, "role": role, "type": kind})
            anchors.append({
                **location,
                "source_record_sha256": record.sha256,
                "source_block_sha256": digest(canonical(block)),
            })
    if not entries:
        raise ValueError("No Stage 1 tool protocol records for the neutral context")
    payload = {
        "kind": "stage1_neutral_operation_metadata",
        "extraction_policy": REPRESENTATION_POLICY,
        "selection_mode": "tool_protocol_metadata",
        "explanation": "Entries mark earlier tool protocol events in source order. Repeated item markers pad length.",
        "entries": entries,
    }
    padding = max(0, target - _token_proxy(_ledger_json(payload)))
    payload["explanation"] += " item" * padding
    return _ledger_json(payload), anchors, padding


def _reduced_metadata(evidence: EvidenceLedger, records: dict[int, _Record]) -> tuple[str, list[dict]]:
    anchors = []
    entries = []
    try:
        evidence_entries = json.loads(evidence.text)["entries"]
        if len(evidence_entries) != evidence.event_count or not evidence_entries:
            raise ValueError
        for entry in evidence_entries:
            anchor = {name: entry[name] for name in _EVIDENCE_ANCHOR_FIELDS}
            record_index = anchor["source_record_index"]
            block_index = anchor["source_block_index"]
            start, end = anchor["start"], anchor["end"]
            if any(type(n) is not int or n < 0 for n in (record_index, block_index, start, end)):
                raise ValueError
            record = records[record_index]
            block = record.blocks[block_index]
            block_text = canonical(block)
            excerpt = entry["verbatim_excerpt"]
            if (
                not isinstance(block, dict)
                or record.sha256 != anchor["source_record_sha256"]
                or digest(block_text) != anchor["source_block_sha256"]
                or len(block_text) != entry["source_block_length"]
                or not start < end <= len(block_text)
                or excerpt != block_text[start:end]
            ):
                raise ValueError
            anchors.append(anchor)
            entries.append({
                "source_record_index": record_index,
                "source_block_index": block_index,
                "role": record.value["role"],
                "type": block["type"],
                "verbatim_excerpt": excerpt,
            })
    except (ValueError, TypeError, KeyError, IndexError):
        raise ValueError("Experiment 4 evidence anchors do not match stripped Stage 1 records") from None
    return _ledger_json({"entries": entries}), anchors


def prepare_representations(stripped_stage1: str, evidence: EvidenceLedger) -> dict[str, Representation]:
    """Return four session bodies with score-only provenance and length diagnostics."""
    if digest(evidence.text) != evidence.sha256:
        raise ValueError("Experiment 4 evidence digest does not match its rendered ledger")
    records = _records(stripped_stage1)
    target = _token_proxy(evidence.text)
    full, full_anchors = _full_history(records)
    neutral, neutral_anchors, padding = _neutral_context(records, target)
    reduced, evidence_anchors = _reduced_metadata(evidence, records)
    bodies = (
        (full, full_anchors, 0),
        (neutral, neutral_anchors, padding),
        (reduced, evidence_anchors, 0),
        (evidence.text, evidence_anchors, 0),
    )
    representations = {}
    for mode, (body, anchors, padding_units) in zip(ABLATION_MODES, bodies):
        units = _token_proxy(body)
        representations[mode] = Representation(body, {
            "representation_policy": REPRESENTATION_POLICY,
            "representation_sha256": digest(body),
            "representation_source_anchors_sha256": digest(canonical(anchors)),
            "representation_entry_count": len(anchors),
            "representation_bytes": len(body.encode("utf-8")),
            "representation_token_proxy_policy": TOKEN_PROXY_POLICY,
            "representation_token_proxy": units,
            "evidence_token_proxy": target,
            "representation_token_proxy_delta": units - target,
            "neutral_padding_units": padding_units,
        })
    return representations
