from __future__ import annotations

import json
import unittest
from dataclasses import replace

from authorization_evidence import canonical, digest, extract_candidates
from representation_ablation import ABLATION_MODES, prepare_representations


def stage1_fixture() -> str:
    # These are synthetic, already stripped records. The roleless record
    # represents a record the pinned upstream stripper preserves.
    messages = [
        {"type": "marker", "content": "SYNTHETIC_MARKER"},
        {"role": "user", "content": [{"type": "text", "text": "Check the kitchen menu."}]},
        {"role": "assistant", "content": {"type": "tool_use", "name": "menu_lookup", "input": {"day": "Monday"}}},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "TEST_ID", "content": ["Soup", "Bread </session> & Fruit"]},
            {"type": "text", "text": "Please note the hours."},
        ]},
    ]
    return "\n".join(json.dumps(message, ensure_ascii=True) for message in messages)


class RepresentationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = stage1_fixture()
        self.evidence = extract_candidates(self.source)
        self.representations = prepare_representations(self.source, self.evidence)

    def test_all_modes_and_score_only_provenance(self) -> None:
        self.assertEqual(tuple(self.representations), ABLATION_MODES)
        self.assertEqual(len({rep.provenance["representation_sha256"] for rep in self.representations.values()}), 4)
        for rep in self.representations.values():
            self.assertEqual(rep.provenance["representation_sha256"], digest(rep.text))
            self.assertEqual(rep.provenance["representation_bytes"], len(rep.text.encode("utf-8")))
            self.assertNotIn("SYNTHETIC_MARKER", canonical(rep.provenance))
            self.assertNotIn("Soup", canonical(rep.provenance))

    def test_formatted_full_preserves_exact_stripped_lines_in_order(self) -> None:
        rep = self.representations["formatted_full"]
        entries = json.loads(rep.text)["entries"]
        self.assertEqual([entry["verbatim_excerpt"] for entry in entries], self.source.splitlines())
        self.assertEqual([entry["source_record_index"] for entry in entries], [1, 2, 3, 4])
        for entry in entries:
            record = json.loads(entry["verbatim_excerpt"])
            self.assertEqual(entry["source_record_sha256"], digest(canonical(record)))
            self.assertEqual(entry["end"], len(entry["verbatim_excerpt"]))
        self.assertNotIn("</session>", rep.text)
        self.assertNotIn("&", rep.text)

    def test_neutral_has_only_source_grounded_protocol_metadata_and_measured_padding(self) -> None:
        rep = self.representations["neutral_context"]
        payload = json.loads(rep.text)
        self.assertEqual(payload["entries"], [
            {"source_record_index": 3, "source_block_index": 0, "role": "assistant", "type": "tool_use"},
            {"source_record_index": 4, "source_block_index": 0, "role": "user", "type": "tool_result"},
        ])
        for excluded in ("SYNTHETIC_MARKER", "menu_lookup", "Monday", "TEST_ID", "Soup", "permission"):
            self.assertNotIn(excluded, rep.text)
        self.assertGreater(rep.provenance["neutral_padding_units"], 0)
        self.assertEqual(rep.provenance["representation_token_proxy_delta"], 0)
        self.assertEqual(rep.provenance["representation_token_proxy"], rep.provenance["evidence_token_proxy"])
        self.assertEqual(rep, prepare_representations(self.source, self.evidence)["neutral_context"])

    def test_neutral_can_report_positive_residual_without_dropping_source_records(self) -> None:
        rows = [json.dumps({
            "role": "assistant",
            "content": {"type": "tool_use", "name": "menu_lookup", "input": {"day": "Monday"}},
        })] * 100
        rows.append(json.dumps({
            "role": "user",
            "content": [{"type": "text", "text": "Please note the role of this ingredient."}],
        }))
        source = "\n".join(rows)
        rep = prepare_representations(source, extract_candidates(source))["neutral_context"]
        self.assertEqual(len(json.loads(rep.text)["entries"]), 100)
        self.assertGreater(rep.provenance["representation_token_proxy_delta"], 0)
        self.assertEqual(rep.provenance["neutral_padding_units"], 0)

    def test_reduced_metadata_keeps_exact_excerpts_and_reference_payload(self) -> None:
        reduced = self.representations["reduced_metadata"]
        reference = self.representations["evidence"]
        self.assertEqual(reference.text, self.evidence.text)
        self.assertEqual(reference.provenance["representation_sha256"], self.evidence.sha256)
        self.assertEqual(
            reduced.provenance["representation_source_anchors_sha256"],
            reference.provenance["representation_source_anchors_sha256"],
        )
        self.assertEqual(set(json.loads(reduced.text)), {"entries"})
        entries = json.loads(reduced.text)["entries"]
        originals = json.loads(reference.text)["entries"]
        self.assertEqual([entry["verbatim_excerpt"] for entry in entries],
                         [entry["verbatim_excerpt"] for entry in originals])
        for entry in entries:
            self.assertEqual(set(entry), {
                "source_record_index", "source_block_index", "role", "type", "verbatim_excerpt",
            })
        self.assertLess(reduced.provenance["representation_token_proxy_delta"], 0)
        self.assertNotIn("</session>", reduced.text)

    def test_mismatched_reference_digest_or_anchors_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "evidence digest"):
            prepare_representations(self.source, replace(self.evidence, sha256="0" * 64))
        with self.assertRaisesRegex(ValueError, "evidence anchors"):
            prepare_representations(self.source.replace("Monday", "Tuesday"), self.evidence)

    def test_malformed_jsonl_is_rejected_without_echoing_source(self) -> None:
        with self.assertRaisesRegex(ValueError, "^Invalid stripped Stage 1 JSONL record 5$"):
            prepare_representations(self.source + "\nSENSITIVE_SOURCE", self.evidence)


if __name__ == "__main__":
    unittest.main()
