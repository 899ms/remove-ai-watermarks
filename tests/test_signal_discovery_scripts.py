"""Contracts for the signal-discovery corpus harnesses."""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import corpus_gap_scan as gap
import sidecar_regression as regression
import visible_eval
import visible_positives as visible


def _report(**overrides):
    values = {
        "is_ai_generated": None,
        "platform": None,
        "signals": [],
        "integrity_clashes": [],
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_gap_markers_are_case_insensitive():
    assert "ComfyUI" in gap._marker_hits(b"COMFYUI WORKFLOW")


def test_gap_markers_require_structural_provenance_evidence():
    noise = b"c2pa digitalSourceType Samsung Galaxy Doubao PhotoEditor_Re_Edit Signature: " + b"A" * 80
    assert gap._marker_hits(noise) == []

    tc260 = b'{"AIGC":{"label":"1","contentProducer":"synthetic"}}'
    assert "TC260 AIGC" in gap._marker_hits(tc260)
    assert "Samsung genAIType" in gap._marker_hits(b'PhotoEditor_Re_Edit_Data{"genAIType":1}')


def test_gap_candidates_cover_more_than_blind_unknowns():
    signal = SimpleNamespace(name="iptc")
    assert gap._candidate_classes(_report(signals=[signal]), []) == ["unattributed_signal"]
    assert gap._candidate_classes(_report(is_ai_generated=True), []) == ["unattributed_ai"]
    assert gap._candidate_classes(_report(), ["SynthID"]) == ["blind_marker"]


def test_since_filter_uses_corpus_date_directories(tmp_path):
    old = tmp_path / "2026-09-01" / "old.png"
    current = tmp_path / "2026-09-10" / "current.png"
    undated = tmp_path / "loose.png"
    for path in (old, current, undated):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    assert gap._files(tmp_path, date(2026, 9, 10)) == [current]


def test_gap_checkpoint_keeps_complete_rows_and_ignores_a_torn_tail(tmp_path):
    checkpoint = tmp_path / "report.csv.progress.jsonl"
    row = gap._base_row("2026-09-10/example.png", ".png", "1.2.3")
    checkpoint.write_text(json.dumps(row) + '\n{"path":"torn', encoding="utf-8")

    assert gap._read_checkpoint(checkpoint) == {row["path"]: row}

    gap._repair_checkpoint(checkpoint)
    second = gap._base_row("2026-09-10/second.png", ".png", "1.2.3")
    with checkpoint.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(second) + "\n")

    assert gap._read_checkpoint(checkpoint) == {row["path"]: row, second["path"]: second}


def test_visible_checkpoint_is_repaired_before_resume(tmp_path):
    checkpoint = tmp_path / "visible.jsonl"
    first = {"path": "/corpus/2026-09-10/first.png", "keys": [], "status": "ok"}
    checkpoint.write_text(json.dumps(first) + '\n{"path":"torn', encoding="utf-8")

    visible._repair_checkpoint(checkpoint)
    with checkpoint.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"path": "/corpus/2026-09-10/second.png", "keys": [], "status": "ok"}) + "\n")

    assert [row["path"] for row in visible._read_records(checkpoint)] == [
        "/corpus/2026-09-10/first.png",
        "/corpus/2026-09-10/second.png",
    ]


def test_visible_since_filter_uses_corpus_date_directories(tmp_path):
    old = tmp_path / "2026-09-01" / "old.png"
    current = tmp_path / "2026-09-10" / "current.png"
    nested = tmp_path / "2026-09-10" / "nested" / "current.webp"
    undated = tmp_path / "loose.png"
    for path in (old, current, nested, undated):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    assert visible._files(tmp_path, date(2026, 9, 10)) == [current, nested]


def test_visible_discovery_candidates_include_unmapped_provenance_and_unbiased_quiet_rows():
    rows = [
        {"path": "known.png", "keys": ["qwen"], "status": "ok", "uscc": "unmapped-a"},
        {"path": "a.png", "keys": [], "status": "ok", "uscc": "unmapped-a"},
        {"path": "b.png", "keys": [], "status": "ok", "platform": "Google Gemini"},
        {"path": "c.png", "keys": [], "status": "ok"},
        {"path": "bad.png", "keys": [], "status": "unreadable", "uscc": "unmapped-b"},
    ]

    selected = visible._discovery_candidates(rows, per_cohort=2, random_quiet=10, seed=7)

    assert {row["path"] for row in selected} == {"a.png", "b.png", "c.png"}
    assert next(row for row in selected if row["path"] == "a.png")["discovery_strata"] == [
        "tc260:unmapped-a",
        "unbiased-quiet",
    ]


def test_visible_evaluation_inventory_follows_the_registry():
    from remove_ai_watermarks.watermark_registry import mark_keys

    assert tuple(mark_keys()) == visible_eval.MARKS


def test_gap_worker_error_is_a_complete_resumable_row(monkeypatch, tmp_path):
    path = tmp_path / "bad.bin"
    path.write_bytes(b"not an image")
    calls = []

    def fail(_path, **kwargs):
        calls.append(kwargs)
        raise ValueError("broken\ninput")

    monkeypatch.setattr(gap, "identify", fail)
    row = gap._scan_one((str(path), "bad.bin", "1.2.3"))

    assert set(row) == set(gap.REPORT_FIELDS)
    assert row["candidate_classes"] == "identify_error"
    assert row["error"] == "ValueError: broken input"
    assert calls == [{"check_visible": False, "check_invisible": False}]


def test_gap_summary_does_not_report_a_negative_hidden_count(capsys):
    row = gap._base_row("candidate.png", ".png", "1.2.3")
    row["candidate_classes"] = "blind_marker"

    gap._summarize([row])

    output = capsys.readouterr()
    assert "more candidate row" not in output.out + output.err


def test_sidecar_regression_prefers_stable_signal_names():
    sidecar = {"signals": ["visible_qwen"], "watermarks": ["wording may change"]}
    assert regression.sidecar_families(sidecar) == {"visible_qwen"}


def test_current_report_uses_signal_names_not_watermark_prose():
    report = SimpleNamespace(
        signals=[SimpleNamespace(name="visible_liblib")],
        watermarks=["new wording not known to the legacy mapper"],
    )
    assert regression.report_families(report) == {"visible_liblib"}


def test_current_report_keeps_watermark_only_synthid_family():
    report = SimpleNamespace(signals=[], watermarks=["SynthID watermark (Google)"])
    assert regression.report_families(report) == {"synthid"}


def test_registry_drives_current_visible_families():
    from remove_ai_watermarks.watermark_registry import known_marks

    for mark in known_marks():
        expected = "visible_sparkle" if mark.key == "gemini" else f"visible_{mark.key}"
        assert regression.family_of(mark.label) == expected


def test_registry_keys_cover_legacy_visible_wording():
    assert regression.family_of("Qwen 千问 AI label") == "visible_qwen"
    assert regression.family_of("Microsoft AI badge") == "visible_microsoft"
