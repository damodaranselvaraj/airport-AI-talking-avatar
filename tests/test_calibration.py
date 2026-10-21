"""Task 3: calibration maths and script plumbing (no network)."""
import json

import pytest

from scripts.calibrate_evidence import load_questions, percentile, suggest_thresholds, top_scores
from tests.fakes import GATE_C22, PRAYER


def test_percentile_interpolates():
    assert percentile([1, 2, 3, 4, 5], 0) == 1
    assert percentile([1, 2, 3, 4, 5], 100) == 5
    assert percentile([1, 2, 3, 4, 5], 50) == 3
    assert percentile([0.0, 1.0], 25) == pytest.approx(0.25)
    with pytest.raises(ValueError):
        percentile([], 50)


def test_separated_distributions_use_in_scope_p10():
    s = suggest_thresholds([0.9, 0.85, 0.8, 0.75, 0.7, 0.95, 0.88, 0.82, 0.79, 0.91], [0.05, 0.1, 0.2])
    assert not s["overlap"]
    assert 0.7 <= s["strong"] <= 0.8
    assert s["medium"] <= s["strong"]
    assert s["in_scope_coverage"] >= 0.9


def test_overlap_raises_strong_above_best_trap():
    s = suggest_thresholds([0.3, 0.4, 0.5, 0.6, 0.7], [0.55, 0.2])
    assert s["overlap"] and s["strong"] > 0.55
    assert s["medium"] <= s["strong"]


def test_no_traps_still_works():
    s = suggest_thresholds([0.6, 0.7, 0.8], [])
    assert s["strong"] >= 0.6 and s["max_trap_score"] == 0.0


def test_top_scores_records_source_hits_and_rejects_rerank_failure():
    def retrieve(q, k):
        return ([GATE_C22, PRAYER], None)

    rows = top_scores(retrieve, [{"id": "a", "question": "q", "expected_source": "gates.md"},
                                 {"id": "b", "question": "q", "expected_source": "hours.md"}], "question", 5)
    assert rows[0]["top"] == pytest.approx(0.93) and rows[0]["hit"] is True
    assert rows[1]["hit"] is False
    with pytest.raises(RuntimeError):
        top_scores(lambda q, k: ([], object()), [{"question": "q"}], "question", 5)


def test_load_questions_accepts_both_shapes(tmp_path):
    p = tmp_path / "q.json"
    p.write_text(json.dumps([{"question": "a"}]), encoding="utf-8")
    assert load_questions(p) == [{"question": "a"}]
    p.write_text(json.dumps({"questions": [{"question": "b"}]}), encoding="utf-8")
    assert load_questions(p) == [{"question": "b"}]
    p.write_text(json.dumps([{"id": "x"}]), encoding="utf-8")
    with pytest.raises(ValueError):
        load_questions(p)


def test_script_exits_cleanly_when_gold_set_missing(tmp_path, capsys):
    from scripts.calibrate_evidence import main

    assert main(["--questions", str(tmp_path / "missing.json")]) == 2
    assert "format" in capsys.readouterr().err.lower()
