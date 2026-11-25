"""``report``: the page is made from its JSON alone and shows every section."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from loki_survey.collect import collect
from loki_survey.config import Analysis, Collect, Evaluate
from loki_survey.evaluate import evaluate
from loki_survey.report import render_html, write_report
from loki_survey.rundir import atomic_write_json

from .synthetic import PULSAR, make_run, resolved

if TYPE_CHECKING:
    from pathlib import Path


def test_report_shows_units_cost_candidates_catalogue_and_ledger(
    tmp_path: Path,
) -> None:
    run, units = make_run(tmp_path)
    (run.root / "cat.csv").write_text(f"name,f0,dm\nPSR_A,{PULSAR},10.0\n")
    analysis = Analysis(
        collect=Collect(drop_undispersed=True),
        evaluate=Evaluate(catalogue="cat.csv"),
    )
    res = resolved(analysis)
    atomic_write_json(run.resolved_json, {"versions": {"loki": "test"}})
    collect(run, res, units)
    evaluate(run, res)
    doc = write_report(run, res, units)

    assert doc["schema"] == "loki-survey report 1"
    assert doc["cost"]["core_hours"] == len(units) * 10.0 * 2 / 3600.0
    rows = doc["candidates"]["rows"]
    matched = [
        (r["catalogue"]["name"], r["catalogue"]["p"], r["catalogue"]["q"])
        for r in rows
        if r["catalogue"]
    ]
    assert matched == [("PSR_A", 1, 1), ("PSR_A", 2, 1)]  # the pulsar and 2 f
    assert doc["ledger"]["by_rule"] == {"drop_undispersed": 1}
    assert doc["catalogue"]["pulsars"][0]["found"]

    page = (run.report_dir / "index.html").read_text()
    for text in ("Units", "Cost", "Candidates", "Catalogue", "Ledger", "PSR_A"):
        assert text in page
    assert "drop_undispersed" in page
    assert "<svg" in page  # the pulsar's S/N against DM
    # The page is a function of the JSON alone.
    saved = json.loads((run.report_dir / "report.json").read_text())
    assert render_html(saved) == page


def test_report_leaves_out_an_evaluation_of_an_earlier_collect(tmp_path: Path) -> None:
    run, units = make_run(tmp_path)
    (run.root / "cat.csv").write_text(f"name,f0,dm\nPSR_A,{PULSAR},10.0\n")
    res = resolved(Analysis(evaluate=Evaluate(catalogue="cat.csv")))
    atomic_write_json(run.resolved_json, {"versions": {}})
    collect(run, res, units)
    evaluate(run, res)
    assert write_report(run, res, units)["stale"] == []

    # collect again under other rules: its groups are renumbered, and the
    # evaluation's labels would land on other groups.
    rules = Analysis(
        collect=Collect(drop_undispersed=True),
        evaluate=Evaluate(catalogue="cat.csv"),
    )
    collect(run, resolved(rules), units)
    doc = write_report(run, resolved(rules), units)
    assert doc["stale"] == ["evaluate"]
    assert doc["catalogue"] is None
    assert doc["lines"] is None
    assert not any(r["catalogue"] for r in doc["candidates"]["rows"])
    assert (
        "Left out, made from an earlier collect: evaluate"
        in (run.report_dir / "index.html").read_text()
    )

    evaluate(run, resolved(rules))
    doc = write_report(run, resolved(rules), units)
    assert doc["stale"] == []
    assert doc["catalogue"]["pulsars"][0]["found"]


def test_report_escapes_names(tmp_path: Path) -> None:
    run, units = make_run(tmp_path)
    (run.root / "cat.csv").write_text(f"name,f0\n<script>x</script>,{PULSAR}\n")
    res = resolved(Analysis(evaluate=Evaluate(catalogue="cat.csv")))
    atomic_write_json(run.resolved_json, {"versions": {}})
    collect(run, res, units)
    evaluate(run, res)
    write_report(run, res, units)
    page = (run.report_dir / "index.html").read_text()
    assert "<script>" not in page
    assert "&lt;script&gt;" in page
