import asyncio
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "artifacts" / "codex"))

import smart_update_lollipop_facts_merge_family_v2_16_2_iter5_2026_03_09 as merge


def test_bucket_result_matches_merge_records_when_record_sets_align() -> None:
    bucket_result = {
        "payload": {
            "decisions": [
                {"record_id": "R_B01"},
                {"record_id": "R_E01"},
            ]
        }
    }
    merge_records = [
        {"record_id": "R_E01"},
        {"record_id": "R_B01"},
    ]

    assert merge._bucket_result_matches_merge_records(bucket_result, merge_records) is True


def test_bucket_result_matches_merge_records_detects_stale_hydrated_bucket() -> None:
    bucket_result = {
        "payload": {
            "decisions": [
                {"record_id": "R_B01"},
                {"record_id": "R_E02"},
            ]
        }
    }
    merge_records = [
        {"record_id": "R_B01"},
        {"record_id": "R_E01"},
    ]

    assert merge._bucket_result_matches_merge_records(bucket_result, merge_records) is False


def test_load_or_rerun_bucket_result_recomputes_when_hydrated_bucket_is_stale(tmp_path: Path, monkeypatch) -> None:
    trace_root = tmp_path / "trace"
    event_id = 2657
    stale_stage_dir = trace_root / str(event_id) / "facts.merge.bucket.v2"
    stale_stage_dir.mkdir(parents=True)
    (stale_stage_dir / "result.json").write_text("{}", encoding="utf-8")

    iter3_stage_dir = tmp_path / "iter3" / str(event_id) / "facts.merge.bucket.v2"
    iter3_stage_dir.mkdir(parents=True)

    monkeypatch.setattr(merge, "ITER3_TRACE_ROOT", tmp_path / "iter3")
    monkeypatch.setattr(
        merge,
        "_hydrate_bucket_result",
        lambda event_id, trace_root: {
            "payload": {"decisions": [{"record_id": "R_B01"}, {"record_id": "R_E02"}]}
        },
    )

    called = {}

    async def fake_run_bucket_stage(*, event_ctx, trace_root, baseline_facts, merge_records):
        called["event_id"] = event_ctx["event"]["id"]
        called["record_ids"] = [record["record_id"] for record in merge_records]
        return {"payload": {"decisions": [{"record_id": "R_B01"}, {"record_id": "R_E01"}]}}

    monkeypatch.setattr(merge.base, "_run_bucket_stage", fake_run_bucket_stage)

    result = asyncio.run(
        merge._load_or_rerun_bucket_result(
            event_ctx={"event": {"id": event_id}},
            trace_root=trace_root,
            baseline_facts=[],
            merge_records=[{"record_id": "R_B01"}, {"record_id": "R_E01"}],
        )
    )

    assert called["event_id"] == event_id
    assert called["record_ids"] == ["R_B01", "R_E01"]
    assert result["payload"]["decisions"][1]["record_id"] == "R_E01"
    assert not stale_stage_dir.exists()


def test_filter_event_ids_preserves_requested_subset_order() -> None:
    filtered = merge.base._filter_event_ids([2673, 2657, 2759], {2759, 2673})
    assert filtered == [2673, 2759]
