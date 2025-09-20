from main import build_festival_merge_selection, _festival_merge_tokens
from models import Festival


def _fest(fest_id: int, name: str) -> Festival:
    return Festival(id=fest_id, name=name)


def test_festival_merge_candidates_sorted_by_token_overlap():
    source = _fest(1, "Geek Picnic Moscow")
    targets = [
        _fest(2, "Alpha Fest"),
        _fest(3, "Geek Picnic Moscow 2024"),
        _fest(4, "Geek Picnic"),
        _fest(5, "Beta Fest"),
        _fest(6, "Moscow Music"),
    ]

    _, markup = build_festival_merge_selection(source, targets, page=1)

    candidate_ids: list[int] = []
    for row in markup.inline_keyboard:
        button = row[0]
        callback = button.callback_data or ""
        if callback.startswith("festmerge_to:"):
            candidate_ids.append(int(callback.split(":")[2]))

    assert len(candidate_ids) == len(targets)

    source_tokens = _festival_merge_tokens(source)
    targets_by_id = {fest.id: fest for fest in targets}

    for first_id, second_id in zip(candidate_ids, candidate_ids[1:]):
        first = targets_by_id[first_id]
        second = targets_by_id[second_id]
        first_overlap = len(source_tokens & _festival_merge_tokens(first))
        second_overlap = len(source_tokens & _festival_merge_tokens(second))
        assert first_overlap >= second_overlap
        if first_overlap == second_overlap:
            first_name = (first.name or "").lower()
            second_name = (second.name or "").lower()
            assert first_name <= second_name
