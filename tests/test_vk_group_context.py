import pytest
import vk_intake
import main


@pytest.mark.asyncio
async def test_vk_build_event_uses_group_title(monkeypatch):
    captured = {}

    async def fake_parse(text, **kwargs):
        captured['channel_title'] = kwargs.get('channel_title')
        captured['festival_names'] = kwargs.get('festival_names')
        return [{
            'title': 'T',
            'date': '2099-01-01',
            'time': '18:00',
            'location_name': 'Venue'
        }]

    monkeypatch.setattr(main, 'parse_event_via_4o', fake_parse)

    draft, festival_payload = await vk_intake.build_event_payload_from_vk(
        'text', source_name='Group'
    )

    assert captured['channel_title'] == 'Group'
    assert 'festival_names' in captured
    assert captured['festival_names'] is None
    assert draft.venue == 'Venue'
    assert festival_payload is None
