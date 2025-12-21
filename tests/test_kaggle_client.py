import importlib
import sys
import types


class DummyDataset:
    def __init__(self, title=None):
        self.title = title


def _install_dummy_kaggle(monkeypatch):
    dummy_kaggle = types.ModuleType("kaggle")
    dummy_kaggle.__path__ = []

    dummy_api_pkg = types.ModuleType("kaggle.api")
    dummy_api_pkg.__path__ = []

    dummy_api_extended = types.ModuleType("kaggle.api.kaggle_api_extended")

    class DummyKaggleApi:
        def authenticate(self):
            return None

    dummy_api_extended.KaggleApi = DummyKaggleApi
    dummy_api_pkg.kaggle_api_extended = dummy_api_extended
    dummy_kaggle.api = dummy_api_pkg

    monkeypatch.setitem(sys.modules, "kaggle", dummy_kaggle)
    monkeypatch.setitem(sys.modules, "kaggle.api", dummy_api_pkg)
    monkeypatch.setitem(
        sys.modules, "kaggle.api.kaggle_api_extended", dummy_api_extended
    )


def test_kaggle_test_skips_max_size(monkeypatch):
    _install_dummy_kaggle(monkeypatch)
    KaggleClient = importlib.import_module("video_announce.kaggle_client").KaggleClient

    called_kwargs = {}

    class StubApi:
        def dataset_list(self, **kwargs):
            nonlocal called_kwargs
            called_kwargs = kwargs
            return [DummyDataset(title="Sample Title")]

    client = KaggleClient()
    monkeypatch.setattr(client, "_get_api", lambda: StubApi())

    result = client.kaggle_test()

    assert result == "Sample Title"
    assert called_kwargs == {"page": 1}


def test_kaggle_test_handles_missing_titles(monkeypatch):
    _install_dummy_kaggle(monkeypatch)
    KaggleClient = importlib.import_module("video_announce.kaggle_client").KaggleClient

    class StubApi:
        def dataset_list(self, **kwargs):
            assert "max_size" not in kwargs
            return [DummyDataset(), DummyDataset(title=None)]

    client = KaggleClient()
    monkeypatch.setattr(client, "_get_api", lambda: StubApi())

    result = client.kaggle_test()

    assert result == "ok (datasets=2)"
