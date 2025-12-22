from __future__ import annotations

import json
import logging
import os
import random
import shutil
import tempfile
from pathlib import Path
from typing import Iterable

KaggleApi = None  # type: ignore[assignment]
_KAGGLE_IMPORT_ERROR: Exception | None = None

try:  # pragma: no cover - optional dependency
    from kaggle.api.kaggle_api_extended import KaggleApi as ImportedKaggleApi
except SystemExit as exc:  # pragma: no cover - missing credentials trigger sys.exit
    _KAGGLE_IMPORT_ERROR = exc
except Exception as exc:  # pragma: no cover - handled at runtime
    _KAGGLE_IMPORT_ERROR = exc
else:
    KaggleApi = ImportedKaggleApi  # type: ignore[assignment]
    _KAGGLE_IMPORT_ERROR = None

from models import Event

logger = logging.getLogger(__name__)

DEFAULT_KERNEL_PATH = Path(__file__).resolve().parent.parent / "kaggle" / "VideoAfisha"


class KaggleClient:
    """Helper for interacting with Kaggle kernels and datasets.

    Besides providing lightweight scoring for local ranking, this client wraps
    a few Kaggle API calls needed to publish kernels that render the video
    announcement.
    """

    def __init__(self, seed: int | None = None):
        self._rand = random.Random(seed)
        self._api: KaggleApi | None = None

    # --- Local scoring fallback used in selection.py ---
    def score(self, events: Iterable[Event]) -> dict[int, float]:
        scores: dict[int, float] = {}
        for e in events:
            weight = e.video_include_count or 0
            weight += min(e.photo_count, 4) * 0.5
            if e.is_free:
                weight += 0.25
            rarity = 1.0 / (1 + (len(e.topics or []))) if hasattr(e, "topics") else 1.0
            jitter = self._rand.random() * 0.1
            scores[e.id] = round(weight + rarity + jitter, 3)
        return scores

    def rank(self, events: Iterable[Event]) -> list[Event]:
        scored = self.score(events)
        return sorted(
            events,
            key=lambda ev: (-scored.get(ev.id, 0.0), ev.date, ev.time, ev.id),
        )

    # --- Kaggle API helpers ---
    def _get_api(self) -> KaggleApi:
        if self._api is None:
            if KaggleApi is None:
                raise RuntimeError(
                    "Kaggle API is unavailable. Install kaggle and configure credentials."
                ) from _KAGGLE_IMPORT_ERROR
            api = KaggleApi()
            api.authenticate()
            self._api = api
        return self._api

    def create_dataset(
        self,
        folder: str | Path,
        *,
        public: bool = False,
        quiet: bool = True,
        convert_to_csv: bool = False,
        dir_mode: str = "zip",
    ) -> None:
        api = self._get_api()
        api.dataset_create_new(
            str(folder),
            public=public,
            quiet=quiet,
            convert_to_csv=convert_to_csv,
            dir_mode=dir_mode,
        )

    def delete_dataset(self, dataset: str, *, no_confirm: bool = True) -> None:
        api = self._get_api()
        if "/" in dataset:
            owner_slug, dataset_slug = dataset.split("/", 1)
        else:
            owner_slug = os.getenv("KAGGLE_USERNAME") or ""
            dataset_slug = dataset
        api.dataset_delete(owner_slug, dataset_slug, no_confirm=no_confirm)

    def push_kernel(
        self,
        *,
        dataset_sources: list[str] | None = None,
        kernel_path: str | Path | None = None,
        timeout: str | None = None,
    ) -> None:
        base_path = Path(kernel_path) if kernel_path else DEFAULT_KERNEL_PATH
        if not base_path.exists():
            raise FileNotFoundError(f"Kernel path not found: {base_path}")
        api = self._get_api()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            for item in base_path.iterdir():
                dest = tmp_path / item.name
                if item.is_dir():
                    shutil.copytree(item, dest)
                else:
                    shutil.copy2(item, dest)
            meta_path = tmp_path / "kernel-metadata.json"
            meta_data = json.loads(meta_path.read_text(encoding="utf-8"))
            if dataset_sources is not None:
                meta_data["dataset_sources"] = dataset_sources
                meta_path.write_text(json.dumps(meta_data, ensure_ascii=False, indent=2))
            api.kernels_push(str(tmp_path), timeout=timeout)

    def get_kernel_status(self, kernel_ref: str) -> dict:
        api = self._get_api()
        return api.kernels_status(kernel_ref)

    def download_kernel_output(
        self, kernel_ref: str, *, path: str | Path, force: bool = True, quiet: bool = False
    ) -> list[str]:
        api = self._get_api()
        files, _ = api.kernels_output(
            kernel_ref, path=str(path), force=force, quiet=quiet
        )
        return files

    def kaggle_test(self) -> str:
        api = self._get_api()
        datasets = api.dataset_list(page=1) or []
        titles = [d.title for d in datasets if getattr(d, "title", None)]
        if titles:
            return titles[0]
        return f"ok (datasets={len(datasets)})"
