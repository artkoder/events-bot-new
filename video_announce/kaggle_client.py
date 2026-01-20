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

# Root directory containing all kernel folders
KERNELS_ROOT_PATH = Path(__file__).resolve().parent.parent / "kaggle"
# Default kernel (first local one added)
DEFAULT_KERNEL_PATH = KERNELS_ROOT_PATH / "VideoAfishaEventsBot"
# Prefix to identify local kernels in kernel_ref
LOCAL_KERNEL_PREFIX = "local:"


def list_local_kernels() -> list[dict]:
    """List all valid kernel folders in the repository's kaggle/ directory.
    
    Returns list of dicts with 'ref', 'title', 'path' keys.
    A valid kernel folder must contain kernel-metadata.json.
    """
    if not KERNELS_ROOT_PATH.exists():
        return []
    
    kernels = []
    for folder in KERNELS_ROOT_PATH.iterdir():
        if not folder.is_dir():
            continue
        meta_path = folder / "kernel-metadata.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            title = meta.get("title") or folder.name
            # Use local: prefix to distinguish from Kaggle kernels
            ref = f"{LOCAL_KERNEL_PREFIX}{folder.name}"
            kernels.append({
                "ref": ref,
                "title": title,
                "path": str(folder),
                "is_local": True,
            })
        except Exception:
            logger.warning("Failed to parse kernel metadata in %s", folder)
            continue
    return kernels


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
        logger.info("kaggle: creating dataset from folder=%s", folder)
        api.dataset_create_new(
            str(folder),
            public=public,
            quiet=quiet,
            convert_to_csv=convert_to_csv,
            dir_mode=dir_mode,
        )
        logger.info("kaggle: dataset created successfully from folder=%s", folder)

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
        logger.info("kaggle: preparing kernel push from %s", base_path.resolve())
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
            files = sorted(
                (f.relative_to(tmp_path).as_posix(), f.stat().st_size)
                for f in tmp_path.rglob("*")
                if f.is_file()
            )
            logger.info("kaggle: pushing kernel files=%s", files)
            api.kernels_push(str(tmp_path), timeout=timeout)

    def kernels_list(self, user: str, page_size: int = 20) -> list[dict]:
        api = self._get_api()
        # api.kernels_list returns a list of objects, convert to dict for easier usage
        kernels = api.kernels_list(user=user, page_size=page_size)
        return [
            {
                "ref": getattr(k, "ref", ""),
                "title": getattr(k, "title", ""),
                "slug": getattr(k, "slug", ""),
                "lastRunTime": getattr(k, "lastRunTime", None),
            }
            for k in kernels
        ]

    def kernels_pull(
        self, kernel_ref: str, path: Path | str, metadata: bool = True
    ) -> None:
        api = self._get_api()
        api.kernels_pull(kernel_ref, path=str(path), metadata=metadata)

    def deploy_kernel_update(self, kernel_ref: str, dataset_slug: str) -> str:
        """Deploy kernel with dataset sources updated.
        
        HYBRID approach:
        - If kernel_ref starts with 'local:', use code from repository
        - Otherwise, pull from Kaggle (original behavior)
        """
        import time
        api = self._get_api()
        
        is_local = kernel_ref.startswith(LOCAL_KERNEL_PREFIX)
        
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            
            if is_local:
                # Extract folder name from local:FolderName
                folder_name = kernel_ref[len(LOCAL_KERNEL_PREFIX):]
                local_kernel_path = KERNELS_ROOT_PATH / folder_name
                
                if not local_kernel_path.exists():
                    raise FileNotFoundError(f"Local kernel path not found: {local_kernel_path}")
                
                logger.info(
                    "kaggle: deploying LOCAL kernel folder=%s dataset=%s",
                    folder_name, dataset_slug
                )
                logger.info(
                    "kaggle: local kernel path resolved=%s",
                    local_kernel_path.resolve(),
                )
                
                # Copy local kernel files to temp directory
                for item in local_kernel_path.iterdir():
                    dest = tmp_path / item.name
                    if item.is_dir():
                        shutil.copytree(item, dest)
                    else:
                        shutil.copy2(item, dest)
                logger.info("kaggle: copied local kernel from %s", local_kernel_path)
            else:
                # Pull from Kaggle (original behavior)
                logger.info(
                    "kaggle: deploying REMOTE kernel ref=%s dataset=%s",
                    kernel_ref, dataset_slug
                )
                api.kernels_pull(kernel_ref, path=str(tmp_path), metadata=True)
                logger.info("kaggle: pulled kernel from Kaggle")
            
            meta_path = tmp_path / "kernel-metadata.json"
            if not meta_path.exists():
                raise FileNotFoundError(f"kernel-metadata.json not found")

            meta_data = json.loads(meta_path.read_text(encoding="utf-8"))
            
            # Set dataset sources for this session
            meta_data["dataset_sources"] = [dataset_slug]
            # Ensure internet is enabled for pip installs
            meta_data["enable_internet"] = True
            
            logger.info(
                "kaggle: kernel metadata updated id=%s dataset_sources=%s",
                meta_data.get("id"),
                meta_data.get("dataset_sources"),
            )

            meta_path.write_text(json.dumps(meta_data, ensure_ascii=False, indent=2))

            files = sorted(
                (f.relative_to(tmp_path).as_posix(), f.stat().st_size)
                for f in tmp_path.rglob("*")
                if f.is_file()
            )
            logger.info("kaggle: pushing kernel files=%s", files)
            api.kernels_push(str(tmp_path))
            result_ref = str(meta_data.get("id") or meta_data.get("slug") or kernel_ref)
            logger.info("kaggle: kernel deployed successfully ref=%s", result_ref)
            
            # Wait for Kaggle to propagate metadata changes before kernel starts
            logger.info("kaggle: waiting 10s for metadata to propagate...")
            time.sleep(10)
            
            return result_ref


    def get_kernel_status(self, kernel_ref: str) -> dict:
        api = self._get_api()
        logger.debug("kaggle: getting kernel status for %s", kernel_ref)
        response = api.kernels_status(kernel_ref)
        
        # Convert API response object to dict for .get() access
        # Priority: to_dict() > parse string repr > getattr status
        if hasattr(response, 'to_dict'):
            result = response.to_dict()
        elif hasattr(response, '__str__'):
            # Response might be like {"status": "COMPLETE", "failureMessage": null}
            try:
                result = json.loads(str(response))
            except (json.JSONDecodeError, TypeError):
                result = {}
        else:
            result = {}
        
        # Fallback: get status directly from response object
        if not result.get("status"):
            status_val = getattr(response, 'status', None)
            if status_val is not None:
                # Handle enum values like KernelWorkerStatus.COMPLETE
                result["status"] = status_val.name if hasattr(status_val, 'name') else str(status_val)
        
        # Also try to get failure message
        if not result.get("failureMessage"):
            fail_msg = getattr(response, 'failure_message', None) or getattr(response, 'failureMessage', None)
            if fail_msg:
                result["failureMessage"] = fail_msg
        
        logger.info(
            "kaggle: kernel status kernel=%s status=%s failure=%s",
            kernel_ref,
            result.get("status"),
            result.get("failureMessage") or result.get("failure_message"),
        )
        return result

    def download_kernel_output(
        self, kernel_ref: str, *, path: str | Path, force: bool = True, quiet: bool = False
    ) -> list[str]:
        api = self._get_api()
        logger.info("kaggle: downloading kernel output kernel=%s path=%s", kernel_ref, path)
        files, _ = api.kernels_output(
            kernel_ref, path=str(path), force=force, quiet=quiet
        )
        logger.info("kaggle: downloaded %s files: %s", len(files), files)
        return files

    def kaggle_test(self) -> str:
        api = self._get_api()
        datasets = api.dataset_list(page=1) or []
        titles = [d.title for d in datasets if getattr(d, "title", None)]
        if titles:
            return titles[0]
        return f"ok (datasets={len(datasets)})"
