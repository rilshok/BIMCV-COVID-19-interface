from contextlib import suppress
from functools import lru_cache, wraps
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, List, Dict

import numpy as np
from deli import load  # type: ignore

from .data import BIMCVCOVID19Root
from .postprocessing import clean_ct_image  # , process_ct_image
from .tools import load_numpy, load_json_gz


def _none_if_not_found(func: Callable[..., Any]):
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Optional[Any]:
        with suppress(FileNotFoundError):
            result = func(*args, **kwargs)
            return result
        return None

    return wrapper


class BIMCV_COVID19(BIMCVCOVID19Root):
    @staticmethod
    @lru_cache
    def _glob(path: Path) -> Tuple[str, ...]:
        if not path.exists() or not path.is_dir():
            raise ValueError("")
        return tuple(sorted(p.name for p in path.iterdir()))

    @property
    def ids_series(self) -> Tuple[str, ...]:
        return self._glob(self.prepared_series)

    @property
    def ids_series_ct(self) -> Tuple[str, ...]:
        return tuple(uid for uid in self.ids_series if uid.endswith("_ct"))

    @property
    def ids_sessions(self) -> Tuple[str, ...]:
        return self._glob(self.prepared_sessions)

    @property
    def ids_subjects(self) -> Tuple[str, ...]:
        return self._glob(self.prepared_subjects)

    def _assert_exsists(self, path: Path) -> None:
        assert path.exists()

    def _series_dir(self, series_id: str) -> Path:
        root = self.prepared_series / series_id
        self._assert_exsists(root)
        return root

    def _session_dir(self, session_id: str) -> Path:
        root = self.prepared_sessions / session_id
        self._assert_exsists(root)
        return root

    def _subject_dir(self, subject_id: str) -> Path:
        root = self.prepared_subjects / subject_id
        self._assert_exsists(root)
        return root

    def image_raw(self, series_id: str) -> np.ndarray:
        return load_numpy(self._series_dir(series_id) / "image.npy.gz", decompress=True)

    def image(self, series_id: str) -> np.ndarray:
        raw_image = self.image_raw(series_id)
        if series_id.endswith("_ct"):
            processed = clean_ct_image(raw_image)
            # todo: transformatin by process_ct_image
            return processed
        return raw_image

    @_none_if_not_found
    def spacing(self, series_id: str) -> Tuple[float, ...]:
        return tuple(map(float, load(self._series_dir(series_id) / "spacing.json")))

    @_none_if_not_found
    def raw_shape(self, series_id: str) -> Tuple[int, ...]:
        return tuple(map(int, load(self._series_dir(series_id) / "shape.json")))

    @_none_if_not_found
    def modality(self, series_id: str) -> str:
        return load(self._series_dir(series_id) / "modality.json")

    @_none_if_not_found
    def tags(self, series_id: str) -> Optional[dict]:
        return load_json_gz(self._series_dir(series_id) / "tags.json.gz")

    @_none_if_not_found
    def session(self, series_id: str) -> str:
        return load(self._series_dir(series_id) / "session_id.json")

    @_none_if_not_found
    def subject(self, series_id: str) -> str:
        return load(self._series_dir(series_id) / "subject_id.json")

    # session methods
    @_none_if_not_found
    def session_subject(self, session_id: str) -> str:
        return load(self._session_dir(session_id) / "subject_id.json")

    @_none_if_not_found
    def session_series(self, session_id: str) -> List[str]:
        return load(self._session_dir(session_id) / "series_ids.json")

    @_none_if_not_found
    def session_date(self, session_id: str) -> Optional[str]:
        path = self._session_dir(session_id) / "study_date.json"
        return load(path)

    @_none_if_not_found
    def session_medical_evaluation(self, session_id: str) -> Optional[str]:
        return load(self._session_dir(session_id) / "medical_evaluation.json")

    @_none_if_not_found
    def session_modalities(self, session_id: str) -> List[str]:
        return load(self._session_dir(session_id) / "series_modalities.json")

    @_none_if_not_found
    def session_labels(self, session_id: str) -> Dict[str, Any]:
        return load(self._session_dir(session_id) / "labels.json")

    # subject methods

    @_none_if_not_found
    def subject_sessions(self, subject_id: str) -> List[str]:
        path = self._subject_dir(subject_id) / "sessions_ids.json"
        return load(path)

    @_none_if_not_found
    def subject_series(self, subject_id: str) -> List[str]:
        return load(self._subject_dir(subject_id) / "series_ids.json")

    @_none_if_not_found
    def subject_modalities(self, subject_id: str) -> List[str]:
        return load(self._subject_dir(subject_id) / "series_modalities.json")

    @_none_if_not_found
    def subject_age(self, subject_id: str) -> Optional[float]:
        return load(self._subject_dir(subject_id) / "age.json")

    @_none_if_not_found
    def subject_gender(self, subject_id: str) -> str:
        return load(self._subject_dir(subject_id) / "gender.json")
