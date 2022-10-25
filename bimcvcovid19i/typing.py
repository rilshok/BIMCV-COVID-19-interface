__all__ = [
    "LikePath",
    "DatasetRoot",
    "Subject",
    "Session",
    "Labels",
    "Test",
    "SeriesRawPath",
]


import typing as tp
from dataclasses import dataclass
from pathlib import Path

import deli  # type: ignore
import numpy as np

from . import tools

LikePath = tp.Union[str, Path]


class DatasetRoot:
    def __init__(
        self,
        root: tp.Optional[LikePath] = None,
        original: tp.Optional[LikePath] = None,
        prepared: tp.Optional[LikePath] = None,
    ):
        if root is not None:
            self._root = Path(root).absolute()
            self._original = self._root / "original"
            self._prepared = self._root / "prepared"
        if original is not None:
            self._original = Path(original).absolute()
        if prepared is not None:
            self._prepared = Path(prepared).absolute()

        if any(p is None for p in [self._original, self._prepared]):
            raise ValueError

    @property
    def original(self) -> Path:
        return self._original

    @property
    def prepared(self) -> Path:
        return self._prepared


@dataclass
class Test:
    subject_id: str
    date: str
    test: str
    result: str

    def to_dict(self):
        state = self.__dict__.copy()
        state.pop("subject_id")
        return state


@dataclass
class Subject:
    uid: str
    age: tp.Optional[float]
    gender: tp.Optional[str]
    tests: tp.Optional[tp.List[Test]]
    sessions_ids: tp.Set[str]
    series_ids: tp.Set[str]
    series_modalities: tp.Set[str]

    def save(self, root: LikePath):
        root = Path(root)
        root.mkdir(exist_ok=True, parents=False)
        deli.save(self.uid, root / "uid.json")
        if self.age is not None and self.age == self.age:
            deli.save(self.age, root / "age.json")
        if self.gender is not None and self.gender == self.gender:
            deli.save(self.gender, root / "gender.json")
        if self.tests is not None:
            tests = [t.to_dict() for t in self.tests if t.subject_id == self.uid]
            deli.save(tests, root / "tests.json")
        deli.save(sorted(self.sessions_ids), root / "sessions_ids.json")
        deli.save(sorted(self.series_ids), root / "series_ids.json")
        deli.save(sorted(self.series_modalities), root / "series_modalities.json")


@dataclass
class Labels:
    subject_id: str
    session_id: str
    report: str
    labels: tp.List[str]
    localizations: tp.List[str]
    labels_localizations_by_sentence: tp.List[str]
    label_CUIS: tp.List[str]
    localizations_CUIS: tp.List[str]

    def to_dict(self):
        state = self.__dict__.copy()
        state.pop("subject_id")
        state.pop("session_id")
        return state


@dataclass
class Session:
    uid: str
    subject_id: str
    study_date: tp.Optional[str]
    medical_evaluation: tp.Optional[str]
    series_modalities: tp.Set[str]
    series_ids: tp.Set[str]
    labels: tp.Optional[Labels]

    def save(self, root: LikePath):
        root = Path(root)
        root.mkdir(exist_ok=True, parents=False)
        deli.save(self.uid, root / "uid.json")
        deli.save(self.subject_id, root / "subject_id.json")
        if self.study_date is not None:
            deli.save(self.study_date, root / "study_date.json")
        if self.medical_evaluation is not None and len(self.medical_evaluation) > 0:
            deli.save(self.medical_evaluation, root / "medical_evaluation.json")
        deli.save(sorted(self.series_modalities), root / "series_modalities.json")
        deli.save(sorted(self.series_ids), root / "series_ids.json")
        if self.labels is not None:
            deli.save(self.labels.to_dict(), root / "labels.json")

    @classmethod
    def load(cls, root: LikePath):
        raise NotImplementedError


Spacing = tp.Union[tp.Tuple[float, float], tp.Tuple[float, float]]


@dataclass
class Series:
    uid: str
    image: tp.Optional[np.ndarray]
    spacing: tp.Optional[tp.Tuple[float, ...]]
    tags: tp.Optional[tp.Dict]
    subject_id: str
    session_id: str
    modality: str

    def save(self, root: LikePath):
        root = Path(root)
        root.mkdir(exist_ok=True, parents=False)
        deli.save(self.uid, root / "uid.json")
        if self.image is not None:
            deli.save(self.image, root / "image.npy.gz", compression=3, timestamp=0)
            deli.save(self.image.shape, root / "shape.json")
        if self.spacing is not None:
            deli.save(list(map(float, self.spacing)), root / "spacing.json")
        if self.tags is not None:
            tools.save_json_gz(self.tags, root / "tags.json.gz", compression=3)
        deli.save(self.subject_id, root / "subject_id.json")
        deli.save(self.session_id, root / "session_id.json")
        deli.save(self.modality, root / "modality.json")

    @classmethod
    def load(cls, root: LikePath) -> "Series":
        raise NotImplementedError


class EmptyFileError(RuntimeError):
    pass


@dataclass
class SeriesRawPath:
    uid: str
    image_path: tp.Optional[Path]
    tags_path: tp.Optional[Path]

    def read_item(self) -> Series:
        image = None
        spacing = None
        if self.image_path is not None:
            try:
                if str(self.image_path).endswith(".png"):
                    image = tools.png2numpy(self.image_path)
                elif str(self.image_path).endswith(".nii.gz"):
                    image = tools.nifty2numpy(self.image_path)
                    spacing = tools.spacing_from_nifty(self.image_path)
                else:
                    raise NotImplementedError(self.image_path)
            except RuntimeError as exc:
                with open(self.image_path, "rb") as file:
                    if file.read() == b"":
                        raise EmptyFileError from exc
                raise exc
            image = tools.down_type(image)

        tags = None
        if self.tags_path is not None:
            tags_data = deli.load(self.tags_path)
            assert isinstance(tags_data, dict)
            tags = tools.parse_dicom_tags(tags_data)
        assert tags is None or isinstance(tags, dict)

        if spacing is None:
            spacing = tools.spacing_from_tags(tags)

        subject_id = None
        session_id = None
        for name_element in self.uid.split("_"):
            if name_element.startswith("sub-"):
                subject_id = name_element
            elif name_element.startswith("ses-"):
                session_id = name_element
            if subject_id is not None and session_id is not None:
                break

        assert subject_id is not None
        assert session_id is not None

        modality = self.uid.rsplit("_", maxsplit=1)[-1].upper()

        return Series(
            uid=self.uid,
            image=image,
            spacing=spacing,
            tags=tags,
            subject_id=subject_id,
            session_id=session_id,
            modality=modality,
        )
