import typing as tp
from pathlib import Path
from dataclasses import dataclass

LikePath = tp.Union[str, Path]


class DatasetRoot:
    def __init__(self, root: LikePath):
        self._root = Path(root).absolute()
        self._original = self._root / "original"
        self._prepared = self._root / "bimcvcovid19i"

    @property
    def root(self) -> Path:
        return self._root

    @property
    def original(self) -> Path:
        return self._original

    @property
    def prepared(self) -> Path:
        return self._prepared


class BIMCVCOVID19Root(DatasetRoot):
    def __init__(self, root):
        super().__init__(root)
        self._prepared_series = self.prepared / "series"
        self._prepared_sessions = self.prepared / "sessions"
        self._prepared_subjects = self.prepared / "subjects"

    @property
    def prepared_series(self) -> Path:
        return self._prepared_series

    @property
    def prepared_sessions(self) -> Path:
        return self._prepared_sessions

    @property
    def prepared_subjects(self) -> Path:
        return self._prepared_subjects

@dataclass
class Subject:
    subject_id: str
    modalities: tp.List[str]
    age: float
    gender: str


@dataclass
class Session:
    subject_id: str
    session_id: str
    study_date: str
    medical_evaluation: str


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
class Test:
    subject_id: str
    date: str
    test: str
    result: str

    def to_dict(self):
        state = self.__dict__.copy()
        state.pop("subject_id")
        return state
