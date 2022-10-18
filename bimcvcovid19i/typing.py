import typing as tp
from pathlib import Path
from dataclasses import dataclass

LikePath = tp.Union[str, Path]


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
