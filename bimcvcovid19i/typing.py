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
