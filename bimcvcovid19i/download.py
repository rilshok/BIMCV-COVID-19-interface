import tarfile
import typing as tp
from pathlib import Path

import pandas as pd

from .typing import BIMCVCOVID19Root, LikePath, Session, Subject
from .webdav import webdav_download_all

from .tools import derepr_medical_evaluation_text


def download_bimcv_covid19_positive(root: LikePath):
    return webdav_download_all(
        root=BIMCVCOVID19Root(root).original,
        webdav_hostname="https://b2drop.bsc.es/public.php/webdav",
        webdav_login="BIMCV-COVID19-cIter_1_2",
        webdav_password="maybeempty",
    )


def download_bimcv_covid19_negative(root: LikePath):
    return webdav_download_all(
        root=BIMCVCOVID19Root(root).original,
        webdav_hostname="https://b2drop.bsc.es/public.php/webdav",
        webdav_login="BIMCV-COVID19-cIter_1_2-Negative",
        webdav_password="maybeempty",
    )


def read_subjects_bimcv_covid19_positive(root: LikePath) -> tp.List[Subject]:
    root = Path(root)
    subjects_file = tarfile.open(root / "covid19_posi_subjects.tar.gz")
    participants_file = subjects_file.extractfile("covid19_posi/participants.tsv")
    dataframe = pd.read_csv(participants_file, sep="\t")
    subjects = []
    for row in dataframe.itertuples():
        if not row.participant.startswith("sub-"):
            continue
        assert row.body_parts == "[['CHEST']]"

        subject_uid = row.participant

        modalities = row.modality_dicom
        for character in "[']":
            modalities = modalities.replace(character, "")
        modalities = modalities.split(", ")

        age = row.age
        for character in "[']":
            age = age.replace(character, "")
        age = list(map(float, age.split(", "))) if age else None
        age = sum(age) / len(age) if age else None

        gender = row.gender
        if gender == "None":
            gender = None
        if modalities == [""]:
            modalities = []

        subject = Subject(
            subject_id=subject_uid,
            modalities=modalities,
            age=age,
            gender=gender,
        )
        subjects.append(subject)
    return subjects


def read_sessions_bimcv_covid19_positive(root_bimcv_covid19) -> tp.List[Session]:
    root = Path(root_bimcv_covid19)
    sessions = []

    all_sessions_file = tarfile.open(root / "covid19_posi_sessions_tsv.tar.gz")
    for sesions_file_member in all_sessions_file.getmembers():
        subject_id = sesions_file_member.name.split("/")[1]
        assert subject_id.startswith("sub-")

        sesions_file = all_sessions_file.extractfile(sesions_file_member)
        sesions_dataframe = pd.read_csv(sesions_file, sep="\t")

        for sesion_row in sesions_dataframe.itertuples():
            session_id = sesion_row.session_id
            study_date = sesion_row.study_date
            medical_evaluation = sesion_row.medical_evaluation

            assert session_id.startswith("ses-")

            if study_date != study_date:
                study_date = None
            else:
                study_date = str(int(study_date))
                assert len(study_date) == 8, study_date
                study_date = f"{study_date[:4]}-{study_date[4:6]}-{study_date[6:]}"

            medical_evaluation = derepr_medical_evaluation_text(medical_evaluation)

            session = Session(
                subject_id=subject_id,
                session_id=session_id,
                study_date=study_date,
                medical_evaluation=medical_evaluation,
            )
            sessions.append(session)

    return sessions


def extract_bimcv_covid19_positive(root: LikePath):
    dsroot = BIMCVCOVID19Root(root)
    assert dsroot.original.exists()
    # TODO: check sums
    # contains empty objects
    subjects = read_subjects_bimcv_covid19_positive(dsroot.original)
    sessions = read_sessions_bimcv_covid19_positive(dsroot.original)
    assert {sub.subject_id for sub in subjects} == {ses.subject_id for ses in sessions}
