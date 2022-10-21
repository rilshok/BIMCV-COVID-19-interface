import itertools as it
import logging
import operator as op
import shutil
import tarfile
import typing as tp
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd  # type: ignore
from tqdm import tqdm  # type: ignore

from . import tools
from .typing import (
    BIMCVCOVID19Root,
    Labels,
    LikePath,
    SeriesRawPath,
    Session,
    Subject,
    Test,
)
from .webdav import webdav_download_all


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


def bimcv_covid19_extract_subjects(dataframe: pd.DataFrame) -> tp.List[Subject]:
    subjects = []
    for row in dataframe.itertuples():
        if not row.participant.startswith("sub-"):
            continue
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
            uid=subject_uid,
            modalities=modalities,
            age=age,
            gender=gender,
            series_ids=[],
            sessions_ids=[],
            tests=None,
        )
        subjects.append(subject)
    return subjects


def subjects_bimcv_covid19_positive(root: LikePath) -> tp.List[Subject]:
    path = Path(root) / "covid19_posi_subjects.tar.gz"
    subpath = "covid19_posi/participants.tsv"
    with tools.open_from_tar(path, subpath) as file:
        dataframe = pd.read_csv(file, sep="\t")
    return bimcv_covid19_extract_subjects(dataframe)


def subjects_bimcv_covid19_negative(root: LikePath) -> tp.List[Subject]:
    path = Path(root) / "covid19_neg_metadata.tar.gz"
    subpath = "covid19_neg/participants.tsv"
    with tools.open_from_tar(path, subpath) as file:
        dataframe = pd.read_csv(file, sep="\t")
    return bimcv_covid19_extract_subjects(dataframe)


def bimcv_covid19_read_sessions(path: LikePath) -> tp.List[Session]:
    """read sessions from *sessions_tsv.tar.gz file"""
    path = Path(path)
    sessions = []
    all_sessions_file = tarfile.open(path)
    for sesions_file_member in all_sessions_file.getmembers():
        subject_id = sesions_file_member.name.split("/")[1]
        assert subject_id.startswith("sub-")

        sesions_file = all_sessions_file.extractfile(sesions_file_member)
        assert sesions_file is not None
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

            medical_evaluation = tools.derepr_medical_evaluation_text(
                medical_evaluation
            )

            session = Session(
                uid=session_id,
                subject_id=subject_id,
                study_date=study_date,
                medical_evaluation=medical_evaluation,
                series_modalities=[],
                series_ids=[],
                labels=None,
            )
            sessions.append(session)
    all_sessions_file.close()
    return sessions


def bimcv_covid19_positive_read_sessions(root: LikePath) -> tp.List[Session]:
    path = Path(root) / "covid19_posi_sessions_tsv.tar.gz"
    return bimcv_covid19_read_sessions(path)


def bimcv_covid19_negative_read_sessions(root: LikePath) -> tp.List[Session]:
    "processing covid19_neg_sessions_tsv.tar.gz"
    path = Path(root) / "covid19_neg_sessions_tsv.tar.gz"
    return bimcv_covid19_read_sessions(path)


def iterate_sessions_bimcv_covid19_positive(root: LikePath) -> tp.Iterator[Path]:
    root = Path(root)
    part_paths = sorted(list(root.glob("*part*.tar.gz")))
    with TemporaryDirectory() as temp_root:
        temp_root_ = Path(temp_root)
        for part_path in part_paths:
            logging.info("Start processing tar file: %s", part_path.name)
            part_file = tarfile.open(part_path)
            members = part_file.getmembers()

            # extracting root paths for sessions inside an archive
            sessions_root_paths = {}
            for member in members:
                member_path = member.name
                member_path_split = member_path.split("/")
                if not member_path_split[-1].startswith("ses-"):
                    continue
                if not member_path_split[-2].startswith("sub-"):
                    continue
                session_id = member_path_split[-1]
                assert member.isdir()
                sessions_root_paths[session_id] = member_path

            # extract paths to each file inside an archive.
            # distribution of paths by sessions
            sessions_element_paths: tp.Dict[str, tp.List[str]] = {
                uid: [] for uid in sessions_root_paths
            }

            for member in members:
                if not member.isfile():
                    continue
                member_path = member.name

                for uid, sessions_root in sessions_root_paths.items():
                    if member_path.startswith(sessions_root):
                        sessions_element_paths[uid].append(member_path)
                        break
                else:
                    raise AssertionError

            for session_id, part_file_paths in sessions_element_paths.items():
                # Extract all files of one session to a temporary folder
                session_root = temp_root_ / session_id
                session_root.mkdir()
                part_file_session_root = Path(sessions_root_paths[session_id])
                for path in part_file_paths:
                    file_path = session_root / Path(path).relative_to(
                        part_file_session_root
                    )
                    file_path.parent.mkdir(parents=True, exist_ok=True)

                    with open(file_path, "wb") as file:
                        data = part_file.extractfile(path)
                        assert data is not None
                        file.write(data.read())
                yield session_root
                shutil.rmtree(session_root)
            part_file.close()


def group_series_files_by_name(session_root: Path) -> tp.Iterator[SeriesRawPath]:
    paths = list(session_root.rglob("*"))
    groups = defaultdict(list)

    extensions = [".json", ".tsv", ".nii.gz", ".png"]

    for path in paths:
        if path.is_dir():
            continue
        str_path = str(path)
        assert any(str_path.endswith(ext) for ext in extensions)
        for ext in extensions:
            if str_path.endswith(ext):
                group_name = str_path[: -len(ext)]
                groups[group_name].append(path)
                break

    for group_name, group in groups.items():
        if len(group) == 1:
            check = str(group[0]).endswith("_scans.tsv")
            check &= str(group[0].relative_to(session_root)) == group[0].name
            check &= all(substring in group[0].name for substring in ["sub-", "_ses-"])
            if check:
                continue

        if len(group) > 2:
            logging.warning("skip group. group too long. %s", group)
            continue

        image_path = None
        meta_path = None
        for path in group:
            if str(path).endswith(".json"):
                meta_path = path
                continue
            image_path = path

        yield SeriesRawPath(
            uid=Path(group_name).name,
            image_path=image_path,
            tags_path=meta_path,
        )


def extract_tests(dataframe: pd.DataFrame) -> tp.Dict[str, tp.List[Test]]:
    """Subject grouped tests (PCR, ACT, etc.)"""
    results_map = dict(
        INDETERMINADO="indeterminate",
        NEGATIVO="negative",
        POSITIVO="positive",
    )
    all_tests = list(
        Test(
            subject_id=row.participant,
            date="-".join(row.date.split(".")[::-1]),
            test=row.test,
            result=results_map[row.result],
        )
        for row in dataframe.itertuples()
    )
    # grouping test results by subject ID
    # with sorting by date
    return {
        subject_id: sorted(group, key=op.attrgetter("date"))
        for subject_id, group in it.groupby(all_tests, op.attrgetter("subject_id"))
    }


def extract_labels(dataframe: pd.DataFrame) -> tp.Dict[str, Labels]:
    return {
        row.ReportID: Labels(
            subject_id=row.PatientID,
            session_id=row.ReportID,
            report=tools.derepr_medical_evaluation_text(row.Report),
            labels=tools.derepr_strings_list(row.Labels),
            localizations=tools.derepr_strings_list(row.Localizations),
            labels_localizations_by_sentence=tools.derepr_strings_list(
                row.LabelsLocalizationsBySentence
            ),
            label_CUIS=tools.derepr_CUIS(row.labelCUIS),
            localizations_CUIS=tools.derepr_CUIS(row.LocalizationsCUIS),
        )
        for row in dataframe.itertuples()
    }


def tests_bimcv_covid19_positive(root: LikePath) -> tp.Dict[str, tp.List[Test]]:
    path = Path(root) / "covid19_posi_head.tar.gz"
    subpath = "covid19_posi/derivatives/EHR/sil_reg_covid_posi.tsv"
    with tools.open_from_tar(path, subpath) as file:
        dataframe = pd.read_csv(file, sep="\t")
    return extract_tests(dataframe)


def labels_bimcv_covid19_positive(root: LikePath) -> tp.Dict[str, Labels]:
    path = Path(root) / "covid19_posi_head.tar.gz"
    subpath = "covid19_posi/derivatives/labels/labels_covid_posi.tsv"
    with tools.open_from_tar(path, subpath) as file:
        dataframe = pd.read_csv(file, sep="\t")
    return extract_labels(dataframe)


def extract_bimcv_covid19_positive(root: LikePath):
    dsroot = BIMCVCOVID19Root(root)
    assert dsroot.original.exists()
    logging.info("Source directory: %s", str(dsroot.original))
    logging.info("Destination directory: %s", str(dsroot.prepared))

    logging.info("Extracting information about subjects")
    subjects = subjects_bimcv_covid19_positive(dsroot.original)

    logging.info("Extracting information about sessions")
    sessions = bimcv_covid19_positive_read_sessions(dsroot.original)

    assert set(map(op.attrgetter("uid"), subjects)) == set(
        map(op.attrgetter("subject_id"), sessions)
    )
    subjects_map = {sub.uid: sub for sub in subjects}
    sessions_map = {ses.uid: ses for ses in sessions}

    logging.info("Extracting information about COVID test results")
    tests = tests_bimcv_covid19_positive(dsroot.original)

    logging.info("Extracting session semantic markup")
    labels = labels_bimcv_covid19_positive(dsroot.original)

    logging.info("Creating root directories")
    for directory in [
        dsroot.prepared,
        dsroot.prepared_series,
        dsroot.prepared_sessions,
        dsroot.prepared_subjects,
    ]:
        directory.mkdir(parents=False, exist_ok=True)

    subject_s_series_ids = defaultdict(set)
    subjects_s_sessions = defaultdict(set)

    logging.info("Start extracting sessions")
    session_dirs = iterate_sessions_bimcv_covid19_positive(dsroot.original)
    session_dirs_verbose = tqdm(session_dirs, total=len(sessions))
    for session_root in session_dirs_verbose:
        session_id = session_root.name
        session_s_modalities = set()
        session_s_series_ids = set()
        for series_raw_path in group_series_files_by_name(session_root):
            series = series_raw_path.read_item()
            assert series.session_id == session_id
            assert series.subject_id == sessions_map[session_id].subject_id
            if series.image is None:
                continue
            series.save(dsroot.prepared_series / series.uid)
            session_s_modalities.add(series.modality)
            session_s_series_ids.add(series.uid)
        if not session_s_series_ids:
            continue

        session = sessions_map[session_id]

        session.series_modalities = sorted(session_s_modalities)
        session.series_ids = sorted(session_s_series_ids)
        session.labels = labels.get("session_id")
        session.save(dsroot.prepared_sessions / session.uid)

        subject_s_series_ids[session.subject_id].update(session.series_ids)
        subjects_s_sessions[session.subject_id].add(session.uid)

    for subject_id, series_ids in tqdm(subject_s_series_ids.items()):
        assert series_ids
        subject = subjects_map[subject_id]
        subject.series_ids = list(series_ids)
        subject.sessions_ids = list(subjects_s_sessions[subject_id])
        subject.tests = tests.get(subject_id)
        subject.save(dsroot.prepared_subjects / subject_id)


def extract_bimcv_covid19_negative(root: LikePath):
    dsroot = BIMCVCOVID19Root(root)
    assert dsroot.original.exists()
    logging.info("Source directory: %s", str(dsroot.original))
    logging.info("Destination directory: %s", str(dsroot.prepared))

    logging.info("Extracting information about subjects")
    subjects = subjects_bimcv_covid19_negative(dsroot.original)

    logging.info("Extracting information about sessions")
    sessions = bimcv_covid19_negative_read_sessions(dsroot.original)
