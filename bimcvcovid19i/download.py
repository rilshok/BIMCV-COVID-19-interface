import contextlib
import functools as ft
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

from . import tools
from .typing import (
    DatasetRoot,
    Labels,
    LikePath,
    Series,
    SeriesRawPath,
    Session,
    Subject,
    Test,
)
from .webdav import webdav_download_all


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


class BIMCVCOVID19:
    webdav_hostname: str
    webdav_login: str
    webdav_password: str

    subjects_tarfile_name: str
    subjects_tarfile_subpath: str

    sessions_tarfile_name: str

    tests_tarfile_name: str
    tests_tarfile_subpath: str

    labels_tarfile_name: str
    labels_tarfile_subpath: str

    def __init__(self, root: LikePath):
        root = Path(root).absolute()
        apply_default_root = [
            "download",
            "subjects",
            "sessions",
            "tests",
            "labels",
        ]
        for method in apply_default_root:
            func_ = getattr(self, method)
            func = ft.partial(func_, root=root)
            setattr(self, method, func)

    @classmethod
    def download(cls, root: LikePath):
        return webdav_download_all(
            root=BIMCVCOVID19Root(root).original,
            webdav_hostname=cls.webdav_hostname,
            webdav_login=cls.webdav_login,
            webdav_password=cls.webdav_password,
        )

    @classmethod
    def subjects(cls, root: LikePath) -> tp.List[Subject]:
        return cls._subjects(
            path=BIMCVCOVID19Root(root).original / cls.subjects_tarfile_name,
            subpath=cls.subjects_tarfile_subpath,
        )

    @classmethod
    def sessions(cls, root: LikePath) -> tp.List[Session]:
        return cls._sessions(
            path=BIMCVCOVID19Root(root).original / cls.sessions_tarfile_name
        )

    @classmethod
    def tests(cls, root: LikePath) -> tp.Dict[str, tp.List[Test]]:
        with contextlib.suppress(FileNotFoundError):
            return cls._tests(
                path=BIMCVCOVID19Root(root).original / cls.tests_tarfile_name,
                subpath=cls.tests_tarfile_subpath,
            )
        return {}

    @classmethod
    def labels(cls, root: LikePath) -> tp.Dict[str, Labels]:
        return cls._labels(
            path=BIMCVCOVID19Root(root).original / cls.labels_tarfile_name,
            subpath=cls.labels_tarfile_subpath,
        )

    @classmethod
    def sessions_iter(cls, root: LikePath) -> tp.Iterator[Path]:
        """Unpacks the next session into a temporary folder and returns the path to it"""
        root = BIMCVCOVID19Root(root).original
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
                    logging.info("Extracting session %s files", session_id)
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

    @classmethod
    def series_iter(cls, root: LikePath) -> tp.Iterator[Series]:
        session_dirs = cls.sessions_iter(root)
        for session_root in session_dirs:
            for series_raw_path in _group_series_files_by_name(session_root):
                logging.info("Series %s reading", series_raw_path.uid)
                yield series_raw_path.read_item()

    @staticmethod
    def _subjects(path: LikePath, subpath: LikePath) -> tp.List[Subject]:
        with tools.open_from_tar(path, subpath) as file:
            dataframe = pd.read_csv(file, sep="\t")

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
                age=age,
                gender=gender,
                tests=None,
                sessions_ids=set(),
                series_ids=set(),
                series_modalities=set(modalities),
            )
            subjects.append(subject)
        return subjects

    @staticmethod
    def _sessions(path: LikePath) -> tp.List[Session]:
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
                    series_modalities=set(),
                    series_ids=set(),
                    labels=None,
                )
                sessions.append(session)
        all_sessions_file.close()
        return sessions

    @staticmethod
    def _tests(path: LikePath, subpath: LikePath) -> tp.Dict[str, tp.List[Test]]:
        """Subject grouped tests (PCR, ACT, etc.)"""
        with tools.open_from_tar(path, subpath) as file:
            dataframe = pd.read_csv(file, sep="\t")

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

    @staticmethod
    def _labels(path: LikePath, subpath: LikePath) -> tp.Dict[str, Labels]:
        with tools.open_from_tar(path, subpath) as file:
            dataframe = pd.read_csv(file, sep="\t")
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

    @classmethod
    def prepare(cls, root: LikePath):
        """
        Extracts the dataset into a new folder structure.
        Makes minor changes to text data.
        """
        dsroot = BIMCVCOVID19Root(root)
        assert dsroot.original.exists()
        logging.info("Source directory: %s", str(dsroot.original))
        logging.info("Destination directory: %s", str(dsroot.prepared))

        logging.info("Extracting information about subjects")
        subjects: tp.List[Subject] = cls.subjects(root)

        logging.info("Extracting information about sessions")
        sessions: tp.List[Session] = cls.sessions(root)

        series_iterator = cls.series_iter(root)
        assert set(map(op.attrgetter("uid"), subjects)) == set(
            map(op.attrgetter("subject_id"), sessions)
        )
        subjects_map = {sub.uid: sub for sub in subjects}
        sessions_map = {ses.uid: ses for ses in sessions}

        logging.info("Extracting information about COVID test results")
        tests: tp.Dict[str, tp.List[Test]] = cls.tests(root)

        logging.info("Extracting session semantic markup")
        labels: tp.Dict[str, Labels] = cls.labels(root)

        logging.info("Creating root directories")
        for directory in [
            dsroot.prepared,
            dsroot.prepared_series,
            dsroot.prepared_sessions,
            dsroot.prepared_subjects,
        ]:
            directory.mkdir(parents=False, exist_ok=True)

        logging.info("Start extracting sessions")

        _sessions_prepared = set()
        for series in series_iterator:
            _sessions_prepared.add(series.session_id)
            logging.info(
                "Processing series %s from session %s. Progress: %s/%s",
                series.uid,
                series.session_id,
                len(_sessions_prepared),
                len(sessions),
            )

            if series.image is None:
                continue

            series.save(dsroot.prepared_series / series.uid)

            sessions_map[series.session_id].series_modalities.add(series.modality)
            sessions_map[series.session_id].series_ids.add(series.uid)
            subjects_map[series.subject_id].series_modalities.add(series.modality)
            subjects_map[series.subject_id].series_ids.add(series.uid)

        for i, session in enumerate(sessions_map.values()):
            logging.info(
                "%s series metadata processing. Progress: %s/%s",
                session.uid,
                i,
                len(sessions),
            )

            if not session.series_ids:
                continue

            session.labels = labels.get(session.uid)
            session.save(dsroot.prepared_sessions / session.uid)

            subjects_map[session.subject_id].sessions_ids.add(session.uid)

        for i, subject in enumerate(subjects_map.values()):
            logging.info(
                "%s subject metadata processing. Progress: %s/%s",
                subject.uid,
                i,
                len(subjects),
            )

            if not subject.series_ids:
                continue

            subject.tests = tests.get(subject.uid)
            subject.save(dsroot.prepared_subjects / subject.uid)


class BIMCVCOVID19positive(BIMCVCOVID19):
    webdav_hostname = "https://b2drop.bsc.es/public.php/webdav"
    webdav_login = "BIMCV-COVID19-cIter_1_2"
    webdav_password = "maybeempty"

    subjects_tarfile_name = "covid19_posi_subjects.tar.gz"
    subjects_tarfile_subpath = "covid19_posi/participants.tsv"

    sessions_tarfile_name = "covid19_posi_sessions_tsv.tar.gz"

    tests_tarfile_name = "covid19_posi_head.tar.gz"
    tests_tarfile_subpath = "covid19_posi/derivatives/EHR/sil_reg_covid_posi.tsv"

    labels_tarfile_name = "covid19_posi_head.tar.gz"
    labels_tarfile_subpath = "covid19_posi/derivatives/labels/labels_covid_posi.tsv"


class BIMCVCOVID19negative(BIMCVCOVID19):
    webdav_hostname = "https://b2drop.bsc.es/public.php/webdav"
    webdav_login = "BIMCV-COVID19-cIter_1_2-Negative"
    webdav_password = "maybeempty"

    subjects_tarfile_name = "covid19_neg_metadata.tar.gz"
    subjects_tarfile_subpath = "covid19_neg/participants.tsv"

    sessions_tarfile_name = "covid19_neg_sessions_tsv.tar.gz"

    # NOTE: it is missing from the dataset
    tests_tarfile_name = "---"
    tests_tarfile_subpath = "---"

    labels_tarfile_name = "covid19_neg_derivative.tar.gz"
    labels_tarfile_subpath = "covid19_neg/derivatives/labels/Labels_covid_NEG_JAN21.tsv"


def download_bimcv_covid19_positive(root: LikePath):
    return BIMCVCOVID19positive.download(root)


def download_bimcv_covid19_negative(root: LikePath):
    return BIMCVCOVID19negative.download(root)


def extract_bimcv_covid19_positive(root: LikePath):
    return BIMCVCOVID19positive.prepare(root)


def extract_bimcv_covid19_negative(root: LikePath):
    return BIMCVCOVID19negative.prepare(root)


def _group_series_files_by_name(session_root: Path) -> tp.Iterator[SeriesRawPath]:
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
            logging.warning(
                "skip group. a group of files does not look like a series. %s",
                group,
            )
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
