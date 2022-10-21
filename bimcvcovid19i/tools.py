""" """

__all__ = [
    "save_json_gz",
    "load_json_gz",
    "resolve_escape_char",
    "remove_double_space",
    "derepr_list",
    "derepr_medical_evaluation_text",
    "skip_empty",
    "derepr_CUIS",
    "nifty2numpy",
    "nifty2numpy",
    "png2numpy",
    "spacing_from_nifty",
    "down_type",
    "spacing_from_tags",
    "parse_dicom_tags",
    "derepr_strings_list",
]

import contextlib
import gzip
import itertools as it
import json
import typing as tp

import nibabel as nib  # type: ignore
import numpy as np
import pydicom
import SimpleITK as sitk

from .typing import LikePath, Spacing


def save_json_gz(data: tp.Dict, path: LikePath, *, compression: int = 1):
    dumps = json.dumps(data).encode()
    gzdumps = gzip.compress(dumps, compresslevel=compression, mtime=0)
    with open(path, "wb") as file:
        file.write(gzdumps)


def load_json_gz(path: LikePath) -> tp.Dict:
    with open(path, "rb") as f:
        gzdumps = f.read()

    dumps = gzip.decompress(gzdumps)
    return json.loads(dumps.decode())


def resolve_escape_char(string: str) -> str:
    string = string.replace("\\n", "\n")
    string = string.replace("\\t", "\t")
    string = string.replace("\\'", "'")
    string = string.replace('\\"', '"')
    string = string.replace('\\"', '"')
    return string


def remove_double_space(string: str) -> str:
    def join(characters: tp.Iterable[str]):
        return "".join(characters)

    doublespaces = list(map(join, it.product(" \t\v\b\r", " \t\v\b\r")))
    newline_pairs = it.chain(
        it.product(" \t\v\b\r\n", "\n"), it.product("\n", " \t\v\b\r")
    )
    newlines = list(map(join, newline_pairs))

    replace = dict.fromkeys(doublespaces, " ") | dict.fromkeys(newlines, "\n")
    while any(ch in string for ch in replace.keys()):
        for old, new in replace.items():
            string = string.replace(old, new)

    while any(string.startswith(ch) for ch in " \t\v\b\r\n"):
        string = string[1:]
    while any(string.endswith(ch) for ch in " \t\v\b\r\n"):
        string = string[:-1]
    return string


def derepr_list(string) -> str:
    return string.replace("['", "").replace("']", "").split("', '")


def derepr_medical_evaluation_text(text: str) -> str:
    if text != text:
        return ""
    medical_evaluation_list = derepr_list(text)
    without_escape_list = map(resolve_escape_char, medical_evaluation_list)
    without_spaces_list = map(remove_double_space, without_escape_list)
    pure_text = "\n".join(without_spaces_list)
    for char in ";,.!?":
        pure_text = pure_text.replace(f" {char}", char)
    return pure_text


def skip_empty(sequence) -> tp.List:
    return list(item for item in sequence if item == item and item)


def derepr_CUIS(string) -> tp.List[str]:
    if string != string:
        string = "[]"
    string = string.replace("[", "").replace("]", "")
    return skip_empty(string.split(","))


def nifty2numpy(nifti_path: LikePath) -> np.ndarray:
    nii = nib.load(nifti_path)
    return np.array(nii.dataobj)


def png2numpy(png_path: LikePath) -> np.ndarray:
    img = sitk.ReadImage(png_path)
    return sitk.GetArrayFromImage(img)


def down_type(data: np.ndarray):
    for dtype in [np.int8, np.uint8, np.int16, np.uint16, np.float16]:
        if np.all(data == data.astype(dtype)):
            data = data.astype(dtype)
            break
    return data


def spacing_from_nifty(nifti_path: LikePath) -> tp.Optional[Spacing]:
    with contextlib.suppress(Exception):
        return nib.load(nifti_path).header.get_zooms()
    return None


def spacing_from_tags(tags: tp.Optional[dict]) -> tp.Optional[Spacing]:
    if tags is None:
        return None

    if not isinstance(tags, dict):
        return None

    if "PixelSpacing" not in tags:
        return None

    spacing = tags["PixelSpacing"]

    if "Modality" not in tags:
        return None
    if tags["Modality"].lower() in ["cr", "dx"]:
        return spacing

    if tags["Modality"].lower() == "ct":
        raise NotImplementedError

    raise NotImplementedError


def parse_dicom_tags(tags: tp.Dict[str, tp.Any]) -> tp.Optional[tp.Union[dict, list]]:
    if not isinstance(tags, dict):
        return tags
    if len(tags) == 1:
        if "vr" in tags:
            return None
        return parse_dicom_tags(list(tags.values())[0])

    if set(tags.keys()) == {"Value", "vr"}:
        value = tags["Value"]
        # vr = tags["vr"]
        assert isinstance(value, list)
        result = [parse_dicom_tags(v) for v in value]
        if not result:
            return None
        if len(result) == 1:
            return result[0]
        return result
    result_: tp.Dict[str, tp.Optional[tp.Union[dict, list]]] = {}
    for tag, value in tags.items():
        try:
            keyword = pydicom.datadict.keyword_for_tag(tag)
        except ValueError:
            keyword = str(tag)
        assert isinstance(result_, dict)
        result_[keyword] = parse_dicom_tags(value)
    return result_


def derepr_strings_list(string):
    strings_list = derepr_list(string if string == string else "[]")
    without_double_spaces = list(map(remove_double_space, strings_list))
    pure_strings = skip_empty(without_double_spaces)
    pure_strings = list(string for string in pure_strings if string != "[]")
    return pure_strings
