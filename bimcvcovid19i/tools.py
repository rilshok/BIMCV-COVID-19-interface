import itertools as it
import typing as tp

import nibabel as nib
import numpy as np
import SimpleITK as sitk

from .typing import LikePath


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


def skip_empty(sequence) -> tp.Sequence:
    return type(sequence)(item for item in sequence if item == item and item)


def derepr_CUIS(string):
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


def spacing_from_nifty(nifti_path: LikePath):
    try:
        return nib.load(nifti_path).header.get_zooms()
    except Exception:
        return None
