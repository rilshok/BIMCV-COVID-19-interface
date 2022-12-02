__all__ = [
    "rotate_ct_transform",
    "clean_ct_image",
]

import contextlib
from typing import Tuple

import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter

from .assets import mapping_bimcv_covid19_ct_rotate_transforms

Image = np.ndarray
Spacing = Tuple[float, ...]


def assert_ndim(data: Image, dim: int):
    if data.ndim != dim:
        msg = f"expected {dim} axes"
        raise ValueError(msg)


class Transform:
    def __call__(self, image: Image, spacing: Spacing):
        img = self.transform_image(image)
        spc = self.transform_spacing(spacing)
        return img, spc

    def transform_image(self, image: Image):
        return image

    def transform_spacing(self, spacing: Spacing):
        return spacing


class RotateCTTransform(Transform):
    pass


class RotateCTTransformType0(RotateCTTransform):
    pass


class RotateCTTransformType1(RotateCTTransform):
    def transform_image(self, image: Image):
        return np.rot90(np.rot90(image, 1, (1, 2)), 1)[..., ::-1]

    def transform_spacing(self, spacing: Spacing):
        return spacing[2], spacing[0], spacing[1]


class RotateCTTransformType2(RotateCTTransform):
    def transform_image(self, image: Image):
        return np.rot90(image, 1, (1, 2))[..., ::-1]

    def transform_spacing(self, spacing: Spacing):
        return spacing[0], spacing[2], spacing[1]


class RotateCTTransformType3(RotateCTTransform):
    def transform_image(self, image: Image):
        return np.rot90(image, 1, (0, 1))[..., ::-1]

    def transform_spacing(self, spacing: Spacing):
        return spacing[1], spacing[0], spacing[2]


class RotateCTTransformType4(RotateCTTransform):
    def transform_image(self, image: Image):
        return np.rot90(image, 1, (1, 2))[..., ::-1, ::-1]

    def transform_spacing(self, spacing: Spacing):
        return spacing[0], spacing[2], spacing[1]


class RotateCTTransformType5(RotateCTTransform):
    def transform_image(self, image: Image):
        return np.rot90(image, 1, (0, 1))

    def transform_spacing(self, spacing: Spacing):
        return spacing[1], spacing[0], spacing[2]


def get_rotate_ct_transform(transform_type: str) -> RotateCTTransform:
    return dict(
        type_0=RotateCTTransformType0,
        type_1=RotateCTTransformType1,
        type_2=RotateCTTransformType2,
        type_3=RotateCTTransformType3,
        type_4=RotateCTTransformType4,
        type_5=RotateCTTransformType5,
    )[transform_type]()


def rotate_ct_transform(image: Image, spacing: Spacing, transform_type: str):
    return get_rotate_ct_transform(transform_type)(image, spacing)


def _clean_3dimage(image: Image, filter_fn):
    assert_ndim(image, 3)
    skip = slice(None)
    first = slice(1, None)
    last = slice(None, -1)
    edge_idxs = [
        ((0, skip, skip), (first, skip, skip)),
        ((-1, skip, skip), (last, skip, skip)),
        ((skip, 0, skip), (skip, first, skip)),
        ((skip, -1, skip), (skip, last, skip)),
        ((skip, skip, 0), (skip, skip, first)),
        ((skip, skip, -1), (skip, skip, last)),
    ]
    while True:
        for edge_idx, get_idx in edge_idxs:
            img = image[edge_idx]
            if filter_fn(img):
                image = image[get_idx]
                break
        else:
            break
    return image


def _blank_filter(image: Image):
    assert_ndim(image, 2)
    return len(np.unique(image)) == 1


def _estimate_regularity(image: Image, k: int = 10) -> float:
    assert_ndim(image, 2)
    min_, max_ = np.quantile(image, 0.03), np.quantile(image, 0.97)
    with contextlib.suppress(Exception):
        if max_ - min_ <= 0:
            raise ValueError()
        one = (image.mean(0) - min_) / (max_ - min_)
        two = (image.mean(1) - min_) / (max_ - min_)
        minimum = np.concatenate(
            tuple(minimum_filter(vec, size=k) for vec in [one, two])
        )
        maximum = np.concatenate(
            tuple(maximum_filter(vec, size=k) for vec in [one, two])
        )
        return (maximum - minimum).mean()
    return 0.0


def _regularity_filter(image: Image):
    return _estimate_regularity(image) > 0.1


def _blank_and_regularity_filter(image: Image):
    return _blank_filter(image) or _regularity_filter(image)


def clean_ct_image(image: Image):
    return _clean_3dimage(image, _blank_and_regularity_filter)


def process_ct_image(series_id: str, image: Image, spacing: Spacing):
    if not series_id.endswith("ct"):
        raise ValueError("CT image identifier was expected")
    transform_type = mapping_bimcv_covid19_ct_rotate_transforms().get(series_id)
    if transform_type:
        image, spacing = rotate_ct_transform(
            image=image, spacing=spacing, transform_type=transform_type
        )
    image = clean_ct_image(image)
    return image, spacing
