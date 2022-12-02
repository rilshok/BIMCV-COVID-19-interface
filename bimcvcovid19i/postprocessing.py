__all__ = [
    "rotate_ct_transform",
]

import numpy as np


class Transform:
    def __call__(self, image, spacing):
        img = self.transform_image(image)
        spc = self.transform_spacing(spacing)
        return img, spc

    def transform_image(self, image):
        return image

    def transform_spacing(self, spacing):
        return spacing


class RotateCTTransform(Transform):
    pass


class RotateCTTransformType0(RotateCTTransform):
    pass


class RotateCTTransformType1(RotateCTTransform):
    def transform_image(self, image):
        return np.rot90(np.rot90(image, 1, (1, 2)), 1)[..., ::-1]

    def transform_spacing(self, spacing):
        return spacing[2], spacing[0], spacing[1]


class RotateCTTransformType2(RotateCTTransform):
    def transform_image(self, image):
        return np.rot90(image, 1, (1, 2))[..., ::-1]

    def transform_spacing(self, spacing):
        return spacing[0], spacing[2], spacing[1]


class RotateCTTransformType3(RotateCTTransform):
    def transform_image(self, image):
        return np.rot90(image, 1, (0, 1))[..., ::-1]

    def transform_spacing(self, spacing):
        return spacing[1], spacing[0], spacing[2]


class RotateCTTransformType4(RotateCTTransform):
    def transform_image(self, image):
        return np.rot90(image, 1, (1, 2))[..., ::-1, ::-1]

    def transform_spacing(self, spacing):
        return spacing[0], spacing[2], spacing[1]


class RotateCTTransformType5(RotateCTTransform):
    def transform_image(self, image):
        return np.rot90(image, 1, (0, 1))

    def transform_spacing(self, spacing):
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


def rotate_ct_transform(image, spacing, transform_type: str):
    return get_rotate_ct_transform(transform_type)(image, spacing)
