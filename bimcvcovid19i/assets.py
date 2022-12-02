import functools
from pathlib import Path
from typing import Dict

import deli


def load_asset(path):
    root = Path(__file__).parent.parent / "assets"
    return deli.load(root / path)


def mapping_bimcv_covid19_negative_ct_rotate_transforms() -> Dict[str, str]:
    asset = load_asset("bimcv-covid19-negative-ct-rotate-transforms.csv")
    return {row.series_id: row.transform_type for row in asset.itertuples()}


def mapping_bimcv_covid19_positive_ct_rotate_transforms() -> Dict[str, str]:
    asset = load_asset("bimcv-covid19-positive-ct-rotate-transforms.csv")
    return {row.series_id: row.transform_type for row in asset.itertuples()}


@functools.lru_cache
def mapping_bimcv_covid19_ct_rotate_transforms() -> Dict[str, str]:
    return (
        mapping_bimcv_covid19_negative_ct_rotate_transforms()
        | mapping_bimcv_covid19_positive_ct_rotate_transforms()
    )
