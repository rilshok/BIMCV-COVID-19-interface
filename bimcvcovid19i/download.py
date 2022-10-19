from .typing import DatasetRoot, LikePath
from .webdav import webdav_download_all


def download_bimcv_covid19_positive(root: LikePath):
    return webdav_download_all(
        root=DatasetRoot(root).original,
        webdav_hostname="https://b2drop.bsc.es/public.php/webdav",
        webdav_login="BIMCV-COVID19-cIter_1_2",
        webdav_password="maybeempty",
    )


def download_bimcv_covid19_negative(root: LikePath):
    return webdav_download_all(
        root=DatasetRoot(root).original,
        webdav_hostname="https://b2drop.bsc.es/public.php/webdav",
        webdav_login="BIMCV-COVID19-cIter_1_2-Negative",
        webdav_password="maybeempty",
    )


def extract_bimcv_covid19_positive(root: LikePath):
    dsroot = DatasetRoot(root)
    assert dsroot.original.exists()
    # TODO: check sums
