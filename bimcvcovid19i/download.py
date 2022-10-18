from pathlib import Path

from .typing import LikePath
from .webdav import webdav_download_all


def download_bimcv_covid19_positive(root: LikePath):
    return webdav_download_all(
        root=Path(root).absolute(),
        webdav_hostname="https://b2drop.bsc.es/public.php/webdav",
        webdav_login="BIMCV-COVID19-cIter_1_2",
        webdav_password="maybeempty",
    )


def download_bimcv_covid19_negative(root: LikePath):
    return webdav_download_all(
        root=Path(root).absolute(),
        webdav_hostname="https://b2drop.bsc.es/public.php/webdav",
        webdav_login="BIMCV-COVID19-cIter_1_2-Negative",
        webdav_password="maybeempty",
    )
