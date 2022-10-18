"""Downloading files via WEBDAV protocol"""

import hashlib
import typing as tp
import warnings
from pathlib import Path

from tqdm import tqdm  # type: ignore
from webdav3.client import Client  # type: ignore
from webdav3.exceptions import WebDavException  # type: ignore

from .typing import LikePath


def get_sha1(path: LikePath) -> str:
    """SHA1 checksum calculation"""
    hash_ = hashlib.sha1(b"")
    with open(path, "rb") as file:
        data = b" "
        while data:
            data = file.read(2**20)
            hash_.update(data)
    return hash_.hexdigest()


def read_checksums(path: LikePath, sep=None) -> tp.Dict[str, str]:
    """Reading a checksum file"""
    with open(path, encoding="utf-8") as file:
        sum_lines = file.readlines()
    result = {}
    for line in sum_lines:
        filehash, filename = line.split(sep)
        result[filename] = filehash
    return result


def webdav_download_file(
    client: Client,
    remote_path: LikePath,
    local_path: LikePath,
    sha1sum: tp.Optional[str] = None,
) -> bool:
    """Download file via WEBDAV protocol"""
    local_path = Path(local_path).absolute()
    backup_path = Path(str(local_path) + ".old")
    exists = local_path.exists()
    if exists:
        if sha1sum is None:
            return True
        if get_sha1(local_path) == sha1sum:
            return True
        local_path.replace(backup_path)

    try:
        client.download_sync(remote_path=str(remote_path), local_path=str(local_path))
    except WebDavException as exc:
        if exists:
            if local_path.exists():
                local_path.unlink()
            backup_path.replace(local_path)
        raise exc

    if backup_path.exists():
        backup_path.unlink()
    if sha1sum is None:
        return True
    if get_sha1(local_path) != sha1sum:
        warnings.warn(f"Checksum mismatch for file '{str(remote_path)}'")
        return False
    return True


def webdav_download_all(
    root: LikePath,
    webdav_hostname: str,
    webdav_login: str,
    webdav_password: str,
):
    """Download all files via WEBDAV protocol"""
    download_path = Path(root).absolute() / "original"

    download_path.parent.mkdir(exist_ok=True, parents=False)
    download_path.mkdir(exist_ok=True, parents=False)

    client = Client(
        dict(
            webdav_hostname=webdav_hostname,
            webdav_login=webdav_login,
            webdav_password=webdav_password,
        )
    )

    names = {Path(info["path"]).name for info in client.list(get_info=True)}
    if "webdav" in names:
        names.remove("webdav")

    sha1sums = {}
    if "sha1sums.txt" in names:
        names.remove("sha1sums.txt")
        webdav_download_file(
            client=client,
            remote_path="sha1sums.txt",
            local_path=download_path / "sha1sums.txt",
        )
        sha1sums.update(read_checksums(download_path / "sha1sums.txt"))

    names_bar = tqdm(names)
    for name in names_bar:
        names_bar.set_description(f"donwloading {name:50}")
        webdav_download_file(
            client=client,
            remote_path=name,
            local_path=str(download_path / name),
            sha1sum=sha1sums.get(name, None),
        )

    for name, value in sha1sums.items():
        file_path = download_path / name
        if file_path.exists():
            pass
            # if get_sha1(file_path) == value:
            #     pass
            # else:
            #     pass
        else:
            warnings.warn(
                f"file '{name}' is listed in sha1sums.txt but has not been downloaded"
            )
