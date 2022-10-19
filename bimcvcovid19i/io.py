import json
import gzip

import typing as tp
from .typing import LikePath


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
