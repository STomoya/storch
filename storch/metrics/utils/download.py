"""Download URL."""

import os
import shutil
import urllib.request

from storch.path import Path


def download_url(url: str, filename: str, folder: str = './.cache/storch/metrics') -> str:
    """Download url.

    Args:
    ----
        url (str): URL to the file to download.
        filename (str): filename to be saved as
        folder (str, optional): the folder to save the downloaded file to. Default: './.cache/storch/metrics'.

    Returns:
    -------
        str: the path to the downloaded file.
    """
    folder = Path(folder)
    ckpt_path = folder / filename
    print(f'Downloading: "{url}" to {ckpt_path.resolve()}')
    if not ckpt_path.exists():
        if not folder.exists():
            os.makedirs(folder)
        with urllib.request.urlopen(url) as response, open(ckpt_path, 'wb') as fp:
            shutil.copyfileobj(response, fp)

    return ckpt_path
