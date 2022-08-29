from typing import Set, Optional
from google.cloud import storage
import os
import sys
import torch
import urllib.request
import zipfile
import tarfile
import gzip
import shutil
import corefai

CACHE = os.path.expanduser('~/.cache/corefai')

NORMALIZE_DICT = {"/.": ".", "/?": "?",
                  "-LRB-": "(", "-RRB-": ")",
                  "-LCB-": "{", "-RCB-": "}",
                  "-LSB-": "[", "-RSB-": "]"}
REMOVED_CHAR = ["/", "%", "*"]

def conll_clean_token(token: str) -> str:
    """ Substitute in /?(){}[] for equivalent CoNLL-2012 representations,
    remove /%* 
    Args:
        token (str): token to clean
    Returns:
        str: cleaned token
    """
    cleaned_token = token
    if cleaned_token in NORMALIZE_DICT:
        cleaned_token = NORMALIZE_DICT[cleaned_token]

    if cleaned_token not in REMOVED_CHAR:
        for char in REMOVED_CHAR:
            cleaned_token = cleaned_token.replace(char, u'')

    if len(cleaned_token) == 0:
        cleaned_token = ","
    return cleaned_token

def read_file(filename:str) -> Set[str]:
    """Read file and return  string object
        Args:
            filename (str): path to the file
        Returns:
            Set[str]: set of tokens in the file
    """
    with open(filename, 'r') as f:
        return set(f.read().split('\n'))

def download(src: str, url:str, path: Optional[str] = None, reload: bool = False, clean: bool = False) -> str:
    if path is None:
        path = CACHE
    file = os.path.basename(urllib.parse.urlparse(url).path)
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, file)
    if reload and os.path.exists(path):
        os.remove(path)

    if src == 'gcp':
        url = corefai.SRC[src]
        try:
            download_public_file(url, file, path)
        except:
            raise RuntimeError(f"Could not download {file} from bucket {url}")

    else:
        if not os.path.exists(path):
            sys.stderr.write(f"Downloading {url} to {path}\n")
            try:
                torch.hub.download_url_to_file(url, path, progress=True)
            except (ValueError, urllib.error.URLError):
                raise RuntimeError(f"File {url} unavailable. Please try other sources.")

    return extract(path, reload, clean)

def download_public_file(bucket_name, source_blob_name, destination_file_name):
    """Downloads a public blob from the bucket."""
    # bucket_name = "your-bucket-name"
    # source_blob_name = "storage-object-name"
    # destination_file_name = "local/path/to/file"

    storage_client = storage.Client.create_anonymous_client()

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    
    blob.download_to_filename(destination_file_name)


def extract(path: str, reload: bool = False, clean: bool = False) -> str:
    extracted = path
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as f:
            extracted = os.path.join(os.path.dirname(path), f.infolist()[0].filename)
            if reload or not os.path.exists(extracted):
                f.extractall(os.path.dirname(path))
    elif tarfile.is_tarfile(path):
        with tarfile.open(path) as f:
            extracted = os.path.join(os.path.dirname(path), f.getnames()[0])
            if reload or not os.path.exists(extracted):
                f.extractall(os.path.dirname(path))
    elif path.endswith('.gz'):
        extracted = path[:-3]
        with gzip.open(path) as fgz:
            with open(extracted, 'wb') as f:
                shutil.copyfileobj(fgz, f)
    if clean:
        os.remove(path)
    return extracted