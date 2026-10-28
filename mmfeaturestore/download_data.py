#!/usr/bin/env python3
"""
Download GTZAN from Kaggle into sample_data/ (1.2 GB; not part of the repository).

    python download_data.py                       # downloads and unpacks, then checks the files
    python download_data.py --zip <file>.zip      # use an already downloaded zip

GTZAN (Tzanetakis and Cook, 2002): 1,000 30-second clips, 100 per genre, and the Kaggle version's mel
spectrogram of each (999: jazz.00054.wav is unreadable). The audio comes from commercial recordings; it
is available for research and can't be redistributed, so each reader downloads it. Kaggle needs an
account: put your API token in ~/.kaggle/kaggle.json, or set KAGGLE_USERNAME and KAGGLE_KEY.
"""
import argparse
import hashlib
import os
import shutil
import sys
import zipfile
from pathlib import Path

DATASET = "andradaolteanu/gtzan-dataset-music-genre-classification"
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "sample_data"
FOLDERS = ("genres_original", "images_original")
EXPECTED = (1999, "af64e5d1a58d2f8b5bf308081c98458cbfb5eb5eed2b140caf09ba228be709ed")


def digest(base: Path):
    """SHA-256 over every file's path and SHA-256, as recorded in EXPECTED."""
    total, count = hashlib.sha256(), 0
    cwd = os.getcwd()
    os.chdir(base)
    try:
        for top in FOLDERS:
            for root, _, files in sorted(os.walk(top)):
                for name in sorted(files):
                    path = os.path.join(root, name)
                    with open(path, "rb") as f:
                        total.update(f"{path}:{hashlib.sha256(f.read()).hexdigest()}\n".encode())
                    count += 1
    finally:
        os.chdir(cwd)
    return count, total.hexdigest()


def download(cache: Path) -> Path:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        sys.exit("The kaggle package is missing: pip install -r requirements.txt")
    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as e:
        sys.exit(f"Kaggle login failed ({e}). Put your API token in ~/.kaggle/kaggle.json "
                 "or set KAGGLE_USERNAME and KAGGLE_KEY.")
    cache.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {DATASET} from Kaggle (1.2 GB)...")
    api.dataset_download_files(DATASET, path=str(cache), quiet=False)
    return cache / (DATASET.split("/")[1] + ".zip")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zip", type=Path, help="an already downloaded gtzan-dataset-music-genre-classification.zip")
    args = parser.parse_args()
    archive = args.zip or download(ROOT / ".download")
    print(f"Unpacking {archive.name} into {OUT.relative_to(ROOT)}/ ...")
    with zipfile.ZipFile(archive) as z:
        for member in z.namelist():
            parts = Path(member).parts
            if len(parts) > 2 and parts[1] in FOLDERS and not member.endswith("/"):
                target = OUT.joinpath(*parts[1:])
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(member) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
    count, value = digest(OUT)
    if (count, value) != EXPECTED:
        sys.exit(f"Checksum mismatch: {count} files, {value}; expected {EXPECTED[0]} files, {EXPECTED[1]}")
    print(f"  {count} files verified (SHA-256): 1,000 clips and 999 spectrograms in 10 genres")
    if not args.zip:
        shutil.rmtree(ROOT / ".download", ignore_errors=True)
    print("Next: docker compose up -d --wait, then python ingest_gtzan.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
