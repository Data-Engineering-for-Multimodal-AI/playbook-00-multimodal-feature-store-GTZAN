"""Checks for the "Use your own data" loader (scripts/own_data.py). No Docker, no GTZAN."""
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mmfeaturestore"))
sys.path.insert(0, str(ROOT / "scripts"))
import quickstart  # noqa: E402
from own_data import scan  # noqa: E402


@pytest.fixture(scope="module")
def clean(tmp_path_factory):
    folder = tmp_path_factory.mktemp("clean")
    quickstart.make_synthetic_data(folder, per_genre=1)
    return folder


@pytest.fixture
def folder(clean, tmp_path):
    shutil.copytree(clean, tmp_path / "music")
    return tmp_path / "music"


def wav(path: Path, samples, sr=22050):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.asarray(samples, dtype=np.float32), sr)


def test_synthetic_folder_passes(folder):
    counts, problems, notes = scan(folder)
    assert problems == [] and notes == []
    assert counts == {"clips": 3, "images": 3, "genres": 3}


def test_audio_without_image_is_a_note(folder):
    wav(folder / "genres_original" / "rock" / "mine.00001.wav", np.sin(np.arange(44100) / 5) * .3)
    counts, problems, notes = scan(folder)
    assert problems == [] and counts["clips"] == 4
    assert notes == ["genres_original/rock/mine.00001.wav: no image (looked for images_original/rock/rock00001.png); audio only"]


@pytest.mark.parametrize("name,make,expected", [
    ("silent.00001.wav", lambda p: wav(p, np.zeros(44100)), "silent (every sample is 0)"),
    ("short.00001.wav", lambda p: wav(p, np.ones(1000) * .1), "0.05 s long, needs at least 1 s"),
    ("broken.00001.wav", lambda p: p.write_text("not audio"), "can't read as WAV audio"),
    ("empty.00001.wav", lambda p: p.write_bytes(b""), "empty file"),
    ("lfs.00001.wav", lambda p: p.write_text("version https://git-lfs.github.com/spec/v1\n"), "Git LFS pointer"),
    ("song.mp3", lambda p: p.write_bytes(b"ID3"), "not a .wav file"),
])
def test_bad_audio_is_reported(folder, name, make, expected):
    path = folder / "genres_original" / "synthetic_bass" / name
    make(path)
    _, problems, _ = scan(folder)
    assert len(problems) == 1 and problems[0].startswith(f"genres_original/synthetic_bass/{name}: ")
    assert expected in problems[0], problems


def test_bad_images_are_reported(folder):
    images = folder / "images_original" / "synthetic_bass"
    (images / "synthetic_bass00000.png").write_text("version https://git-lfs.github.com/spec/v1\n")
    _, problems, _ = scan(folder)
    assert problems == ["images_original/synthetic_bass/synthetic_bass00000.png: a Git LFS pointer, not the image (run git lfs pull)"]
    Image.new("RGB", (16, 16)).save(images / "synthetic_bass00000.png")
    _, problems, _ = scan(folder)
    assert problems == ["images_original/synthetic_bass/synthetic_bass00000.png: too small (16x16), needs at least 32x32"]


def test_duplicate_song_ids(folder):
    shutil.copy(folder / "genres_original" / "synthetic_bass" / "synthetic_bass.00000.wav",
                folder / "genres_original" / "synthetic_noise" / "synthetic_bass.00000.wav")
    _, problems, _ = scan(folder)
    assert len(problems) == 1 and "same song ID 'synthetic_bass.00000'" in problems[0]


def test_long_names_are_truncated_ids(folder):
    long = "a" * 40 + ".00001.wav"
    wav(folder / "genres_original" / "synthetic_bass" / long, np.sin(np.arange(44100) / 5) * .3)
    _, problems, notes = scan(folder)
    assert problems == []
    assert any("name longer than 36 characters" in n for n in notes)


def test_layout_errors(tmp_path):
    assert scan(tmp_path / "nope")[1] == [f"{tmp_path / 'nope'}: not a folder"]
    assert scan(tmp_path)[1] == [f"{tmp_path}: no genres_original/ folder"]
    (tmp_path / "genres_original" / "jazz").mkdir(parents=True)
    assert scan(tmp_path)[1] == ["genres_original/jazz/: no .wav files"]
