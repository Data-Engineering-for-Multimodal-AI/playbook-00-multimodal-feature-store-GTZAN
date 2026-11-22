"""Try this 3: which modality copes with a noisy recording? (make try-noise)

Each song has two feature vectors: 28 audio features from the WAV file, and 512 ResNet-18 features
from an image of its mel spectrogram (GTZAN's images are mel spectrograms too). This stores the clean
synthetic clips, then adds white noise to each query clip at a falling signal-to-noise ratio, makes
its spectrogram image the same way quickstart.py does, and searches both ways. No Docker, no GTZAN.

Change SNR_DB or PER_GENRE and run it again.
"""

import sys

import librosa
import numpy as np
import soundfile as sf
from PIL import Image

from common import OUT, synthetic, timed
from src.encoders.audio import AudioEncoder
from src.encoders.image import ImageEncoder

PER_GENRE = 6
SNR_DB = [None, 30, 20, 10, 0, -10]


def spectrogram_png(y, sr, path):
    mel = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128), ref=np.max)
    pixels = ((mel - mel.min()) / (mel.max() - mel.min() + 1e-9) * 255).astype(np.uint8)[::-1]
    Image.fromarray(pixels).convert("RGB").resize((432, 288)).save(path)


def hits(queries, stored, genres):
    """Nearest stored clip other than the query's own clean version: same genre? And is the
    query's own clean clip the nearest of all?"""
    s = queries @ stored.T
    own = float(np.mean(np.argmax(s, axis=1) == np.arange(len(genres))))
    np.fill_diagonal(s, -np.inf)
    same = float(np.mean([genres[int(np.argmax(s[i]))] == genres[i] for i in range(len(genres))]))
    return own, same


def main() -> int:
    songs = synthetic(PER_GENRE, folder="noise_clean")
    genres = [s["genre"] for s in songs]
    audio, image = AudioEncoder(), ImageEncoder()
    stored_audio, _ = timed(lambda s: audio.encode(str(s["audio"])), songs)
    stored_image, _ = timed(lambda s: image.encode(str(s["image"])), songs)
    rng = np.random.default_rng(1)
    folder = OUT / "noisy"
    folder.mkdir(parents=True, exist_ok=True)
    print(f"{len(songs)} clean synthetic clips stored; each query is a noisy copy of one of them\n")
    print(f"{'SNR':>7}  {'audio: finds itself':>19}  {'same genre':>10}  {'image: finds itself':>19}  {'same genre':>10}")
    for snr in SNR_DB:
        wavs, pngs = [], []
        for s in songs:
            y, sr = sf.read(s["audio"])
            if snr is not None:
                noise = rng.standard_normal(len(y))
                noise *= np.sqrt(np.mean(y ** 2) / (np.mean(noise ** 2) * 10 ** (snr / 10)))
                y = y + noise
            wav, png = folder / f"{s['id']}_{snr}.wav", folder / f"{s['id']}_{snr}.png"
            sf.write(wav, y.astype(np.float32), sr)
            spectrogram_png(y, sr, png)
            wavs.append(str(wav))
            pngs.append(str(png))
        qa, _ = timed(audio.encode, wavs)
        qi, _ = timed(image.encode, pngs)
        a_own, a_same = hits(qa, stored_audio, genres)
        i_own, i_same = hits(qi, stored_image, genres)
        label = "clean" if snr is None else f"{snr} dB"
        print(f"{label:>7}  {a_own:>19.0%}  {a_same:>10.0%}  {i_own:>19.0%}  {i_same:>10.0%}")
    print("\nfinds itself: the clean version of the same clip is the nearest stored vector.")
    print("same genre: the nearest stored clip other than its own clean version is the same genre.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
