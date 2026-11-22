"""Try this 2: how much of the spectrogram does the image encoder see? (make try-image-crop)

The image encoder (src/encoders/image.py) resizes each image so its short side is 256 pixels, then
keeps the center 224 x 224 for ResNet-18. A spectrogram image has time across and frequency up and
down, so the crop drops the start and end of the clip and the highest and lowest bands. This
blacks out parts of each synthetic clip's 432 x 288 spectrogram image and measures how much its
vector changes. No Docker, no GTZAN.

Change REGIONS and run it again.
"""

import sys

import numpy as np
from PIL import Image

from common import OUT, synthetic, timed
from src.encoders.image import ImageEncoder

PER_GENRE = 4


def kept_box(width, height):
    """The part of the original image that Resize(256) + CenterCrop(224) keeps, in original pixels."""
    scale = 256 / min(width, height)
    w, h = width * scale, height * scale
    left, top = (w - 224) / 2 / scale, (h - 224) / 2 / scale
    return left, top, width - left, height - top


def main() -> int:
    songs = synthetic(PER_GENRE, folder="crop")
    encoder = ImageEncoder()
    width, height = Image.open(songs[0]["image"]).size
    left, top, right, bottom = kept_box(width, height)
    print(f"{len(songs)} spectrogram images of {width} x {height}; the encoder keeps x {left:.0f}-{right:.0f}, "
          f"y {top:.0f}-{bottom:.0f}")
    print(f"That is {(right - left) / width:.0%} of the clip's duration and {(bottom - top) / height:.0%} of its mel bands.\n")
    w, h = width, height
    # (what is blacked out, share of the image, columns, rows); columns and rows are (start, end) fractions
    regions = [
        ("first and last 20% of the time", 0.4, [(0, .2), (.8, 1)], [(0, 1)]),
        ("top and bottom 5% of the bands", 0.1, [(0, 1)], [(0, .05), (.95, 1)]),
        ("middle 40% of the time", 0.4, [(.3, .7)], [(0, 1)]),
        ("middle 10% of the bands", 0.1, [(0, 1)], [(.45, .55)]),
    ]
    original, _ = timed(lambda s: encoder.encode(str(s["image"])), songs)
    folder = OUT / "crop_edited"
    folder.mkdir(parents=True, exist_ok=True)
    print(f"{'blacked out':<32}  {'share of pixels':>15}  {'similarity to the original vector':>33}")
    for name, share, columns, rows in regions:
        paths = []
        for s in songs:
            pixels = np.array(Image.open(s["image"]).convert("RGB"))
            for c0, c1 in columns:
                for r0, r1 in rows:
                    pixels[int(h * r0):int(h * r1), int(w * c0):int(w * c1)] = 0
            path = folder / f"{s['id']}.png"
            Image.fromarray(pixels).save(path)
            paths.append(str(path))
        edited, _ = timed(encoder.encode, paths)
        similarity = float(np.mean(np.sum(edited * original, axis=1)))
        print(f"{name:<32}  {share:>15.0%}  {similarity:>33.4f}")
    print("\n1.0000 means the encoder never saw the change.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
