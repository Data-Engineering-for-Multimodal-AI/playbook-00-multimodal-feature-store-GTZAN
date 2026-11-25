# Sample Data

This directory holds the audio and image files the feature store uses. They are not in the repository. From the repository root, `make data` downloads them here (needs a Kaggle API token).

## Important Note About GitHub

**The actual audio and image files may not be visible in GitHub's web interface** because:
1. They are binary files (WAV, PNG) which GitHub doesn't display directly
2. Large binary files may be tracked with Git LFS or excluded from the repository
3. Binary files in general don't display content previews in GitHub

If you don't see these files locally, run `make data` (see "Data Sources" below).

## Directory Structure

- `genres_original/`: Contains WAV audio files organized by genre
  - `blues/`: Blues music samples
  - `classical/`: Classical music samples
  - `country/`: Country music samples
  - `disco/`: Disco music samples
  - `hiphop/`: Hip-hop music samples
  - `jazz/`: Jazz music samples
  - `metal/`: Metal music samples
  - `pop/`: Pop music samples
  - `reggae/`: Reggae music samples
  - `rock/`: Rock music samples

- `images_original/`: Contains spectrogram images (PNG format), one per audio clip, organized by genre
  - The directory structure mirrors the `genres_original/` directory

## Sample Usage

These samples are used by the test scripts in the repository root, such as:
- `test_new_song.py`
- `explore_genres.py`
- `test_api_curl.sh`

## Data Sources

The audio samples are from the GTZAN dataset, which is a collection of 1000 audio tracks, each 30 seconds long, with 100 tracks per genre. The dataset is widely used for music genre classification tasks.

You can download the GTZAN dataset from: [GTZAN on Kaggle](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification). From the repository root, `make data` downloads it here (needs a Kaggle API token).

The images come in the same Kaggle download. Each one is a mel spectrogram of one audio clip, not album artwork.

## Expected File Names

When adding your own samples, follow these naming conventions:
- Audio files: `{genre}.{number}.wav` (e.g., `rock.00000.wav`)
- Image files: `{genre}{number}.png` (e.g., `rock00000.png`) 