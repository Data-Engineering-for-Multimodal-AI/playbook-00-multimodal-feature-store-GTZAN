# Sample Data

This directory contains the sample audio and image files used for testing and demonstrating the multimodal feature store.

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

- `images_original/`: Contains album artwork images (PNG format) organized by genre
  - The directory structure mirrors the `genres_original/` directory

## Sample Usage

These samples are used by the test scripts in the repository root, such as:
- `test_new_song.py`
- `explore_genres.py`
- `test_api_curl.sh`

## Data Sources

The audio samples are from the GTZAN dataset, which is a collection of 1000 audio tracks, each 30 seconds long, with 100 tracks per genre. The dataset is widely used for music genre classification tasks.

The image samples are generated artwork that visually represents each genre's aesthetic. 