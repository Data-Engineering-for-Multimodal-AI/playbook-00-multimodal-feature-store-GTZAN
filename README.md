# Multimodal Feature Store

A robust feature store for audio and image data that enables efficient extraction, storage, and similarity search across multimodal music data.

## Overview

This feature store provides a comprehensive solution for processing and retrieving music data across multiple modalities (audio and image). The system extracts meaningful features from audio files and album artwork, stores them in optimized vector databases, and enables powerful similarity searches to find music with similar acoustic properties or visual aesthetics.

## Architecture

- **PostgreSQL**: Stores metadata and job status information
- **Milvus**: Vector database for storing and querying high-dimensional feature vectors
- **FastAPI**: REST API for interacting with the feature store
- **Encoder Models**: Specialized audio and image encoders that extract features from raw files

## Setup & Installation

### Prerequisites

- Python 3.8+
- PostgreSQL
- Milvus

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/mmfeaturestore.git
   cd mmfeaturestore
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Start the PostgreSQL database:
   ```bash
   # Setup PostgreSQL with user "featurestore_user" and database "featurestore"
   createuser --createdb featurestore_user
   createdb -O featurestore_user featurestore
   ```

4. Start the Milvus service:
   ```bash
   # If using Docker
   docker-compose up -d
   ```

5. Start the API server:
   ```bash
   python run.py
   ```

## Sample Data

The repository includes sample audio and image files for testing, but these **do not appear in GitHub's web interface** because they are binary files. The sample data is organized as follows:

- **Audio samples**: Located in `/mmfeaturestore/sample_data/genres_original/{genre}/*.wav`
- **Image samples**: Located in `/mmfeaturestore/sample_data/images_original/{genre}/*.png`

Genres include: pop, rock, jazz, classical, metal, blues, disco, reggae, hiphop, and country.

### Obtaining Sample Data

If the sample data is missing when you clone the repository, you can:

1. Download the GTZAN dataset for audio samples: [GTZAN Genre Collection](http://marsyas.info/downloads/datasets.html)
2. Place audio files in the appropriate genre folders under `mmfeaturestore/sample_data/genres_original/`
3. Create or download album artwork images and place them in the corresponding folders under `mmfeaturestore/sample_data/images_original/`

Or contact the repository maintainer for access to the full sample dataset.

## Usage

### REST API Endpoints

- `POST /songs`: Create a new song record
- `PUT /songs/{song_id}/data/{modality}`: Add data path for a specific modality
- `POST /songs/{song_id}/process`: Process features for a song
- `GET /jobs/{job_id}`: Get processing job status
- `GET /songs/{song_id}/features/{modality}`: Get extracted features for a modality
- `POST /songs/{song_id}/similar`: Find similar songs based on features

### Example: Processing a New Song

```python
import requests
import json
import uuid

# API base URL
BASE_URL = "http://localhost:8000"

# Create a new song
song_id = str(uuid.uuid4())
song_data = {
    "id": song_id,
    "title": "Rock Sample",
    "artist": "Sample Artist",
    "album": "Rock Collection",
    "genre": "rock",
    "year": 2024
}

response = requests.post(f"{BASE_URL}/songs", json=song_data)
print(f"Created song: {response.json()}")

# Set audio and image paths
audio_path = "/path/to/audio.wav"
image_path = "/path/to/cover.png"

requests.put(
    f"{BASE_URL}/songs/{song_id}/data/audio", 
    json={"path": audio_path}
)
requests.put(
    f"{BASE_URL}/songs/{song_id}/data/image", 
    json={"path": image_path}
)

# Process features
response = requests.post(
    f"{BASE_URL}/songs/{song_id}/process", 
    json={"modalities": ["audio", "image"]}
)
job_id = response.json()["job_id"]

# Check job status
response = requests.get(f"{BASE_URL}/jobs/{job_id}")
print(f"Job status: {response.json()}")

# Find similar songs
response = requests.post(
    f"{BASE_URL}/songs/{song_id}/similar",
    json={"modality": "audio", "top_k": 5}
)
print(f"Similar songs: {response.json()}")
```

## Testing Scripts

The repository includes several test scripts for exploring and validating the feature store functionality:

1. `test_new_song.py`: Tests creating a single song, extracting features, and performing similarity search
2. `explore_genres.py`: Processes samples from different music genres and compares similarities between them
3. `test_features.py`: Validates feature extraction for different modalities
4. `test_similarity.py`: Tests similarity search functionality
5. `test_api_curl.sh`: Example cURL commands for interacting with the API

## Feature Extraction Process

1. **Audio Features**:
   - Processes WAV files to extract a 28-dimensional feature vector
   - Features capture timbral, rhythmic, and harmonic characteristics

2. **Image Features**:
   - Processes PNG/JPEG album artwork using deep learning models
   - Extracts a 512-dimensional feature vector capturing visual characteristics

## Troubleshooting

### Known Issues

1. **Milvus Connection**: Ensure Milvus is running before starting the server:
   ```bash
   docker ps | grep milvus
   ```
   If Milvus isn't running, restart it with:
   ```bash
   docker-compose up -d
   ```

2. **Numpy Serialization Error**: When accessing feature endpoints directly, you may encounter:
   ```
   TypeError: 'numpy.float32' object is not iterable
   ```
   This is due to serialization issues with raw numpy vectors. Use the `/similar` endpoint instead which properly serializes the results.

3. **Bottleneck Warning**: You may see a warning about the bottleneck library version:
   ```
   UserWarning: Pandas requires version '1.3.6' or newer of 'bottleneck' (version '1.3.5' currently installed).
   ```
   This is a non-critical warning and can be safely ignored, or you can update bottleneck with:
   ```bash
   pip install --upgrade bottleneck
   ```

4. **Port Already In Use**: If you see an error like:
   ```
   [Errno 48] error while attempting to bind on address ('0.0.0.0', 8000): address already in use
   ```
   Kill the existing process using port 8000:
   ```bash
   lsof -i :8000
   kill <PID>
   ```

## Dependencies

- `fastapi`: Web framework for the API
- `pymilvus`: Client for Milvus vector database
- `asyncpg`: Async PostgreSQL client
- `numpy`: Numerical computing
- `librosa`: Audio processing
- `torchvision`: Image processing
- `pandas`: Data manipulation
- `pydantic`: Data validation
- `uvicorn`: ASGI server

## License

MIT


