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

## What to Expect When Running

### Starting the Server

When running `python run.py` to start the server, you should see:

```
INFO: NumExpr defaulting to 8 threads.
INFO: Started server process [XXXXX]
INFO: Waiting for application startup.
INFO: Connected to PostgreSQL at localhost:5432/featurestore
INFO: Created tables if they didn't exist
INFO: Connected to Milvus at localhost:19530
INFO: Connected to Milvus
INFO: Application startup complete.
INFO: Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

If Milvus is not running, you'll see an error like:
```
ERROR: Failed to connect to Milvus: <MilvusException: (code=2, message=Fail connecting to server on localhost:19530)>
```

### Creating and Processing a Song

When creating and processing a song through the API, expect:

1. **Creating a song**: Returns a JSON response with the song ID
2. **Setting data paths**: Returns a confirmation message
3. **Processing features**: Returns a job ID for tracking
4. **Job status polling**: Shows progression from "pending" to "processing" to "processed" for each modality
5. **Similarity search**: Returns a list of similar songs with IDs and similarity scores

If you run the included `test_new_song.py` script, you'll see output like:
```
INFO: Created song: 7f218e3d-cb99-4fb4-b9ea-d0ff85c18ace
INFO: Updated data path for song 7f218e3d-cb99-4fb4-b9ea-d0ff85c18ace / audio
INFO: Updated data path for song 7f218e3d-cb99-4fb4-b9ea-d0ff85c18ace / image
INFO: Created job: 991240fa-3b3e-42a8-bdf2-5ec80add9732
INFO: Generated audio features with shape: (28,)
INFO: Inserted embedding for song in modality audio
INFO: Generated image features with shape: (512,)
INFO: Inserted embedding for song in modality image
```

### Common Error Patterns

1. **Milvus not running**:
   ```
   WARNING: Failed to connect to Milvus, vector operations will be disabled
   WARNING: Milvus not connected, skipping feature storage for audio
   ```
   Fix by starting the Milvus server.

2. **Feature retrieval errors**:
   Direct feature access may show errors due to numpy serialization issues:
   ```
   TypeError: 'numpy.float32' object is not iterable
   ```
   This is why the `/similar` endpoint should be used instead of direct feature access.

3. **Port conflicts**:
   ```
   ERROR: [Errno 48] error while attempting to bind on address ('0.0.0.0', 8000): address already in use
   ```
   Fix by killing other processes using port 8000.

## Step-by-Step Implementation Guide

Follow this guide to set up and use the feature store with the included sample data.

### 1. Setting Up The Environment

First, ensure all dependencies are installed:

```bash
pip install -r requirements.txt
```

Expected output:
```
Successfully installed pymilvus-2.3.1 fastapi-0.103.1 uvicorn-0.23.2 asyncpg-0.28.0 librosa-0.10.1 torchvision-0.16.0 ...
```

### 2. Starting the PostgreSQL Database

Ensure PostgreSQL is running and create the necessary database:

```bash
createuser --createdb featurestore_user
createdb -O featurestore_user featurestore
```

Expected output:
No output if successful.

### 3. Starting Milvus with Docker

Start the Milvus vector database:

```bash
docker-compose up -d
```

Expected output:
```
Creating network "mmfeaturestore_default" with the default driver
Creating mmfeaturestore_etcd_1         ... done
Creating mmfeaturestore_minio_1        ... done
Creating mmfeaturestore_standalone_1   ... done
```

Verify Milvus is running:
```bash
docker ps | grep milvus
```

Expected output:
```
d6f3e0a9e0b2   milvusdb/milvus:v2.3.1   "/tini -- milvus run…"   2 hours ago    Up 2 hours    0.0.0.0:19530->19530/tcp, 0.0.0.0:9091->9091/tcp   mmfeaturestore_standalone_1
```

### 4. Starting the API Server

Start the FastAPI server:

```bash
cd mmfeaturestore
python run.py
```

Expected output:
```
INFO: NumExpr defaulting to 8 threads.
INFO: Started server process [58006]
INFO: Waiting for application startup.
INFO: Connected to PostgreSQL at localhost:5432/featurestore
INFO: Created tables if they didn't exist
INFO: Connected to Milvus at localhost:19530
INFO: Connected to Milvus
INFO: Application startup complete.
INFO: Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

### 5. Processing Sample Data

Run the sample processing script:

```bash
python test_new_song.py
```

This script processes a rock sample from the sample data. You'll see output like:

```
INFO: NumExpr defaulting to 8 threads.
INFO: Connected to PostgreSQL at localhost:5432/featurestore
INFO: Using song ID: 610cd810-735b-4996-a94e-f5ed65ea1738
INFO: Updated audio path: mmfeaturestore/sample_data/genres_original/rock/rock.00000.wav
INFO: Updated image path: mmfeaturestore/sample_data/images_original/rock/rock00000.png
INFO: Processing features for audio and image
INFO: Generated audio features with shape: (28,)
INFO: Inserted embedding for audio in Milvus
INFO: Generated image features with shape: (512,)
INFO: Inserted embedding for image in Milvus
INFO: Processing complete
```

### 6. Testing Similarity Search

After processing some samples, you can test similarity search:

```bash
python test_similarity.py
```

Expected output:
```
INFO: Finding similar songs to Rock Sample 00000 by audio...
INFO: Found 3 similar songs:
1. Test Rock Song API (Score: 0.82)
2. Rock Sample 00000 (Score: 0.78)
3. Pop Sample 00032 (Score: 0.65)

INFO: Finding similar songs to Rock Sample 00000 by image...
INFO: Found 2 similar songs:
1. Pop Sample 00032 (Score: 0.71)
2. Direct Test Song (Score: 0.68)
```

### 7. Exploring Multiple Genres

To compare similarities across music genres:

```bash
python explore_genres.py
```

This will process sample files from each available genre and compare their similarities. Expected output:

```
INFO: Found 10 genres: ['pop', 'rock', 'jazz', 'classical', 'metal', 'blues', 'disco', 'reggae', 'hiphop', 'country']
INFO: Processing pop sample...
INFO: Generated audio features with shape: (28,)
INFO: Generated image features with shape: (512,)
INFO: Processing rock sample...
INFO: Generated audio features with shape: (28,)
INFO: Generated image features with shape: (512,)
...
INFO: Processing country sample...
INFO: Finding audio similarities between genres...
INFO: Pop and Rock similarity: 0.65
INFO: Pop and Jazz similarity: 0.47
...
INFO: Finding image similarities between genres...
INFO: Pop and Rock similarity: 0.72
INFO: Pop and Jazz similarity: 0.53
...
```

### 8. Using the API Directly

You can also interact with the API directly using curl or other HTTP clients:

```bash
# Check server health
curl http://localhost:8000/health
# Expected: {"status":"ok"}

# Register a sample
curl -X POST http://localhost:8000/songs \
  -H "Content-Type: application/json" \
  -d '{"id":"sample123", "title":"Rock Sample", "artist":"Sample Artist", "album":"Rock Collection", "genre":"rock", "year":2024}'
# Expected: {"id":"sample123","title":"Rock Sample","artist":"Sample Artist","album":"Rock Collection","genre":"rock","year":2024}

# Set data paths
curl -X PUT http://localhost:8000/songs/sample123/data/audio \
  -H "Content-Type: application/json" \
  -d '{"path":"sample_data/genres_original/rock/rock.00000.wav"}'
# Expected: {"message":"Updated data path for song sample123 / audio"}

curl -X PUT http://localhost:8000/songs/sample123/data/image \
  -H "Content-Type: application/json" \
  -d '{"path":"sample_data/images_original/rock/rock00000.png"}'
# Expected: {"message":"Updated data path for song sample123 / image"}

# Process features
curl -X POST http://localhost:8000/songs/sample123/process \
  -H "Content-Type: application/json" \
  -d '{"modalities":["audio","image"]}'
# Expected: {"job_id":"991240fa-3b3e-42a8-bdf2-5ec80add9732"}

# Check job status
curl http://localhost:8000/jobs/991240fa-3b3e-42a8-bdf2-5ec80add9732
# Expected: {"id":"991240fa-3b3e-42a8-bdf2-5ec80add9732","status":{"audio":"processed","image":"processed"}}

# Find similar songs
curl -X POST http://localhost:8000/songs/sample123/similar \
  -H "Content-Type: application/json" \
  -d '{"modality":"audio","top_k":3}'
# Expected: {"similar":[{"id":"4f512e5a-...","score":0.82},{"id":"7f218e3d-...","score":0.78},{"id":"3a71b2c9-...","score":0.65}]}
```

## Common Issues and Troubleshooting

### 1. Milvus Connection Issues

If you see:
```
ERROR: Failed to connect to Milvus: <MilvusException: (code=2, message=Fail connecting to server on localhost:19530)>
WARNING: Failed to connect to Milvus, vector operations will be disabled
```

**Solution:**
```bash
# Check if Milvus containers are running
docker ps | grep milvus

# If containers are not running, start them
docker-compose up -d
```

### 2. Port Already in Use

If you see:
```
ERROR: [Errno 48] error while attempting to bind on address ('0.0.0.0', 8000): address already in use
```

**Solution:**
```bash
# Find process using port 8000
lsof -i :8000

# Kill the process
kill <PID>
```

### 3. Feature Retrieval Errors

If you see:
```
ValueError: [TypeError("'numpy.float32' object is not iterable"), TypeError('vars() argument must have __dict__ attribute')]
```

**Solution:** This is a known issue with direct feature retrieval. Use the `/similar` endpoint instead, which properly handles the serialization:

```bash
# Instead of:
# curl http://localhost:8000/songs/{song_id}/features/{modality}

# Use:
curl -X POST http://localhost:8000/songs/{song_id}/similar \
  -H "Content-Type: application/json" \
  -d '{"modality":"audio","top_k":3}'
```


