# Multimodal Feature Store (MMFeatureStore)

A comprehensive multimodal feature store system for processing, storing, and retrieving feature vectors from audio and image data. This project enables efficient similarity search across different modalities using state-of-the-art vector database technology.

## Overview

MMFeatureStore is designed to extract meaningful features from different types of media (audio files, images, and text) and store them in a vector database for efficient similarity search. The system is built with a microservice architecture using FastAPI for the API layer, PostgreSQL for metadata storage, and Milvus for vector similarity search.

### Key Features

- **Multimodal Processing**: Extract feature vectors from audio (WAV), image (PNG), and text data
- **Feature Extraction**: Uses specialized encoders for each modality:
  - Audio: MFCCs, spectral centroids, chroma features, zero crossing rate, and spectral rolloff
  - Image: Deep features from a pre-trained ResNet18 model
  - Text: Embeddings from Sentence Transformers
- **Vector Database**: Store and search feature vectors using Milvus' similarity search capabilities
- **RESTful API**: Well-structured API with FastAPI for easy integration
- **Asynchronous Processing**: Background job processing for handling large files
- **Job Tracking**: Track the status of processing jobs in PostgreSQL
- **Similar Content Discovery**: Find similar content across your dataset based on acoustic or visual similarity

## Requirements

- **Python 3.10+** (Recommended: Python 3.13)
- **Docker** and **Docker Compose** for running Milvus
- **PostgreSQL** for metadata storage
- At least 4GB of RAM for running the services
- Internet connection for downloading pre-trained models (first run only)

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Data-Engineering-for-Multimodal-AI/pb2-mmfeaturestore.git
cd pb2-mmfeaturestore
```

### 2. Set Up Python Environment

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Unix/macOS
# OR
.venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Start Milvus with Docker Compose

```bash
docker-compose up -d
```

This will start:
- Milvus vector database
- Etcd (required by Milvus)
- MinIO (required by Milvus for storage)

### 4. Set Up PostgreSQL

Ensure PostgreSQL is installed and running. Create a database for the feature store:

```bash
# Access PostgreSQL
psql -U postgres

# In PostgreSQL prompt
CREATE DATABASE featurestore;
CREATE USER featurestore_user WITH PASSWORD 'featurestore_password';
GRANT ALL PRIVILEGES ON DATABASE featurestore TO featurestore_user;
```

## Testing the Installation

Run the test script to verify that all components are working properly:

```bash
cd mmfeaturestore
python run_test.py
```

This will test:
- Audio encoder functionality
- Image encoder functionality
- Text encoder functionality

If all tests pass, your setup is complete and ready to use!

## Running the API Server

Start the FastAPI server:

```bash
cd mmfeaturestore
python run.py
```

The API server will be available at http://localhost:8000

You can access the interactive API documentation at http://localhost:8000/docs

## API Usage Guide

### Creating a Song Entry

```bash
curl -X POST http://localhost:8000/songs \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Song Title",
    "artist": "Artist Name",
    "album": "Album Name",
    "genre": "rock",
    "year": 2023
  }'
```

This will return a response with the song ID.

### Setting Data Paths

```bash
# For audio file
curl -X PUT http://localhost:8000/songs/YOUR_SONG_ID/data/audio \
  -H "Content-Type: application/json" \
  -d '{
    "path": "/absolute/path/to/your/audio.wav"
  }'

# For album artwork
curl -X PUT http://localhost:8000/songs/YOUR_SONG_ID/data/image \
  -H "Content-Type: application/json" \
  -d '{
    "path": "/absolute/path/to/your/cover.png"
  }'
```

### Processing Features

```bash
curl -X POST http://localhost:8000/songs/YOUR_SONG_ID/process \
  -H "Content-Type: application/json" \
  -d '{
    "modalities": ["audio", "image"]
  }'
```

This will return a job ID which you can use to check the status.

### Checking Job Status

```bash
curl -X GET http://localhost:8000/jobs/YOUR_JOB_ID
```

### Finding Similar Songs

```bash
curl -X POST http://localhost:8000/songs/YOUR_SONG_ID/similar \
  -H "Content-Type: application/json" \
  -d '{
    "modality": "audio",
    "top_k": 5
  }'
```

## Project Structure

```
mmfeaturestore/
├── src/                # Main source code
│   ├── api/            # FastAPI application
│   │   └── main.py     # API endpoints
│   ├── db/             # Database clients
│   │   ├── milvus.py   # Milvus vector database client
│   │   └── postgres.py # PostgreSQL metadata database client
│   ├── encoders/       # Feature extractors
│   │   ├── audio.py    # Audio feature extraction
│   │   ├── image.py    # Image feature extraction
│   │   ├── text.py     # Text feature extraction
│   │   └── factory.py  # Factory pattern for encoders
│   ├── models/         # Data models
│   │   └── pipeline.py # Pydantic models
│   └── worker/         # Background processing
│       └── tasks.py    # Asynchronous tasks
├── sample_data/        # Sample files for testing
├── volumes/            # Mounted volumes for Docker
├── run.py              # Entry point for the API server
└── run_test.py         # Testing script
```

## Sample Data

Sample data is included in the `sample_data` directory for testing purposes. This includes:
- Sample audio files in WAV format (in `genres_original`)
- Sample image files in PNG format (in `images_original`)

## Known Limitations

1. Currently optimized for WAV and PNG files only
2. Text encoding requires internet access on first run to download models
3. Large files (>50MB) may cause memory issues on systems with less than 4GB RAM

## What to Expect

When running this code, you should expect:

1. **Docker Startup**: Setting up Milvus and its dependencies takes approximately 30-60 seconds
2. **Initial Processing**: The first feature extraction may be slow as models are loaded
3. **API Performance**: Typical API response times:
   - Creating songs/metadata: ~50ms
   - Processing features: Varies by file size (~1-10s)
   - Similarity search: ~100-200ms

## Troubleshooting

### Common Issues

1. **Milvus Connection Error**:
   - Ensure Docker is running and the containers are up (`docker ps`)
   - Check if Milvus is accessible on port 19530 (`telnet localhost 19530`)

2. **PostgreSQL Connection Error**:
   - Verify PostgreSQL is running (`pg_isready`)
   - Confirm database credentials in `src/db/postgres.py`

3. **Encoder Errors**:
   - For audio encoding errors, ensure librosa and its dependencies are correctly installed
   - For image encoding errors, check if torch and torchvision are properly installed

## Performance Considerations

- Audio encoding: ~1-5 seconds per file
- Image encoding: ~0.5-2 seconds per file
- Text encoding: ~0.1-0.5 seconds per text
- Vector search: ~50-200ms for top-k results

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 