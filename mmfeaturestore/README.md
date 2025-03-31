# Multimodal Feature Store

A system for extracting, storing, and retrieving feature vectors from multimodal data sources (audio, images, and text).

## Overview

This project implements a feature store for multimodal data, specifically focusing on music data with three modalities:

1. **Audio** - extracts features from WAV files using librosa
2. **Images** - extracts features from album art or spectrograms using ResNet18
3. **Lyrics** - extracts features from song lyrics using SentenceTransformers

The system consists of:
- Feature extractors for each modality
- A PostgreSQL database for metadata storage
- A Milvus vector database for storing embeddings
- A FastAPI web service for accessing the feature store

## Directory Structure

```
mmfeaturestore/
├── src/                    # Source code
│   ├── api/                # FastAPI web service
│   ├── db/                 # Database clients
│   ├── encoders/           # Feature encoders
│   ├── models/             # Data models
│   └── worker/             # Background processing
├── sample_data/            # Sample data for testing
├── test_sample_data.py     # Script to test with sample data
└── requirements.txt        # Python dependencies
```

## Setup

### Prerequisites

- Python 3.10+
- PostgreSQL
- Milvus (or Milvus Lite)

### Installation

1. Clone this repository

2. Create a virtual environment and activate it:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables (optional):
   ```bash
   export POSTGRES_USER=postgres
   export POSTGRES_PASSWORD=postgres
   export POSTGRES_HOST=localhost
   export POSTGRES_PORT=5432
   export POSTGRES_DB=mmfeaturestore
   export MILVUS_HOST=localhost
   export MILVUS_PORT=19530
   ```

## Usage

### Running the API

Start the FastAPI server:

```bash
cd mmfeaturestore
python -m src.api.main
```

The API will be available at http://localhost:8000, with interactive documentation at http://localhost:8000/docs.

### API Endpoints

- `POST /songs` - Create a new song
- `GET /songs/{song_id}` - Get a song by ID
- `PUT /songs/{song_id}/data/{modality}` - Update a song's data path
- `POST /songs/{song_id}/process` - Process song features
- `GET /jobs/{job_id}` - Get a processing job
- `GET /songs/{song_id}/features/{modality}` - Get feature vectors
- `POST /songs/{song_id}/similar` - Find similar songs

### Testing with Sample Data

To test the feature extractors with sample data:

```bash
python test_sample_data.py
```

## Modalities and Feature Extraction

### Audio Features

Audio features are extracted using librosa and include:
- MFCCs (Mel-frequency cepstral coefficients)
- Spectral centroid
- Chroma features
- Zero crossing rate
- Spectral rolloff

The combined feature vector has 28 dimensions.

### Image Features

Image features are extracted using a pre-trained ResNet18 model with the classification head removed. The features are 512-dimensional vectors representing the visual content.

### Text/Lyrics Features

Lyrics features are extracted using a SentenceTransformer model (all-MiniLM-L6-v2), which produces 384-dimensional embeddings representing the semantic content of the lyrics.

## Vector Similarity Search

Milvus provides efficient vector similarity search capabilities. The system uses cosine similarity to find songs with similar features in any modality.

## License

This project is open-source software. 