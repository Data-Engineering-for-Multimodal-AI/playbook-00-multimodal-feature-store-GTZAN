# Multimodal FeatureStore

A multimodal feature store for processing and storing features from audio and image data. This project uses PostgreSQL for metadata storage and Milvus for vector similarity search.

## Features

- Process audio (WAV) and image (PNG) files
- Extract feature vectors using custom encoders
- Store feature vectors in Milvus for similarity search
- Track processing jobs with PostgreSQL
- Asynchronous processing with FastAPI
- Batch processing support
- RESTful API interface

## Requirements

- Python 3.13+
- PostgreSQL
- Milvus
- Docker (for running Milvus)

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/yourusername/mmfeaturestore.git
cd mmfeaturestore
```

2. Create and activate virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Unix/macOS
# OR
.venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Start PostgreSQL and Milvus services:
```bash
# PostgreSQL (if not running as a service)
pg_ctl -D /path/to/data start

# Milvus (using docker-compose)
docker-compose up -d
```

5. Start the API server:
```bash
cd mmfeaturestore
PYTHONPATH=$PWD uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --log-level debug
```

## Usage

### Single Song Processing

1. Create a song entry:
```bash
curl -X POST http://localhost:8000/songs -H "Content-Type: application/json" -d '{"title": "Song Title", "artist": "Artist Name", "genre": "genre"}'
```

2. Update data paths:
```bash
curl -X PUT http://localhost:8000/songs/{song_id}/data/audio -H "Content-Type: application/json" -d '{"path": "/path/to/audio.wav"}'
curl -X PUT http://localhost:8000/songs/{song_id}/data/image -H "Content-Type: application/json" -d '{"path": "/path/to/image.png"}'
```

3. Process features:
```bash
curl -X POST http://localhost:8000/songs/{song_id}/process -H "Content-Type: application/json" -d '{"modalities": ["audio", "image"]}'
```

### Batch Processing

Process all songs in a folder:
```bash
cd mmfeaturestore
PYTHONPATH=$PWD python scripts/process_all_songs.py --genre-folder /path/to/genre/folder
```

## Project Structure

```
mmfeaturestore/
├── src/
│   ├── api/          # FastAPI application
│   ├── db/           # Database clients
│   ├── encoders/     # Feature extractors
│   └── worker/       # Background tasks
├── scripts/          # Utility scripts
├── tests/           # Test files
└── playbook.md      # Detailed documentation
```

## Feature Vectors

- Audio features: shape (28,)
- Image features: shape (512,)

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Documentation

For more detailed information, please refer to the [playbook](mmfeaturestore/playbook.md). 
