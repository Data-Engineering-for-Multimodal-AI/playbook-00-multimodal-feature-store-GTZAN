#!/usr/bin/env bash
# The API with curl (python run.py must be running).
set -euo pipefail
API=${API:-http://127.0.0.1:8000}
curl -s "$API/health"; echo
curl -s -X POST "$API/songs" -H "Content-Type: application/json" \
  -d '{"id": "rock.00001", "title": "rock.00001", "genre": "rock"}'; echo
curl -s -X PUT "$API/songs/rock.00001/data/audio" -H "Content-Type: application/json" \
  -d '{"path": "sample_data/genres_original/rock/rock.00001.wav"}'; echo
curl -s -X PUT "$API/songs/rock.00001/data/image" -H "Content-Type: application/json" \
  -d '{"path": "sample_data/images_original/rock/rock00001.png"}'; echo
curl -s -X POST "$API/songs/rock.00001/process" -H "Content-Type: application/json" \
  -d '{"modalities": ["audio", "image"]}'; echo
sleep 3
curl -s -X POST "$API/songs/rock.00001/similar" -H "Content-Type: application/json" \
  -d '{"modality": "audio", "top_k": 3}'; echo
