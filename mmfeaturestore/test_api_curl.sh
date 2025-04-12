#!/bin/bash

# This script demonstrates using the Feature Store API with cURL

# Set base URL
BASE_URL="http://localhost:8000"

# Health check
echo "Checking API health..."
curl -s "${BASE_URL}/health"
echo -e "\n"

# Create a new song
echo "Creating a new song..."
SONG_RESPONSE=$(curl -s -X POST "${BASE_URL}/songs" \
    -H "Content-Type: application/json" \
    -d '{
        "title": "Test Rock Song API",
        "artist": "API Tester",
        "album": "cURL Test Album",
        "genre": "rock",
        "year": 2024
    }')

echo $SONG_RESPONSE
SONG_ID=$(echo $SONG_RESPONSE | sed -n 's/.*"id":"\([^"]*\)".*/\1/p')
echo "Created song ID: $SONG_ID"
echo -e "\n"

# Set audio path
echo "Setting audio path..."
curl -s -X PUT "${BASE_URL}/songs/${SONG_ID}/data/audio" \
    -H "Content-Type: application/json" \
    -d '{
        "path": "mmfeaturestore/sample_data/genres_original/rock/rock.00001.wav"
    }'
echo -e "\n"

# Set image path
echo "Setting image path..."
curl -s -X PUT "${BASE_URL}/songs/${SONG_ID}/data/image" \
    -H "Content-Type: application/json" \
    -d '{
        "path": "mmfeaturestore/sample_data/images_original/rock/rock00001.png"
    }'
echo -e "\n"

# Process features
echo "Processing features..."
JOB_RESPONSE=$(curl -s -X POST "${BASE_URL}/songs/${SONG_ID}/process" \
    -H "Content-Type: application/json" \
    -d '{
        "modalities": ["audio", "image"]
    }')

echo $JOB_RESPONSE
JOB_ID=$(echo $JOB_RESPONSE | sed -n 's/.*"job_id":"\([^"]*\)".*/\1/p')
echo "Job ID: $JOB_ID"
echo -e "\n"

# Check job status (poll until complete)
echo "Checking job status..."
STATUS="pending"
while [[ "$STATUS" != *"processed"* ]]; do
    sleep 2
    JOB_STATUS_RESPONSE=$(curl -s -X GET "${BASE_URL}/jobs/${JOB_ID}")
    echo $JOB_STATUS_RESPONSE
    
    if [[ "$JOB_STATUS_RESPONSE" == *"image\":\"processed"* && "$JOB_STATUS_RESPONSE" == *"audio\":\"processed"* ]]; then
        STATUS="processed"
        echo "Processing complete!"
        break
    fi
    
    echo "Still processing, waiting..."
done
echo -e "\n"

# Find similar songs by audio
echo "Finding similar songs by audio..."
curl -s -X POST "${BASE_URL}/songs/${SONG_ID}/similar" \
    -H "Content-Type: application/json" \
    -d '{
        "modality": "audio",
        "top_k": 5
    }'
echo -e "\n"

# Find similar songs by image
echo "Finding similar songs by image..."
curl -s -X POST "${BASE_URL}/songs/${SONG_ID}/similar" \
    -H "Content-Type: application/json" \
    -d '{
        "modality": "image",
        "top_k": 5
    }'
echo -e "\n"

echo "API test completed!" 