import asyncio
import numpy as np
import json
from src.db.milvus import MilvusClient
from src.models.pipeline import Modality

async def test_feature_conversion():
    """Test feature conversion and serialization"""
    song_id = "46d05664-0323-4b20-8a32-79af214edaf5"  # Use an existing song ID
    modality = Modality.IMAGE  # Test with image features
    
    # Connect to Milvus and get features
    milvus_client = MilvusClient()
    try:
        milvus_client.connect()
        features = milvus_client.get_embedding(song_id, modality)
        
        print(f"Original features type: {type(features)}")
        if features is not None:
            print(f"First feature type: {type(features[0])}")
        
        # Convert numpy array to list
        if isinstance(features, np.ndarray):
            features = features.tolist()
            
        # Convert numpy float32 to Python float - deeper conversion
        if features:
            if isinstance(features, list):
                features = [float(x) if isinstance(x, np.number) else x for x in features]
            
        # Test JSON serialization
        try:
            json_str = json.dumps({"features": features})
            print("Successfully serialized to JSON")
            print(f"First few values: {features[:5]}")
            return True
        except Exception as e:
            print(f"Failed to serialize to JSON: {e}")
            return False
    finally:
        if milvus_client.connected:
            milvus_client.disconnect()

if __name__ == "__main__":
    asyncio.run(test_feature_conversion()) 