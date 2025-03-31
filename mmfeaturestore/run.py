import uvicorn
import os
import sys

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.api.main import app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 