#!/usr/bin/env python3
"""Start the API: python run.py  (http://127.0.0.1:8000/docs; HOST=0.0.0.0 to listen on the network)"""
import os
import sys

import uvicorn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.api.main import app  # noqa: E402

if __name__ == "__main__":
    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "8000")))
