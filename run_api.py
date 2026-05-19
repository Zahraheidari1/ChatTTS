"""
run_api.py  –  start the ChatTTS OpenAI-compatible API server.

Usage
-----
    conda activate speech
    cd ChatTTS-main
    python run_api.py

The server will be available at  http://localhost:8000
Health check:  http://localhost:8000/health
"""
import os
import sys

# Make sure imports find tools/ and ChatTTS/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "examples.api.openai_api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
