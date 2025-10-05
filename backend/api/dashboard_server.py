"""
Minimal Dashboard Server
Serves the frontend dashboard and static assets via FastAPI.
"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

app = FastAPI(title="Levi UI Dashboard", version="1.0.0")

frontend_dir = Path(__file__).resolve().parents[2] / "frontend"

app.mount(
    "/static", StaticFiles(directory=str(frontend_dir), html=False), name="static"
)


@app.get("/")
async def index():
    index_file = frontend_dir / "index.html"
    return FileResponse(str(index_file))
