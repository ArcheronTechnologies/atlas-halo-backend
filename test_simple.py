"""Minimal FastAPI app for testing Railway deployment"""
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"status": "ok", "message": "Simple test app working"}

@app.get("/health")
def health():
    return {"status": "healthy"}
