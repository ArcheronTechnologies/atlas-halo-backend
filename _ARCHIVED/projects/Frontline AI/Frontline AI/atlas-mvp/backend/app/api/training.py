from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Request, BackgroundTasks
from typing import List, Dict, Optional
import os
import shutil
import logging
from pathlib import Path
import uuid
from datetime import datetime

from app.database import db_manager
from app.training.automl import AutoMLTrainer

logger = logging.getLogger(__name__)

training_router = APIRouter()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@training_router.get("/models")
async def get_models():
    try:
        models = await db_manager.get_models()
        return {"models": models}
        
    except Exception as e:
        logger.error(f"Error getting models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@training_router.get("/models/active")
async def get_active_model():
    try:
        active_model = await db_manager.get_active_model()
        if not active_model:
            raise HTTPException(status_code=404, detail="No active model found")
        
        return {"model": active_model}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting active model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@training_router.post("/models/{model_id}/activate")
async def activate_model(model_id: int, request: Request):
    try:
        success = await db_manager.set_active_model(model_id)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to activate model")
        
        models = await db_manager.get_models()
        active_model = next((m for m in models if m['id'] == model_id), None)
        
        if active_model:
            inference_engine = request.app.state.inference_engine
            if inference_engine:
                model_switched = await inference_engine.switch_model(active_model['path'])
                if not model_switched:
                    logger.warning(f"Failed to switch to model {active_model['path']}")
        
        return {"message": f"Model {model_id} activated successfully"}
        
    except Exception as e:
        logger.error(f"Error activating model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@training_router.post("/upload-images")
async def upload_training_images(
    files: List[UploadFile] = File(...),
    project_name: str = Form(...)
):
    try:
        if not project_name:
            raise HTTPException(status_code=400, detail="Project name is required")
        
        project_dir = UPLOAD_DIR / project_name
        project_dir.mkdir(exist_ok=True)
        images_dir = project_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        uploaded_files = []
        
        for file in files:
            if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                logger.warning(f"Skipping non-image file: {file.filename}")
                continue
            
            file_id = str(uuid.uuid4())
            file_extension = Path(file.filename).suffix
            safe_filename = f"{file_id}{file_extension}"
            file_path = images_dir / safe_filename
            
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            uploaded_files.append({
                "original_name": file.filename,
                "saved_name": safe_filename,
                "file_path": str(file_path),
                "file_size": len(content)
            })
        
        return {
            "message": f"Successfully uploaded {len(uploaded_files)} images",
            "project_name": project_name,
            "uploaded_files": uploaded_files,
            "project_path": str(project_dir)
        }
        
    except Exception as e:
        logger.error(f"Error uploading images: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@training_router.post("/projects/{project_name}/annotate")
async def save_annotations(
    project_name: str,
    annotations: Dict = None
):
    try:
        project_dir = UPLOAD_DIR / project_name
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")
        
        annotations_file = project_dir / "annotations.json"
        
        import json
        with open(annotations_file, 'w') as f:
            json.dump(annotations, f, indent=2)
        
        return {
            "message": "Annotations saved successfully",
            "annotations_file": str(annotations_file)
        }
        
    except Exception as e:
        logger.error(f"Error saving annotations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@training_router.post("/projects/{project_name}/train")
async def start_training(
    project_name: str,
    background_tasks: BackgroundTasks,
    epochs: int = 10,
    batch_size: int = 16
):
    try:
        project_dir = UPLOAD_DIR / project_name
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")
        
        annotations_file = project_dir / "annotations.json"
        if not annotations_file.exists():
            raise HTTPException(status_code=400, detail="No annotations found for project")
        
        trainer = AutoMLTrainer()
        
        background_tasks.add_task(
            trainer.train_model,
            project_dir,
            epochs,
            batch_size
        )
        
        return {
            "message": f"Training started for project '{project_name}'",
            "project_name": project_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "status": "training_started"
        }
        
    except Exception as e:
        logger.error(f"Error starting training: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@training_router.get("/projects")
async def list_projects():
    try:
        projects = []
        if UPLOAD_DIR.exists():
            for project_dir in UPLOAD_DIR.iterdir():
                if project_dir.is_dir():
                    images_dir = project_dir / "images"
                    annotations_file = project_dir / "annotations.json"
                    
                    image_count = 0
                    if images_dir.exists():
                        image_count = len(list(images_dir.glob("*.{jpg,jpeg,png}")))
                    
                    projects.append({
                        "name": project_dir.name,
                        "path": str(project_dir),
                        "image_count": image_count,
                        "has_annotations": annotations_file.exists(),
                        "created_at": datetime.fromtimestamp(project_dir.stat().st_ctime).isoformat()
                    })
        
        return {"projects": projects}
        
    except Exception as e:
        logger.error(f"Error listing projects: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@training_router.delete("/projects/{project_name}")
async def delete_project(project_name: str):
    try:
        project_dir = UPLOAD_DIR / project_name
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")
        
        shutil.rmtree(project_dir)
        
        return {"message": f"Project '{project_name}' deleted successfully"}
        
    except Exception as e:
        logger.error(f"Error deleting project: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@training_router.get("/projects/{project_name}/images")
async def get_project_images(project_name: str):
    try:
        project_dir = UPLOAD_DIR / project_name
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")
        
        images_dir = project_dir / "images"
        if not images_dir.exists():
            return {"images": []}
        
        images = []
        for image_file in images_dir.glob("*"):
            if image_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                images.append({
                    "filename": image_file.name,
                    "path": str(image_file),
                    "size": image_file.stat().st_size
                })
        
        return {"images": images}
        
    except Exception as e:
        logger.error(f"Error getting project images: {e}")
        raise HTTPException(status_code=500, detail=str(e))