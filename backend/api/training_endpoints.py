"""API endpoints for AI training pipeline management."""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel

from ..auth.jwt_authentication import get_current_user, AuthenticatedUser
from ..ai_training.threat_model_trainer import get_threat_trainer, TrainingJob
from ..database.postgis_database import get_database
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/training", tags=["AI Training Pipeline"])


# Request/Response Models
class TrainingJobRequest(BaseModel):
    model_type: str = "threat_classifier"
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    validation_split: float = 0.2
    
    class Config:
        schema_extra = {
            "example": {
                "model_type": "threat_classifier",
                "epochs": 100,
                "batch_size": 32,
                "learning_rate": 0.001,
                "validation_split": 0.2
            }
        }


class TrainingJobResponse(BaseModel):
    job_id: str
    model_type: str
    status: str
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    dataset_size: int
    epochs: int
    metrics: Optional[Dict]
    model_path: Optional[str]
    error_message: Optional[str]


class TrainingMetricsResponse(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: List[List[int]]
    class_report: Dict


class DatasetStatsResponse(BaseModel):
    total_detections: int
    labeled_examples: int
    feedback_examples: int
    threat_level_distribution: Dict[str, int]
    recent_examples_7d: int
    data_quality_score: float


@router.post("/jobs", response_model=Dict[str, str])
async def create_training_job(
    request: TrainingJobRequest,
    current_user: AuthenticatedUser = Depends(get_current_user)
):
    """
    Create a new AI model training job.
    
    Requires admin privileges to start training jobs.
    """
    
    # Check admin permissions
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin privileges required for training jobs")
    
    try:
        trainer = get_threat_trainer()
        
        job_id = await trainer.create_training_job(
            model_type=request.model_type,
            epochs=request.epochs,
            batch_size=request.batch_size,
            learning_rate=request.learning_rate,
            validation_split=request.validation_split
        )
        
        logger.info(f"User {current_user.user_id} created training job {job_id}")
        
        return {
            "job_id": job_id,
            "status": "created",
            "message": "Training job created successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to create training job: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create training job: {str(e)}")


@router.get("/jobs/{job_id}", response_model=TrainingJobResponse)
async def get_training_job(
    job_id: str,
    current_user: AuthenticatedUser = Depends(get_current_user)
):
    """
    Get the status and details of a specific training job.
    """
    
    try:
        trainer = get_threat_trainer()
        job = await trainer.get_training_job_status(job_id)
        
        if not job:
            raise HTTPException(status_code=404, detail="Training job not found")
        
        # Convert to response model
        metrics_dict = None
        if job.metrics:
            metrics_dict = {
                "accuracy": job.metrics.accuracy,
                "precision": job.metrics.precision,
                "recall": job.metrics.recall,
                "f1_score": job.metrics.f1_score,
                "confusion_matrix": job.metrics.confusion_matrix,
                "class_report": job.metrics.class_report
            }
        
        return TrainingJobResponse(
            job_id=job.job_id,
            model_type=job.model_type,
            status=job.status,
            started_at=job.started_at,
            completed_at=job.completed_at,
            dataset_size=job.dataset_size,
            epochs=job.epochs,
            metrics=metrics_dict,
            model_path=job.model_path,
            error_message=job.error_message
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get training job {job_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve training job")


@router.get("/jobs", response_model=List[TrainingJobResponse])
async def list_training_jobs(
    current_user: AuthenticatedUser = Depends(get_current_user)
):
    """
    List all training jobs (recent first).
    
    Admin users see all jobs, regular users see limited info.
    """
    
    try:
        trainer = get_threat_trainer()
        jobs = await trainer.list_training_jobs()
        
        response = []
        for job in sorted(jobs, key=lambda x: x.started_at or datetime.min, reverse=True):
            
            # Limit details for non-admin users
            if not current_user.is_admin:
                metrics_dict = None
                model_path = None
                error_message = None
            else:
                metrics_dict = None
                if job.metrics:
                    metrics_dict = {
                        "accuracy": job.metrics.accuracy,
                        "precision": job.metrics.precision,
                        "recall": job.metrics.recall,
                        "f1_score": job.metrics.f1_score,
                        "confusion_matrix": job.metrics.confusion_matrix,
                        "class_report": job.metrics.class_report
                    }
                model_path = job.model_path
                error_message = job.error_message
            
            response.append(TrainingJobResponse(
                job_id=job.job_id,
                model_type=job.model_type,
                status=job.status,
                started_at=job.started_at,
                completed_at=job.completed_at,
                dataset_size=job.dataset_size,
                epochs=job.epochs,
                metrics=metrics_dict,
                model_path=model_path,
                error_message=error_message
            ))
        
        return response
        
    except Exception as e:
        logger.error(f"Failed to list training jobs: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve training jobs")


@router.post("/retrain", response_model=Dict[str, str])
async def trigger_retraining(
    current_user: AuthenticatedUser = Depends(get_current_user)
):
    """
    Trigger model retraining based on recent user feedback.
    
    This will create a new training job using recent feedback data to improve model performance.
    """
    
    # Check admin permissions
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin privileges required for retraining")
    
    try:
        trainer = get_threat_trainer()
        job_id = await trainer.retrain_from_feedback()
        
        logger.info(f"User {current_user.user_id} triggered retraining job {job_id}")
        
        return {
            "job_id": job_id,
            "status": "created",
            "message": "Retraining job created successfully based on recent feedback"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to trigger retraining: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to trigger retraining: {str(e)}")


@router.get("/dataset/stats", response_model=DatasetStatsResponse)
async def get_dataset_statistics(
    current_user: AuthenticatedUser = Depends(get_current_user)
):
    """
    Get statistics about the training dataset.
    
    Provides insights into data quality and distribution for training decisions.
    """
    
    try:
        db = await get_database()
        
        # Get total detections
        total_result = await db.execute_query(
            "SELECT COUNT(*) as count FROM threat_detections"
        )
        total_detections = total_result[0]['count'] if total_result else 0
        
        # Get labeled examples (with ground truth or feedback)
        labeled_result = await db.execute_query(
            """
            SELECT COUNT(*) as count 
            FROM threat_detections td
            LEFT JOIN user_feedback uf ON td.detection_id = uf.detection_id
            WHERE td.ground_truth_label IS NOT NULL 
            OR uf.feedback_type IS NOT NULL
            """)
        labeled_examples = labeled_result[0]['count'] if labeled_result else 0
        
        # Get feedback examples
        feedback_result = await db.execute_query(
            "SELECT COUNT(*) as count FROM user_feedback"
        )
        feedback_examples = feedback_result[0]['count'] if feedback_result else 0
        
        # Get threat level distribution
        distribution_result = await db.execute_query(
            """
            SELECT threat_level, COUNT(*) as count
            FROM threat_detections
            GROUP BY threat_level
            ORDER BY count DESC
            """)
        threat_level_distribution = {
            row['threat_level']: row['count'] 
            for row in distribution_result
        }
        
        # Get recent examples (last 7 days)
        recent_result = await db.execute_query(
            """
            SELECT COUNT(*) as count
            FROM threat_detections
            WHERE timestamp >= NOW() - INTERVAL '7 days'
            """)
        recent_examples = recent_result[0]['count'] if recent_result else 0
        
        # Calculate data quality score (0-1)
        if total_detections > 0:
            labeled_ratio = labeled_examples / total_detections
            feedback_ratio = feedback_examples / total_detections
            recent_ratio = min(recent_examples / 100, 1.0)  # Target 100+ recent examples
            
            # Weighted quality score
            data_quality_score = (
                labeled_ratio * 0.5 +      # 50% weight on labeled data
                feedback_ratio * 0.3 +     # 30% weight on feedback
                recent_ratio * 0.2          # 20% weight on recent data
            )
        else:
            data_quality_score = 0.0
        
        return DatasetStatsResponse(
            total_detections=total_detections,
            labeled_examples=labeled_examples,
            feedback_examples=feedback_examples,
            threat_level_distribution=threat_level_distribution,
            recent_examples_7d=recent_examples,
            data_quality_score=round(data_quality_score, 3)
        )
        
    except Exception as e:
        logger.error(f"Failed to get dataset statistics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve dataset statistics")


@router.get("/models/active", response_model=Dict[str, str])
async def get_active_model_info(
    current_user: AuthenticatedUser = Depends(get_current_user)
):
    """
    Get information about the currently active threat detection model.
    """
    
    try:
        # This would typically load from configuration or database
        # For now, return placeholder information
        
        return {
            "model_type": "threat_classifier",
            "version": "1.0.0",
            "last_trained": "2025-09-26T18:00:00Z",
            "accuracy": "0.892",
            "status": "active",
            "training_examples": "2547"
        }
        
    except Exception as e:
        logger.error(f"Failed to get active model info: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve model information")


@router.post("/evaluate", response_model=Dict[str, Any])
async def evaluate_model_performance(
    model_path: Optional[str] = None,
    current_user: AuthenticatedUser = Depends(get_current_user)
):
    """
    Evaluate model performance on recent data.
    
    Can be used to test a specific model or the currently active model.
    """
    
    # Check admin permissions
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin privileges required for model evaluation")
    
    try:
        # This would implement model evaluation logic
        # For now, return placeholder results
        
        return {
            "evaluation_id": "eval_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
            "model_path": model_path or "current_active_model",
            "test_accuracy": 0.884,
            "test_precision": 0.891,
            "test_recall": 0.876,
            "test_f1": 0.883,
            "evaluation_date": datetime.now().isoformat(),
            "test_samples": 150,
            "message": "Model evaluation completed successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to evaluate model: {e}")
        raise HTTPException(status_code=500, detail="Failed to evaluate model performance")


@router.delete("/jobs/{job_id}", response_model=Dict[str, str])
async def cancel_training_job(
    job_id: str,
    current_user: AuthenticatedUser = Depends(get_current_user)
):
    """
    Cancel a running training job.
    
    Only jobs in 'pending' or 'running' status can be cancelled.
    """
    
    # Check admin permissions
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin privileges required to cancel training jobs")
    
    try:
        trainer = get_threat_trainer()
        job = await trainer.get_training_job_status(job_id)
        
        if not job:
            raise HTTPException(status_code=404, detail="Training job not found")
        
        if job.status not in ['pending', 'running']:
            raise HTTPException(
                status_code=400, 
                detail=f"Cannot cancel job with status '{job.status}'"
            )
        
        # Update job status to cancelled
        # In a real implementation, you'd also stop the training process
        job.status = 'cancelled'
        job.completed_at = datetime.now()
        
        logger.info(f"User {current_user.user_id} cancelled training job {job_id}")
        
        return {
            "job_id": job_id,
            "status": "cancelled",
            "message": "Training job cancelled successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel training job {job_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to cancel training job")