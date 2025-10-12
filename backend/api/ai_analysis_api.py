"""
AI Analysis API Endpoints
Photo/Video/Audio analysis with auto-classification
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, List
import logging
from io import BytesIO

from backend.ai_processing.photo_analyzer import get_photo_analyzer
from backend.ai_processing.incident_classifier import get_incident_classifier
from backend.ai_processing.video_analyzer import get_video_analyzer
from backend.ai_processing.audio_analyzer import get_audio_analyzer

router = APIRouter(
    prefix="/api/v1/ai",
    tags=["AI Analysis"]
)

logger = logging.getLogger(__name__)


# Response models
class PhotoAnalysisResponse(BaseModel):
    """Response from photo analysis"""
    success: bool
    analysis: Dict
    classification: Dict
    summary: str
    processing_time_ms: float

    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "analysis": {
                    "objects": [
                        {"class": "person", "confidence": 0.89, "bbox": [120, 45, 340, 280]},
                        {"class": "knife", "confidence": 0.76, "bbox": [200, 150, 230, 200]}
                    ],
                    "people_count": 1,
                    "weapons": [{"class": "knife", "confidence": 0.76}],
                    "suggested_category": "weapons_offense",
                    "confidence": 0.75
                },
                "classification": {
                    "backend_category": "weapons_offense",
                    "backend_subcategory": "threatening_with_weapon",
                    "mobile_category": "violence",
                    "polisen_type": "Vapenbrott",
                    "confidence": 0.75,
                    "reasoning": ["Weapons detected: knife", "Person detected in frame"]
                },
                "summary": "Detected weapons offense with 75% confidence",
                "processing_time_ms": 156.3
            }
        }


class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: str
    models_loaded: bool
    device: str
    message: str


@router.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """
    Check if AI services are ready

    Returns health status and model information
    """
    try:
        analyzer = get_photo_analyzer()

        return {
            "status": "healthy",
            "models_loaded": True,
            "device": analyzer.device,
            "message": f"AI services ready on {analyzer.device}"
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "models_loaded": False,
            "device": "unknown",
            "message": f"Error: {str(e)}"
        }


@router.post("/analyze/photo", response_model=PhotoAnalysisResponse)
async def analyze_photo(
    file: UploadFile = File(..., description="Photo to analyze (JPEG, PNG)"),
    return_detailed: bool = Form(default=True, description="Return detailed analysis")
):
    """
    Analyze a photo for crime indicators and auto-classify incident

    **Features**:
    - Object detection (people, weapons, vehicles, etc.)
    - Violence indicator detection
    - Vandalism indicator detection
    - Weapon detection
    - Auto-classification to incident categories
    - Multi-level taxonomy mapping (Backend, Mobile, Polisen.se)

    **Performance**: ~100-200ms on M1 MacBook Air

    **Example Usage**:
    ```bash
    curl -X POST "http://localhost:8000/api/v1/ai/analyze/photo" \\
      -F "file=@/path/to/image.jpg" \\
      -F "return_detailed=true"
    ```
    """
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {file.content_type}. Must be an image."
            )

        # Read file data
        image_data = await file.read()

        if len(image_data) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        if len(image_data) > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(status_code=400, detail="File too large (max 10MB)")

        logger.info(f"Analyzing photo: {file.filename} ({len(image_data)} bytes)")

        # Get AI services
        analyzer = get_photo_analyzer()
        classifier = get_incident_classifier()

        # Analyze photo
        analysis = analyzer.analyze_photo(image_data)

        if 'error' in analysis:
            raise HTTPException(status_code=500, detail=f"Analysis failed: {analysis['error']}")

        # Classify incident
        classification = classifier.classify_from_analysis(analysis)

        # Validate classification
        is_valid, errors = classifier.validate_classification(classification)
        if not is_valid:
            logger.warning(f"Classification validation errors: {errors}")

        # Generate summary
        summary = _generate_summary(classification, analysis)

        # Prepare response
        response = {
            "success": True,
            "analysis": analysis if return_detailed else {
                "suggested_category": analysis.get('suggested_category'),
                "confidence": analysis.get('confidence'),
                "people_count": analysis.get('people_count'),
                "weapons_detected": len(analysis.get('weapons', [])) > 0
            },
            "classification": classification,
            "summary": summary,
            "processing_time_ms": analysis.get('processing_time_ms', 0)
        }

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in analyze_photo: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.post("/analyze/photo/batch")
async def analyze_photo_batch(files: List[UploadFile] = File(...)):
    """
    Analyze multiple photos in batch

    **Use case**: User submits multiple evidence photos for a single incident

    Returns combined analysis with the highest-confidence classification
    """
    try:
        if len(files) == 0:
            raise HTTPException(status_code=400, detail="No files provided")

        if len(files) > 10:
            raise HTTPException(status_code=400, detail="Maximum 10 files per batch")

        analyzer = get_photo_analyzer()
        classifier = get_incident_classifier()

        results = []
        total_time = 0

        for file in files:
            # Validate file
            if not file.content_type or not file.content_type.startswith('image/'):
                continue

            # Read and analyze
            image_data = await file.read()
            analysis = analyzer.analyze_photo(image_data)

            if 'error' not in analysis:
                classification = classifier.classify_from_analysis(analysis)
                results.append({
                    'filename': file.filename,
                    'analysis': analysis,
                    'classification': classification
                })
                total_time += analysis.get('processing_time_ms', 0)

        if len(results) == 0:
            raise HTTPException(status_code=400, detail="No valid images to analyze")

        # Find highest confidence classification
        best_result = max(results, key=lambda x: x['classification']['confidence'])

        return {
            "success": True,
            "total_photos": len(results),
            "best_classification": best_result['classification'],
            "all_results": results,
            "total_processing_time_ms": total_time
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch analysis: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.get("/categories")
async def get_categories():
    """
    Get all available incident categories across taxonomies

    Returns:
    - Backend API categories (10 main types)
    - Mobile app categories (6 main types)
    - Polisen.se Swedish mappings
    """
    classifier = get_incident_classifier()

    return {
        "backend_categories": classifier.BACKEND_CATEGORIES,
        "mobile_categories": classifier.MOBILE_CATEGORIES,
        "polisen_mapping": classifier.POLISEN_MAPPING
    }


@router.get("/categories/{category}")
async def get_category_details(category: str):
    """
    Get detailed information about a specific category

    Args:
        category: Backend category name (e.g., 'assault', 'vandalism')
    """
    classifier = get_incident_classifier()

    try:
        details = classifier.get_category_details(category)
        return details
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Category not found: {category}")


@router.post("/analyze/video")
async def analyze_video(
    file: UploadFile = File(..., description="Video to analyze (MP4, MOV, AVI)"),
    keyframe_fps: float = Form(default=1.0, description="Extract keyframes at this FPS (lower = faster)"),
    max_duration: int = Form(default=60, description="Maximum video duration to process (seconds)")
):
    """
    Analyze a video for crime indicators using keyframe extraction

    **Features**:
    - Memory-efficient keyframe extraction (1 FPS = 30x memory reduction)
    - Timeline of events throughout video
    - Aggregated analysis across all frames
    - Weapon/people detection over time
    - Violence/vandalism indicators

    **Performance**: ~2-5 seconds for 60-second video on M1

    **Example Usage**:
    ```bash
    curl -X POST "http://localhost:8000/api/v1/ai/analyze/video" \\
      -F "file=@/path/to/video.mp4" \\
      -F "keyframe_fps=1.0" \\
      -F "max_duration=60"
    ```
    """
    try:
        # Log incoming request for debugging
        logger.info(f"Received video analysis request: filename={file.filename}, content_type={file.content_type}")

        # Validate file type
        if not file.content_type or not file.content_type.startswith('video/'):
            logger.error(f"Invalid content type: {file.content_type}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {file.content_type}. Must be a video."
            )

        # Read file data
        video_data = await file.read()

        if len(video_data) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        if len(video_data) > 100 * 1024 * 1024:  # 100MB limit
            raise HTTPException(status_code=400, detail="File too large (max 100MB)")

        logger.info(f"Analyzing video: {file.filename} ({len(video_data)} bytes)")

        # Save to temporary file (required for OpenCV)
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(video_data)
            tmp_path = tmp_file.name

        try:
            # Get video analyzer
            analyzer = get_video_analyzer()
            classifier = get_incident_classifier()

            # Analyze video
            analysis = await analyzer.analyze(video_data, filename=file.filename)

            if 'error' in analysis:
                raise HTTPException(status_code=500, detail=f"Analysis failed: {analysis['error']}")

            # Classify based on aggregated results
            # Create a pseudo-analysis for classifier
            pseudo_analysis = {
                'suggested_category': analysis['suggested_category'],
                'confidence': analysis['confidence'],
                'people_count': analysis['aggregated_results']['max_people_count'],
                'weapons': analysis['aggregated_results']['weapons_detected'],
                'violence_indicators': {'weapons_present': len(analysis['aggregated_results']['weapons_detected']) > 0},
                'vandalism_indicators': {},
                'objects': []
            }

            classification = classifier.classify_from_analysis(pseudo_analysis)

            # Generate summary
            summary = _generate_video_summary(analysis, classification)

            return {
                "success": True,
                "video_analysis": {
                    "duration_seconds": analysis['duration_seconds'],
                    "keyframes_analyzed": analysis['keyframes_analyzed'],
                    "resolution": analysis['resolution'],
                    "people_detected": analysis['people_detected'],
                    "weapons_detected": analysis['weapons_detected'],
                    "violence_indicators": analysis['violence_indicators'],
                    "max_people_count": analysis['aggregated_results']['max_people_count'],
                    "avg_people_count": analysis['aggregated_results']['avg_people_count'],
                    "violence_score": analysis['aggregated_results']['violence_score'],
                    "vandalism_score": analysis['aggregated_results']['vandalism_score']
                },
                "timeline": analysis['timeline'],
                "classification": classification,
                "summary": summary,
                "processing_time_ms": analysis['processing_time_ms']
            }

        finally:
            # Clean up temp file
            import os
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in analyze_video: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


def _generate_summary(classification: Dict, analysis: Dict) -> str:
    """Generate human-readable summary of analysis"""
    cat = classification['backend_category']
    conf = classification['confidence']

    summary_parts = []

    # Main classification
    summary_parts.append(f"Classified as {cat} ({conf:.0%} confidence)")

    # Key details
    people = analysis.get('people_count', 0)
    if people > 0:
        summary_parts.append(f"{people} person(s) detected")

    weapons = analysis.get('weapons', [])
    if weapons:
        weapon_types = ', '.join(set(w['class'] for w in weapons))
        summary_parts.append(f"Weapons: {weapon_types}")

    return ". ".join(summary_parts) + "."


def _generate_video_summary(analysis: Dict, classification: Dict) -> str:
    """Generate human-readable summary of video analysis"""
    cat = classification['backend_category']
    conf = classification['confidence']
    agg = analysis['aggregated_results']

    summary_parts = []

    # Main classification
    summary_parts.append(f"Video classified as {cat} ({conf:.0%} confidence)")

    # Duration
    duration = analysis['duration_seconds']
    keyframes = analysis['keyframes_analyzed']
    summary_parts.append(f"Analyzed {keyframes} frames from {duration:.1f}s video")

    # People
    if agg['max_people_count'] > 0:
        summary_parts.append(f"Up to {agg['max_people_count']} people visible")

    # Weapons
    if agg['weapons_detected']:
        weapon_types = ', '.join(set(w['class'] for w in agg['weapons_detected']))
        summary_parts.append(f"⚠️ Weapons detected: {weapon_types}")

    # Violence score
    if agg['violence_score'] > 0.5:
        summary_parts.append(f"🚨 High violence indicators ({agg['violence_score']:.0%})")

    # Vandalism score
    if agg['vandalism_score'] > 0.4:
        summary_parts.append(f"🎨 Vandalism indicators ({agg['vandalism_score']:.0%})")

    return ". ".join(summary_parts) + "."


@router.post("/analyze/audio")
async def analyze_audio(
    file: UploadFile = File(..., description="Audio to analyze (MP3, WAV, M4A)"),
    language: str = Form(default='sv', description="Language code ('sv' or 'en')"),
    model_size: str = Form(default='tiny', description="Whisper model size")
):
    """
    Analyze audio for speech transcription and threat detection

    **Features**:
    - Speech-to-text transcription (Swedish & English)
    - Threat keyword detection
    - Aggression/urgency scoring
    - Speaker count estimation
    - Auto-classification based on content

    **Performance**: ~5-15 seconds for 60-second audio on M1 (Whisper-tiny)

    **Example Usage**:
    ```bash
    curl -X POST "http://localhost:8000/api/v1/ai/analyze/audio" \\
      -F "file=@/path/to/audio.mp3" \\
      -F "language=sv" \\
      -F "model_size=tiny"
    ```
    """
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('audio/'):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {file.content_type}. Must be audio."
            )

        # Read file data
        audio_data = await file.read()

        if len(audio_data) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        if len(audio_data) > 50 * 1024 * 1024:  # 50MB limit
            raise HTTPException(status_code=400, detail="File too large (max 50MB)")

        logger.info(f"Analyzing audio: {file.filename} ({len(audio_data)} bytes)")

        # Save to temporary file (required for Whisper)
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
            tmp_file.write(audio_data)
            tmp_path = tmp_file.name

        try:
            # Get audio analyzer
            analyzer = get_audio_analyzer(model_size=model_size)
            classifier = get_incident_classifier()

            # Analyze audio
            analysis = analyzer.analyze_audio(tmp_path, language=language)

            if 'error' in analysis:
                raise HTTPException(status_code=500, detail=f"Analysis failed: {analysis['error']}")

            # Classify based on audio analysis
            pseudo_analysis = {
                'suggested_category': analysis['suggested_category'],
                'confidence': analysis['confidence'],
                'people_count': analysis['speaker_count_estimate'],
                'weapons': [],
                'violence_indicators': {
                    'weapons_present': analysis['threat_level'] in ['high', 'medium']
                },
                'vandalism_indicators': {},
                'objects': []
            }

            classification = classifier.classify_from_analysis(pseudo_analysis)

            # Generate summary
            summary = _generate_audio_summary(analysis, classification)

            return {
                "success": True,
                "audio_analysis": {
                    "transcription": analysis['transcription'],
                    "language": analysis['language'],
                    "duration_seconds": analysis['duration_seconds'],
                    "threat_level": analysis['threat_level'],
                    "threat_score": analysis['threat_score'],
                    "aggression_score": analysis['aggression_score'],
                    "distress_detected": analysis['distress_detected'],
                    "threat_keywords": analysis['threat_keywords_found'],
                    "speaker_estimate": analysis['speaker_count_estimate']
                },
                "classification": classification,
                "summary": summary,
                "processing_time_ms": analysis['processing_time_ms']
            }

        finally:
            # Clean up temp file
            import os
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in analyze_audio: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


def _generate_audio_summary(analysis: Dict, classification: Dict) -> str:
    """Generate human-readable summary of audio analysis"""
    cat = classification['backend_category']
    conf = classification['confidence']

    summary_parts = []

    # Main classification
    summary_parts.append(f"Audio classified as {cat} ({conf:.0%} confidence)")

    # Transcription snippet
    transcription = analysis['transcription']
    if transcription:
        snippet = transcription[:100] + "..." if len(transcription) > 100 else transcription
        summary_parts.append(f"Transcription: \"{snippet}\"")

    # Threat level
    threat_level = analysis['threat_level']
    if threat_level != 'none':
        summary_parts.append(f"⚠️ Threat level: {threat_level}")

    # Threat keywords
    if analysis['threat_keywords_found']:
        keywords = ', '.join(analysis['threat_keywords_found'][:3])
        summary_parts.append(f"Keywords detected: {keywords}")

    # Distress
    if analysis['distress_detected']:
        summary_parts.append("🚨 Distress indicators detected")

    # Aggression
    if analysis['aggression_score'] > 0.6:
        summary_parts.append(f"High aggression ({analysis['aggression_score']:.0%})")

    # Speakers
    if analysis['speaker_count_estimate'] > 1:
        summary_parts.append(f"~{analysis['speaker_count_estimate']} speakers detected")

    return ". ".join(summary_parts) + "."
