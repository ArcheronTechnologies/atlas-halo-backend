"""Incident classification AI service - Atlas Intelligence Integration"""
import logging
from typing import Dict, Optional
from backend.services.atlas_client import get_atlas_client

logger = logging.getLogger(__name__)

class IncidentClassifier:
    """AI-powered incident classification (powered by Atlas Intelligence)"""

    # Polisen.se incident categories (Swedish)
    POLISEN_CATEGORIES = [
        "Rån",
        "Våld mot person",
        "Mord/dråp",
        "Sexualbrott",
        "Narkotikabrott",
        "Vapenbrott",
        "Skadegörelse",
        "Inbrott",
        "Stöld",
        "Motorfordon, stöld",
        "Brand",
        "Trafikolycka",
        "Trafikhinder",
        "Bråk",
        "Övrigt"
    ]

    def __init__(self):
        self.atlas = get_atlas_client()
        logger.info("IncidentClassifier initialized (Atlas Intelligence)")

    async def classify(
        self,
        description: str,
        context: Optional[Dict] = None
    ) -> Dict:
        """
        Classify incident type using Atlas Intelligence

        Returns:
            {
                "classification": {
                    "threat_category": "weapons",
                    "threat_subcategory": "weapon_possession",
                    "severity": 5,
                    "confidence": 0.85
                },
                "product_mappings": {
                    "halo_incident_type": "weapon_possession",
                    ...
                },
                "recommendations": [...]
            }
        """
        try:
            result = await self.atlas.classify_threat(description, context)

            # Map to Halo's expected format
            classification = result.get("classification", {})
            mappings = result.get("product_mappings", {})

            return {
                "incident_type": mappings.get("halo_incident_type", "other"),
                "confidence": classification.get("confidence", 0.0),
                "severity": classification.get("severity", 1),
                "threat_level": classification.get("threat_category", "unknown"),
                "recommendations": result.get("recommendations", []),
                "atlas_classification": classification  # Full Atlas response
            }

        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return {
                "incident_type": "other",
                "confidence": 0.0,
                "severity": 1,
                "error": str(e)
            }

_incident_classifier = None

def get_incident_classifier() -> IncidentClassifier:
    global _incident_classifier
    if _incident_classifier is None:
        _incident_classifier = IncidentClassifier()
    return _incident_classifier
