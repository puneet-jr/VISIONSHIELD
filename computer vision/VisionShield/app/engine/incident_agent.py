"""Multimodal Agentic Incident Dispatcher
Produces real-time, human-grade Tactical Incident Dossiers from security threat events.

Features:
1. Automatically captures the threat keyframe and encodes it to base64.
2. Leverages Google Gemini Flash Multimodal API when GEMINI_API_KEY is available.
3. Automatically falls back to an offline Tactical Expert Synthesizer if offline,
   guaranteeing 100% demo uptime on stage during the hackathon!
"""
import base64
import json
import logging
import cv2
import numpy as np
import time
from typing import Dict, Any, Optional
from app.config import config
from app.db.database import db

logger = logging.getLogger("VisionShield.IncidentAgent")

class IncidentDispatcher:
    def __init__(self):
        self.api_key = config.GEMINI_API_KEY
        self._gemini_client = None
        self._init_client()

    def _init_client(self):
        if self.api_key:
            try:
                from google import genai
                self._gemini_client = genai.Client(api_key=self.api_key, http_options={"timeout": 15000})
                logger.info("Multimodal AI Dispatcher successfully configured with Gemini API.")
            except Exception as e:
                logger.warning(f"Failed to configure Gemini API client: {e}. Will use Offline Tactical Expert.")
                self._gemini_client = None

    def generate_dossier(self, event: Dict[str, Any], keyframe: Optional[np.ndarray]) -> Dict[str, Any]:
        """Generate structured tactical incident dossier for a HIGH severity alert."""
        b64_image = None
        if keyframe is not None:
            try:
                # Resize thumbnail to save memory and network bandwidth
                h, w = keyframe.shape[:2]
                scale = min(1.0, 480 / max(h, w))
                thumb = cv2.resize(keyframe, (int(w * scale), int(h * scale)))
                _, buffer = cv2.imencode(".jpg", thumb, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                b64_image = base64.b64encode(buffer).decode("utf-8")
            except Exception as e:
                logger.error(f"Thumbnail encoding error: {e}")

        # If Gemini is configured and online, attempt Multimodal analysis
        if self._gemini_client and b64_image:
            try:
                dossier = self._call_gemini(event, b64_image)
                if dossier:
                    dossier["snapshot_base64"] = b64_image
                    db.insert_incident(dossier)
                    return dossier
            except Exception as e:
                logger.warning(f"Gemini API call failed ({e}). Falling back to Tactical Synthesizer.")

        # Offline Tactical Expert Synthesizer (Zero-latency fallback)
        dossier = self._synthesize_offline(event)
        dossier["report_mode"] = "offline_template"
        dossier["snapshot_base64"] = b64_image
        db.insert_incident(dossier)
        return dossier

    def _call_gemini(self, event: Dict[str, Any], b64_image: str) -> Optional[Dict[str, Any]]:
        prompt = f"""
        You are a Tactical Security Operations Center AI Commander analyzing a live security alert.
        Alert Type: {event.get('alert_type')}
        Object: {event.get('object_label')} (ID #{event.get('track_id')})
        Threat Score: {event.get('threat_score')} / 100
        Explanation: {event.get('explanation')}

        Analyze this surveillance snapshot and provide a strictly formatted JSON response with:
        {{
            "incident_title": "Short tactical code name (e.g. OPERATION SHIELD: UNATTENDED PACKAGE)",
            "severity": "HIGH",
            "executive_summary": "2-sentence clear situation breakdown for field dispatch",
            "subject_description": "Visual details observable in image (clothing, object color, environment)",
            "threat_level_rating": integer between 1 and 10,
            "recommended_action": "Immediate tactical protocol (e.g. Dispatch Sector 3 officer, isolate perimeter 15m)"
        }}
        Output ONLY the raw JSON without markdown formatting.
        """
        from google.genai import types
        response = self._gemini_client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents=[prompt, types.Part.from_bytes(data=base64.b64decode(b64_image), mime_type="image/jpeg")],
            config={"response_mime_type": "application/json"})
        text = response.text.strip()
        if text.startswith("```json"):
            text = text[7:-3].strip()
        elif text.startswith("```"):
            text = text[3:-3].strip()

        data = json.loads(text)
        data["timestamp"] = event.get("timestamp", time.strftime("%Y-%m-%d %H:%M:%S"))
        return data

    def _synthesize_offline(self, event: Dict[str, Any]) -> Dict[str, Any]:
        alert_type = event.get("alert_type", "SECURITY_ALERT")
        obj = event.get("object_label", "Entity")
        tid = event.get("track_id", 0)
        score = event.get("threat_score", 85.0)

        if alert_type == "UNATTENDED_LUGGAGE":
            title = f"UNATTENDED BAG CANDIDATE [{obj.upper()} #{tid}]"
            summary = event.get("explanation", "A stationary bag remained separated from its associated owner.")
            subject = f"Detected {obj}, track #{tid}. Ownership is a spatial association, not verified identity."
            rating = 8
            action = "Review live footage and contact the site supervisor to verify ownership."
        elif alert_type == "FALL_DETECTED":
            title = f"POSSIBLE FALL [PERSON #{tid}]"
            summary = event.get("explanation", "Bounding-box geometry changed in a way consistent with a possible fall.")
            subject = f"Person track #{tid}. Physical condition has not been verified."
            rating = 8
            action = "Check the person's condition and follow the site's assistance procedure."
        elif alert_type == "PERIMETER_BREACH":
            title = f"RESTRICTED AREA ENTRY [PERSON #{tid}]"
            summary = event.get("explanation", "Person remained within an operator-defined zone.")
            subject = f"Person track #{tid}. Access authorization cannot be inferred from video."
            rating = 8
            action = "Verify the person's authorization before escalating to site security."
        elif alert_type == "POSSIBLE_WEAPON":
            title = f"POSSIBLE KNIFE NEAR PERSON [OBJECT #{tid}]"
            summary = event.get("explanation", "Repeated knife-class detections near a person.")
            subject = "Object classified as knife. Hand contact, intent and danger are not established."
            rating = 8
            action = "Have an operator review the live scene promptly and follow site policy if a threat is verified."
        else:
            title = f"OBSERVATION: {alert_type} [ENTITY #{tid}]"
            summary = event.get("explanation", "Behavioral deviation detected.")
            subject = f"Entity: {obj}, track #{tid}."
            rating = 6
            action = "Review the observation in context before taking action."

        return {
            "timestamp": event.get("timestamp", time.strftime("%Y-%m-%d %H:%M:%S")),
            "incident_title": title,
            "severity": "HIGH",
            "executive_summary": summary,
            "subject_description": subject,
            "threat_level_rating": rating,
            "recommended_action": action
        }

# Global singleton dispatcher
incident_dispatcher = IncidentDispatcher()
