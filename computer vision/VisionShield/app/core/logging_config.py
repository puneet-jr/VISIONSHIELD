"""Structured application logs, without pairing secrets or camera image payloads."""

from datetime import datetime, timezone
import json
import logging
from app.config import config


class JsonFormatter(logging.Formatter):
    def format(self, record):
        message = record.getMessage()
        for secret in [config.ACCESS_TOKEN, config.CAMERA_TOKEN, config.GEMINI_API_KEY]:
            if secret:
                message = message.replace(secret, "[REDACTED]")
        return json.dumps(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": message,
            }
        )


def configure_logging():
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("VisionShield")
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)
    logger.propagate = False
