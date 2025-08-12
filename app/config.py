"""
Configuration settings for the Gemini AI REST API
"""

import os
from typing import Optional
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings(BaseModel):
    """Application settings"""
    
    # Gemini API credentials
    gemini_secure_1psid: str = os.getenv("GEMINI_SECURE_1PSID", "")
    gemini_secure_1psidts: str = os.getenv("GEMINI_SECURE_1PSIDTS", "")
    
    # API configuration
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    
    # Queue management
    max_queue_size: int = int(os.getenv("MAX_QUEUE_SIZE", "100"))
    request_timeout: int = int(os.getenv("REQUEST_TIMEOUT", "300"))
    
    # Redis configuration (optional)
    redis_url: Optional[str] = os.getenv("REDIS_URL")
    
    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    
    # FastAPI configuration
    app_name: str = "Gemini AI REST API"
    app_version: str = "1.0.0"
    description: str = "A full-featured REST API for Google Gemini AI with queue management"


# Global settings instance
settings = Settings()


def validate_settings() -> bool:
    """Validate required settings"""
    if not settings.gemini_secure_1psid:
        raise ValueError("GEMINI_SECURE_1PSID is required. Please set it in your .env file")
    
    if len(settings.gemini_secure_1psid) < 10:
        raise ValueError("GEMINI_SECURE_1PSID appears to be invalid")
    
    return True
