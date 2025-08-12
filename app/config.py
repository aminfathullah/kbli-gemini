"""
Configuration settings for the Gemini AI REST API
"""

import os
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field, validator


class Settings(BaseSettings):
    """Application settings with validation"""
    
    # Gemini API Configuration
    gemini_secure_1psid: str = Field(..., description="Gemini __Secure-1PSID cookie")
    gemini_secure_1psidts: str = Field(..., description="Gemini __Secure-1PSIDTS cookie")
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")
    app_name: str = Field(default="Gemini AI REST API", description="Application name")
    app_version: str = Field(default="1.0.0", description="Application version")
    description: str = Field(default="FastAPI-based REST API for Google Gemini AI", description="App description")
    log_level: str = Field(default="INFO", description="Logging level")
    
    # Queue Management
    max_queue_size: int = Field(default=100, description="Maximum queue size")
    request_timeout: int = Field(default=300, description="Request timeout in seconds")
    
    # Redis Configuration (optional)
    redis_url: Optional[str] = Field(default=None, description="Redis connection URL")
    redis_password: Optional[str] = Field(default=None, description="Redis password")
    redis_db: int = Field(default=0, description="Redis database number")
    
    # Security
    secret_key: str = Field(default="dev-secret-key", description="Secret key for sessions")
    allowed_hosts: List[str] = Field(default=["*"], description="Allowed hosts")
    
    # File Upload Configuration
    max_file_size: int = Field(default=10485760, description="Maximum file size in bytes (10MB)")
    upload_directory: str = Field(default="./uploads", description="Upload directory")
    allowed_file_types: List[str] = Field(
        default=[
            "image/jpeg", "image/png", "image/gif", "image/webp",
            "application/pdf", "text/plain", "text/markdown",
            "application/msword", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ],
        description="Allowed file MIME types"
    )
    
    # Rate Limiting
    rate_limit_requests: int = Field(default=100, description="Rate limit requests per window")
    rate_limit_window: int = Field(default=3600, description="Rate limit window in seconds")
    
    # Environment
    environment: str = Field(default="development", description="Environment (development/production)")
    debug: bool = Field(default=False, description="Debug mode")
    
    @validator('log_level')
    def validate_log_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'Invalid log level. Must be one of: {valid_levels}')
        return v.upper()
    
    @validator('environment')
    def validate_environment(cls, v):
        valid_envs = ['development', 'staging', 'production']
        if v.lower() not in valid_envs:
            raise ValueError(f'Invalid environment. Must be one of: {valid_envs}')
        return v.lower()
    
    @validator('debug')
    def validate_debug_in_production(cls, v, values):
        if values.get('environment') == 'production' and v:
            raise ValueError('Debug mode cannot be enabled in production')
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Initialize settings
settings = Settings()


def validate_settings() -> bool:
    """Validate that all required settings are properly configured"""
    try:
        # Check required Gemini credentials
        if not settings.gemini_secure_1psid or not settings.gemini_secure_1psidts:
            raise ValueError("Gemini credentials are required")
        
        # Create upload directory if it doesn't exist
        os.makedirs(settings.upload_directory, exist_ok=True)
        
        return True
    except Exception as e:
        raise ValueError(f"Configuration validation failed: {e}")


def get_cors_origins() -> List[str]:
    """Get CORS origins based on environment"""
    if settings.environment == "production":
        return settings.allowed_hosts
    else:
        return ["*"]  # Allow all origins in development