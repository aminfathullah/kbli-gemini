"""
Pydantic models for request/response validation
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class GeminiModel(str, Enum):
    """Available Gemini models"""
    UNSPECIFIED = "unspecified"
    GEMINI_2_5_FLASH = "gemini-2.5-flash"
    GEMINI_2_5_PRO = "gemini-2.5-pro"
    GEMINI_2_0_FLASH = "gemini-2.0-flash"
    GEMINI_2_0_FLASH_THINKING = "gemini-2.0-flash-thinking"


class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    message: str = Field(..., description="The message to send to Gemini", min_length=1)
    model: Optional[GeminiModel] = Field(
        default=GeminiModel.UNSPECIFIED,
        description="The Gemini model to use"
    )
    gem_id: Optional[str] = Field(
        default=None,
        description="Gem ID for applying system prompt"
    )
    reset_conversation: Optional[bool] = Field(
        default=False,
        description="Whether to reset the conversation before sending the message"
    )


class ImageInfo(BaseModel):
    """Information about an image in the response"""
    title: Optional[str] = None
    url: Optional[str] = None
    alt: Optional[str] = None
    image_type: Optional[str] = None  # "web" or "generated"


class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    success: bool = Field(..., description="Whether the request was successful")
    message: str = Field(..., description="The response text from Gemini")
    model_used: Optional[str] = Field(default=None, description="The model that was used")
    thoughts: Optional[str] = Field(default=None, description="The model's thought process")
    images: Optional[List[ImageInfo]] = Field(default=[], description="Images in the response")
    conversation_id: Optional[str] = Field(default=None, description="The conversation ID")
    response_time: Optional[float] = Field(default=None, description="Response time in seconds")
    queue_position: Optional[int] = Field(default=None, description="Position in queue when processed")


class ErrorResponse(BaseModel):
    """Error response model"""
    success: bool = Field(default=False)
    error: str = Field(..., description="Error message")
    error_type: Optional[str] = Field(default=None, description="Type of error")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="API status")
    uptime: float = Field(..., description="Uptime in seconds")
    queue_size: int = Field(..., description="Current queue size")
    gemini_connected: bool = Field(..., description="Whether Gemini client is connected")


class QueueStatus(BaseModel):
    """Queue status information"""
    current_size: int = Field(..., description="Current number of requests in queue")
    max_size: int = Field(..., description="Maximum queue size")
    processing: bool = Field(..., description="Whether a request is currently being processed")
    average_wait_time: Optional[float] = Field(default=None, description="Average wait time in seconds")


class ModelInfo(BaseModel):
    """Information about a Gemini model"""
    name: str = Field(..., description="Model name")
    display_name: str = Field(..., description="Display name")
    description: Optional[str] = Field(default=None, description="Model description")
    available: bool = Field(..., description="Whether the model is available")


class ModelsResponse(BaseModel):
    """Response for models endpoint"""
    models: List[ModelInfo] = Field(..., description="List of available models")


class FileUploadRequest(BaseModel):
    """Request model for file upload chat"""
    message: str = Field(..., description="The message to send with the file")
    model: Optional[GeminiModel] = Field(default=GeminiModel.UNSPECIFIED)
    gem_id: Optional[str] = Field(default=None)


class ConversationResetResponse(BaseModel):
    """Response for conversation reset"""
    success: bool = Field(..., description="Whether the reset was successful")
    message: str = Field(..., description="Confirmation message")
    new_conversation_id: Optional[str] = Field(default=None, description="New conversation ID")
