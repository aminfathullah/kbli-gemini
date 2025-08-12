"""
FastAPI application for Gemini AI REST API
"""

import asyncio
import time
import logging
import sys
from contextlib import asynccontextmanager
from typing import List, Optional
import tempfile
import aiofiles
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer
import structlog

from .config import settings, validate_settings, get_cors_origins
from .models import (
    ChatRequest, ChatResponse, ErrorResponse, HealthResponse,
    QueueStatus, ModelsResponse, ModelInfo, FileUploadRequest,
    ConversationResetResponse, GeminiModel
)
from .gemini_service import gemini_service
from .queue_manager import RequestQueue

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(message)s",
    stream=sys.stdout,
)
logger = structlog.get_logger(__name__)

# Global variables
request_queue: Optional[RequestQueue] = None
app_start_time = time.time()

# Security
security = HTTPBearer(auto_error=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global request_queue
    
    logger.info("Starting application", version=settings.app_version, environment=settings.environment)
    
    try:
        # Validate configuration
        validate_settings()
        logger.info("Configuration validated successfully")
        
        # Initialize request queue
        request_queue = RequestQueue(max_size=settings.max_queue_size)
        logger.info("Request queue initialized", max_size=settings.max_queue_size)
        
        # Initialize Gemini service
        await gemini_service.initialize()
        logger.info("Gemini service initialized")
        
        yield
        
    except Exception as e:
        logger.error("Failed to start application", error=str(e))
        raise
    finally:
        logger.info("Shutting down application")
        if request_queue:
            await request_queue.cleanup()


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=settings.description,
    lifespan=lifespan,
    docs_url="/docs" if settings.environment != "production" else None,
    redoc_url="/redoc" if settings.environment != "production" else None,
)

# Add security middleware
if settings.environment == "production":
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.allowed_hosts
    )

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests"""
    start_time = time.time()
    
    # Log request
    logger.info(
        "Request started",
        method=request.method,
        url=str(request.url),
        client_ip=request.client.host if request.client else None,
    )
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # Log response
        logger.info(
            "Request completed",
            method=request.method,
            url=str(request.url),
            status_code=response.status_code,
            process_time=process_time,
        )
        
        response.headers["X-Process-Time"] = str(process_time)
        return response
        
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(
            "Request failed",
            method=request.method,
            url=str(request.url),
            error=str(e),
            process_time=process_time,
        )
        raise


async def process_gemini_request(request_data: dict) -> dict:
    """Process a Gemini request (used by queue manager)"""
    try:
        request_type = request_data.get("type")
        
        if request_type == "chat":
            return await gemini_service.send_message(
                request_data["message"]
            )
        elif request_type == "reset":
            return await gemini_service.reset_conversation()
        else:
            raise ValueError(f"Unknown request type: {request_type}")
            
    except Exception as e:
        logger.error(f"Error processing Gemini request: {e}")
        raise


def get_queue() -> RequestQueue:
    """Dependency to get the request queue"""
    if request_queue is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    return request_queue


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler with structured logging"""
    logger.error(
        "Unhandled exception",
        url=str(request.url),
        method=request.method,
        error=str(exc),
        exc_info=True,
    )
    
    if settings.environment == "production":
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "detail": "An unexpected error occurred"}
        )
    else:
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "detail": str(exc)}
        )

@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information"""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "description": settings.description,
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "uptime": time.time() - app_start_time,
        "status": "running"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        uptime = time.time() - app_start_time
        queue_size = request_queue.size() if request_queue else 0
        
        # Get Gemini service health
        gemini_health = await gemini_service.health_check()
        
        return HealthResponse(
            status="healthy" if gemini_health["service_initialized"] else "unhealthy",
            uptime=uptime,
            queue_size=queue_size,
            gemini_connected=gemini_health["client_connected"]
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Health check failed: {e}")


@app.get("/queue/status", response_model=QueueStatus)
async def get_queue_status(queue: RequestQueue = Depends(get_queue)):
    """Get current queue status"""
    try:
        status = queue.get_status()
        
        return QueueStatus(
            current_size=status["queue_size"],
            max_size=status["max_size"],
            processing=status["processing"],
            average_wait_time=queue.get_average_wait_time()
        )
        
    except Exception as e:
        logger.error(f"Error getting queue status: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting queue status: {e}")


@app.get("/models", response_model=ModelsResponse)
async def get_models():
    """Get available Gemini models"""
    try:
        models_data = await gemini_service.get_available_models()
        models = [ModelInfo(**model) for model in models_data]
        
        return ModelsResponse(models=models)
        
    except Exception as e:
        logger.error(f"Error getting models: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting models: {e}")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, queue: RequestQueue = Depends(get_queue)):
    """Send a message to Gemini AI"""
    try:
        if queue.is_full():
            raise HTTPException(status_code=503, detail="Request queue is full")
        
        # Prepare request data
        request_data = {
            "type": "chat",
            "message": request.message,
            "model": request.model,
            "gem_id": request.gem_id,
            "reset_conversation": request.reset_conversation
        }
        
        # Add to queue and wait for result
        result = await queue.process_request_and_wait(
            request_data=request_data,
            timeout=settings.request_timeout
        )
        
        # Add queue position info
        result["queue_position"] = queue.size()
        
        return ChatResponse(**result)
        
    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail="Request timed out")
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing chat request: {e}")


@app.post("/chat/file", response_model=ChatResponse)
async def chat_with_file(
    background_tasks: BackgroundTasks,
    message: str = Form(...),
    model: Optional[GeminiModel] = Form(default=GeminiModel.UNSPECIFIED),
    gem_id: Optional[str] = Form(default=None),
    file: UploadFile = File(...),
    queue: RequestQueue = Depends(get_queue)
):
    """Send a message to Gemini AI with file upload"""
    try:
        if queue.is_full():
            raise HTTPException(status_code=503, detail="Request queue is full")
        
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Create temporary file
        temp_file = None
        try:
            # Create temporary file with original extension
            file_extension = Path(file.filename).suffix
            temp_file = tempfile.NamedTemporaryFile(
                delete=False, 
                suffix=file_extension
            )
            
            # Save uploaded file to temporary location
            content = await file.read()
            async with aiofiles.open(temp_file.name, 'wb') as f:
                await f.write(content)
            
            # Prepare request data
            request_data = {
                "type": "chat",
                "message": message,
                "model": model,
                "gem_id": gem_id,
                "files": [temp_file.name],
                "reset_conversation": False
            }
            
            # Add to queue and wait for result
            result = await queue.process_request_and_wait(
                request_data=request_data,
                timeout=settings.request_timeout
            )
            
            # Add queue position info
            result["queue_position"] = queue.size()
            
            # Schedule cleanup of temp file
            background_tasks.add_task(cleanup_temp_file, temp_file.name)
            
            return ChatResponse(**result)
            
        except Exception as e:
            # Clean up temp file on error
            if temp_file and Path(temp_file.name).exists():
                Path(temp_file.name).unlink()
            raise e
            
    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail="Request timed out")
    except Exception as e:
        logger.error(f"Error in chat with file endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing chat request: {e}")


@app.delete("/conversation/reset", response_model=ConversationResetResponse)
async def reset_conversation(queue: RequestQueue = Depends(get_queue)):
    """Reset the current conversation"""
    try:
        if queue.is_full():
            raise HTTPException(status_code=503, detail="Request queue is full")
        
        # Prepare request data
        request_data = {
            "type": "reset"
        }
        
        # Add to queue and wait for result
        result = await queue.process_request_and_wait(
            request_data=request_data,
            timeout=30  # Shorter timeout for reset
        )
        
        return ConversationResetResponse(**result)
        
    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail="Reset request timed out")
    except Exception as e:
        logger.error(f"Error in reset conversation endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Error resetting conversation: {e}")


async def cleanup_temp_file(file_path: str):
    """Clean up temporary file"""
    try:
        file_obj = Path(file_path)
        if file_obj.exists():
            file_obj.unlink()
            logger.debug(f"Cleaned up temporary file: {file_path}")
    except Exception as e:
        logger.warning(f"Could not clean up temporary file {file_path}: {e}")


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level=settings.log_level.lower()
    )