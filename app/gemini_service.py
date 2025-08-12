"""
Gemini API service with persistent chat session and advanced features
"""

import asyncio
import time
import logging
from typing import Optional, List, Dict, Any, Union
from pathlib import Path
import json
import aiofiles

from gemini_webapi import GeminiClient
from gemini_webapi.constants import Model

from .config import settings
from .models import GeminiModel, ImageInfo

logger = logging.getLogger(__name__)


class GeminiService:
    """
    Service class for managing Gemini API interactions
    Features:
    - Persistent chat session with metadata storage
    - File upload support
    - Model selection
    - Error handling and retry logic
    - Response caching
    """
    
    def __init__(self):
        self.client: Optional[GeminiClient] = None
        self.chat_session = None
        self.chat_metadata = None
        self.is_initialized = False
        self.initialization_time: Optional[float] = None
        self.last_request_time: Optional[float] = None
        self.request_count = 0
        self.error_count = 0
        self.metadata_file = Path("chat_metadata.json")
        
    async def initialize(self) -> bool:
        """
        Initialize the Gemini client and restore chat session
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            logger.info("Initializing Gemini client...")
            
            # Create client with credentials
            self.client = GeminiClient(
                settings.gemini_secure_1psid,
                settings.gemini_secure_1psidts,
                proxy=None
            )
            
            # Initialize client with settings
            await self.client.init(
                timeout=60,
                auto_close=True,  # Enable auto-close for better resource management
                close_delay=600,  # 10 minutes of inactivity before closing
                auto_refresh=True  # Keep cookies fresh
            )
            
            # Load existing chat metadata if available
            await self._load_chat_metadata()
            
            # Start or restore chat session
            if self.chat_metadata:
                logger.info("Restoring previous chat session")
                self.chat_session = self.client.start_chat(metadata=self.chat_metadata)
            else:
                logger.info("Starting new chat session")
                self.chat_session = self.client.start_chat()
                await self.chat_session.send_message("Please become my KBLI 2020 classifier. untuk format jawaban, kurang lebih tampilkan kategori, kode KBLI 5 digit, judul KBLI, dan alasan kenapa memilih kode tersebut. Sebagai knowledge base, berikut saya lampirkan daftar kode KBLI aktif, yg berisi kode KBLI 5 digit beserta judulnya|Uraian dari kode KBLI 5 digit tersebut|parent1: berisi parent KBLI 4 digit nya|parent2: berisi parent KBLI 3 digit nya|parent3: berisi parent KBLI 2 digit nya|kategori: berisi kategori KBLI 5 digit tersebut. Dan aku menambahkan juga file kasus batas untuk kasus2 yg biasa jadi perdebatan", 
                                        files=["kbli.txt", Path('kasus-batas.pdf')])
                print(self.chat_session.metadata)
                # await self.chat_session.send_message("Please become my KBLI 2020 classifier. untuk format jawaban, kurang lebih tampilkan kategori, kode KBLI 5 digit, judul KBLI, dan alasan kenapa memilih kode tersebut. Sebagai knowledge base, berikut saya lampirkan daftar kode KBLI aktif, yg berisi kode KBLI 5 digit beserta judulnya|Uraian dari kode KBLI 5 digit tersebut|parent1: berisi parent KBLI 4 digit nya|parent2: berisi parent KBLI 3 digit nya|parent3: berisi parent KBLI 2 digit nya|kategori: berisi kategori KBLI 5 digit tersebut", 
                #                         files=["kbli.txt", Path('kasus-batas.pdf')])
                
            self.is_initialized = True
            self.initialization_time = time.time()
            
            logger.info("Gemini client initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            self.error_count += 1
            return False
            
    async def _load_chat_metadata(self):
        """Load chat metadata from file"""
        try:
            if self.metadata_file.exists():
                async with aiofiles.open(self.metadata_file, 'r') as f:
                    content = await f.read()
                    print(content)
                    self.chat_metadata = json.loads(content)
                logger.info("Chat metadata loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load chat metadata: {e}")
            self.chat_metadata = None
            
    async def _save_chat_metadata(self):
        """Save chat metadata to file"""
        try:
            if self.chat_session and hasattr(self.chat_session, 'metadata'):
                async with aiofiles.open(self.metadata_file, 'w') as f:
                    await f.write(json.dumps(self.chat_session.metadata, indent=2))
                logger.debug("Chat metadata saved successfully")
        except Exception as e:
            logger.warning(f"Could not save chat metadata: {e}")
            
    async def send_message(
        self,
        message: str,
        model: Optional[GeminiModel] = None,
        files: Optional[List[Union[str, Path]]] = None,
        gem_id: Optional[str] = None,
        reset_conversation: bool = False
    ) -> Dict[str, Any]:
        """
        Send a message to Gemini
        
        Args:
            message: The message to send
            model: The model to use
            files: List of file paths to include
            gem_id: Gem ID for system prompt
            reset_conversation: Whether to reset the conversation
            
        Returns:
            Dictionary containing response data
        """
        if not self.is_initialized:
            raise RuntimeError("Gemini service not initialized")
            
        start_time = time.time()
        
        try:
            # Reset conversation if requested
            if reset_conversation:
                await self.reset_conversation()
                
            # Convert model enum to string if provided
            model_str = None
            if model and model != GeminiModel.UNSPECIFIED:
                model_str = model.value
                
            # Prepare arguments for send_message
            send_args = {}
            
            if files:
                send_args["files"] = files
                
            if model_str:
                send_args["model"] = model_str
                
            if gem_id:
                send_args["gem"] = gem_id
                
            logger.debug(f"Sending message with args: {send_args}")
            
            # Send message to Gemini
            response = await self.chat_session.send_message(message)
            
            # Save updated metadata
            await self._save_chat_metadata()
            
            # Process response
            result = await self._process_response(response, start_time, model_str)
            
            self.request_count += 1
            self.last_request_time = time.time()
            
            return result
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Error sending message to Gemini: {e}")
            raise
            
    async def _process_response(self, response, start_time: float, model_used: Optional[str]) -> Dict[str, Any]:
        """Process Gemini response into standardized format"""
        processing_time = time.time() - start_time
        
        # Extract text response
        response_text = str(response) if response else ""
        
        # Extract images if available
        images = []
        if hasattr(response, 'images') and response.images:
            for img in response.images:
                image_info = ImageInfo(
                    title=getattr(img, 'title', None),
                    url=getattr(img, 'url', None),
                    alt=getattr(img, 'alt', None),
                    image_type=getattr(img, 'image_type', 'unknown')
                )
                images.append(image_info.model_dump())
                
        # Extract thoughts if available
        thoughts = None
        if hasattr(response, 'thoughts') and response.thoughts:
            thoughts = str(response.thoughts)
            
        # Get conversation ID
        conversation_id = None
        if self.chat_session and hasattr(self.chat_session, 'metadata'):
            conversation_id = self.chat_session.metadata[0] if self.chat_session.metadata else None
            
        return {
            "success": True,
            "message": response_text,
            "model_used": model_used,
            "thoughts": thoughts,
            "images": images,
            "conversation_id": conversation_id,
            "response_time": processing_time
        }
        
    async def reset_conversation(self) -> Dict[str, Any]:
        """
        Reset the current conversation and start a new one
        
        Returns:
            Dictionary with reset confirmation
        """
        try:
            logger.info("Resetting conversation")
            
            # Start a new chat session
            self.chat_session = self.client.start_chat()
            
            # Clear metadata
            self.chat_metadata = None
            
            # Remove metadata file
            if self.metadata_file.exists():
                self.metadata_file.unlink()
                
            # Get new conversation ID
            new_conversation_id = None
            if hasattr(self.chat_session, 'metadata'):
                new_conversation_id = self.chat_session.metadata.get('conversation_id')
                
            return {
                "success": True,
                "message": "Conversation reset successfully",
                "new_conversation_id": new_conversation_id
            }
            
        except Exception as e:
            logger.error(f"Error resetting conversation: {e}")
            return {
                "success": False,
                "message": f"Failed to reset conversation: {e}"
            }
            
    async def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available Gemini models"""
        models = [
            {
                "name": GeminiModel.UNSPECIFIED.value,
                "display_name": "Default Model",
                "description": "Default Gemini model (recommended)",
                "available": True
            },
            {
                "name": GeminiModel.GEMINI_2_5_FLASH.value,
                "display_name": "Gemini 2.5 Flash",
                "description": "Fast and efficient model for quick responses",
                "available": True
            },
            {
                "name": GeminiModel.GEMINI_2_5_PRO.value,
                "display_name": "Gemini 2.5 Pro",
                "description": "Advanced model with enhanced capabilities (may have usage limits)",
                "available": True
            },
            {
                "name": GeminiModel.GEMINI_2_0_FLASH.value,
                "display_name": "Gemini 2.0 Flash (Deprecated)",
                "description": "Previous generation Flash model",
                "available": True
            },
            {
                "name": GeminiModel.GEMINI_2_0_FLASH_THINKING.value,
                "display_name": "Gemini 2.0 Flash Thinking (Deprecated)",
                "description": "Previous generation thinking model",
                "available": True
            }
        ]
        
        return models
        
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on Gemini service"""
        health_status = {
            "service_initialized": self.is_initialized,
            "client_connected": self.client is not None,
            "chat_session_active": self.chat_session is not None,
            "initialization_time": self.initialization_time,
            "last_request_time": self.last_request_time,
            "request_count": self.request_count,
            "error_count": self.error_count,
            "success_rate": (
                (self.request_count - self.error_count) / self.request_count
                if self.request_count > 0 else 1.0
            )
        }
        
        return health_status
        
    async def cleanup(self):
        """Clean up resources"""
        try:
            if self.client:
                # Save metadata before closing
                await self._save_chat_metadata()
                
                # Close client if it has a close method
                if hasattr(self.client, 'close'):
                    await self.client.close()
                    
            logger.info("Gemini service cleaned up successfully")
            
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")


# Global instance
gemini_service = GeminiService()
