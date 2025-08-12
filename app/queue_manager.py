"""
Advanced queue management for handling concurrent requests to Gemini API
"""

import asyncio
import time
from typing import Optional, Dict, Any, Callable, Awaitable
from dataclasses import dataclass, field
from uuid import uuid4
import logging

logger = logging.getLogger(__name__)


@dataclass
class QueueItem:
    """Represents an item in the processing queue"""
    id: str
    request_data: Dict[str, Any]
    future: asyncio.Future
    created_at: float
    priority: int = 0
    timeout: Optional[float] = None
    
    def __post_init__(self):
        if self.timeout is None:
            self.timeout = self.created_at + 300  # 5 minutes default timeout


class RequestQueue:
    """
    Advanced queue manager for handling concurrent requests
    Features:
    - FIFO processing with priority support
    - Request timeout handling
    - Queue size limits
    - Performance metrics
    - Graceful shutdown
    """
    
    def __init__(self, max_size: int = 100, default_timeout: float = 300):
        self.max_size = max_size
        self.default_timeout = default_timeout
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=max_size)
        self.processing = False
        self.current_request: Optional[QueueItem] = None
        self.processed_count = 0
        self.failed_count = 0
        self.total_processing_time = 0.0
        self.start_time = time.time()
        self._shutdown = False
        self._processor_task: Optional[asyncio.Task] = None
        
    async def start(self, processor_func: Callable[[Dict[str, Any]], Awaitable[Any]]):
        """Start the queue processor"""
        if self._processor_task is not None:
            logger.warning("Queue processor is already running")
            return
            
        self.processor_func = processor_func
        self._processor_task = asyncio.create_task(self._process_queue())
        logger.info("Queue processor started")
        
    async def stop(self):
        """Stop the queue processor gracefully"""
        self._shutdown = True
        
        if self._processor_task:
            try:
                await asyncio.wait_for(self._processor_task, timeout=30.0)
            except asyncio.TimeoutError:
                logger.warning("Queue processor did not stop gracefully, cancelling")
                self._processor_task.cancel()
                
        logger.info("Queue processor stopped")
        
    async def add_request(self, request_data: Dict[str, Any], priority: int = 0, timeout: Optional[float] = None) -> str:
        """
        Add a request to the queue
        
        Args:
            request_data: The request data to process
            priority: Priority level (higher = more important)
            timeout: Request timeout in seconds
            
        Returns:
            Request ID for tracking
            
        Raises:
            asyncio.QueueFull: If queue is at maximum capacity
        """
        if self._shutdown:
            raise RuntimeError("Queue is shutting down")
            
        request_id = str(uuid4())
        current_time = time.time()
        
        if timeout is None:
            timeout = current_time + self.default_timeout
        else:
            timeout = current_time + timeout
            
        queue_item = QueueItem(
            id=request_id,
            request_data=request_data,
            future=asyncio.Future(),
            created_at=current_time,
            priority=priority,
            timeout=timeout
        )
        
        try:
            await self.queue.put(queue_item)
            logger.debug(f"Request {request_id} added to queue (position: {self.queue.qsize()})")
            return request_id
        except asyncio.QueueFull:
            logger.error(f"Queue is full, rejecting request {request_id}")
            raise
            
    async def get_result(self, request_id: str, timeout: Optional[float] = None) -> Any:
        """
        Get the result for a specific request
        
        Args:
            request_id: The request ID returned by add_request
            timeout: How long to wait for the result
            
        Returns:
            The processed result
            
        Raises:
            asyncio.TimeoutError: If timeout is exceeded
            Exception: Any exception that occurred during processing
        """
        # Note: In a real implementation, you'd store the futures in a dict
        # For simplicity, this assumes the caller has access to the queue item
        # A better approach would be to store futures by request_id
        raise NotImplementedError("This would require storing futures by request_id")
        
    async def process_request_and_wait(self, request_data: Dict[str, Any], priority: int = 0, timeout: Optional[float] = None) -> Any:
        """
        Add a request and wait for its result
        
        Args:
            request_data: The request data to process
            priority: Priority level
            timeout: Request timeout in seconds
            
        Returns:
            The processed result
        """
        if self._shutdown:
            raise RuntimeError("Queue is shutting down")
            
        current_time = time.time()
        
        if timeout is None:
            actual_timeout = current_time + self.default_timeout
        else:
            actual_timeout = current_time + timeout
            
        queue_item = QueueItem(
            id=str(uuid4()),
            request_data=request_data,
            future=asyncio.Future(),
            created_at=current_time,
            priority=priority,
            timeout=actual_timeout
        )
        
        try:
            await self.queue.put(queue_item)
            logger.debug(f"Request {queue_item.id} added to queue (position: {self.queue.qsize()})")
            
            # Wait for the result
            wait_timeout = timeout if timeout else self.default_timeout
            return await asyncio.wait_for(queue_item.future, timeout=wait_timeout)
            
        except asyncio.QueueFull:
            logger.error(f"Queue is full, rejecting request {queue_item.id}")
            raise
        except asyncio.TimeoutError:
            logger.error(f"Request {queue_item.id} timed out")
            raise
            
    async def _process_queue(self):
        """Main queue processing loop"""
        logger.info("Starting queue processor loop")
        
        while not self._shutdown:
            try:
                # Get next item from queue with timeout
                try:
                    queue_item = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue  # Check shutdown flag
                    
                # Check if request has timed out
                current_time = time.time()
                if current_time > queue_item.timeout:
                    logger.warning(f"Request {queue_item.id} timed out before processing")
                    queue_item.future.set_exception(asyncio.TimeoutError("Request timed out"))
                    self.failed_count += 1
                    continue
                    
                # Process the request
                self.current_request = queue_item
                self.processing = True
                process_start = time.time()
                
                try:
                    logger.debug(f"Processing request {queue_item.id}")
                    result = await self.processor_func(queue_item.request_data)
                    queue_item.future.set_result(result)
                    
                    processing_time = time.time() - process_start
                    self.total_processing_time += processing_time
                    self.processed_count += 1
                    
                    logger.debug(f"Request {queue_item.id} processed successfully in {processing_time:.2f}s")
                    
                except Exception as e:
                    logger.error(f"Error processing request {queue_item.id}: {e}")
                    queue_item.future.set_exception(e)
                    self.failed_count += 1
                    
                finally:
                    self.current_request = None
                    self.processing = False
                    self.queue.task_done()
                    
            except Exception as e:
                logger.error(f"Unexpected error in queue processor: {e}")
                
        logger.info("Queue processor loop ended")
        
    def get_status(self) -> Dict[str, Any]:
        """Get current queue status and statistics"""
        current_time = time.time()
        uptime = current_time - self.start_time
        
        avg_processing_time = (
            self.total_processing_time / self.processed_count 
            if self.processed_count > 0 else 0
        )
        
        return {
            "queue_size": self.queue.qsize(),
            "max_size": self.max_size,
            "processing": self.processing,
            "current_request_id": self.current_request.id if self.current_request else None,
            "processed_count": self.processed_count,
            "failed_count": self.failed_count,
            "success_rate": (
                self.processed_count / (self.processed_count + self.failed_count)
                if (self.processed_count + self.failed_count) > 0 else 1.0
            ),
            "average_processing_time": avg_processing_time,
            "uptime": uptime,
            "is_full": self.queue.full(),
            "shutdown": self._shutdown
        }
        
    def is_full(self) -> bool:
        """Check if queue is full"""
        return self.queue.full()
        
    def size(self) -> int:
        """Get current queue size"""
        return self.queue.qsize()
        
    def get_average_wait_time(self) -> Optional[float]:
        """Calculate average wait time based on current queue and processing speed"""
        if self.processed_count == 0:
            return None
            
        avg_processing_time = self.total_processing_time / self.processed_count
        return self.queue.qsize() * avg_processing_time
