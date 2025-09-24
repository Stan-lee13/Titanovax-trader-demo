"""
AsyncManager - Centralized async event loop management for TitanovaX
Fixes nested asyncio.run() issues and provides consistent async handling
"""

import asyncio
import threading
from typing import Optional, Coroutine, Any
import logging

logger = logging.getLogger(__name__)


class AsyncManager:
    """Manages async event loops across the entire TitanovaX system"""
    
    def __init__(self):
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()
        
    def start(self):
        """Start the async manager with a dedicated event loop"""
        with self._lock:
            if self._running:
                return
                
            self._loop = asyncio.new_event_loop()
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()
            self._running = True
            logger.info("AsyncManager started with dedicated event loop")
    
    def _run_loop(self):
        """Run the event loop in a separate thread"""
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_forever()
        except Exception as e:
            logger.error(f"Event loop error: {e}")
        finally:
            self._cleanup()
    
    def stop(self):
        """Stop the async manager and cleanup resources"""
        with self._lock:
            if not self._running:
                return
                
            self._running = False
            if self._loop and not self._loop.is_closed():
                self._loop.call_soon_threadsafe(self._loop.stop)
            
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=5)
                
            self._cleanup()
            logger.info("AsyncManager stopped")
    
    def _cleanup(self):
        """Cleanup event loop resources"""
        if self._loop and not self._loop.is_closed():
            try:
                # Cancel all pending tasks
                pending = asyncio.all_tasks(self._loop)
                for task in pending:
                    task.cancel()
                
                # Run until all tasks are cancelled
                if pending:
                    self._loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
                
                self._loop.close()
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")
        
        self._loop = None
        self._thread = None
    
    def ensure_task(self, coro: Coroutine) -> Any:
        """
        Ensure a coroutine runs properly regardless of current event loop state
        
        Args:
            coro: The coroutine to run
            
        Returns:
            The result of the coroutine
        """
        if not self._running:
            self.start()
        
        try:
            # Check if we're already in an event loop
            current_loop = asyncio.get_running_loop()
            # If we're in the same loop, create task directly
            if current_loop == self._loop:
                return asyncio.create_task(coro)
            else:
                # Schedule on our managed loop
                future = asyncio.run_coroutine_threadsafe(coro, self._loop)
                return future.result(timeout=30)
        except RuntimeError:
            # No event loop running, use our managed one
            future = asyncio.run_coroutine_threadsafe(coro, self._loop)
            return future.result(timeout=30)
    
    async def create_task(self, coro: Coroutine) -> asyncio.Task:
        """Create a task on the managed event loop"""
        if not self._running:
            self.start()
        
        if asyncio.get_event_loop() == self._loop:
            return asyncio.create_task(coro)
        else:
            # Schedule on managed loop
            return asyncio.run_coroutine_threadsafe(
                self._wrap_coro(coro), self._loop
            ).result()
    
    async def _wrap_coro(self, coro: Coroutine):
        """Wrap coroutine for threadsafe execution"""
        return await coro
    
    def run_coroutine(self, coro: Coroutine, timeout: float = 30) -> Any:
        """
        Run a coroutine synchronously, handling event loop conflicts
        
        Args:
            coro: The coroutine to run
            timeout: Maximum time to wait
            
        Returns:
            The result of the coroutine
        """
        if not self._running:
            self.start()
        
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)
    
    @property
    def loop(self) -> Optional[asyncio.AbstractEventLoop]:
        """Get the managed event loop"""
        return self._loop
    
    @property
    def is_running(self) -> bool:
        """Check if the manager is running"""
        return self._running


# Global instance for easy access
async_manager = AsyncManager()


def async_safe(coro_func):
    """
    Decorator to make async functions safe to call from any context
    
    Usage:
        @async_safe
        async def my_async_function():
            # Your async code here
            pass
    """
    def wrapper(*args, **kwargs):
        coro = coro_func(*args, **kwargs)
        return async_manager.ensure_task(coro)
    
    return wrapper


def initialize_async_manager():
    """Initialize the global async manager"""
    if not async_manager.is_running:
        async_manager.start()
    return async_manager


def cleanup_async_manager():
    """Cleanup the global async manager"""
    if async_manager.is_running:
        async_manager.stop()