"""Performance tracking utilities"""
import functools
import time
import logging

logger = logging.getLogger(__name__)

def performance_tracked(name_or_func=None):
    """Decorator to track function performance"""
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.time()
            result = await func(*args, **kwargs)
            duration = time.time() - start
            logger.debug(f"{func.__name__} took {duration:.3f}s")
            return result

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start
            logger.debug(f"{func.__name__} took {duration:.3f}s")
            return result

        import inspect
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    # Allow both @performance_tracked and @performance_tracked("name")
    if callable(name_or_func):
        return decorator(name_or_func)
    return decorator
