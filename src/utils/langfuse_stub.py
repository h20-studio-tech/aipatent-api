import asyncio
from functools import wraps

# Simple no-op observe decorator used during testing when langfuse.decorators
# is unavailable.
def observe(func=None, *, name=None):
    def decorator(f):
        if asyncio.iscoroutinefunction(f):
            @wraps(f)
            async def wrapped(*args, **kwargs):
                return await f(*args, **kwargs)
            return wrapped
        else:
            @wraps(f)
            def wrapped(*args, **kwargs):
                return f(*args, **kwargs)
            return wrapped
    if func and callable(func):
        return decorator(func)
    return decorator
