"""
Manages AI response buffering using a simple queue implementation.
"""
from typing import List, Optional
from collections import deque


class ResponseBuffer:
    """Simple queue-based buffer for managing AI responses."""
    
    def __init__(self):
        """Initialize an empty response buffer."""
        self._queue: deque[str] = deque()

    def add(self, response: str) -> None:
        """Add a response to the buffer."""
        self._queue.append(response)

    def get(self) -> Optional[str]:
        """Get the next response from the buffer if available."""
        return self._queue.popleft() if self._queue else None

    def clear(self) -> None:
        """Clear all responses from the buffer."""
        self._queue.clear()

    def is_empty(self) -> bool:
        """Check if the buffer is empty."""
        return len(self._queue) == 0

    def __len__(self) -> int:
        """Get the number of responses in the buffer."""
        return len(self._queue)
