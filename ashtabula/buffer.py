"""
Manages AI session state including response buffering, partial transcriptions, 
and predicted sentences with timestamps.
"""
from typing import List, Optional, Dict, Any, Tuple
from collections import deque
import time
from dataclasses import dataclass


@dataclass
class TranscriptionItem:
    """Item containing a transcription text and metadata."""
    text: str
    timestamp: float  # Unix timestamp
    is_final: bool = False


@dataclass
class PredictionItem:
    """Item containing a predicted sentence and metadata."""
    text: str
    timestamp: float  # Unix timestamp
    source_transcription: str  # The partial transcription that led to this prediction
    embedding: Optional[List[float]] = None  # Optional embedding vector


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


class SessionBuffer:
    """
    Manages session state including partial transcriptions, predicted sentences,
    and embeddings.
    
    This buffer is designed to maintain the state of an ongoing conversation,
    tracking both partial and final transcriptions, along with predicted completions
    and their embeddings.
    """
    
    def __init__(self, max_transcription_history: int = 10, max_prediction_history: int = 5):
        """
        Initialize a session buffer.
        
        Args:
            max_transcription_history: Maximum number of transcription items to keep
            max_prediction_history: Maximum number of prediction items to keep
        """
        self.transcriptions: deque[TranscriptionItem] = deque(maxlen=max_transcription_history)
        self.predictions: deque[PredictionItem] = deque(maxlen=max_prediction_history)
        self.current_partial_text: str = ""
        self.session_id: str = ""
        self.last_activity_time: float = time.time()
    
    def add_transcription(self, text: str, is_final: bool = False) -> None:
        """
        Add a transcription to the buffer.
        
        Args:
            text: The transcribed text
            is_final: Whether this is a final transcription
        """
        item = TranscriptionItem(
            text=text,
            timestamp=time.time(),
            is_final=is_final
        )
        
        self.transcriptions.append(item)
        self.current_partial_text = text if not is_final else ""
        self.last_activity_time = item.timestamp
    
    def add_prediction(self, text: str, source_transcription: str, 
                       embedding: Optional[List[float]] = None) -> None:
        """
        Add a prediction to the buffer.
        
        Args:
            text: The predicted text
            source_transcription: The partial transcription that led to this prediction
            embedding: Optional embedding vector for the prediction
        """
        item = PredictionItem(
            text=text,
            timestamp=time.time(),
            source_transcription=source_transcription,
            embedding=embedding
        )
        
        self.predictions.append(item)
        self.last_activity_time = item.timestamp
    
    def get_last_prediction(self) -> Optional[PredictionItem]:
        """Get the most recent prediction if available."""
        return self.predictions[-1] if self.predictions else None
    
    def get_prediction_by_similarity(self, text: str, similarity_func) -> Optional[PredictionItem]:
        """
        Find the most similar prediction to the given text.
        
        Args:
            text: The text to compare against predictions
            similarity_func: A function that takes two strings and returns a similarity score
            
        Returns:
            The most similar prediction, or None if no predictions exist
        """
        if not self.predictions:
            return None
        
        # Find the prediction with the highest similarity
        best_match = max(
            self.predictions,
            key=lambda pred: similarity_func(text, pred.text),
            default=None
        )
        
        return best_match
    
    def get_transcription_history(self, limit: Optional[int] = None) -> List[TranscriptionItem]:
        """
        Get the transcription history, optionally limited to the most recent items.
        
        Args:
            limit: Maximum number of items to return, or None for all
            
        Returns:
            List of transcription items, newest first
        """
        history = list(self.transcriptions)
        if limit:
            history = history[-limit:]
        return history
    
    def get_prediction_history(self, limit: Optional[int] = None) -> List[PredictionItem]:
        """
        Get the prediction history, optionally limited to the most recent items.
        
        Args:
            limit: Maximum number of items to return, or None for all
            
        Returns:
            List of prediction items, newest first
        """
        history = list(self.predictions)
        if limit:
            history = history[-limit:]
        return history
    
    def clear(self) -> None:
        """Clear all session data."""
        self.transcriptions.clear()
        self.predictions.clear()
        self.current_partial_text = ""
        self.last_activity_time = time.time()
    
    def is_stale(self, timeout_seconds: float) -> bool:
        """
        Check if the session has been inactive for longer than the timeout.
        
        Args:
            timeout_seconds: Seconds of inactivity to consider the session stale
            
        Returns:
            True if the session is stale, False otherwise
        """
        return (time.time() - self.last_activity_time) > timeout_seconds
