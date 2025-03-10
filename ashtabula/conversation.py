"""
Handles the full AI conversation pipeline, integrating STT, LLM, buffering, and TTS.

This module manages the conversation state, processes audio streams through STT,
generates responses via LLM, and orchestrates the entire conversation flow.
"""
from typing import Optional, AsyncGenerator, Dict, Any, List, Set, Callable, Tuple
from dataclasses import dataclass
import asyncio
import time
import uuid
from difflib import SequenceMatcher
import logging

from .llm import LLMProvider
from .stt import STTProvider
from .buffer import ResponseBuffer, SessionBuffer, TranscriptionItem, PredictionItem

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class ConversationConfig:
    """Configuration for conversation parameters."""
    prediction_threshold: float = 0.85  # Minimum similarity for prediction match
    silence_timeout: float = 3.0  # Seconds of silence before completing sentence
    max_history_length: int = 5  # Number of previous utterances to keep
    session_timeout: float = 300.0  # Seconds before a session is considered stale (5 minutes)
    max_transcription_history: int = 20  # Max number of transcription items to keep
    max_prediction_history: int = 10  # Max number of prediction items to keep
    embed_predictions: bool = True  # Whether to generate embeddings for predictions


class ConversationManager:
    """Manages real-time AI conversations with streaming STT and response prediction."""
    
    def __init__(self,
                 llm_provider: LLMProvider,
                 stt_provider: STTProvider,
                 response_buffer: ResponseBuffer,
                 config: Optional[ConversationConfig] = None):
        """
        Initialize the conversation manager.
        
        Args:
            llm_provider: Provider for LLM text generation
            stt_provider: Provider for speech-to-text conversion
            response_buffer: Buffer for managing responses
            config: Optional configuration parameters
        """
        self.llm = llm_provider
        self.stt = stt_provider
        self.response_buffer = response_buffer
        self.config = config or ConversationConfig()
        
        # Session management
        self.sessions: Dict[str, SessionBuffer] = {}
        self.active_session_id: Optional[str] = None
        
        # Legacy state (kept for backwards compatibility)
        self.conversation_history: list[str] = []
        self.current_sentence: list[str] = []
        self.last_speech_time: float = 0
        self.predicted_text: str = ""
        self.is_interrupted: bool = False
        
        # Initialize but don't start session cleanup task
        self._cleanup_task = None
        
    def start_cleanup_task(self):
        """Start the session cleanup task if it's not already running."""
        if not self._cleanup_task:
            try:
                self._cleanup_task = asyncio.create_task(self._cleanup_stale_sessions())
            except RuntimeError:
                # No running event loop, skip task creation
                pass
    
    async def _cleanup_stale_sessions(self) -> None:
        """Periodically clean up stale sessions."""
        while True:
            try:
                # Check for stale sessions every minute
                await asyncio.sleep(60)
                
                stale_sessions = []
                for session_id, session in self.sessions.items():
                    if session.is_stale(self.config.session_timeout):
                        stale_sessions.append(session_id)
                
                # Remove stale sessions
                for session_id in stale_sessions:
                    logger.info(f"Removing stale session: {session_id}")
                    del self.sessions[session_id]
                
            except asyncio.CancelledError:
                # Task was cancelled, exit
                break
            except Exception as e:
                logger.error(f"Error in session cleanup: {e}")
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity ratio between two texts."""
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

    def _update_history(self, text: str) -> None:
        """Update conversation history, maintaining max length."""
        self.conversation_history.append(text)
        if len(self.conversation_history) > self.config.max_history_length:
            self.conversation_history.pop(0)
    
    def _get_or_create_session(self, session_id: Optional[str] = None) -> Tuple[str, SessionBuffer]:
        """
        Get an existing session or create a new one.
        
        Args:
            session_id: Optional existing session ID
            
        Returns:
            Tuple of (session_id, session_buffer)
        """
        if session_id and session_id in self.sessions:
            return session_id, self.sessions[session_id]
        
        # Create new session
        new_id = session_id or str(uuid.uuid4())
        new_session = SessionBuffer(
            max_transcription_history=self.config.max_transcription_history,
            max_prediction_history=self.config.max_prediction_history
        )
        new_session.session_id = new_id
        self.sessions[new_id] = new_session
        
        return new_id, new_session

    async def _handle_silence(self) -> bool:
        """Check if silence duration exceeds timeout."""
        if not self.current_sentence:
            return False
        
        silence_duration = time.time() - self.last_speech_time
        return silence_duration >= self.config.silence_timeout
    
    def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate an embedding vector for the given text.
        
        Args:
            text: The text to embed
            
        Returns:
            Embedding vector, or None if embedding is not supported
        """
        # In a real implementation, this would call an embedding model
        # For now, we'll just use a simple mock
        if not self.config.embed_predictions:
            return None
        
        # Mock embedding: just convert character positions to floats
        # In a real system, this would use a proper embedding model
        mock_embedding = [ord(c) / 255.0 for c in text[:20]]
        return mock_embedding

    async def stream_audio(self, audio_source: str, session_id: Optional[str] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process streaming audio input and generate AI responses.
        
        Args:
            audio_source: Source of the audio stream
            session_id: Optional session ID for continuing an existing conversation
            
        Yields:
            Dict containing:
                - 'text': Transcribed/predicted text
                - 'is_final': Whether this is a final transcription
                - 'response': AI response if available
                - 'confidence': Confidence score for predictions
        """
        # Get or create session
        session_id, session = self._get_or_create_session(session_id)
        self.active_session_id = session_id
        
        try:
            async for transcription in self.stt.stream_audio(audio_source):
                # Reset interrupt flag for new input
                self.is_interrupted = False
                
                # Extract text and completion status from transcription
                current_text = (transcription.text if hasattr(transcription, 'text')
                              else transcription['text'] if isinstance(transcription, dict)
                              else str(transcription))
                
                is_final = (transcription.get('is_final', False) if isinstance(transcription, dict)
                          else getattr(transcription, 'is_final', False))
                
                # Update timing for silence detection
                self.last_speech_time = time.time()
                
                # Update session state
                session.add_transcription(current_text, is_final=is_final)
                
                # Legacy state (for backwards compatibility)
                self.current_sentence.append(current_text)
                partial_sentence = " ".join(self.current_sentence)
                
                # Get prediction for sentence completion
                self.predicted_text = self.llm.predict(partial_sentence)
                
                # Add prediction to session state
                embedding = self._generate_embedding(self.predicted_text) if is_final else None
                session.add_prediction(
                    text=self.predicted_text,
                    source_transcription=current_text,
                    embedding=embedding
                )
                
                # Check for sentence completion (either from STT or silence)
                is_sentence_complete = is_final or await self._handle_silence()
                
                if is_sentence_complete:
                    # Compare actual vs predicted text
                    actual_text = partial_sentence
                    similarity = self._calculate_similarity(actual_text, self.predicted_text)
                    
                    # Find best matching prediction
                    best_prediction = session.get_prediction_by_similarity(
                        actual_text, self._calculate_similarity
                    )
                    
                    # Check if we have a good match
                    if best_prediction and self._calculate_similarity(
                        actual_text, best_prediction.text
                    ) >= self.config.prediction_threshold:
                        logger.info(f"Using prediction match: similarity={similarity:.2f}")
                        # In a real implementation, we might use a precomputed response
                        # based on the prediction, but for now we'll generate a new one
                    
                    # Generate response
                    context = " ".join(self.conversation_history[-3:])  # Use recent history
                    response = self.llm.generate(
                        f"{context}\nUser: {actual_text}\nAssistant:",
                        temperature=0.7
                    )
                    
                    # Buffer the response
                    self._buffer_response(response)
                    
                    # Update conversation history
                    self._update_history(actual_text)
                    self._update_history(response)
                    
                    # Clear current sentence buffer
                    self.current_sentence = []
                    
                    # Get response from buffer
                    buffered_response = self._get_buffered_response()
                    
                    yield {
                        'text': actual_text,
                        'is_final': True,
                        'response': buffered_response,
                        'confidence': similarity,
                        'session_id': session_id
                    }
                else:
                    # Yield intermediate results
                    yield {
                        'text': partial_sentence,
                        'is_final': False,
                        'predicted_completion': self.predicted_text,
                        'confidence': None,
                        'session_id': session_id
                    }

        except Exception as e:
            # Log error and yield error state
            logger.error(f"Error in conversation stream: {str(e)}")
            yield {
                'text': None,
                'is_final': True,
                'error': str(e),
                'session_id': session_id
            }

    def handle_interrupt(self) -> None:
        """Handle user interruption of AI response."""
        self.is_interrupted = True
        self.response_buffer.clear()  # Clear pending responses
        self.current_sentence = []  # Reset current sentence
        self.predicted_text = ""  # Clear prediction
        
        # Clear session state if we have an active session
        if self.active_session_id and self.active_session_id in self.sessions:
            self.sessions[self.active_session_id].clear()

    def _buffer_response(self, response: str) -> None:
        """Add a response to the buffer."""
        self.response_buffer.add(response)

    def _get_buffered_response(self) -> Optional[str]:
        """Get the next response from the buffer if available."""
        return self.response_buffer.get()
    
    def get_session(self, session_id: str) -> Optional[SessionBuffer]:
        """Get a session by ID."""
        return self.sessions.get(session_id)
    
    def reset_session(self, session_id: Optional[str] = None) -> None:
        """
        Reset a session or the active session.
        
        Args:
            session_id: The session ID to reset, or None for the active session
        """
        target_id = session_id or self.active_session_id
        if target_id and target_id in self.sessions:
            self.sessions[target_id].clear()
    
    def close(self) -> None:
        """Clean up resources when the manager is no longer needed."""
        if hasattr(self, '_cleanup_task') and self._cleanup_task:
            self._cleanup_task.cancel()
    
    async def wait_closed(self) -> None:
        """Wait for cleanup to complete."""
        if hasattr(self, '_cleanup_task') and self._cleanup_task:
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
