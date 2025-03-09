"""
Handles the full AI conversation pipeline, integrating STT, LLM, buffering, and TTS.
"""
from typing import Optional, AsyncGenerator, Dict, Any
from dataclasses import dataclass
import asyncio
import time
from difflib import SequenceMatcher

from .llm import LLMProvider
from .stt import STTProvider
from .buffer import ResponseBuffer


@dataclass
class ConversationConfig:
    """Configuration for conversation parameters."""
    prediction_threshold: float = 0.85  # Minimum similarity for prediction match
    silence_timeout: float = 3.0  # Seconds of silence before completing sentence
    max_history_length: int = 5  # Number of previous utterances to keep


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
        
        self.conversation_history: list[str] = []
        self.current_sentence: list[str] = []
        self.last_speech_time: float = 0
        self.predicted_text: str = ""
        self.is_interrupted: bool = False

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity ratio between two texts."""
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

    def _update_history(self, text: str) -> None:
        """Update conversation history, maintaining max length."""
        self.conversation_history.append(text)
        if len(self.conversation_history) > self.config.max_history_length:
            self.conversation_history.pop(0)

    async def _handle_silence(self) -> bool:
        """Check if silence duration exceeds timeout."""
        if not self.current_sentence:
            return False
        
        silence_duration = time.time() - self.last_speech_time
        return silence_duration >= self.config.silence_timeout

    async def stream_audio(self, audio_source: str) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process streaming audio input and generate AI responses.
        
        Args:
            audio_source: Source of the audio stream
            
        Yields:
            Dict containing:
                - 'text': Transcribed/predicted text
                - 'is_final': Whether this is a final transcription
                - 'response': AI response if available
                - 'confidence': Confidence score for predictions
        """
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
                
                # Add to current sentence buffer
                self.current_sentence.append(current_text)
                partial_sentence = " ".join(self.current_sentence)
                
                # Get prediction for sentence completion
                self.predicted_text = self.llm.predict(partial_sentence)
                
                # Check for sentence completion (either from STT or silence)
                is_sentence_complete = is_final or await self._handle_silence()
                
                if is_sentence_complete:
                    # Compare actual vs predicted text
                    actual_text = partial_sentence
                    similarity = self._calculate_similarity(actual_text, self.predicted_text)
                    
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
                        'confidence': similarity
                    }
                else:
                    # Yield intermediate results
                    yield {
                        'text': partial_sentence,
                        'is_final': False,
                        'predicted_completion': self.predicted_text,
                        'confidence': None
                    }

        except Exception as e:
            # Log error and yield error state
            print(f"Error in conversation stream: {str(e)}")
            yield {
                'text': None,
                'is_final': True,
                'error': str(e)
            }

    def handle_interrupt(self) -> None:
        """Handle user interruption of AI response."""
        self.is_interrupted = True
        self.response_buffer.clear()  # Clear pending responses
        self.current_sentence = []  # Reset current sentence
        self.predicted_text = ""  # Clear prediction

    def _buffer_response(self, response: str) -> None:
        """Add a response to the buffer."""
        self.response_buffer.add(response)

    def _get_buffered_response(self) -> Optional[str]:
        """Get the next response from the buffer if available."""
        return self.response_buffer.get()
