"""
Conversation State Machine for Ashtabula

This module implements the Ashtabula conversation flow as a Finite State Machine (FSM),
handling the transitions between states and managing the prediction-based response system.
"""

from enum import Enum
import logging
import time
from typing import Optional, Dict, Any, List, Callable, Union, cast
from transitions import Machine  # type: ignore
import asyncio

from .llm import LLMProvider
from .stt import STTProvider
from .buffer import ResponseBuffer, SessionBuffer

# Configure logging
logger = logging.getLogger(__name__)


class ConversationState(Enum):
    """States for the conversation state machine."""
    IDLE = 'idle'                         # No active conversation
    LISTENING = 'listening'               # Receiving audio but no speech detected
    SPEECH_ACTIVE = 'speech_active'       # Active speech being transcribed
    PROCESSING_UTTERANCE = 'processing'   # Processing complete utterance
    GENERATING_RESPONSE = 'generating'    # Generating response with LLM
    SPEAKING = 'speaking'                 # Delivering response to user
    INTERRUPTED = 'interrupted'           # User interrupted the flow


class ConversationFSM:
    """
    Finite State Machine implementation of the Ashtabula conversation flow.
    
    This class manages the transitions between conversation states and implements
    the core prediction-based response generation logic.
    """
    
    def __init__(
        self,
        llm_provider: LLMProvider,
        stt_provider: STTProvider,
        response_buffer: ResponseBuffer,
        session_buffer: SessionBuffer,
        prediction_threshold: float = 0.85,
        silence_timeout: float = 3.0,
        max_history_length: int = 5
    ):
        """
        Initialize the conversation state machine.
        
        Args:
            llm_provider: The LLM provider for generating responses
            stt_provider: The STT provider for transcribing speech
            response_buffer: Buffer for storing and retrieving responses
            session_buffer: Buffer for session state including transcriptions and predictions
            prediction_threshold: Threshold for considering predictions a match
            silence_timeout: Seconds of silence to trigger utterance completion
            max_history_length: Maximum conversation history entries to maintain
        """
        self.llm = llm_provider
        self.stt = stt_provider
        self.response_buffer = response_buffer
        self.session_buffer = session_buffer
        self.prediction_threshold = prediction_threshold
        self.silence_timeout = silence_timeout
        self.max_history_length = max_history_length
        
        # Conversation state data
        self.session_id: Optional[str] = None
        self.conversation_history: List[str] = []
        self.current_partial_text: str = ""
        self.current_prediction: str = ""
        self.final_transcription: str = ""
        self.current_response: str = ""
        
        # Timing data
        self.last_speech_time: float = 0
        self.speech_start_time: Optional[float] = None
        self.silence_start_time: Optional[float] = None
        
        # Create state machine with states and transitions
        states = [state.value for state in ConversationState]  # Use string values for states
        self.machine = Machine(
            model=self,
            states=states,
            initial=ConversationState.IDLE.value,
            auto_transitions=False,
        )
        self._state = ConversationState.IDLE.value  # Track current state
        
        # Define transitions
        self._define_transitions()
    
    def _define_transitions(self) -> None:
        """Define the state machine transitions."""
        # From IDLE state
        self.machine.add_transition(
            trigger='start_listening',
            source=ConversationState.IDLE.value,
            dest=ConversationState.LISTENING.value,
            before='_on_start_listening'
        )
        
        # From LISTENING state
        self.machine.add_transition(
            trigger='detect_speech',
            source=ConversationState.LISTENING.value,
            dest=ConversationState.SPEECH_ACTIVE.value,
            before='_on_detect_speech'
        )
        
        self.machine.add_transition(
            trigger='timeout',
            source=ConversationState.LISTENING.value,
            dest=ConversationState.IDLE.value,
            before='_on_listening_timeout'
        )
        
        # From SPEECH_ACTIVE state
        self.machine.add_transition(
            trigger='detect_silence',
            source=ConversationState.SPEECH_ACTIVE.value,
            dest=ConversationState.PROCESSING_UTTERANCE.value,
            before='_on_detect_silence',
            conditions=['_is_silence_duration_exceeded']
        )
        
        self.machine.add_transition(
            trigger='finalize_transcription',
            source=ConversationState.SPEECH_ACTIVE.value,
            dest=ConversationState.PROCESSING_UTTERANCE.value,
            before='_on_finalize_transcription'
        )
        
        # From PROCESSING_UTTERANCE state
        self.machine.add_transition(
            trigger='generate_response',
            source=ConversationState.PROCESSING_UTTERANCE.value,
            dest=ConversationState.GENERATING_RESPONSE.value,
            before='_on_generate_response'
        )
        
        # From GENERATING_RESPONSE state
        self.machine.add_transition(
            trigger='deliver_response',
            source=ConversationState.GENERATING_RESPONSE.value,
            dest=ConversationState.SPEAKING.value,
            before='_on_deliver_response'
        )
        
        # From SPEAKING state
        self.machine.add_transition(
            trigger='complete_speaking',
            source=ConversationState.SPEAKING.value,
            dest=ConversationState.LISTENING.value,
            before='_on_complete_speaking'
        )
        
        # Interrupt can happen from multiple states
        self.machine.add_transition(
            trigger='interrupt',
            source=[
                ConversationState.SPEECH_ACTIVE.value,
                ConversationState.PROCESSING_UTTERANCE.value,
                ConversationState.GENERATING_RESPONSE.value,
                ConversationState.SPEAKING.value
            ],
            dest=ConversationState.INTERRUPTED.value,
            before='_on_interrupt'
        )
        
        # From INTERRUPTED state
        self.machine.add_transition(
            trigger='resume',
            source=ConversationState.INTERRUPTED.value,
            dest=ConversationState.LISTENING.value,
            before='_on_resume'
        )
    
    # State transition handlers
    
    def _on_start_listening(self) -> None:
        """Handle transition to LISTENING state."""
        logger.debug("Starting to listen")
        self.silence_start_time = time.time()
        self.speech_start_time = None
        self._state = ConversationState.LISTENING.value
    
    def _on_detect_speech(self) -> None:
        """Handle transition to SPEECH_ACTIVE state."""
        logger.debug("Speech detected")
        self.speech_start_time = time.time()
        self.silence_start_time = None
        self.current_partial_text = ""
        self.current_prediction = ""
        self._state = ConversationState.SPEECH_ACTIVE.value
    
    def _on_detect_silence(self) -> None:
        """Handle transition to PROCESSING_UTTERANCE due to silence."""
        logger.debug("Silence detected, finalizing utterance")
        self.final_transcription = self.current_partial_text
        
        # Add final transcription to session
        self.session_buffer.add_transcription(
            self.final_transcription, 
            is_final=True
        )
        self._state = ConversationState.PROCESSING_UTTERANCE.value
    
    def _on_finalize_transcription(self) -> None:
        """Handle transition to PROCESSING_UTTERANCE due to final transcription."""
        logger.debug("Finalizing transcription")
        # This is called when STT indicates the transcription is final
        # The final transcription is already stored in self.final_transcription
        self._state = ConversationState.PROCESSING_UTTERANCE.value
    
    def _on_generate_response(self) -> None:
        """Handle transition to GENERATING_RESPONSE state."""
        logger.debug("Generating response")
        # This is where we implement the core prediction matching logic
        
        # Step 1: Try to find a matching prediction
        best_prediction = self.session_buffer.get_prediction_by_similarity(
            self.final_transcription,
            self._calculate_similarity
        )
        
        # Step 2: Check if prediction match is good enough
        if best_prediction and self._calculate_similarity(
            self.final_transcription, best_prediction.text
        ) >= self.prediction_threshold:
            logger.info(f"Using prediction match: similarity={self._calculate_similarity(self.final_transcription, best_prediction.text):.2f}")
            # In a real implementation, we would have pre-generated a response
            # when creating the prediction, and could reuse it here
            # For now, we'll just generate a new response
        
        # Step 3: Generate response (whether from prediction or new)
        context = " ".join(self.conversation_history[-3:])
        self.current_response = self.llm.generate(
            f"{context}\nUser: {self.final_transcription}\nAssistant:",
            temperature=0.7
        )
        
        # Step 4: Add to history
        self._update_history(self.final_transcription)
        self._update_history(self.current_response)
        self._state = ConversationState.GENERATING_RESPONSE.value
    
    def _on_deliver_response(self) -> None:
        """Handle transition to SPEAKING state."""
        logger.debug("Delivering response")
        # In a full implementation, this would trigger TTS
        self.response_buffer.add(self.current_response)
        self._state = ConversationState.SPEAKING.value
    
    def _on_complete_speaking(self) -> None:
        """Handle transition back to LISTENING after speaking."""
        logger.debug("Completed speaking, ready for next input")
        # Reset for next utterance
        self.current_partial_text = ""
        self.current_prediction = ""
        self.final_transcription = ""
        self.current_response = ""
        self.silence_start_time = time.time()
        self._state = ConversationState.LISTENING.value
    
    def _on_interrupt(self) -> None:
        """Handle interruption."""
        logger.debug("Handling interruption")
        # Clear buffers and state
        self.response_buffer.clear()
        self.current_partial_text = ""
        self.current_prediction = ""
        self.final_transcription = ""
        self.current_response = ""
        self._state = ConversationState.INTERRUPTED.value
    
    def _on_resume(self) -> None:
        """Handle resuming after interruption."""
        logger.debug("Resuming after interruption")
        self.silence_start_time = time.time()
        self._state = ConversationState.LISTENING.value
    
    def _on_listening_timeout(self) -> None:
        """Handle timeout in LISTENING state."""
        logger.debug("Listening timeout")
        # Optionally reset any state here
        self._state = ConversationState.IDLE.value
    
    # Helper methods
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate the similarity between two texts."""
        from difflib import SequenceMatcher
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    
    def _is_silence_duration_exceeded(self) -> bool:
        """Check if silence duration exceeds the configured timeout."""
        if not self.silence_start_time:
            return False
        
        return (time.time() - self.silence_start_time) >= self.silence_timeout
    
    def _update_history(self, text: str) -> None:
        """Update conversation history."""
        self.conversation_history.append(text)
        while len(self.conversation_history) > self.max_history_length:
            self.conversation_history.pop(0)
    
    # Public API methods
    
    async def process_transcription(self, text: str, is_final: bool) -> Dict[str, Any]:
        """
        Process a transcription update.
        
        Args:
            text: The transcribed text
            is_final: Whether this is a final transcription
            
        Returns:
            A dict with processing results
        """
        # Update state based on transcription
        self.current_partial_text = text
        
        # Update last speech time
        self.last_speech_time = time.time()
        self.silence_start_time = None
        
        # If we're in LISTENING state, transition to SPEECH_ACTIVE
        if self._state == ConversationState.LISTENING.value:
            self.detect_speech()
        
        # If this is a final transcription, finalize it
        if is_final and self._state == ConversationState.SPEECH_ACTIVE.value:
            self.final_transcription = text
            self.finalize_transcription()
            self.generate_response()
            self.deliver_response()
            
            # Get response from buffer
            response = self.response_buffer.get()
            
            # Return final result
            return {
                'text': self.final_transcription,
                'is_final': True,
                'response': response,
                'state': self._state
            }
        
        # If we're in SPEECH_ACTIVE, generate a prediction
        if self._state == ConversationState.SPEECH_ACTIVE.value:
            # Generate prediction
            self.current_prediction = self.llm.predict(text)
            
            # Add prediction to session
            embedding = None  # We only generate embeddings for final utterances
            self.session_buffer.add_prediction(
                text=self.current_prediction,
                source_transcription=text,
                embedding=embedding
            )
            
            # Return intermediate result
            return {
                'text': text,
                'is_final': False,
                'predicted_completion': self.current_prediction,
                'state': self._state
            }
        
        # Default response
        return {
            'text': text,
            'is_final': False,
            'state': self._state
        }
    
    async def check_silence(self) -> Optional[Dict[str, Any]]:
        """
        Check for silence and trigger state transitions if needed.
        
        Returns:
            A dict with processing results if a transition occurred, or None
        """
        # If we're in SPEECH_ACTIVE and silence is detected
        if (self._state == ConversationState.SPEECH_ACTIVE.value and 
            self.silence_start_time is None and 
            (time.time() - self.last_speech_time) >= self.silence_timeout):
            
            # Mark the start of silence
            self.silence_start_time = time.time()
            
            # Transition to processing state - manual state changes to handle the flow
            self.detect_silence()
            # We need to manually set these states since direct transitions may not be valid
            self._state = ConversationState.PROCESSING_UTTERANCE.value
            self._on_generate_response()
            self._state = ConversationState.GENERATING_RESPONSE.value
            self._on_deliver_response()
            
            # Get response from buffer
            response = self.response_buffer.get()
            
            # Return final result
            return {
                'text': self.final_transcription,
                'is_final': True,
                'response': response,
                'state': self._state,
                'trigger': 'silence'
            }
        
        return None
    
    def handle_interrupt(self) -> None:
        """Handle user interruption."""
        # Trigger interrupt transition
        if self._state != ConversationState.IDLE.value and self._state != ConversationState.LISTENING.value:
            self.interrupt()
            self.resume()
    
    def get_state(self) -> str:
        """Get the current state."""
        return self._state