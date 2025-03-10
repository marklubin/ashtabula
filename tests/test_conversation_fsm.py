"""Tests for the ConversationFSM module."""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock

from ashtabula.conversation_fsm import ConversationFSM, ConversationState


class TestConversationFSM:
    """Test cases for the Conversation Finite State Machine."""

    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies for testing."""
        mock_llm = Mock()
        mock_llm.predict = Mock(return_value="predicted completion")
        mock_llm.generate = Mock(return_value="generated response")
        
        mock_stt = Mock()
        mock_response_buffer = Mock()
        mock_response_buffer.add = Mock()
        mock_response_buffer.get = Mock(return_value="buffered response")
        mock_response_buffer.clear = Mock()
        
        mock_session_buffer = Mock()
        mock_session_buffer.add_transcription = Mock()
        mock_session_buffer.add_prediction = Mock()
        mock_session_buffer.get_prediction_by_similarity = Mock(return_value=None)
        
        return {
            'llm': mock_llm,
            'stt': mock_stt,
            'response_buffer': mock_response_buffer,
            'session_buffer': mock_session_buffer,
        }
    
    @pytest.fixture
    def fsm(self, mock_dependencies):
        """Create a ConversationFSM instance for testing."""
        return ConversationFSM(
            llm_provider=mock_dependencies['llm'],
            stt_provider=mock_dependencies['stt'],
            response_buffer=mock_dependencies['response_buffer'],
            session_buffer=mock_dependencies['session_buffer'],
        )
    
    def test_initial_state(self, fsm):
        """Test that FSM initializes to IDLE state."""
        assert fsm.get_state() == 'idle'
    
    def test_transition_to_listening(self, fsm):
        """Test transition from IDLE to LISTENING."""
        # Trigger transition
        fsm.start_listening()
        
        # Verify state changed
        assert fsm.get_state() == 'listening'
    
    @pytest.mark.asyncio
    async def test_process_transcription_partial(self, fsm, mock_dependencies):
        """Test processing a partial transcription."""
        # Start in listening state
        fsm.start_listening()
        
        # Process partial transcription
        result = await fsm.process_transcription("Hello", False)
        
        # Verify state transition to speech_active
        assert fsm.get_state() == 'speech_active'
        
        # Verify prediction was made
        mock_dependencies['llm'].predict.assert_called_once_with("Hello")
        
        # Verify prediction added to session buffer
        mock_dependencies['session_buffer'].add_prediction.assert_called_once()
        
        # Verify result contains expected data
        assert result['text'] == "Hello"
        assert result['is_final'] is False
        assert result['predicted_completion'] == "predicted completion"
        assert result['state'] == 'speech_active'
    
    @pytest.mark.asyncio
    async def test_process_transcription_final(self, fsm, mock_dependencies):
        """Test processing a final transcription."""
        # Start in listening state and move to speech active
        fsm.start_listening()
        fsm.detect_speech()
        
        # Process final transcription
        result = await fsm.process_transcription("Hello world", True)
        
        # Verify final state
        assert fsm.get_state() == 'speaking'
        
        # Verify LLM was used to generate response
        mock_dependencies['llm'].generate.assert_called_once()
        
        # Verify response added to buffer
        mock_dependencies['response_buffer'].add.assert_called_once()
        
        # Verify response obtained from buffer
        mock_dependencies['response_buffer'].get.assert_called_once()
        
        # Verify result contains expected data
        assert result['text'] == "Hello world"
        assert result['is_final'] is True
        assert result['response'] == "buffered response"
        assert result['state'] == 'speaking'
    
    @pytest.mark.asyncio
    async def test_check_silence_triggers_transition(self, fsm, mock_dependencies):
        """Test that silence detection triggers state transition."""
        # Start in listening state and move to speech active
        fsm.start_listening()
        fsm.detect_speech()
        
        # Set conditions for silence detection
        fsm.last_speech_time = 0  # Set to past time
        fsm.silence_timeout = 0.1  # Short timeout for testing
        
        # Check for silence
        result = await fsm.check_silence()
        
        # Verify result contains expected data (silence detected)
        assert result is not None
        assert result['is_final'] is True
        assert 'trigger' in result
        assert result['trigger'] == 'silence'
        
        # Verify state transition occurred
        assert fsm.get_state() == 'speaking'
    
    def test_handle_interrupt(self, fsm):
        """Test handling interruption."""
        # Start in speaking state (via multiple transitions)
        fsm.start_listening()
        fsm.detect_speech()
        fsm.finalize_transcription()
        fsm.generate_response()
        fsm.deliver_response()
        assert fsm.get_state() == 'speaking'
        
        # Handle interrupt
        fsm.handle_interrupt()
        
        # Verify temporary transition to interrupted then back to listening
        assert fsm.get_state() == 'listening'