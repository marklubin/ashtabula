#!/usr/bin/env python
"""Tests for the demo CLI application."""

import os
import sys
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import numpy as np
from pathlib import Path

# Add parent directory to path to import the demo module
sys.path.insert(0, str(Path(__file__).parent))

import demo_cli


@pytest.fixture
def mock_providers():
    """Create mock providers for testing."""
    mock_llm = Mock()
    mock_llm.predict = Mock(return_value="predicted completion")
    mock_llm.generate = Mock(return_value="This is a generated response from the assistant")
    
    mock_stt = AsyncMock()
    mock_stt.transcribe_file = AsyncMock(return_value="Hello, how are you?")
    
    mock_tts = Mock()
    mock_tts.synthesize = Mock(return_value="/tmp/test_audio.wav")
    
    mock_response_buffer = Mock()
    mock_session_buffer = Mock()
    
    return {
        'llm': mock_llm,
        'stt': mock_stt,
        'tts': mock_tts,
        'response_buffer': mock_response_buffer,
        'session_buffer': mock_session_buffer,
    }


@patch('demo_cli.HuggingFaceLLMProvider')
@patch('demo_cli.HFWhisperSTTProvider')
@patch('demo_cli.SpeechT5TTSProvider')
@patch('demo_cli.ResponseBuffer')
@patch('demo_cli.SessionBuffer')
@patch('demo_cli.ConversationFSM')
@patch('demo_cli.sd')
@patch('wave.open')
@patch('tempfile.NamedTemporaryFile')
class TestDemoCLI:
    """Tests for the demo CLI application."""
    
    def create_mock_wave_file(self):
        """Create a mock wave file for testing."""
        mock_wave = Mock()
        mock_wave.__enter__ = Mock(return_value=mock_wave)
        mock_wave.__exit__ = Mock(return_value=None)
        mock_wave.getframerate = Mock(return_value=16000)
        mock_wave.getnframes = Mock(return_value=16000)
        mock_wave.readframes = Mock(return_value=bytes(16000 * 2))  # 16-bit audio
        return mock_wave
    
    @pytest.mark.asyncio
    async def test_process_audio_chunk(
        self,
        mock_tempfile,
        mock_wave_open,
        mock_sd,
        mock_fsm_class,
        mock_session_buffer,
        mock_response_buffer,
        mock_tts_provider,
        mock_stt_provider,
        mock_llm_provider,
        mock_providers
    ):
        """Test processing an audio chunk."""
        # Set up mocks
        mock_temp = Mock()
        mock_temp.name = "/tmp/test_audio.wav"
        mock_tempfile.return_value.__enter__.return_value = mock_temp
        
        mock_wave_file = Mock()
        mock_wave_file.__enter__ = Mock(return_value=mock_wave_file)
        mock_wave_file.__exit__ = Mock(return_value=None)
        mock_wave_open.return_value = mock_wave_file
        
        # Create mock FSM
        mock_fsm = Mock()
        mock_fsm.stt = mock_providers['stt']
        mock_fsm.process_transcription = AsyncMock()
        mock_fsm.process_transcription.return_value = {
            'text': 'Hello',
            'is_final': False,
            'predicted_completion': 'Hello, how are you?',
            'state': 'speech_active'
        }
        mock_fsm.check_silence = AsyncMock()
        mock_fsm.check_silence.return_value = {
            'text': 'Hello, how are you?',
            'is_final': True,
            'response': 'I am doing well, thank you for asking!',
            'state': 'speaking',
            'trigger': 'silence'
        }
        
        # Set the global FSM to our mock
        demo_cli.fsm = mock_fsm
        
        # Create a test audio chunk
        audio_chunk = np.zeros(16000, dtype=np.float32)
        
        # Mock the os.unlink function to prevent file not found errors
        with patch('os.unlink') as mock_unlink, patch('demo_cli.play_response', new_callable=AsyncMock) as mock_play:
            # Call the function
            await demo_cli.process_audio_chunk(audio_chunk)
            
            # Verify calls
            mock_fsm.stt.transcribe_file.assert_called_once()
            mock_fsm.process_transcription.assert_called_once()
            mock_fsm.check_silence.assert_called_once()
            mock_play.assert_called_once_with('I am doing well, thank you for asking!')
    
    @pytest.mark.asyncio
    async def test_play_response(
        self,
        mock_tempfile,
        mock_wave_open,
        mock_sd,
        mock_fsm_class,
        mock_session_buffer,
        mock_response_buffer,
        mock_tts_provider,
        mock_stt_provider,
        mock_llm_provider,
        mock_providers
    ):
        """Test playing a response."""
        # Set up mocks
        mock_tts = Mock()
        mock_tts.synthesize = Mock()  # Now takes output_path parameter
        mock_tts_provider.return_value = mock_tts
        
        # Set up tempfile mock
        mock_tmp = Mock()
        mock_tmp.name = "/tmp/test_response.wav"
        mock_tempfile.return_value.__enter__.return_value = mock_tmp
        
        # Mock wave file
        mock_wave = self.create_mock_wave_file()
        mock_wave_open.return_value = mock_wave
        
        # Mock os.path.exists and os.unlink to avoid file not found errors
        with patch('os.path.exists') as mock_exists, patch('os.unlink') as mock_unlink:
            # Mock file existence checks
            mock_exists.side_effect = lambda path: path == "/tmp/test_response.wav"
            
            # Call the function
            await demo_cli.play_response("Hello, world!")
            
            # Verify calls - with updated parameter signature
            mock_tts.synthesize.assert_called_once_with("Hello, world!", output_path="/tmp/test_response.wav")
            mock_wave_open.assert_called_once_with("/tmp/test_response.wav", 'rb')
            mock_sd.play.assert_called_once()
            mock_sd.wait.assert_called_once()
            mock_unlink.assert_called_once_with("/tmp/test_response.wav")


def test_signal_handler():
    """Test the signal handler."""
    # Save original value
    original_running = demo_cli.running
    
    try:
        # Set running to True
        demo_cli.running = True
        
        # Call the signal handler
        demo_cli.signal_handler(None, None)
        
        # Verify running is set to False
        assert not demo_cli.running
    finally:
        # Restore original value
        demo_cli.running = original_running


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])