#!/usr/bin/env python
"""
Basic tests for the demo CLI application that doesn't rely on pytest.
"""

import os
import sys
import asyncio
import unittest
from unittest.mock import Mock, patch, AsyncMock
import numpy as np
from pathlib import Path

# Add parent directory to path to import the demo module
sys.path.insert(0, str(Path(__file__).parent))

import demo_cli


class MockWave:
    """Mock for wave.open."""
    def __init__(self):
        self.setnchannels = Mock()
        self.setsampwidth = Mock()
        self.setframerate = Mock()
        self.writeframes = Mock()
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class TestDemoCLI(unittest.TestCase):
    """Basic tests for the demo CLI application."""
    
    def test_signal_handler(self):
        """Test the signal handler."""
        # Save original value
        original_running = demo_cli.running
        
        try:
            # Set running to True
            demo_cli.running = True
            
            # Call the signal handler
            demo_cli.signal_handler(None, None)
            
            # Verify running is set to False
            self.assertFalse(demo_cli.running)
        finally:
            # Restore original value
            demo_cli.running = original_running


async def test_process_audio_chunk():
    """Test processing an audio chunk."""
    # Create mock components
    mock_stt = AsyncMock()
    mock_stt.transcribe_file = AsyncMock(return_value="Hello, how are you?")
    
    mock_fsm = Mock()
    mock_fsm.stt = mock_stt
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
    original_fsm = demo_cli.fsm
    demo_cli.fsm = mock_fsm
    
    # Create a test audio chunk
    audio_chunk = np.zeros(16000, dtype=np.float32)
    
    # Patch tempfile, wave, and play_response
    with patch('tempfile.NamedTemporaryFile') as mock_tempfile, \
         patch('wave.open') as mock_wave_open, \
         patch('demo_cli.play_response', new_callable=AsyncMock) as mock_play:
        
        # Set up mock tempfile
        mock_temp = Mock()
        mock_temp.name = "/tmp/test_audio.wav"
        mock_tempfile.return_value.__enter__.return_value = mock_temp
        
        # Set up mock wave
        mock_wave_file = MockWave()
        mock_wave_open.return_value = mock_wave_file
        
        # Call the function
        await demo_cli.process_audio_chunk(audio_chunk)
        
        # Verify calls
        mock_fsm.stt.transcribe_file.assert_called_once()
        mock_fsm.process_transcription.assert_called_once()
        mock_fsm.check_silence.assert_called_once()
        mock_play.assert_called_once_with('I am doing well, thank you for asking!')
    
    # Restore original FSM
    demo_cli.fsm = original_fsm


def run_async_test(coroutine):
    """Run an async test function."""
    loop = asyncio.get_event_loop()
    loop.run_until_complete(coroutine)


def main():
    """Run all tests."""
    print("Running basic tests for demo_cli.py...")
    
    # Run unittest tests
    unittest.main(argv=[sys.argv[0], 'TestDemoCLI'], exit=False)
    
    # Run async tests
    print("\nRunning async tests...")
    run_async_test(test_process_audio_chunk())
    print("✓ test_process_audio_chunk passed")
    
    print("\nAll tests passed!")


if __name__ == "__main__":
    main()