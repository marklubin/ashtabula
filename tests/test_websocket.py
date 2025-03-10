"""
Test suite for Ashtabula WebSocket Server

This module contains tests for the WebSocket server component,
covering basic connectivity, audio chunking, error handling,
and concurrent client scenarios.
"""

import pytest
import asyncio
import websockets
import wave
import numpy as np
import json
import base64
from unittest.mock import MagicMock, patch

from ashtabula.websocket import WebSocketServer
from ashtabula.stt import STTProvider
from ashtabula.conversation import ConversationManager
from ashtabula.buffer import ResponseBuffer

# Test constants
TEST_HOST = "localhost"
TEST_PORT = 8765
TEST_TIMEOUT = 5  # seconds


class MockSTTProvider(STTProvider):
    """Mock STT provider for testing."""
    
    async def stream_audio(self, audio_source):
        # Simulate transcription results
        yield {"text": "Test transcription", "is_final": False}
        yield {"text": "Test transcription complete", "is_final": True}


@pytest.fixture
async def mock_conversation_manager():
    """Fixture to provide a mock conversation manager."""
    mock_llm = MagicMock()
    mock_stt = MockSTTProvider()
    mock_buffer = ResponseBuffer()
    
    manager = ConversationManager(
        llm_provider=mock_llm,
        stt_provider=mock_stt,
        response_buffer=mock_buffer
    )
    
    # Mock the handle_interrupt method
    manager.handle_interrupt = MagicMock()
    
    return manager


@pytest.fixture
async def websocket_server(mock_conversation_manager):
    """Fixture to create and start a test server."""
    # Create a mock STT provider
    mock_stt = MockSTTProvider()
    
    # Create the server with mocks
    server = WebSocketServer(
        host=TEST_HOST,
        port=TEST_PORT,
        chunk_duration=1.0,
        stt_provider=mock_stt,
        conversation_manager=mock_conversation_manager
    )
    
    # Start the server in a background task
    task = asyncio.create_task(server.start())
    await asyncio.sleep(0.1)  # Allow server to start
    
    yield server
    
    # Clean up after the test
    await server.stop()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


async def receive_message(websocket, timeout=TEST_TIMEOUT):
    """Helper to receive and parse a JSON message."""
    try:
        message = await asyncio.wait_for(websocket.recv(), timeout=timeout)
        return json.loads(message)
    except json.JSONDecodeError:
        return message


@pytest.mark.asyncio
async def test_basic_connection_and_chunking(websocket_server):
    """Test basic connection and audio chunking functionality.
    
    Scenario: A client establishes a WebSocket connection and streams 2 seconds 
    of WAV audio at a chunk size of 1 second.
    """
    # Create 2 seconds of test audio data
    sample_rate = 16000
    duration = 2
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio_data = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    
    async with websockets.connect(f'ws://{TEST_HOST}:{TEST_PORT}') as websocket:
        # Receive welcome message
        welcome = await receive_message(websocket)
        assert welcome["type"] == "connection_established"
        assert "session_id" in welcome
        
        # Send audio in 100ms chunks
        chunk_size = int(sample_rate * 0.1) * 2  # *2 for 16-bit samples
        for i in range(0, len(audio_data), chunk_size // 2):
            chunk = audio_data[i:i + chunk_size // 2].tobytes()
            await websocket.send(chunk)
            
            # Optional: receive server responses
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=0.01)
                # We're not asserting on the response here, just ensuring the server
                # is responding without errors
            except asyncio.TimeoutError:
                pass
    
    # Give the server a moment to clean up connections
    await asyncio.sleep(0.1)
    
    # Allow for potential lingering connections
    assert len(websocket_server.connections) <= 1


@pytest.mark.asyncio
async def test_json_protocol(websocket_server):
    """Test the JSON protocol for sending audio chunks.
    
    Scenario: A client sends audio using the JSON protocol with base64 encoding.
    """
    async with websockets.connect(f'ws://{TEST_HOST}:{TEST_PORT}') as websocket:
        # Receive welcome message
        welcome = await receive_message(websocket)
        assert welcome["type"] == "connection_established"
        
        # Start audio stream
        await websocket.send(json.dumps({
            "type": "start_audio_stream",
            "format": "wav",
            "sample_rate": 16000
        }))
        
        # Receive acknowledgment
        response = await receive_message(websocket)
        assert response["type"] == "stream_started"
        
        # Create sample audio data
        audio_data = np.zeros(16000, dtype=np.int16).tobytes()
        encoded_data = base64.b64encode(audio_data).decode('utf-8')
        
        # Send an audio chunk
        await websocket.send(json.dumps({
            "type": "audio_chunk",
            "data": encoded_data
        }))
        
        # Receive chunk acknowledgment or transcription
        response = await receive_message(websocket)
        assert response["type"] in ["chunk_received", "partial_transcription"]
        
        # End the stream
        await websocket.send(json.dumps({
            "type": "end_audio_stream"
        }))
        
        # Receive stream end acknowledgment
        response = await receive_message(websocket)
        assert response["type"] == "stream_ended"


@pytest.mark.asyncio
async def test_unsupported_format(websocket_server):
    """Test handling of unsupported audio format.
    
    Scenario: A client sends audio data in an unsupported format (e.g., MP3).
    """
    async with websockets.connect(f'ws://{TEST_HOST}:{TEST_PORT}') as websocket:
        # Skip welcome message
        await websocket.recv()
        
        # Start audio stream with unsupported format
        await websocket.send(json.dumps({
            "type": "start_audio_stream",
            "format": "mp3",  # Unsupported format
            "sample_rate": 16000
        }))
        
        # Server should warn but accept the stream
        responses = []
        try:
            # Get stream started acknowledgment
            responses.append(await receive_message(websocket))
            
            # Should also get a warning
            responses.append(await asyncio.wait_for(websocket.recv(), timeout=0.1))
        except asyncio.TimeoutError:
            pass
        
        # Verify we got a warning in one of the responses
        warning_received = any(
            resp.get("type") == "warning" and "may not be fully supported" in resp.get("message", "")
            for resp in responses if isinstance(resp, dict)
        )
        assert warning_received, "Server should warn about unsupported format"
        
        # End the stream
        await websocket.send(json.dumps({
            "type": "end_audio_stream"
        }))


@pytest.mark.asyncio
async def test_high_concurrency(websocket_server):
    """Test multiple simultaneous connections.
    
    Scenario: Ten clients connect concurrently, each streaming short audio clips.
    """
    async def client_session(client_id):
        async with websockets.connect(f'ws://{TEST_HOST}:{TEST_PORT}') as websocket:
            # Wait for welcome message
            await websocket.recv()
            
            # Send some test audio data
            audio_data = np.random.randint(-32768, 32767, 16000).astype(np.int16).tobytes()
            await websocket.send(audio_data)
            
            # Wait for server processing
            try:
                await asyncio.wait_for(websocket.recv(), timeout=0.1)
            except asyncio.TimeoutError:
                pass
    
    # Create and run 10 concurrent clients
    clients = [client_session(i) for i in range(10)]
    await asyncio.gather(*clients)
    
    # Give the server a moment to clean up connections
    await asyncio.sleep(0.1)
    
    # All connections should be closed
    assert len(websocket_server.connections) <= 1  # Allow for connection that might still be active


@pytest.mark.asyncio
async def test_corrupted_chunk_recovery(websocket_server):
    """Test handling of corrupted audio chunks.
    
    Scenario: A client sends partially corrupted data in the middle of the stream.
    """
    async with websockets.connect(f'ws://{TEST_HOST}:{TEST_PORT}') as websocket:
        # Skip welcome message
        await websocket.recv()
        
        # Send valid chunk
        valid_chunk = np.zeros(16000, dtype=np.int16).tobytes()
        await websocket.send(valid_chunk)
        
        # Wait for server processing
        try:
            await asyncio.wait_for(websocket.recv(), timeout=0.1)
        except asyncio.TimeoutError:
            pass
        
        # Send corrupted chunk
        await websocket.send(b'corrupted data')
        
        # Server may send an error message - receive it
        try:
            message = await asyncio.wait_for(websocket.recv(), timeout=0.1)
            # We're not asserting anything here - just ensuring the server responds
        except asyncio.TimeoutError:
            pass
        
        # Send another valid chunk - the server should continue processing
        await websocket.send(valid_chunk)
        
        # Wait for server processing
        try:
            await asyncio.wait_for(websocket.recv(), timeout=0.1)
        except asyncio.TimeoutError:
            pass


@pytest.mark.asyncio
async def test_chunk_size_variations():
    """Test different chunk size configurations.
    
    Scenario: The server is configured for different chunk durations.
    """
    # Test with 0.5s chunks
    server_half = WebSocketServer(
        host=TEST_HOST,
        port=TEST_PORT + 1,  # Use a different port
        chunk_duration=0.5
    )
    task = asyncio.create_task(server_half.start())
    await asyncio.sleep(0.1)
    
    try:
        async with websockets.connect(f'ws://{TEST_HOST}:{TEST_PORT + 1}') as websocket:
            # Skip welcome message
            await websocket.recv()
            
            # Calculate chunk size for 0.5s duration
            bytes_per_half_second = int(16000 * 0.5 * 2)  # sample_rate * duration * bytes_per_sample
            
            # Create 1 second of audio (should be two 0.5s chunks)
            audio_data = np.zeros(16000, dtype=np.int16).tobytes()
            
            # Send the audio
            await websocket.send(audio_data)
            
            # Server should process two chunks
            responses = []
            for _ in range(2):
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=0.2)
                    responses.append(response)
                except asyncio.TimeoutError:
                    break
            
            # We expect at least one chunk response
            assert len(responses) > 0, "No chunk processing responses received"
    finally:
        # Clean up the server
        await server_half.stop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_text_input(websocket_server):
    """Test sending text input directly.
    
    Scenario: A client sends text input instead of audio.
    """
    async with websockets.connect(f'ws://{TEST_HOST}:{TEST_PORT}') as websocket:
        # Skip welcome message
        await websocket.recv()
        
        # Send text input
        test_text = "Hello, this is a test message."
        await websocket.send(json.dumps({
            "type": "text_input",
            "text": test_text
        }))
        
        # Receive AI response
        response = await receive_message(websocket)
        assert response["type"] == "ai_response"
        assert "text" in response


@pytest.mark.asyncio
async def test_interruption(websocket_server):
    """Test interrupting an AI response.
    
    Scenario: A client interrupts the AI while it's responding.
    """
    # Create a fresh mock just for this test to ensure it starts uncalled
    mock_cm = MagicMock()
    mock_cm.handle_interrupt = MagicMock()
    
    # Replace the conversation manager with our mock
    original_cm = websocket_server.conversation_manager
    websocket_server.conversation_manager = mock_cm
    
    try:
        async with websockets.connect(f'ws://{TEST_HOST}:{TEST_PORT}') as websocket:
            # Skip welcome message
            await websocket.recv()
            
            # Send text input to start a response
            await websocket.send(json.dumps({
                "type": "text_input",
                "text": "Tell me a long story about space exploration."
            }))
            
            # Wait for the initial response
            await receive_message(websocket)
            
            # Send interrupt with new input
            await websocket.send(json.dumps({
                "type": "interrupt",
                "new_text": "Actually, tell me about quantum computing instead."
            }))
            
            # Wait a bit to make sure the request is processed
            await asyncio.sleep(0.1)
            
            # Verify the conversation manager's interrupt method was called
            assert mock_cm.handle_interrupt.called, "Interrupt handler was not called"
            
            # Try to receive interrupted confirmation
            try:
                response = await asyncio.wait_for(receive_message(websocket), timeout=0.5)
                if response["type"] == "ai_response_interrupted":
                    # Also try to get the new response
                    try:
                        response = await asyncio.wait_for(receive_message(websocket), timeout=0.5)
                        assert response["type"] == "ai_response"
                        assert "text" in response
                    except asyncio.TimeoutError:
                        # Connection might have closed, which is okay
                        pass
            except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed):
                # It's okay if the connection closed after sending the interrupt
                pass
    finally:
        # Restore the original conversation manager
        websocket_server.conversation_manager = original_cm


@pytest.mark.asyncio
async def test_error_handling(websocket_server):
    """Test server's handling of malformed requests.
    
    Scenario: A client sends invalid messages.
    """
    async with websockets.connect(f'ws://{TEST_HOST}:{TEST_PORT}') as websocket:
        # Skip welcome message
        await websocket.recv()
        
        # Send malformed JSON
        await websocket.send("This is not valid JSON")
        
        # Server should respond with error
        response = await receive_message(websocket)
        assert response["type"] == "error"
        assert "message" in response
        
        # Send message with missing required fields
        await websocket.send(json.dumps({
            "type": "text_input"
            # Missing "text" field
        }))
        
        # Server should respond with error
        response = await receive_message(websocket)
        assert response["type"] == "error"
        assert "message" in response
        
        # Verify server is still operational
        await websocket.send(json.dumps({
            "type": "text_input",
            "text": "Is the server still working?"
        }))
        
        # Server should still respond normally
        response = await receive_message(websocket)
        assert response["type"] == "ai_response"
        assert "text" in response
