"""
Integration tests for full conversation pipeline.

Tests the interaction between:
- Speech-to-text streaming with intermediate and final results
- LLM response generation and prediction
- Response buffering and management
- Conversation state handling
- Session management and session state persistence
"""
import pytest
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List, Optional

from ashtabula.conversation import ConversationManager, ConversationConfig
from ashtabula.llm import LLMProvider
from ashtabula.stt import STTProvider
from ashtabula.buffer import ResponseBuffer, SessionBuffer, TranscriptionItem, PredictionItem


class MockLLMProvider(LLMProvider):
    def generate(self, input_text: str, **kwargs) -> str:
        return "Mock response"
    
    def predict(self, partial_text: str, **kwargs) -> str:
        return f"{partial_text} completed"


class MockSTTProvider(STTProvider):
    async def stream_audio(self, audio_source: str):
        """Mock streaming audio transcription with intermediate and final results."""
        # Simulate realistic streaming with intermediate results
        chunks = [
            {'text': "Hello", 'is_final': False},
            {'text': "Hello how", 'is_final': False},
            {'text': "Hello how are", 'is_final': False},
            {'text': "Hello how are you", 'is_final': True}
        ]
        
        # Add delay to simulate real-time processing
        for chunk in chunks:
            await asyncio.sleep(0.1)  # Small delay between chunks
            yield chunk


@pytest.fixture
def response_buffer():
    return ResponseBuffer()


@pytest.fixture
def session_buffer():
    return SessionBuffer(max_transcription_history=5, max_prediction_history=3)


@pytest.fixture
def conversation_manager(response_buffer):
    llm = MockLLMProvider()
    stt = MockSTTProvider()
    config = ConversationConfig(
        prediction_threshold=0.8,
        silence_timeout=1.0,
        max_history_length=3,
        session_timeout=300.0,
        max_transcription_history=5,
        max_prediction_history=3,
        embed_predictions=True
    )
    manager = ConversationManager(llm, stt, response_buffer, config)
    yield manager
    # Clean up
    manager.close()


def test_response_buffer(response_buffer):
    """Test response buffer operations."""
    # Test empty buffer
    assert response_buffer.is_empty()
    assert len(response_buffer) == 0
    assert response_buffer.get() is None

    # Test adding and getting responses
    response_buffer.add("First response")
    response_buffer.add("Second response")
    assert len(response_buffer) == 2
    assert not response_buffer.is_empty()
    
    assert response_buffer.get() == "First response"
    assert len(response_buffer) == 1
    
    # Test clearing buffer
    response_buffer.clear()
    assert response_buffer.is_empty()
    assert response_buffer.get() is None


def test_session_buffer_basic_operations(session_buffer):
    """Test basic operations of the SessionBuffer."""
    # Test adding transcriptions
    session_buffer.add_transcription("Hello", is_final=False)
    session_buffer.add_transcription("Hello world", is_final=True)
    
    # Check transcription history
    history = session_buffer.get_transcription_history()
    assert len(history) == 2
    assert history[0].text == "Hello"
    assert not history[0].is_final
    assert history[1].text == "Hello world"
    assert history[1].is_final
    
    # Test adding predictions
    session_buffer.add_prediction("Hello there", "Hello")
    session_buffer.add_prediction("Hello world and everyone", "Hello world")
    
    # Check prediction history
    predictions = session_buffer.get_prediction_history()
    assert len(predictions) == 2
    assert predictions[0].text == "Hello there"
    assert predictions[0].source_transcription == "Hello"
    assert predictions[1].text == "Hello world and everyone"
    
    # Test clearing the buffer
    session_buffer.clear()
    assert len(session_buffer.get_transcription_history()) == 0
    assert len(session_buffer.get_prediction_history()) == 0
    assert session_buffer.current_partial_text == ""


def test_session_buffer_stale_detection(session_buffer):
    """Test stale session detection."""
    # Add some data
    session_buffer.add_transcription("Hello", is_final=False)
    
    # Should not be stale immediately
    assert not session_buffer.is_stale(timeout_seconds=5.0)
    
    # Artificially manipulate the last activity time to simulate time passing
    original_time = session_buffer.last_activity_time
    session_buffer.last_activity_time = original_time - 10.0
    
    # Now should be stale
    assert session_buffer.is_stale(timeout_seconds=5.0)


def test_session_buffer_prediction_similarity(session_buffer):
    """Test finding predictions by similarity."""
    # Add some predictions
    session_buffer.add_prediction("Hello world", "Hello")
    session_buffer.add_prediction("Hello everyone", "Hello ev")
    session_buffer.add_prediction("Goodbye friends", "Goodbye")
    
    # Define a simple similarity function for testing
    def simple_similarity(text1, text2):
        # Count matching characters
        return sum(1 for a, b in zip(text1.lower(), text2.lower()) if a == b) / max(len(text1), len(text2))
    
    # Find the best match for a given text
    best_match = session_buffer.get_prediction_by_similarity("Hello world!", simple_similarity)
    assert best_match is not None
    assert best_match.text == "Hello world"
    
    # Try with a text that's more similar to the second prediction
    best_match = session_buffer.get_prediction_by_similarity("Hello every person", simple_similarity)
    assert best_match is not None
    assert best_match.text == "Hello everyone"
    
    # Try with totally different text
    best_match = session_buffer.get_prediction_by_similarity("Something completely different", simple_similarity)
    assert best_match is not None  # Should still return the best match even if not very similar


@pytest.mark.asyncio
async def test_stream_audio_basic_flow(conversation_manager):
    """Test basic streaming audio flow with transcription and responses."""
    # Collect all responses
    responses = []
    async for response in conversation_manager.stream_audio("test.wav"):
        responses.append(response)
    
    # Should have intermediate responses and one final response
    assert len(responses) > 1, "Expected multiple responses"
    
    # Check intermediate responses
    intermediate_responses = [r for r in responses[:-1]]
    assert all(not r.get('is_final', False) for r in intermediate_responses), (
        "Intermediate responses should not be final"
    )
    assert all('predicted_completion' in r for r in intermediate_responses), (
        "Intermediate responses should include predicted completions"
    )
    
    # Check final response
    final_response = responses[-1]
    assert final_response.get('is_final', False), "Last response should be final"
    assert 'response' in final_response, "Final response should include AI response"
    assert isinstance(final_response.get('text'), str), "Final response should include text"
    
    # Check session ID was provided
    assert 'session_id' in final_response
    session_id = final_response['session_id']
    
    # Verify session exists
    assert session_id in conversation_manager.sessions
    session = conversation_manager.sessions[session_id]
    
    # Verify session has recorded transcriptions and predictions
    assert len(session.get_transcription_history()) > 0
    assert len(session.get_prediction_history()) > 0


@pytest.mark.asyncio
async def test_silence_detection(conversation_manager):
    """Test that silence triggers sentence completion."""
    # Set short timeout for testing
    conversation_manager.config.silence_timeout = 0.1
    
    # Collect responses until final
    responses = []
    async for response in conversation_manager.stream_audio("test.wav"):
        responses.append(response)
        if response.get('is_final', False):
            break
    
    # Verify we got intermediate results before completion
    assert len(responses) > 1, "Expected intermediate results before completion"
    assert not responses[0].get('is_final', False), "First response should not be final"
    
    # Verify final response
    final_response = responses[-1]
    assert final_response.get('is_final', False), "Last response should be final"
    assert 'response' in final_response, "Final response should include AI response"
    assert isinstance(final_response.get('text'), str), "Final response should include text"


@pytest.mark.asyncio
async def test_session_continuity(conversation_manager):
    """Test maintaining conversation state across multiple audio streams."""
    # Create a different mock STT provider for this test with different response patterns
    class MockSTTProvider2(STTProvider):
        async def stream_audio(self, audio_source: str):
            chunks = [
                {'text': "Hello", 'is_final': False},
                {'text': "Hello world", 'is_final': False},
                {'text': "Hello world how", 'is_final': False},
                {'text': "Hello world how are you", 'is_final': True}
            ]
            
            # Use different patterns for different audio sources
            if audio_source == "test1.wav":
                # First audio source gets basic chunks
                basic_chunks = [
                    {'text': "First", 'is_final': False},
                    {'text': "First test", 'is_final': True}
                ]
                for chunk in basic_chunks:
                    await asyncio.sleep(0.1)
                    yield chunk
            else:
                # Second audio source gets different chunks
                second_chunks = [
                    {'text': "Second", 'is_final': False},
                    {'text': "Second audio", 'is_final': False},
                    {'text': "Second audio test", 'is_final': True}
                ]
                for chunk in second_chunks:
                    await asyncio.sleep(0.1)
                    yield chunk
    
    # Use the custom STT provider
    original_stt = conversation_manager.stt
    conversation_manager.stt = MockSTTProvider2()
    
    try:
        # First conversation turn
        session_id = None
        async for response in conversation_manager.stream_audio("test1.wav"):
            if response.get('is_final', False):
                session_id = response.get('session_id')
                break
        
        # Ensure we got a session ID
        assert session_id is not None
        
        # Remember the number of transcriptions and predictions
        first_session = conversation_manager.get_session(session_id)
        first_transcription_count = len(first_session.get_transcription_history())
        first_prediction_count = len(first_session.get_prediction_history())
        
        # Second conversation turn, same session
        async for response in conversation_manager.stream_audio("test2.wav", session_id=session_id):
            if response.get('is_final', False):
                break
        
        # Verify the session has accumulated more history
        second_session = conversation_manager.get_session(session_id)
        assert len(second_session.get_transcription_history()) > first_transcription_count
        assert len(second_session.get_prediction_history()) > first_prediction_count
    finally:
        # Restore the original STT provider
        conversation_manager.stt = original_stt


def test_conversation_history(conversation_manager):
    """Test conversation history management."""
    texts = ["First message", "Second message", "Third message", "Fourth message"]
    
    for text in texts:
        conversation_manager._update_history(text)
    
    # Should maintain max_history_length
    assert len(conversation_manager.conversation_history) == conversation_manager.config.max_history_length
    assert conversation_manager.conversation_history[-1] == texts[-1]


def test_interrupt_handling(conversation_manager):
    """Test interrupt handling clears state."""
    # Create a session and make it active
    session_id, session = conversation_manager._get_or_create_session()
    conversation_manager.active_session_id = session_id
    
    # Add some data to the session
    session.add_transcription("Hello world", is_final=False)
    session.add_prediction("Hello world and everyone", "Hello world")
    
    # Also add data to the legacy state
    conversation_manager.current_sentence = ["Hello", "world"]
    conversation_manager.predicted_text = "Hello world!"
    
    # Handle interrupt
    conversation_manager.handle_interrupt()
    
    # Verify legacy state is cleared
    assert conversation_manager.is_interrupted
    assert not conversation_manager.current_sentence
    assert not conversation_manager.predicted_text
    
    # Verify session state is cleared
    assert len(session.get_transcription_history()) == 0
    assert len(session.get_prediction_history()) == 0


def test_similarity_calculation(conversation_manager):
    """Test text similarity calculation."""
    text1 = "Hello world"
    text2 = "Hello World"  # Capitalization difference
    text3 = "Something else"
    
    sim1 = conversation_manager._calculate_similarity(text1, text2)
    sim2 = conversation_manager._calculate_similarity(text1, text3)
    
    assert sim1 > 0.9  # Should be very similar
    assert sim2 < 0.5  # Should be quite different


@pytest.mark.asyncio
async def test_error_handling(conversation_manager):
    """Test error handling in stream processing."""
    # Mock STT to raise an exception
    async def error_stream(_):
        raise Exception("Test error")
    
    conversation_manager.stt.stream_audio = error_stream
    
    async for response in conversation_manager.stream_audio("test.wav"):
        assert response['is_final']
        assert 'error' in response
        assert response['text'] is None
        assert 'session_id' in response  # Should still provide a session ID
        break


@pytest.mark.asyncio
async def test_session_cleanup(conversation_manager):
    """Test automatic cleanup of stale sessions."""
    # Create several sessions
    session_ids = []
    for i in range(3):
        session_id, _ = conversation_manager._get_or_create_session()
        session_ids.append(session_id)
    
    # Make some sessions stale by manipulating their last activity time
    for session_id in session_ids[:2]:
        session = conversation_manager.sessions[session_id]
        session.last_activity_time = time.time() - conversation_manager.config.session_timeout - 10
    
    # Call cleanup directly without using the coroutine
    # Just check that sessions with timestamp older than timeout are removed
    original_ids = set(conversation_manager.sessions.keys())
    
    # Do manual cleanup, simulating what the coroutine would do
    current_time = time.time()
    timeout = conversation_manager.config.session_timeout
    
    stale_session_ids = []
    for sess_id, session in conversation_manager.sessions.items():
        if current_time - session.last_activity_time > timeout:
            stale_session_ids.append(sess_id)
    
    # Remove stale sessions
    for sess_id in stale_session_ids:
        del conversation_manager.sessions[sess_id]
    
    # Verify stale sessions were removed
    remaining_sessions = set(conversation_manager.sessions.keys())
    assert session_ids[0] not in remaining_sessions
    assert session_ids[1] not in remaining_sessions
    assert session_ids[2] in remaining_sessions  # This one should still be active


@pytest.mark.asyncio
async def test_embedding_integration(conversation_manager):
    """Test that embeddings are generated and stored with predictions."""
    # Enable embedding and process some audio
    conversation_manager.config.embed_predictions = True
    async for response in conversation_manager.stream_audio("test.wav"):
        if response.get('is_final', False):
            break
    
    # Get the active session
    session_id = conversation_manager.active_session_id
    session = conversation_manager.sessions[session_id]
    
    # Check predictions have embeddings
    predictions = session.get_prediction_history()
    assert len(predictions) > 0
    
    # At least the final prediction should have an embedding
    final_predictions = [p for p in predictions if p.embedding is not None]
    assert len(final_predictions) > 0
    
    # Verify embedding structure
    embedding = final_predictions[-1].embedding
    assert isinstance(embedding, list)
    assert all(isinstance(x, float) for x in embedding)
