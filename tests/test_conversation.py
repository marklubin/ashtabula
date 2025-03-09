"""
Integration tests for full conversation pipeline.

Tests the interaction between:
- Speech-to-text streaming with intermediate and final results
- LLM response generation and prediction
- Response buffering and management
- Conversation state handling
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from typing import Dict, Any

from ashtabula.conversation import ConversationManager, ConversationConfig
from ashtabula.llm import LLMProvider
from ashtabula.stt import STTProvider
from ashtabula.buffer import ResponseBuffer


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
def conversation_manager(response_buffer):
    llm = MockLLMProvider()
    stt = MockSTTProvider()
    config = ConversationConfig(
        prediction_threshold=0.8,
        silence_timeout=1.0,
        max_history_length=3
    )
    return ConversationManager(llm, stt, response_buffer, config)

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
    conversation_manager.current_sentence = ["Hello", "world"]
    conversation_manager.predicted_text = "Hello world!"
    
    conversation_manager.handle_interrupt()
    
    assert conversation_manager.is_interrupted
    assert not conversation_manager.current_sentence
    assert not conversation_manager.predicted_text


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
        break
