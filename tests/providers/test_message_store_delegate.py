import pytest
from unittest.mock import Mock
from ashtabula.llm import LLMProvider
from ashtabula.messages import Message, MessageStore
from ashtabula.providers.message_store_delegate import MessageStoreDelegateLLM

class MockMessageStore(MessageStore):
    """Mock message store for testing."""
    def __init__(self, messages=None):
        self.messages = messages or []

    def write_message(self, message):
        self.messages.append(message)

    def read_messages(self, start_index=0):
        return self.messages[start_index:]

    def get_last_n_messages(self, n):
        return self.messages[-n:]

    def get_message_count(self):
        return len(self.messages)

class MockLLM(LLMProvider):
    """Mock LLM for testing."""
    def generate(self, input_text, max_length=None, temperature=0.7, top_p=0.95, top_k=50):
        # Echo input for testing
        return f"Response to: {input_text}"
        
    def predict(self, partial_text: str, max_length=None, temperature=0.7) -> str:
        # Echo input with completion marker for testing
        return f"{partial_text} [completion]."

def test_delegate_without_history():
    """Test generation with no message history."""
    store = MockMessageStore()
    mock_llm = MockLLM()
    
    delegate = MessageStoreDelegateLLM(
        delegate=mock_llm,
        message_store=store,
        num_messages=3
    )
    
    result = delegate.generate("Hello")
    expected = "Response to: Current input:\nHello"
    assert result == expected

def test_delegate_with_history():
    """Test generation with message history."""
    messages = [
        Message(text="First message", user_id="user1"),
        Message(text="Second message", user_id="user2"),
        Message(text="Third message", user_id="user1")
    ]
    store = MockMessageStore(messages)
    mock_llm = MockLLM()
    
    delegate = MessageStoreDelegateLLM(
        delegate=mock_llm,
        message_store=store,
        num_messages=2
    )
    
    result = delegate.generate("Hello")
    expected = (
        "Response to: Previous conversation:\n"
        "user2: Second message\n"
        "user1: Third message\n\n"
        "Current input:\n"
        "Hello"
    )
    assert result == expected

def test_delegate_with_custom_prefix():
    """Test generation with custom history prefix."""
    messages = [
        Message(text="Test message", user_id="user1")
    ]
    store = MockMessageStore(messages)
    mock_llm = MockLLM()
    
    delegate = MessageStoreDelegateLLM(
        delegate=mock_llm,
        message_store=store,
        num_messages=1,
        history_prefix="History: "
    )
    
    result = delegate.generate("Hello")
    expected = (
        "Response to: History: "
        "user1: Test message\n\n"
        "Current input:\n"
        "Hello"
    )
    assert result == expected

def test_delegate_passes_parameters():
    """Test that generation parameters are passed through."""
    store = MockMessageStore()
    mock_llm = Mock(spec=LLMProvider)
    mock_llm.generate.return_value = "Mock response"
    
    delegate = MessageStoreDelegateLLM(
        delegate=mock_llm,
        message_store=store,
        num_messages=1
    )
    
    delegate.generate(
        input_text="Hello",
        max_length=100,
        temperature=0.5,
        top_p=0.8,
        top_k=40
    )
    
    # Verify the mock was called with correct parameters
    mock_llm.generate.assert_called_once()
    call_args = mock_llm.generate.call_args[1]
    assert call_args["input_text"] == "Current input:\nHello"
    assert call_args["max_length"] == 100
    assert call_args["temperature"] == 0.5
    assert call_args["top_p"] == 0.8
    assert call_args["top_k"] == 40

def test_predict_without_history():
    """Test prediction without message history."""
    store = MockMessageStore()
    mock_llm = Mock(spec=LLMProvider)
    mock_llm.predict.return_value = "predicted completion"
    
    delegate = MessageStoreDelegateLLM(
        delegate=mock_llm,
        message_store=store,
        num_messages=1
    )
    
    result = delegate.predict("Hello")
    
    # Verify prediction is made without history
    mock_llm.predict.assert_called_once_with(
        partial_text="Hello",
        max_length=None,
        temperature=0.7
    )
    assert result == "predicted completion"

def test_predict_passes_parameters():
    """Test that prediction parameters are passed through."""
    store = MockMessageStore()
    mock_llm = Mock(spec=LLMProvider)
    mock_llm.predict.return_value = "predicted completion"
    
    delegate = MessageStoreDelegateLLM(
        delegate=mock_llm,
        message_store=store,
        num_messages=1
    )
    
    delegate.predict(
        partial_text="Hello",
        max_length=50,
        temperature=0.5
    )
    
    # Verify parameters are passed correctly
    mock_llm.predict.assert_called_once_with(
        partial_text="Hello",
        max_length=50,
        temperature=0.5
    )

def test_delegate_with_partial_history():
    """Test when requesting more history than available."""
    messages = [
        Message(text="Only message", user_id="user1")
    ]
    store = MockMessageStore(messages)
    mock_llm = MockLLM()
    
    delegate = MessageStoreDelegateLLM(
        delegate=mock_llm,
        message_store=store,
        num_messages=3  # More than available
    )
    
    result = delegate.generate("Hello")
    expected = (
        "Response to: Previous conversation:\n"
        "user1: Only message\n\n"
        "Current input:\n"
        "Hello"
    )
    assert result == expected
