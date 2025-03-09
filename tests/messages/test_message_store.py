import pytest
from ashtabula.messages import Message, MessageStore

def test_message_creation():
    """Test creating a Message with and without explicit ID."""
    # Test with auto-generated ID
    msg1 = Message(text="Hello", user_id="user1")
    assert msg1.text == "Hello"
    assert msg1.user_id == "user1"
    assert msg1.id is not None
    assert isinstance(msg1.id, str)

    # Test with explicit ID
    explicit_id = "test-id-123"
    msg2 = Message(text="World", user_id="user2", id=explicit_id)
    assert msg2.text == "World"
    assert msg2.user_id == "user2"
    assert msg2.id == explicit_id

def test_message_unique_ids():
    """Test that auto-generated message IDs are unique."""
    msg1 = Message(text="First", user_id="user1")
    msg2 = Message(text="Second", user_id="user1")
    assert msg1.id != msg2.id

# Example concrete implementation for testing
class InMemoryMessageStore(MessageStore):
    def __init__(self):
        self.messages = []

    def write_message(self, message: Message) -> None:
        self.messages.append(message)

    def read_messages(self, start_index: int = 0) -> list[Message]:
        return self.messages[start_index:]

    def get_last_n_messages(self, n: int) -> list[Message]:
        return self.messages[-n:]

    def get_message_count(self) -> int:
        return len(self.messages)

def test_message_store_operations():
    """Test basic operations of a concrete MessageStore implementation."""
    store = InMemoryMessageStore()
    
    # Test initial state
    assert store.get_message_count() == 0
    
    # Test writing messages
    msg1 = Message(text="First message", user_id="user1")
    msg2 = Message(text="Second message", user_id="user2")
    msg3 = Message(text="Third message", user_id="user1")
    
    store.write_message(msg1)
    store.write_message(msg2)
    store.write_message(msg3)
    
    assert store.get_message_count() == 3
    
    # Test reading all messages
    all_messages = store.read_messages()
    assert len(all_messages) == 3
    assert all_messages[0].text == "First message"
    
    # Test reading from index
    messages_from_second = store.read_messages(start_index=1)
    assert len(messages_from_second) == 2
    assert messages_from_second[0].text == "Second message"
    
    # Test getting last N messages
    last_two = store.get_last_n_messages(2)
    assert len(last_two) == 2
    assert last_two[0].text == "Second message"
    assert last_two[1].text == "Third message"
