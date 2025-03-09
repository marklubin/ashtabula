import os
import pytest
from pathlib import Path
from ashtabula.messages import Message
from ashtabula.messages.sqlite_store import SQLiteMessageStore

@pytest.fixture
def db_path(tmp_path):
    """Fixture to provide a temporary database path."""
    return str(tmp_path / "test.db")

@pytest.fixture
def store(db_path):
    """Fixture to provide a SQLiteMessageStore instance."""
    store = SQLiteMessageStore(db_path)
    yield store
    # Cleanup
    if os.path.exists(db_path):
        os.remove(db_path)

def test_sqlite_store_initialization(db_path):
    """Test that the store initializes and creates the database file."""
    store = SQLiteMessageStore(db_path)
    assert os.path.exists(db_path)

def test_write_and_read_message(store):
    """Test writing a message and reading it back."""
    message = Message(text="Test message", user_id="user1")
    store.write_message(message)
    
    messages = store.read_messages()
    assert len(messages) == 1
    assert messages[0].text == "Test message"
    assert messages[0].user_id == "user1"
    assert messages[0].id == message.id

def test_read_messages_with_offset(store):
    """Test reading messages with an offset."""
    messages = [
        Message(text=f"Message {i}", user_id="user1")
        for i in range(5)
    ]
    
    for msg in messages:
        store.write_message(msg)
    
    # Read from index 2
    result = store.read_messages(start_index=2)
    assert len(result) == 3
    assert result[0].text == "Message 2"
    assert result[2].text == "Message 4"

def test_get_last_n_messages(store):
    """Test retrieving the last N messages."""
    messages = [
        Message(text=f"Message {i}", user_id="user1")
        for i in range(5)
    ]
    
    for msg in messages:
        store.write_message(msg)
    
    # Get last 3 messages
    result = store.get_last_n_messages(3)
    assert len(result) == 3
    assert result[0].text == "Message 2"
    assert result[1].text == "Message 3"
    assert result[2].text == "Message 4"

def test_message_count(store):
    """Test getting the total message count."""
    assert store.get_message_count() == 0
    
    messages = [
        Message(text=f"Message {i}", user_id="user1")
        for i in range(3)
    ]
    
    for msg in messages:
        store.write_message(msg)
    
    assert store.get_message_count() == 3

def test_clear_messages(store):
    """Test clearing all messages from the store."""
    messages = [
        Message(text=f"Message {i}", user_id="user1")
        for i in range(3)
    ]
    
    for msg in messages:
        store.write_message(msg)
    
    assert store.get_message_count() == 3
    store.clear()
    assert store.get_message_count() == 0

def test_messages_ordered_by_creation(store):
    """Test that messages are returned in order of creation."""
    messages = [
        Message(text=f"Message {i}", user_id="user1")
        for i in range(3)
    ]
    
    for msg in messages:
        store.write_message(msg)
    
    result = store.read_messages()
    for i, msg in enumerate(result):
        assert msg.text == f"Message {i}"

def test_database_persistence(db_path):
    """Test that messages persist across store instances."""
    # Create first store instance and add messages
    store1 = SQLiteMessageStore(db_path)
    message = Message(text="Persistent message", user_id="user1")
    store1.write_message(message)
    
    # Create new store instance with same database
    store2 = SQLiteMessageStore(db_path)
    messages = store2.read_messages()
    
    assert len(messages) == 1
    assert messages[0].text == "Persistent message"
    assert messages[0].user_id == "user1"
    assert messages[0].id == message.id
