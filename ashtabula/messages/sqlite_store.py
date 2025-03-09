import sqlite3
from pathlib import Path
from typing import List
from .message_store import Message, MessageStore

class SQLiteMessageStore(MessageStore):
    """A MessageStore implementation backed by SQLite."""
    
    def __init__(self, db_path: str):
        """
        Initialize the SQLite message store.
        
        Args:
            db_path (str): Path to the SQLite database file
        """
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the database schema if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def write_message(self, message: Message) -> None:
        """
        Write a message to the SQLite store.
        
        Args:
            message (Message): The message to store
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO messages (id, text, user_id) VALUES (?, ?, ?)",
                (message.id, message.text, message.user_id)
            )
            conn.commit()

    def read_messages(self, start_index: int = 0) -> List[Message]:
        """
        Read messages from the store starting at the given index.
        
        Args:
            start_index (int): Starting index for reading messages
            
        Returns:
            List[Message]: List of messages from the starting index
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT id, text, user_id FROM messages ORDER BY created_at LIMIT -1 OFFSET ?",
                (start_index,)
            )
            return [
                Message(id=row[0], text=row[1], user_id=row[2])
                for row in cursor.fetchall()
            ]

    def get_last_n_messages(self, n: int) -> List[Message]:
        """
        Get the last N messages from the store.
        
        Args:
            n (int): Number of messages to retrieve
            
        Returns:
            List[Message]: The last N messages in chronological order
        """
        with sqlite3.connect(self.db_path) as conn:
            # Get total count first
            cursor = conn.execute("SELECT COUNT(*) FROM messages")
            total = cursor.fetchone()[0]
            
            # Calculate the offset to get the last n messages
            offset = max(0, total - n)
            
            cursor = conn.execute(
                "SELECT id, text, user_id FROM messages ORDER BY created_at LIMIT ? OFFSET ?",
                (n, offset)
            )
            return [
                Message(id=row[0], text=row[1], user_id=row[2])
                for row in cursor.fetchall()
            ]

    def get_message_count(self) -> int:
        """
        Get the total number of messages in the store.
        
        Returns:
            int: Total number of messages
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM messages")
            return cursor.fetchone()[0]

    def clear(self) -> None:
        """Clear all messages from the store."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM messages")
            conn.commit()
