from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
import uuid

@dataclass
class Message:
    """Data class representing a message in the conversation."""
    text: str
    user_id: str
    id: str = None

    def __post_init__(self):
        """Generate a unique ID if none was provided."""
        if self.id is None:
            self.id = str(uuid.uuid4())

class MessageStore(ABC):
    """Abstract base class for storing conversation messages."""

    @abstractmethod
    def write_message(self, message: Message) -> None:
        """
        Write a new message to the store.

        Args:
            message (Message): The message to store.
        """
        pass

    @abstractmethod
    def read_messages(self, start_index: int = 0) -> List[Message]:
        """
        Read messages incrementally from a starting index.

        Args:
            start_index (int): The index to start reading from.

        Returns:
            List[Message]: List of messages from the starting index.
        """
        pass

    @abstractmethod
    def get_last_n_messages(self, n: int) -> List[Message]:
        """
        Retrieve the last N messages from the store.

        Args:
            n (int): Number of messages to retrieve.

        Returns:
            List[Message]: The last N messages.
        """
        pass

    @abstractmethod
    def get_message_count(self) -> int:
        """
        Get the total number of messages in the store.

        Returns:
            int: Total number of messages.
        """
        pass
