from typing import Optional
from ..llm import LLMProvider
from ..messages import MessageStore

class MessageStoreDelegateLLM(LLMProvider):
    """LLM provider that injects message history into input."""
    
    def __init__(self, 
                 delegate: LLMProvider,
                 message_store: MessageStore,
                 num_messages: int,
                 history_prefix: str = "Previous conversation:\n"):
        """
        Initialize the delegate LLM provider.
        
        Args:
            delegate: The LLM provider to delegate to
            message_store: MessageStore containing conversation history
            num_messages: Number of previous messages to include
            history_prefix: Text to prefix the message history with
        """
        self.delegate = delegate
        self.message_store = message_store
        self.num_messages = num_messages
        self.history_prefix = history_prefix

    def _format_message(self, message) -> str:
        """Format a single message for inclusion in history."""
        return f"{message.user_id}: {message.text}"

    def _get_history_text(self) -> str:
        """Get formatted message history text."""
        messages = self.message_store.get_last_n_messages(self.num_messages)
        if not messages:
            return "Current input:\n"
            
        history_lines = [self._format_message(msg) for msg in messages]
        history_text = "\n".join(history_lines)
        return f"{self.history_prefix}{history_text}\n\nCurrent input:\n"

    def generate(self,
                input_text: str,
                max_length: Optional[int] = None,
                temperature: float = 0.7,
                top_p: float = 0.95,
                top_k: int = 50) -> str:
        """
        Generate a response with injected message history.
        
        Args:
            input_text: The text to generate a response for
            max_length: Maximum length of generated response
            temperature: Sampling temperature (0.0-1.0)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            
        Returns:
            Generated text response
        """
        augmented_input = f"{self._get_history_text()}{input_text}"
        
        return self.delegate.generate(
            input_text=augmented_input,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k
        )

    def predict(self,
               partial_text: str,
               max_length: Optional[int] = None,
               temperature: float = 0.7) -> str:
        """
        Predict completion using the delegate LLM.
        
        Args:
            partial_text: The incomplete sentence to predict completion for
            max_length: Maximum length of predicted completion
            temperature: Sampling temperature (0.0-1.0)
            
        Returns:
            Predicted sentence completion
        """
        # For predictions, we don't include message history to keep completions focused
        return self.delegate.predict(
            partial_text=partial_text,
            max_length=max_length,
            temperature=temperature
        )
