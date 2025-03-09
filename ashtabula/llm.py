from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def generate(self, 
                input_text: str,
                max_length: Optional[int] = None,
                temperature: float = 0.7,
                top_p: float = 0.95,
                top_k: int = 50) -> str:
        """
        Generate a response for the given input text.
        
        Args:
            input_text: The text to generate a response for
            max_length: Maximum length of generated response
            temperature: Sampling temperature (0.0-1.0)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            
        Returns:
            Generated text response
        """
        pass

    @abstractmethod
    def predict(self,
               partial_text: str,
               max_length: Optional[int] = None,
               temperature: float = 0.7) -> str:
        """
        Predict how the user will complete their current sentence.
        
        Args:
            partial_text: The incomplete sentence to predict completion for
            max_length: Maximum length of predicted completion
            temperature: Sampling temperature (0.0-1.0)
            
        Returns:
            Predicted sentence completion
        """
        pass
