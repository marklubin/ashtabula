from typing import Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ...llm import LLMProvider

class HuggingFaceLLMProvider(LLMProvider):
    """
    A generic HuggingFace model provider that can work with any model ID.
    Handles model loading and generation using HuggingFace transformers.
    """
    
    def __init__(self, model_id: str):
        """
        Initialize provider with a HuggingFace model ID.
        
        Args:
            model_id: HuggingFace model identifier (e.g. 'mistralai/Mistral-7B-v0.1', 'meta-llama/Llama-2-7b')
        """
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        if self.device == "cpu":
            self.model = self.model.to(self.device)

    def generate(self,
                input_text: str,
                max_length: Optional[int] = None,
                temperature: float = 0.7,
                top_p: float = 0.95,
                top_k: int = 50) -> str:
        """
        Generate a response using the loaded model.
        
        Args:
            input_text: Text to generate a response for
            max_length: Maximum length of generated response
            temperature: Sampling temperature (0.0-1.0)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            
        Returns:
            Generated text response
        """
        if not input_text.strip():
            return ""
            
        try:
            # Tokenize input
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt"
            )
            input_ids = inputs["input_ids"].to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    max_new_tokens=max_length,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode and return response
            full_response = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )
            
            # Remove the input text from response
            response = full_response[len(input_text):].strip()
            return response
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate response: {str(e)}")
