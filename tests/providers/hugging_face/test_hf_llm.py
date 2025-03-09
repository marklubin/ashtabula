"""
Tests for the HuggingFace LLM Provider.

This test suite covers:
1. Model initialization and configuration
2. Response generation with different parameters
3. Error handling and edge cases
"""

import pytest
from unittest.mock import patch, MagicMock
import torch

from ashtabula.providers.hugging_face.hf_llm import HuggingFaceLLMProvider

@pytest.fixture
def mock_model():
    """Create a mock model."""
    model = MagicMock()
    model.generate.return_value = torch.tensor([[1, 2, 3]])
    return model

@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer."""
    tokenizer = MagicMock()
    tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
    tokenizer.decode.return_value = "Input text Generated response"
    tokenizer.eos_token_id = 0
    return tokenizer

@pytest.fixture
def provider(mock_model, mock_tokenizer):
    """Create a provider with mocked dependencies."""
    with patch("torch.cuda.is_available", return_value=False), \
         patch("ashtabula.providers.hugging_face.hf_llm.AutoModelForCausalLM") as mock_auto_model, \
         patch("ashtabula.providers.hugging_face.hf_llm.AutoTokenizer") as mock_auto_tokenizer:
        
        mock_auto_model.from_pretrained.return_value = mock_model
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        
        provider = HuggingFaceLLMProvider("test-model")
        provider.model = mock_model
        provider.tokenizer = mock_tokenizer
        return provider

class TestHuggingFaceLLMProvider:
    def test_initialization(self, mock_model, mock_tokenizer):
        """Test provider initialization."""
        with patch("torch.cuda.is_available", return_value=False), \
             patch("ashtabula.providers.hugging_face.hf_llm.AutoModelForCausalLM") as mock_auto_model, \
             patch("ashtabula.providers.hugging_face.hf_llm.AutoTokenizer") as mock_auto_tokenizer:
            
            mock_auto_model.from_pretrained.return_value = mock_model
            mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
            
            provider = HuggingFaceLLMProvider("mistralai/Mistral-7B-v0.1")
            
            mock_auto_model.from_pretrained.assert_called_once_with(
                "mistralai/Mistral-7B-v0.1",
                torch_dtype=torch.float32,
                device_map=None
            )
            mock_auto_tokenizer.from_pretrained.assert_called_once_with(
                "mistralai/Mistral-7B-v0.1"
            )
    
    def test_generate_basic_response(self, provider):
        """Test basic response generation."""
        response = provider.generate("Input text")
        
        assert provider.model.generate.called
        assert "Generated response" in response
        
        call_kwargs = provider.model.generate.call_args[1]
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["top_p"] == 0.95
        assert call_kwargs["top_k"] == 50
    
    def test_generate_with_parameters(self, provider):
        """Test generation with custom parameters."""
        response = provider.generate(
            "Test input",
            max_length=200,
            temperature=0.8,
            top_p=0.9,
            top_k=40
        )
        
        call_kwargs = provider.model.generate.call_args[1]
        assert call_kwargs["max_new_tokens"] == 200
        assert call_kwargs["temperature"] == 0.8
        assert call_kwargs["top_p"] == 0.9
        assert call_kwargs["top_k"] == 40
    
    def test_empty_input(self, provider):
        """Test handling of empty input."""
        response = provider.generate("")
        assert response == ""
        assert not provider.model.generate.called
    
    def test_generation_error(self, provider):
        """Test error handling during generation."""
        provider.model.generate.side_effect = Exception("Generation failed")
        
        with pytest.raises(RuntimeError, match="Failed to generate response"):
            provider.generate("Test input")
    
    def test_cuda_initialization(self, mock_model, mock_tokenizer):
        """Test initialization with CUDA."""
        with patch("torch.cuda.is_available", return_value=True), \
             patch("ashtabula.providers.hugging_face.hf_llm.AutoModelForCausalLM") as mock_auto_model, \
             patch("ashtabula.providers.hugging_face.hf_llm.AutoTokenizer") as mock_auto_tokenizer:
            
            mock_auto_model.from_pretrained.return_value = mock_model
            mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
            
            provider = HuggingFaceLLMProvider("test-model")
            
            mock_auto_model.from_pretrained.assert_called_once_with(
                "test-model",
                torch_dtype=torch.float16,
                device_map="auto"
            )
            assert provider.device == "cuda"

    def test_predict_basic_completion(self, provider):
        """Test basic sentence completion prediction."""
        provider.tokenizer.decode.return_value = "This is an incomplete sentence and here is the completion"
        
        completion = provider.predict("This is an incomplete sentence")
        
        assert provider.model.generate.called
        assert "completion" in completion
        
        call_kwargs = provider.model.generate.call_args[1]
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["top_p"] == 0.9  # More focused sampling for completion
        assert call_kwargs["top_k"] == 20   # More focused sampling for completion
        assert call_kwargs["max_new_tokens"] == 50  # Default tokens for completion

    def test_predict_with_parameters(self, provider):
        """Test prediction with custom parameters."""
        provider.tokenizer.decode.return_value = "Test input with custom completion"
        
        completion = provider.predict(
            "Test input",
            max_length=100,
            temperature=0.5
        )
        
        call_kwargs = provider.model.generate.call_args[1]
        assert call_kwargs["max_new_tokens"] == 100
        assert call_kwargs["temperature"] == 0.5

    def test_predict_adds_punctuation(self, provider):
        """Test that predictions add sentence-ending punctuation if missing."""
        # Test cases without punctuation
        provider.tokenizer.decode.return_value = "This is a completion without punctuation"
        completion = provider.predict("This is incomplete")
        assert completion.endswith('.')
        
        # Test cases with existing punctuation
        provider.tokenizer.decode.return_value = "This already has punctuation!"
        completion = provider.predict("This is incomplete")
        assert completion.endswith('!')  # Should preserve existing punctuation

    def test_predict_empty_input(self, provider):
        """Test prediction with empty input."""
        completion = provider.predict("")
        assert completion == ""
        assert not provider.model.generate.called

    def test_predict_error_handling(self, provider):
        """Test error handling during prediction."""
        provider.model.generate.side_effect = Exception("Prediction failed")
        
        with pytest.raises(RuntimeError, match="Failed to predict completion"):
            provider.predict("Test input")
