"""
Test Runner for Hugging Face Whisper STT Provider

This script runs the STT validation test suite against the Hugging Face Whisper
implementation of the STT Provider interface.
"""

import os
import sys
import pytest
import asyncio
from pathlib import Path

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import provider
from ashtabula.providers.hugging_face.whisper_sst import HFWhisperSTTProvider, HFWhisperConfig


def run_tests():
    """Run the STT validation tests against the Hugging Face Whisper provider."""
    print("\n" + "="*80)
    print("Running STT Validation Tests for Hugging Face Whisper Provider")
    print("="*80 + "\n")
    
    # Create configuration for small model
    config = HFWhisperConfig(
        model_size="small",
        device="auto",
        compute_dtype="float16",
        language="en",  # Use English for tests
        chunk_length_s=30.0,
        batch_size=1,
        return_timestamps=False
    )
    
    # We need to add the conftest.py to create the --stt-provider option
    # Create a provider instance for testing
    from ashtabula.providers.hugging_face.whisper_sst import HFWhisperSTTProvider, HFWhisperConfig
    
    # Create test command
    test_command = [
        "-xvs",
        "tests/test_stt_validation.py"
    ]
    
    # Run the tests
    return pytest.main(test_command)


if __name__ == "__main__":
    # Set environment variables for better GPU performance if available
    if sys.platform == "darwin":  # macOS
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    
    # Run tests
    sys.exit(run_tests())
