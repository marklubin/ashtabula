"""
Hugging Face Whisper STT Provider for Ashtabula

This module implements speech-to-text functionality using Hugging Face's Transformers
implementation of OpenAI's Whisper model.

Key Features:
- Streaming transcription with incremental results
- Configurable model size for balancing accuracy vs. speed
- Language detection and multi-language support
- Optimized for CPU or GPU inference

Usage:
    provider = HFWhisperSTTProvider(
        model_size="small",
        device="auto",
        compute_dtype="float16"
    )
    
    # Process audio file with streaming results
    async for result in provider.stream_audio("/path/to/audio.wav"):
        print(result.text)
"""

import os
import asyncio
import logging
import tempfile
import numpy as np
from typing import Dict, List, Any, AsyncGenerator, Optional, Union, Tuple, Coroutine
from dataclasses import dataclass, field
import librosa
import torch
import sys
import subprocess

# Configure basic logging
logging.basicConfig(level=logging.INFO)

# Import the abstract base class
from ashtabula.stt import STTProvider
# Import transformers (ignoring missing type stubs)
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq  # type: ignore

@dataclass
class HFWhisperConfig:
    """
    Configuration parameters for Hugging Face Whisper STT.

    Attributes:
        model_size: Size of the Whisper model ('tiny', 'base', 'small', 'medium', 'large-v2', 'large-v3')
        device: Device to run the model on ('cpu', 'cuda', 'mps', 'auto')
        compute_dtype: Data type for computation ('float16', 'float32', 'int8')
        language: Language code (e.g., 'en', 'fr') or None for auto-detection
        chunk_length_s: Chunk size in seconds for processing long audio
        batch_size: Batch size for processing
        return_timestamps: Whether to return timestamps
        log_level: Logging level for the STT provider
    """
    model_size: str = "small"
    device: str = "auto"
    compute_dtype: str = "float16"
    language: Optional[str] = None
    chunk_length_s: float = 30.0
    batch_size: int = 1
    return_timestamps: bool = False
    log_level: int = logging.INFO


@dataclass
class TranscriptionResult:
    """
    Result from STT transcription containing text and metadata.

    Attributes:
        text: The transcribed text
        confidence: Confidence score (0-1) if available
        is_final: Whether this is a final result
        start_time: Start time of this segment in seconds
        end_time: End time of this segment in seconds
        language: Detected language code if available
        metadata: Additional provider-specific metadata
    """
    text: str
    confidence: Optional[float] = None
    is_final: bool = False
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    language: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class HFWhisperSTTProvider(STTProvider):
    """
    Speech-to-Text provider using Hugging Face's Whisper implementation.

    This provider implements the STTProvider interface for the Ashtabula
    framework and offers high-quality, streaming transcription.
    """

    def __init__(self, config: Union[HFWhisperConfig, Dict[str, Any]]) -> None:
        """
        Initialize the Hugging Face Whisper STT provider with given configuration.

        Args:
            config: Either a HFWhisperConfig object or a dictionary with configuration parameters

        Raises:
            ImportError: If the transformers package is not installed
            ValueError: If required parameters are missing or invalid
            RuntimeError: If the model fails to load
        """
        # Check transformers availability
        try:
            import transformers
            TRANSFORMERS_AVAILABLE = True
        except ImportError:
            TRANSFORMERS_AVAILABLE = False

        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "The transformers package is required. "
                "Install it with: pip install transformers torch librosa"
            )

        # Handle dict or config object
        if isinstance(config, dict):
            self.config = HFWhisperConfig(**config)
        else:
            self.config = config

        # Configure logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(self.config.log_level)

        # Validate configuration
        self._validate_config()

        # Model will be initialized lazily
        self.processor: Optional[AutoProcessor] = None
        self.model: Optional[AutoModelForSpeechSeq2Seq] = None
        self.device: Optional[str] = None
        self._is_initialized: bool = False

    def _validate_config(self) -> None:
        """
        Validate the configuration parameters.

        Raises:
            ValueError: If any parameters are invalid
        """
        valid_model_sizes = [
            "tiny", "base", "small", "medium",
            "large-v1", "large-v2", "large-v3"
        ]

        if self.config.model_size not in valid_model_sizes:
            raise ValueError(
                f"Invalid model_size: {self.config.model_size}. "
                f"Must be one of: {', '.join(valid_model_sizes)}"
            )

        valid_devices = ["cpu", "cuda", "mps", "auto"]
        if self.config.device not in valid_devices:
            raise ValueError(
                f"Invalid device: {self.config.device}. "
                f"Must be one of: {', '.join(valid_devices)}"
            )

        valid_compute_dtypes = ["float16", "float32", "int8"]
        if self.config.compute_dtype not in valid_compute_dtypes:
            raise ValueError(
                f"Invalid compute_dtype: {self.config.compute_dtype}. "
                f"Must be one of: {', '.join(valid_compute_dtypes)}"
            )

    async def initialize(self) -> None:
        """
        Initialize the model asynchronously.

        This is done lazily to avoid loading the model at import time.
        """
        if self._is_initialized:
            return

        self.logger.info(f"Initializing Hugging Face Whisper model: {self.config.model_size}")

        # Initialize model in a separate thread to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._initialize_model)

        self._is_initialized = True

    def _initialize_model(self) -> None:
        """
        Initialize the Whisper model with the provided configuration.

        Raises:
            RuntimeError: If model initialization fails
        """
        try:
            # Determine device
            device = self.config.device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

            # Use local model path instead of downloading from HF Hub
            local_model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 
                                        "models", "whisper", "whisper")
            
            self.logger.info(f"Loading local Whisper model from: {local_model_path}")
            
            # Load processor from local path
            self.processor = AutoProcessor.from_pretrained(local_model_path)

            # Always use float32 for stability across devices
            # This avoids dtype mismatches which can cause errors
            torch_dtype = torch.float32

            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                local_model_path,
                torch_dtype=torch_dtype,
                local_files_only=True,
                low_cpu_mem_usage=True,
                use_safetensors=True
            ).to(device)

            self.device = device

        except Exception as e:
            raise RuntimeError(f"Hugging Face Whisper initialization failed: {e}")

    async def cleanup(self) -> None:
        """
        Clean up resources used by the provider.
        """
        # Free memory
        self._is_initialized = False
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None

        # Force garbage collection
        import gc
        gc.collect()
        if hasattr(torch, 'cuda') and torch.cuda.is_available():
            torch.cuda.empty_cache()

    async def stream_audio(self, audio_source: str) -> AsyncGenerator[Any, None]:  # type: ignore[override]
        """
        Stream audio from a file and transcribe it incrementally.

        Args:
            audio_source: Path to the audio file

        Yields:
            TranscriptionResult objects with transcription progress

        Raises:
            ValueError: If the audio file is invalid
            RuntimeError: If transcription fails
        """
        # Ensure model is initialized
        if not self._is_initialized:
            await self.initialize()

        try:
            # Load audio file
            try:
                loop = asyncio.get_event_loop()
                audio, sr = await loop.run_in_executor(
                    None,
                    lambda: librosa.load(audio_source, sr=16000)  # Resample to 16kHz
                )
            except Exception as e:
                raise ValueError(f"Invalid audio file: {audio_source}. Error: {str(e)}") from e

            # Process audio in chunks
            all_text: List[str] = []

            # For now, just process the entire audio file at once
            try:
                # Ensure audio is float32 for consistent dtype
                if audio.dtype != np.float32:
                    audio = audio.astype(np.float32)

                # Get features
                if self.processor is None:
                    raise RuntimeError("Processor not initialized")
                input_features = self.processor(
                    audio,
                    sampling_rate=sr,
                    return_tensors="pt"
                ).input_features.to(self.device)

                # Forced language if specified
                forced_decoder_ids: Optional[List[int]] = None
                if self.config.language and self.processor is not None:
                    forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                        language=self.config.language, task="transcribe"
                    )

                # Generate transcription
                if self.model is None:
                    raise RuntimeError("Model not initialized")
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        input_features,
                        forced_decoder_ids=forced_decoder_ids
                    )

                # Decode the transcription
                if self.processor is None:
                    raise RuntimeError("Processor not initialized")
                transcription = self.processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True
                )[0]

                # Create a result
                yield TranscriptionResult(
                    text=transcription.strip(),
                    is_final=True,
                    language=self.config.language
                )

            except Exception as e:
                raise RuntimeError(f"Transcription failed: {e}")

        except Exception as e:
            if "Invalid audio file" in str(e):
                raise  # Re-raise ValueError for invalid files
            else:
                raise RuntimeError(f"Transcription failed: {e}")

    async def transcribe_file(self, audio_path: str) -> str:
        """
        Transcribe an entire audio file and return the complete text.
        This is a convenience method for batch processing.

        Args:
            audio_path: Path to the audio file

        Returns:
            Complete transcription as a string

        Raises:
            ValueError: If the audio file is invalid
            RuntimeError: If transcription fails
        """
        complete_text: List[str] = []
        async for result in self.stream_audio(audio_path):
            if result.is_final:
                return result.text
            complete_text.append(result.text)

        return " ".join(complete_text)

def run_static_analysis() -> None:
    """Run all static analysis checks for the project."""
    print("\n🚀 Running Static Analysis for Ashtabula...\n" + "=" * 50)

    commands = [
        ("mypy (Type Checking)", "mypy ashtabula/"),
        ("ruff (Linting & Formatting)", "ruff check ashtabula/"),
        ("flake8 (General Linting)", "flake8 ashtabula/"),
        ("pylint (Code Quality Checks)", "pylint ashtabula/"),
        ("bandit (Security Scan)", "bandit -r ashtabula/"),
    ]

    for description, command in commands:
        run_command(description, command)

    print("\n✅ Static Analysis Completed.")

def run_command(description: str, command: str) -> None:
    """Runs a shell command and prints the output."""
    print(f"\n🔍 Running {description}...\n{'=' * 40}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)

if __name__ == "__main__":
    run_static_analysis()
