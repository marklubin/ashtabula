"""
Speech-to-Text (STT) Provider Interface

This module defines the abstract base class for all STT providers in the Ashtabula framework.
Any STT implementation must inherit from and implement the STTProvider interface.

The interface requires implementing asynchronous streaming audio transcription,
allowing for real-time speech processing with partial results.
"""

import asyncio
from typing import Any, AsyncGenerator, Optional, Union


class STTProvider:
    """
    Abstract base class for Speech-to-Text providers.
    
    This interface defines the common API that all STT implementations must support.
    Each provider can implement additional functionality beyond this interface.
    """
    
    async def initialize(self) -> None:
        """
        Initialize the STT provider asynchronously.
        
        This method should be called before using the provider.
        It may download models, establish connections, etc.
        
        Implementation is optional if no async initialization is needed.
        """
        pass  # Optional implementation
    
    async def cleanup(self) -> None:
        """
        Clean up resources used by the STT provider.
        
        This method should be called when the provider is no longer needed.
        It should release any resources, close connections, etc.
        
        Implementation is optional if no cleanup is needed.
        """
        pass  # Optional implementation
    
    async def stream_audio(self, audio_source: str) -> AsyncGenerator[Any, None]:
        """
        Stream audio from the given source and transcribe it incrementally.
        
        This is the primary method that all STT providers must implement.
        It should process the audio source and yield transcription results
        as they become available.
        
        Args:
            audio_source: Path to the audio file or stream identifier
            
        Yields:
            Transcription results which may be:
            - String containing transcribed text
            - Dict with at least a 'text' key
            - Any object with a 'text' attribute
            
        Raises:
            ValueError: If the audio source is invalid or unsupported
            RuntimeError: If transcription fails for any reason
        """
        raise NotImplementedError("STT providers must implement stream_audio")
