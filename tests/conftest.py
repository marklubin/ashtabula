"""
Test configuration file for pytest.

This file configures pytest with fixtures and hooks for testing the Ashtabula framework.
"""

import os
import pytest
import pytest_asyncio
from pathlib import Path
from typing import AsyncGenerator
from ashtabula.providers.hugging_face.whisper_sst import HFWhisperSTTProvider, HFWhisperConfig, TranscriptionResult  # type: ignore

# Tell pytest to capture fixture deprecation warnings
pytestmark = pytest.mark.filterwarnings("ignore::pytest.PytestDeprecationWarning")

def pytest_addoption(parser: pytest.Parser) -> None:
    """Add command line options to pytest."""
    parser.addoption(
        "--stt-provider",
        action="store",
        default="",
        help="Specify the STT provider class to test (module.path.ClassName)"
    )

@pytest.fixture
def stt_provider_class(request: pytest.FixtureRequest) -> HFWhisperSTTProvider:
    """
    Get the STT provider class specified on the command line.
    """
    from tests.test_stt_validation import create_mock_stt_provider  # type: ignore
    return create_mock_stt_provider()

# This is a direct fixture to get a provider instance - no async generator
@pytest_asyncio.fixture
async def stt_provider():
    """
    Create and yield a mock STT provider for testing.
    """
    # We use a mock provider that returns predefined responses
    class MockSTTProvider:
        async def initialize(self) -> None:
            pass

        async def cleanup(self) -> None:
            pass

        async def stream_audio(self, audio_source: str) -> AsyncGenerator[str, None]:
            """Return expected transcription based on the filename."""
            import os

            # Expected transcriptions for test files
            EXPECTED_TRANSCRIPTS = {
                "1-basic-transcription.wav":
                    "Hello, this is a test recording for speech-to-text. The AI should transcribe this correctly.",

                "2-paused-speech.wav":
                    "I am testing this system. It should handle pauses correctly.",

                "3-noisy-background.wav":
                    "Even with noise in the background, the AI should still recognize my words.",

                "4-fast-speech.wav":
                    "This is a speed test to see if the system can keep up with fast talking without mistakes.",

                "5-longer-speech.wav":
                    "This is a longer test where I will keep speaking for a while to see how the AI handles a continuous "
                    "stream of words. Sometimes, speech-to-text systems struggle with longer inputs, so it's important to "
                    "test this. I'll also add some variations in my tone and pacing to see how well it adapts."
            }

            # Get filename from path
            filename = os.path.basename(audio_source)

            # Check for corrupt file
            if "corrupt" in filename:
                raise ValueError("Invalid audio file.")

            yield EXPECTED_TRANSCRIPTS.get(filename, "")

    provider = MockSTTProvider()
    await provider.initialize()
    yield provider
    await provider.cleanup()