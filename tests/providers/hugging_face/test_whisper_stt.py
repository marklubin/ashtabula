"""
Tests for the Hugging Face Whisper STT Provider
"""

import os
import pytest
from ashtabula.providers.hugging_face.whisper_sst import HFWhisperSTTProvider, HFWhisperConfig

def check_whisper_model() -> bool:
    """Check if Whisper model is downloaded."""
    model_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
        "models", "whisper", "whisper"
    )
    return os.path.exists(model_path)

@pytest.fixture(scope="session", autouse=True)
def ensure_models():
    """Ensure models are downloaded before running tests."""
    if not check_whisper_model():
        pytest.skip(
            "Whisper model not found. "
            "Please run 'python -m scripts.download_models' first."
        )

@pytest.fixture
def stt_provider():
    """Create a test STT provider."""
    config = HFWhisperConfig(
        model_size="small",
        device="auto",
        compute_dtype="float32"
    )
    provider = HFWhisperSTTProvider(config)
    yield provider
    # Cleanup after tests
    import asyncio
    asyncio.run(provider.cleanup())

@pytest.mark.asyncio
async def test_basic_transcription(stt_provider):
    """Test basic audio transcription."""
    audio_path = "test-data/test_wav/1-basic-transcription.wav"
    text = await stt_provider.transcribe_file(audio_path)
    
    assert text, "Transcribed text should not be empty"
    assert isinstance(text, str), "Transcribed text should be a string"
    assert len(text) > 0, "Transcribed text should have content"

@pytest.mark.asyncio
async def test_streaming_transcription(stt_provider):
    """Test streaming transcription."""
    audio_path = "test-data/test_wav/2-paused-speech.wav"
    results = []
    
    async for result in stt_provider.stream_audio(audio_path):
        assert result.text, "Each result should have text"
        assert isinstance(result.text, str), "Result text should be a string"
        results.append(result)
    
    assert len(results) > 0, "Should have received multiple results"
    assert any(r.is_final for r in results), "Should have at least one final result"

@pytest.mark.asyncio
async def test_noisy_audio(stt_provider):
    """Test transcription of noisy audio."""
    audio_path = "test-data/test_wav/3-noisy-background.wav"
    text = await stt_provider.transcribe_file(audio_path)
    
    assert text, "Should transcribe despite noise"
    assert len(text) > 0, "Should produce non-empty transcription"

@pytest.mark.asyncio
async def test_fast_speech(stt_provider):
    """Test transcription of fast speech."""
    audio_path = "test-data/test_wav/4-fast-speech.wav"
    text = await stt_provider.transcribe_file(audio_path)
    
    assert text, "Should transcribe fast speech"
    assert len(text) > 0, "Should produce non-empty transcription"

@pytest.mark.asyncio
async def test_long_audio(stt_provider):
    """Test transcription of longer audio."""
    audio_path = "test-data/test_wav/5-longer-speech.wav"
    text = await stt_provider.transcribe_file(audio_path)
    
    assert text, "Should transcribe long audio"
    assert len(text) > 0, "Should produce non-empty transcription"

@pytest.mark.asyncio
async def test_invalid_audio_file(stt_provider):
    """Test error handling with invalid audio file."""
    with pytest.raises(ValueError):
        await stt_provider.transcribe_file("nonexistent.wav")

def test_invalid_config():
    """Test initialization with invalid config."""
    with pytest.raises(ValueError):
        HFWhisperSTTProvider(HFWhisperConfig(model_size="invalid_size"))

@pytest.mark.asyncio
async def test_cleanup(stt_provider):
    """Test cleanup method."""
    await stt_provider.cleanup()
    assert not stt_provider._is_initialized, "Provider should be uninitialized after cleanup"
    assert stt_provider.model is None, "Model should be None after cleanup"
    assert stt_provider.processor is None, "Processor should be None after cleanup"
