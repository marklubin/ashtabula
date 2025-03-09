import os
import pytest
import tempfile
import torch
from ashtabula.providers.hugging_face.speech_t5_tts import SpeechT5TTSProvider
from ashtabula.providers.hugging_face.whisper_sst import HFWhisperSTTProvider, HFWhisperConfig
import difflib


@pytest.fixture
def tts_provider():
    """Initialize the TTS provider."""
    model_dir = os.getenv("SPEECHT5_MODEL_DIR", "microsoft/speecht5_tts")
    return SpeechT5TTSProvider(model_dir=model_dir)


@pytest.fixture
def stt_provider():
    """Initialize the Whisper STT provider."""
    config = HFWhisperConfig(
        model_size="small",
        device="auto",
        compute_dtype="float32"
    )
    return HFWhisperSTTProvider(config)


def similarity_score(expected: str, actual: str) -> float:
    """Calculate similarity score between expected and actual transcription."""
    expected_words = expected.lower().split()
    actual_words = actual.lower().split()
    matcher = difflib.SequenceMatcher(None, expected_words, actual_words)
    return matcher.ratio()


@pytest.mark.asyncio
async def test_basic_synthesis(tts_provider, stt_provider):
    """Test basic text-to-speech synthesis and validation."""
    text = "Hello, this is a test of the text-to-speech system."

    # Generate speech
    audio_path = tts_provider.synthesize(text)
    assert os.path.exists(audio_path), "Audio file was not created"

    # Save audio file to test artifacts directory
    test_output_dir = "test_artifacts"
    os.makedirs(test_output_dir, exist_ok=True)
    saved_audio_path = os.path.join(test_output_dir, "basic_synthesis.wav")
    os.rename(audio_path, saved_audio_path)

    # Validate with STT
    transcribed_text = await stt_provider.transcribe_file(saved_audio_path)

    # Compare with allowed miss rate
    score = similarity_score(text, transcribed_text)
    assert score >= 0.8, f"Transcription mismatch (similarity={score}): Expected '{text}', got '{transcribed_text}'"


@pytest.mark.asyncio
async def test_different_speakers(tts_provider, stt_provider):
    """Test synthesis with different speakers."""
    text = "This is a test of different speakers."
    speakers = ["random1", "random2", "random3"]

    test_output_dir = "test_artifacts"
    os.makedirs(test_output_dir, exist_ok=True)

    for speaker in speakers:
        tts_provider.speaker_embeddings = torch.randn(1, 512).to(tts_provider.device)
        audio_path = tts_provider.synthesize(text, output_path=f"{test_output_dir}/{speaker}.wav")
        assert os.path.exists(audio_path), f"Audio file not created for speaker {speaker}"

        transcribed_text = await stt_provider.transcribe_file(audio_path)

        score = similarity_score(text, transcribed_text)
        assert score >= 0.8, f"Transcription mismatch (similarity={score}): Expected '{text}', got '{transcribed_text}'"


@pytest.mark.asyncio
async def test_custom_output_path(tts_provider, stt_provider):
    """Test specifying a custom output path."""
    text = "This is a test with a custom output path."

    test_output_dir = "test_artifacts"
    os.makedirs(test_output_dir, exist_ok=True)
    output_path = os.path.join(test_output_dir, "custom_output_path.wav")

    audio_path = tts_provider.synthesize(text, output_path=output_path)
    assert os.path.exists(audio_path), "Audio file was not created at custom path"
    assert audio_path == output_path, "Output path does not match requested path"

    transcribed_text = await stt_provider.transcribe_file(audio_path)

    score = similarity_score(text, transcribed_text)
    assert score >= 0.8, f"Transcription mismatch (similarity={score}): Expected '{text}', got '{transcribed_text}'"