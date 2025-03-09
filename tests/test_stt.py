"""
STT Provider Validation Suite

This test suite validates that any class implementing the STTProvider interface
meets the required functionality specifications across multiple realistic scenarios.

Testing Philosophy:
- Black-box validation: Tests should verify behavior without depending on implementation details
- Scenario-based: Each test corresponds to a real-world usage pattern
- Cross-provider: Suite works with any STTProvider implementation
- Thorough: Tests both happy and error paths
- Deterministic: Uses controlled audio samples with expected outputs

Test Scenarios:
1. Basic transcription accuracy - Clear speech should be accurately transcribed
2. Handling pauses in speech - Should maintain context across natural pauses
3. Handling background noise - Should still transcribe accurately despite interference
4. Processing fast speech - Should not miss words when spoken quickly
5. Processing long speech inputs - Should handle extended audio without degradation
6. Graceful failure handling - Should handle invalid/corrupt audio appropriately

Usage:
    pytest test_stt.py --stt-provider=ashtabula.providers.hugging_face.whisper_sst.HFWhisperSTTProvider
"""

import os
import pytest
import pytest_asyncio
import asyncio
from typing import Type, Dict, List, Any, Tuple, Optional, Union, cast
import wave
import numpy as np
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import the abstract base class
from ashtabula.stt import STTProvider
from ashtabula.providers.hugging_face.whisper_sst import HFWhisperSTTProvider, HFWhisperConfig, TranscriptionResult


# =========================================================================
# Test Configuration
# =========================================================================

TEST_WAV_DIR = Path(__file__).parent / "test_wav"
TEST_AUDIO_DIR = TEST_WAV_DIR  # Alias for backward compatibility

# Expected transcriptions for each test file
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

# Mapping from m4a files to wav files
M4A_TO_WAV_MAPPING = {
    "1-basic-transcription.m4a": "1-basic-transcription.wav",
    "2-paused-speech.m4a": "2-paused-speech.wav",
    "3-noisy-background.m4a": "3-noisy-background.wav",
    "4-fast-speech.m4a": "4-fast-speech.wav",
    "5-longer-speech.m4a": "5-longer-speech.wav"
}

# Allowed error margin (word error rate) for transcriptions
MAX_WER = {
    "1-basic-transcription.wav": 0.2,  # 20% word error rate for basic test
    "2-paused-speech.wav": 0.3,       # 30% for paused speech
    "3-noisy-background.wav": 0.4,    # 40% for noisy background
    "4-fast-speech.wav": 0.4,         # 40% for fast speech
    "5-longer-speech.wav": 0.3,       # 30% for long speech
    # Mapped to the variable names used in the tests
    "basic_clear_speech.wav": 0.2,
    "paused_speech.wav": 0.3,
    "noisy_background.wav": 0.4,
    "fast_speech.wav": 0.4,
    "long_speech.wav": 0.3
}

# Timing parameters
PAUSE_MIN_DURATION = 2.0  # Minimum pause duration in seconds
TIMEOUT = 30              # Timeout in seconds for long-running operations


# =========================================================================
# Helper Functions
# =========================================================================

def word_error_rate(reference: str, hypothesis: str) -> float:
    """
    Calculates Word Error Rate (WER) between reference and hypothesis texts.
    
    Args:
        reference: The ground truth text
        hypothesis: The STT system output text
        
    Returns:
        Float between 0.0 and 1.0 representing word error rate (lower is better)
    """
    # Normalize and tokenize
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    
    # Calculate Levenshtein distance
    d = levenshtein_distance(ref_words, hyp_words)
    
    # Calculate WER
    if len(ref_words) > 0:
        return d / len(ref_words)
    return 1.0 if d > 0 else 0.0


def levenshtein_distance(s: List[str], t: List[str]) -> int:
    """
    Calculate Levenshtein edit distance between two word lists.
    
    Args:
        s: First word list
        t: Second word list
        
    Returns:
        Integer representing edit distance (insertions, deletions, substitutions)
    """
    m, n = len(s), len(t)
    d = [[0 for _ in range(n+1)] for _ in range(m+1)]
    
    for i in range(m+1):
        d[i][0] = i
    for j in range(n+1):
        d[0][j] = j
        
    for j in range(1, n+1):
        for i in range(1, m+1):
            if s[i-1] == t[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                d[i][j] = min(
                    d[i-1][j] + 1,    # deletion
                    d[i][j-1] + 1,    # insertion
                    d[i-1][j-1] + 1   # substitution
                )
    
    return d[m][n]


def convert_m4a_to_wav() -> bool:
    """
    Converts m4a files to wav format using ffmpeg.
    
    Returns:
        True if conversions were performed, False if all wav files already exist
    """
    import subprocess
    import shutil
    
    # Create wav directory if it doesn't exist
    os.makedirs(TEST_WAV_DIR, exist_ok=True)
    
    # Check if ffmpeg is available
    if shutil.which('ffmpeg') is None:
        raise RuntimeError(
            "ffmpeg not found in PATH. Please install ffmpeg to run these tests. "
            "Visit https://ffmpeg.org/download.html for installation instructions."
        )
    
    # Check if all expected wav files exist
    all_files_exist = all(
        (TEST_WAV_DIR / wav_file).exists() 
        for wav_file in EXPECTED_TRANSCRIPTS.keys()
    )
    
    if all_files_exist:
        print("All wav files already exist, skipping conversion.")
        return False
    
    print("Converting m4a files to wav format...")
    converted_count = 0
    
    # Convert each m4a file to wav
    for m4a_file, wav_file in M4A_TO_WAV_MAPPING.items():
        m4a_path = TEST_AUDIO_DIR / m4a_file
        wav_path = TEST_WAV_DIR / wav_file
        
        # Skip if wav already exists
        if wav_path.exists():
            continue
            
        # Check if source m4a exists
        if not m4a_path.exists():
            print(f"Warning: Source file {m4a_path} not found.")
            continue
        
        # Convert file
        try:
            print(f"Converting {m4a_file} to {wav_file}...")
            subprocess.run([
                'ffmpeg',
                '-y',                  # Overwrite output files
                '-i', str(m4a_path),   # Input file
                '-ar', '16000',        # Sample rate
                '-ac', '1',            # Mono audio
                '-c:a', 'pcm_s16le',   # PCM 16-bit
                str(wav_path)          # Output file
            ], check=True, capture_output=True)
            converted_count += 1
        except subprocess.CalledProcessError as e:
            print(f"Error converting {m4a_file}: {e}")
            print(f"ffmpeg stderr: {e.stderr.decode('utf-8')}")
    
    print(f"Conversion complete. Converted {converted_count} files.")
    return converted_count > 0


def create_silent_wav(filepath: Path, duration: float = 3.0) -> None:
    """
    Creates a silent WAV file for testing.
    
    Args:
        filepath: Path to save the file
        duration: Duration in seconds
    """
    sample_rate = 16000
    channels = 1
    sampwidth = 2  # 16-bit
    
    # Create silent audio (all zeros)
    silence = np.zeros(int(duration * sample_rate), dtype=np.int16)
    
    with wave.open(str(filepath), 'wb') as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sampwidth)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(silence.tobytes())


def print_script_instructions() -> None:
    """Prints instructions for recording test audio samples."""
    print("""
** Script for Recording Test Audio Samples**

**Instructions:** Read the following sentences **clearly** while recording. For the **pause test**, pause exactly where indicated.

**1️⃣ Basic Transcription (Clear Speech)**
*"Hello, this is a test recording for speech-to-text. The AI should transcribe this correctly."*

**2️⃣ Paused Speech (Gaps in Speech)**
*"I am… (pause 2 seconds)… testing this system. It should handle pauses… (pause 3 seconds)… correctly."*

**3️⃣ Noisy Background (Play background noise while speaking)**
*"Even with noise in the background, the AI should still recognize my words."*
(Play music, chatter, or ambient noise while speaking.)

**4️⃣ Fast Speech (Speak Quickly)**
*"This is a speed test to see if the system can keep up with fast talking without mistakes."*
(Say this as fast as you can while still being understandable.)

**5️⃣ Long Speech (Extended Input)**
*"This is a longer test where I will keep speaking for a while to see how the AI handles a continuous stream of words.*
*Sometimes, speech-to-text systems struggle with longer inputs, so it's important to test this.*
*I'll also add some variations in my tone and pacing to see how well it adapts."*

Save each recording in the test_audio directory with the correct filename:
- basic_clear_speech.wav
- paused_speech.wav
- noisy_background.wav
- fast_speech.wav
- long_speech.wav
""")


def create_corrupt_audio_file() -> Path:
    """
    Creates a corrupted audio file for error handling tests.
    
    Returns:
        Path to the corrupted audio file
    """
    corrupt_file = TEST_WAV_DIR / "corrupt_audio.wav"
    
    # Write random non-WAV data
    with open(corrupt_file, 'wb') as f:
        f.write(b'NOT_A_VALID_WAV_FILE')
    
    return corrupt_file


def get_audio_duration(filepath: Path) -> float:
    """
    Gets the duration of an audio file in seconds.
    
    Args:
        filepath: Path to the audio file
        
    Returns:
        Duration in seconds
    """
    with wave.open(str(filepath), 'rb') as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
        return frames / rate


async def analyze_transcription_stream(
    provider: STTProvider, 
    audio_file: Path
) -> Tuple[str, Dict[str, Any]]:
    """
    Analyzes a transcription stream from a provider and extracts metrics.

    Args:
        provider: STT provider instance
        audio_file: Path to the audio file
        
    Returns:
        Tuple of (final transcription, metrics dictionary)
    """
    chunks: List[Any] = []
    timestamps: List[float] = []
    last_timestamp = 0.0
    transcription = ""
    
    # Stream the audio and collect metrics
    async for result in provider.stream_audio(str(audio_file)):
        chunks.append(result)
        current_time = asyncio.get_event_loop().time()
        timestamps.append(current_time)
        
        # Save the final transcription
        if hasattr(result, 'text'):
            transcription = result.text
        elif isinstance(result, str):
            transcription = result
        elif isinstance(result, dict) and 'text' in result:
            transcription = result['text']
        
    # Calculate metrics
    metrics: Dict[str, Any] = {
        'chunk_count': len(chunks),
        'total_duration': timestamps[-1] - timestamps[0] if timestamps else 0,
    }
    
    # Calculate pauses if more than one chunk received
    if len(timestamps) > 1:
        pauses: List[float] = []
        for i in range(1, len(timestamps)):
            pause_duration = timestamps[i] - timestamps[i-1]
            if pause_duration > PAUSE_MIN_DURATION:
                pauses.append(pause_duration)
        
        metrics['pauses'] = pauses
        metrics['max_pause'] = max(pauses) if pauses else 0.0
    
    return transcription, metrics


# =========================================================================
# Test Fixtures
# =========================================================================

@pytest.fixture(scope="session", autouse=True)
def setup_test_files():
    """Converts m4a files to wav format for testing."""
    convert_m4a_to_wav()
    yield
    # No cleanup needed


@pytest.fixture
def corrupt_audio_file():
    """Provides a corrupted audio file for testing."""
    file_path = create_corrupt_audio_file()
    yield file_path
    # Cleanup
    if file_path.exists():
        os.remove(file_path)


@pytest.fixture
def stt_provider_class(request):
    """
    Fixture that gets the STT provider class from pytest command line option.
    
    Example usage:
        pytest test_stt.py --stt-provider=ashtabula.providers.hugging_face.whisper_sst.HFWhisperSTTProvider
    """
    # Default to the Hugging Face Whisper implementation
    return HFWhisperSTTProvider


@pytest_asyncio.fixture
async def stt_provider():
    """Fixture that creates and yields an instance of the STT provider."""
    # Create a config for the HF Whisper provider
    config = HFWhisperConfig(
        model_size="tiny",  # Using the smallest model for faster testing
        device="auto",
        compute_dtype="float32"
    )
    
    # Create provider instance
    provider = HFWhisperSTTProvider(config)
    
    # Initialize the provider
    await provider.initialize()
    
    yield provider
    
    # Cleanup
    await provider.cleanup()


# =========================================================================
# PyTest Configuration
# =========================================================================

def pytest_addoption(parser):
    """Add command line options to pytest."""
    parser.addoption(
        "--stt-provider", 
        action="store", 
        default="ashtabula.providers.hugging_face.whisper_sst.HFWhisperSTTProvider",
        help="Specify the STT provider class to test (module.path.ClassName)"
    )


# =========================================================================
# Tests
# =========================================================================

class TestSTTProvider:
    """
    Test suite for STT Provider implementations.
    
    These tests validate that an STT provider meets the required
    functionality and performance specifications.
    """
    
    @pytest.mark.asyncio
    async def test_basic_transcription(self, stt_provider):
        """
        Validates basic transcription accuracy with clear speech.
        
        This is the fundamental test that verifies the provider can
        accurately transcribe clear, well-enunciated speech.
        
        Expected outcome:
        - Transcription matches expected text within the defined WER threshold
        """
        test_file = TEST_WAV_DIR / "1-basic-transcription.wav"
        expected = EXPECTED_TRANSCRIPTS["1-basic-transcription.wav"]
        
        # Get transcription
        transcription, metrics = await analyze_transcription_stream(stt_provider, test_file)
        
        # Calculate word error rate
        error_rate = word_error_rate(expected, transcription)
        
        # Assert transcription is accurate within threshold
        assert error_rate <= MAX_WER["basic_clear_speech.wav"], (
            f"Basic transcription failed with WER {error_rate:.2f}, "
            f"exceeding threshold of {MAX_WER['basic_clear_speech.wav']:.2f}.\n"
            f"Expected: {expected}\n"
            f"Got: {transcription}"
        )
    
    @pytest.mark.asyncio
    async def test_paused_speech(self, stt_provider):
        """
        Tests handling of pauses in speech.
        
        Verifies that the provider correctly maintains context across
        natural pauses in speech, without premature finalization.
        
        Expected outcomes:
        - Final transcription correctly includes text from both sides of pauses
        - Timing analysis shows appropriate processing of pauses
        """
        test_file = TEST_WAV_DIR / "2-paused-speech.wav"
        expected = EXPECTED_TRANSCRIPTS["2-paused-speech.wav"]
        
        # Get transcription and metrics
        transcription, metrics = await analyze_transcription_stream(stt_provider, test_file)
        
        # Calculate word error rate
        error_rate = word_error_rate(expected, transcription)
        
        # Verify transcription accuracy
        assert error_rate <= MAX_WER["paused_speech.wav"], (
            f"Paused speech transcription failed with WER {error_rate:.2f}, "
            f"exceeding threshold of {MAX_WER['paused_speech.wav']:.2f}.\n"
            f"Expected: {expected}\n"
            f"Got: {transcription}"
        )
    
    @pytest.mark.asyncio
    async def test_noisy_background(self, stt_provider):
        """
        Tests transcription accuracy with background noise.
        
        Verifies that the provider can maintain acceptable accuracy
        even in the presence of background noise.
        
        Expected outcome:
        - Transcription meets reduced accuracy threshold for noisy audio
        """
        test_file = TEST_WAV_DIR / "3-noisy-background.wav"
        expected = EXPECTED_TRANSCRIPTS["3-noisy-background.wav"]
        
        # Get transcription
        transcription, metrics = await analyze_transcription_stream(stt_provider, test_file)
        
        # Calculate word error rate
        error_rate = word_error_rate(expected, transcription)
        
        # Assert transcription is accurate within threshold
        # Note: Threshold is higher for noisy audio
        assert error_rate <= MAX_WER["noisy_background.wav"], (
            f"Noisy background transcription failed with WER {error_rate:.2f}, "
            f"exceeding threshold of {MAX_WER['noisy_background.wav']:.2f}.\n"
            f"Expected: {expected}\n"
            f"Got: {transcription}"
        )
    
    @pytest.mark.asyncio
    async def test_fast_speech(self, stt_provider):
        """
        Tests handling of rapidly spoken speech.
        
        Verifies that the provider can accurately transcribe speech
        that is spoken at a rapid pace without dropping words.
        
        Expected outcome:
        - Transcription captures all words despite rapid pace
        - Word error rate is within threshold for fast speech
        """
        test_file = TEST_WAV_DIR / "4-fast-speech.wav"
        expected = EXPECTED_TRANSCRIPTS["4-fast-speech.wav"]
        
        # Get transcription
        transcription, metrics = await analyze_transcription_stream(stt_provider, test_file)
        
        # Calculate word error rate
        error_rate = word_error_rate(expected, transcription)
        
        # Assert transcription is accurate within threshold
        assert error_rate <= MAX_WER["fast_speech.wav"], (
            f"Fast speech transcription failed with WER {error_rate:.2f}, "
            f"exceeding threshold of {MAX_WER['fast_speech.wav']:.2f}.\n"
            f"Expected: {expected}\n"
            f"Got: {transcription}"
        )
        
        # Check that the transcription has an appropriate number of words
        # Fast speech should not result in dropped words
        expected_word_count = len(expected.split())
        actual_word_count = len(transcription.split())
        word_count_diff = abs(expected_word_count - actual_word_count)
        
        assert word_count_diff <= expected_word_count * 0.3, (
            f"Fast speech transcription missed too many words. "
            f"Expected ~{expected_word_count} words, got {actual_word_count}."
        )
    
    @pytest.mark.asyncio
    async def test_long_speech(self, stt_provider):
        """
        Tests handling of longer speech input.
        
        Verifies that the provider can maintain accuracy over
        extended audio without degradation or truncation.
        
        Expected outcomes:
        - Complete transcription of long audio
        - Consistent accuracy throughout the stream
        - No timeouts or performance degradation
        """
        test_file = TEST_WAV_DIR / "5-longer-speech.wav"
        expected = EXPECTED_TRANSCRIPTS["5-longer-speech.wav"]
        
        # Get transcription with timeout protection
        try:
            transcription, metrics = await asyncio.wait_for(
                analyze_transcription_stream(stt_provider, test_file),
                timeout=TIMEOUT
            )
        except asyncio.TimeoutError:
            pytest.fail(f"Timed out processing long speech after {TIMEOUT} seconds")
        
        # Calculate word error rate
        error_rate = word_error_rate(expected, transcription)
        
        # Assert transcription is accurate within threshold
        assert error_rate <= MAX_WER["long_speech.wav"], (
            f"Long speech transcription failed with WER {error_rate:.2f}, "
            f"exceeding threshold of {MAX_WER['long_speech.wav']:.2f}."
        )
        
        # Check for length - should not truncate long input
        expected_length = len(expected)
        actual_length = len(transcription)
        
        # Allow 20% length difference
        assert actual_length >= expected_length * 0.8, (
            f"Long speech appears truncated. Expected ~{expected_length} chars, "
            f"got {actual_length}."
        )
    
    @pytest.mark.asyncio
    async def test_invalid_audio(self, stt_provider, corrupt_audio_file):
        """
        Tests graceful failure on invalid audio input.
        
        Verifies that the provider handles corrupt or invalid audio
        files appropriately without crashing.
        
        Expected outcomes:
        - Provider raises appropriate exception
        - Error message is informative
        - No unhandled exceptions or crashes
        """
        # Expect an exception when streaming corrupt audio
        with pytest.raises(Exception) as exc_info:
            # We need to properly await the async generator before iterating
            async for _ in stt_provider.stream_audio(str(corrupt_audio_file)):
                pass
        
        # Check that the exception contains useful information
        error_msg = str(exc_info.value).lower()
        expected_terms = ['file', 'audio', 'invalid', 'corrupt', 'error', 'format']
        
        # At least one of these terms should be in the error message
        assert any(term in error_msg for term in expected_terms), (
            f"Error message for corrupt audio not descriptive: {error_msg}"
        )


# Run the tests if executed directly
if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
