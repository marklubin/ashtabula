# Voice Activity Detection (VAD) in Ashtabula

This document explains the Voice Activity Detection (VAD) system implemented in the Ashtabula conversational AI pipeline.

## Overview

Voice Activity Detection (VAD) is a crucial component that identifies the presence of speech in audio streams. In Ashtabula, it serves as the gatekeeper that determines when to process audio for transcription and when to finalize utterances.

The system is designed to:

1. Process real-time audio streams chunk by chunk
2. Detect the presence or absence of speech in each chunk
3. Track the timing of speech and silence periods
4. Identify complete speech segments with precise start and end times
5. Support different levels of sensitivity for various acoustic environments

## Available VAD Providers

Ashtabula implements a modular VAD system with multiple provider options:

### 1. PyannoteVADProvider

A high-quality VAD implementation based on the [pyannote.audio](https://github.com/pyannote/pyannote-audio) library.

**Features:**
- State-of-the-art neural network-based detection
- Excellent noise robustness
- Highly accurate speech/non-speech boundaries
- Configurable sensitivity parameters

**Usage Example:**
```python
from ashtabula.vad import PyannoteVADProvider

# Create a PyannoteVADProvider with custom settings
vad = PyannoteVADProvider(
    threshold=0.5,              # Detection sensitivity (0-1)
    min_duration_on=0.2,        # Minimum speech segment duration (seconds)
    min_duration_off=0.3        # Minimum silence segment duration (seconds)
)

# Initialize (downloads models the first time)
await vad.initialize()

# Process a chunk of audio
is_speech = await vad.process_chunk(audio_chunk, sample_rate=16000)

# Get all speech segments in a file
segments = await vad.get_speech_segments(audio_data, sample_rate=16000)
```

### 2. SimpleThresholdVADProvider

A lightweight energy-based VAD that doesn't require external models.

**Features:**
- No external dependencies (uses NumPy only)
- Configurable energy threshold
- Simple and fast processing
- Minimal resource usage

**Usage Example:**
```python
from ashtabula.vad import SimpleThresholdVADProvider

# Create a SimpleThresholdVADProvider with custom settings
vad = SimpleThresholdVADProvider(
    energy_threshold=0.01,      # Energy threshold for speech detection
    min_speech_duration=0.3,    # Minimum speech segment duration (seconds)
    min_silence_duration=0.5    # Minimum silence segment duration (seconds)
)

# Process a chunk of audio
is_speech = await vad.process_chunk(audio_chunk, sample_rate=16000)
```

## Core Features

### Speech Segment Detection

The VAD system not only detects speech in real-time but also identifies complete speech segments with precise timing:

```python
# Get speech segments in a file or buffer
segments = await vad.get_speech_segments(audio_data, sample_rate)

for segment in segments:
    print(f"Speech from {segment.start:.2f}s to {segment.end:.2f}s")
```

### Streaming Audio Processing

Process audio streams in real-time with detailed speech information:

```python
async for result in vad.stream_audio(audio_stream, sample_rate):
    if result['is_speech']:
        # Speech detected
        speech_duration = result['speech_duration']
        print(f"Speech detected, duration: {speech_duration:.2f}s")
    else:
        # Silence detected
        silence_duration = result['silence_duration']
        print(f"Silence detected, duration: {silence_duration:.2f}s")
```

## Configuration Options

### PyannoteVADProvider Options

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `threshold` | Detection sensitivity | 0.5 | 0.1-0.9 |
| `min_duration_on` | Min speech duration | 0.1s | 0.05-1.0s |
| `min_duration_off` | Min silence duration | 0.1s | 0.05-1.0s |

### SimpleThresholdVADProvider Options

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `energy_threshold` | Energy threshold | 0.01 | 0.001-0.1 |
| `min_speech_duration` | Min speech duration | 0.3s | 0.1-1.0s |
| `min_silence_duration` | Min silence duration | 0.5s | 0.1-1.0s |

## Tuning Guidelines

### Noise Environments

For environments with different noise levels:

- **Quiet environment**: Use default settings
- **Moderate noise**: Increase threshold to 0.6-0.7
- **Loud environment**: Increase threshold to 0.7-0.8 and increase min_duration_on to 0.2-0.3s

### Speech Patterns

For different speaking styles:

- **Slow, deliberate speech**: Increase min_duration_off to 0.5-1.0s
- **Fast, continuous speech**: Decrease min_duration_off to 0.1-0.2s
- **Conversational with pauses**: Use default settings

## Performance Considerations

### CPU vs. GPU

- PyannoteVADProvider performance benefits significantly from GPU acceleration
- On CPU-only systems, consider using SimpleThresholdVADProvider for lightweight applications
- For real-time applications on CPU, increase chunk size to reduce processing frequency

### Memory Usage

PyannoteVADProvider loads neural network models that require ~500MB of memory. If memory constraints are a concern, SimpleThresholdVADProvider uses minimal memory.

## Integration with Conversation Flow

The VAD component integrates with the conversation manager to trigger important state changes:

1. **Speech Start**: When speech is detected after a period of silence
2. **Speech End**: When silence is detected after a period of speech
3. **Utterance Finalization**: When silence duration exceeds a threshold, indicating the end of an utterance

These events drive the conversation flow, determining when to send audio for transcription and when to finalize utterances for processing by the language model.