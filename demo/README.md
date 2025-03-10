# Ashtabula Demo Applications

This directory contains demo applications that showcase how to use the Ashtabula library in various scenarios.

## CLI Demo

The `demo_cli.py` script provides a command-line interface for conversational interaction with Ashtabula's AI capabilities. It demonstrates:

- Real-time speech processing using sounddevice
- 1-second audio chunk processing
- The state machine conversation flow
- Prediction-based response generation
- Text-to-speech output

### Requirements

In addition to the main Ashtabula dependencies, the demo requires:

```bash
# Always use uv for installing dependencies
uv add sounddevice numpy
# Or for dev dependencies
uv pip install sounddevice numpy
```

### Usage

```bash
# Run the CLI demo
uv run python demo_cli.py

# Enable debug logging
uv run python demo_cli.py --debug
```

**IMPORTANT:** Always use `uv run` to execute Python commands in this project.

### How It Works

1. The application listens for audio input from your microphone in 1-second chunks
2. Each chunk is processed by the Whisper STT model to generate transcriptions
3. Partial transcriptions are used to predict the full sentence
4. When silence is detected, the final transcription is passed to the LLM
5. The LLM generates a response which is converted to speech by the TTS model
6. The response is played through your speakers

### Testing

The demo comes with tests that verify its functionality:

```bash
# Run tests (always use uv)
uv run pytest test_demo_cli.py -v
```