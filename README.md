![logo](https://github.com/user-attachments/assets/f98ef082-8d07-4c0f-99a1-578e1c091437)
# Ashtabula AI Framework

Ashtabula is an AI framework that integrates speech-to-text, text-to-speech, and language model capabilities using state-of-the-art models from Hugging Face, with a focus on real-time conversational AI.

## Features

- Speech-to-Text using Whisper
- Text-to-Speech using Parler TTS and SpeechT5
- Language Model using Phi-2 and other HuggingFace models
- Real-time audio processing with 1-second incremental chunks
- Sentence prediction for faster responses
- Finite State Machine (FSM) implementation for conversation flow
- Streaming transcription support
- Multiple speaker and emotion options for TTS
- Local model support for offline usage

## Installation

### From PyPI (coming soon)

```bash
# Install from PyPI
pip install ashtabula
```

### From Source

```bash
# Clone the repository
git clone https://github.com/niteshift-ai/ashtabula.git
cd ashtabula

# Install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```

### Building the Package

```bash
# Build the package
python scripts/build_package.py
```

## Important: Download Models

Before running tests or using the application, you must download the required models:

```bash
# Download all required models
python -m scripts.download_models
```

This will download:
- Whisper model for speech-to-text
- Parler TTS model for text-to-speech
- Phi-2 model for language model capabilities

The models will be saved locally in the `models/` directory:
- `models/whisper/whisper/` - Whisper model files
- `models/parler/parler/` - Parler TTS model files
- `models/phi/phi/` - Phi-2 model files

## Running Tests

After downloading the models, you can run the tests:

```bash
# Run all tests
pytest

# Run specific test files
pytest tests/providers/hugging_face/test_whisper_stt.py
pytest tests/providers/hugging_face/test_parler_tts.py
pytest tests/providers/hugging_face/test_hf_llm.py
```

Note: If you try to run tests without downloading the models first, the tests will be skipped with a message indicating that you need to run the download script.

## Usage

```python
from ashtabula.providers.hugging_face.whisper_sst import HFWhisperSTTProvider, HFWhisperConfig
from ashtabula.providers.hugging_face.parler_tts import ParlerTTSProvider
from ashtabula.providers.hugging_face.hf_llm import HFLLMProvider

# Initialize providers
stt_provider = HFWhisperSTTProvider(HFWhisperConfig())
tts_provider = ParlerTTSProvider()
llm_provider = HFLLMProvider()

# Speech-to-Text
text = await stt_provider.transcribe_file("audio.wav")

# Text-to-Speech
audio_path = tts_provider.synthesize(
    "Hello, world!",
    speaker="Thomas",
    emotion="happy"
)

# Language Model
response = await llm_provider.generate(
    "What is the capital of France?",
    temperature=0.7
)
```

## Architecture

Ashtabula uses a state machine design to model the conversation flow:

1. **IDLE**: No active conversation
2. **LISTENING**: Receiving audio but no speech detected
3. **SPEECH_ACTIVE**: Active speech being transcribed
4. **PROCESSING_UTTERANCE**: Processing complete utterance
5. **GENERATING_RESPONSE**: Generating response with LLM
6. **SPEAKING**: Delivering response to user
7. **INTERRUPTED**: User interrupted the flow

The system processes 1-second audio chunks, performs predictions on partial utterances, and generates responses based on these predictions for faster interaction.

## License

MIT License - see LICENSE file for details.
