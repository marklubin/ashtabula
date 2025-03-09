# Ashtabula
🚀 **Ashtabula** is an open-source Python framework built for **real-time streaming AI conversations** using Speech-to-Text (STT), predictive LLM sentence completion, buffering, and Text-to-Speech (TTS).

## Features
- 🔥 **Streaming STT input** with real-time transcription
- 🧠 **Predictive LLM-based sentence completion**
- ⏳ **Response buffering & silence detection**
- 🎙️ **TTS output with natural voice streaming**
- 🚀 **Interruptible AI** to prevent unnatural interactions

## Installation
```bash
pip install ashtabula
```

## Quick Start
```python
from ashtabula import ConversationManager

conversation = ConversationManager(
    llm_provider="openai",
    stt_provider="riva",
    tts_provider="coqui"
)

async for response in conversation.stream_audio("user_audio.wav"):
    print("AI Response:", response)
```

## Contributing
We welcome contributions! Fork the repo and submit PRs.
