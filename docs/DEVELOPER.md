# Ashtabula – High-Level Design & Implementation Roadmap

This document outlines the architecture, design, and testing strategy for the **Ashtabula** real-time conversational AI pipeline. It expands upon the tasks in **TODO.md** and correlates them with the existing Python codebase described in the `uv.lock` file and the provided directory structure.

---

## 1. Overview

Ashtabula is a **streaming conversational AI system** that:

1. **Accepts audio input in real-time** over a WebSocket connection.
2. **Applies Voice Activity Detection (VAD)** to identify speech boundaries.
3. **Transcribes audio chunks** incrementally using **Whisper STT**.
4. **Manages session state** including partial transcriptions, embeddings, and responses.
5. **Uses a conversation manager** to generate or retrieve responses from a language model (e.g., Mistral, Parler, or other integrated LLM).
6. **Outputs synthetic speech** via **SpeechT5 TTS** (or other TTS providers).

Below is a conceptual diagram of the entire pipeline:

┌───────────────────────┐
│ Real-time Audio In  │ => │ WebSocket Server (WS)  │
└─────────────────────┘    └───────────────────────┘
|                         |
|   (audio frames/chunks)|
v                         v
┌──────────────────┐     ┌─────────────────────┐
│ VAD (Silero,etc.)│ =>  │ Whisper STT Service │
└──────────────────┘     └─────────────────────┘
|                         |
|(transcribed text/partial)
v                         v
┌────────────────────────────────────────┐
│        Conversation Manager           │
│(Tracks session, embeddings, etc.)     │
└────────────────────────────────────────┘
|
(final utterance identified)
v
┌────────────────┐
│ LLM Generator  │
└────────────────┘
|
(response text)
v
┌────────────────┐
│ TTS (SpeechT5) │
└────────────────┘
|
(audio out)


---

## 2. Components & Detailed Implementation

### 2.1 WebSocket Server (`ashtabula/websocket.py`)

**Purpose:** 
- Accept compressed or uncompressed audio streams over WebSocket.
- Chunk audio into configurable time intervals (e.g., 0.5s, 1s).
- Forward these chunks to the STT module.

**Implementation Guide:**
1. **WebSocket Connection Setup**: Use a Python library (e.g., `websockets` or `uvicorn` + `fastapi` websockets).
2. **Chunking Logic**: Maintain a buffer for incoming audio bytes. When the buffer duration reaches the configured chunk size, forward it to the STT module.
3. **Compression Formats**: Ensure support for formats like Opus, FLAC, WAV. Optionally use libraries such as `soundfile` or `pydub` for decoding.
4. **Error Handling**: If a chunk is corrupted, log the error and skip or request retransmission.

**Test Strategy**:
- **Unit Tests**: Mock WebSocket connections and verify chunking boundaries.
- **Integration Tests**: Send real audio data in Opus and WAV format and confirm the system can decode properly.
- **Performance Tests**: Stress test with simultaneous streams to confirm concurrency handling.

**Developer Docs**:
- **Configuration**: Document environment variables or config files (e.g., chunk size, concurrency limits).
- **API Endpoints**: Provide sample client code to establish a WebSocket connection.

---

### 2.2 Session State Management (`ashtabula/buffer.py`, `ashtabula/conversation.py`)

**Purpose**:
- Maintain incremental transcriptions, predicted sentences, embeddings, and responses for each user session.
- Clear and reset state once the utterance is finalized.

**Implementation Guide**:
1. **Buffer Structure** (`buffer.py`):
   - Track partial text from STT. 
   - Store predicted sentences along with timestamps.
2. **Conversation Context** (`conversation.py`):
   - Upon final utterance detection, compute embeddings (e.g., using a separate embeddings model or LLM).
   - Store partial responses from the LLM for real-time fallback if needed.
   - Finalize once the conversation manager chooses a best response.

**Test Strategy**:
- **Unit Tests**:
  - Validate addition/removal of partial transcriptions.
  - Validate embedding storage and retrieval.
- **Integration Tests**:
  - Simulate a conversation flow to confirm session state is updated properly over multiple utterances.
- **Edge Cases**:
  - Verify that the session resets when no speech is detected for a long interval.

**Developer Docs**:
- **Data Structures**: Explain the in-memory or DB schema used to store partial transcriptions.
- **Concurrency Handling**: Document thread safety if multiple tasks access the session buffer.

---

### 2.3 Voice Activity Detection (VAD) Integration

**Purpose**:
- Identify the start and end of speech to decide when to finalize transcriptions and trigger response generation.

**Implementation Guide**:
1. **VAD Model**:
   - Integrate a Python-based VAD library (e.g., `webrtcvad`, `pyannote`, or `silero`).
2. **Configurable Sensitivity**:
   - Expose parameters (e.g., threshold or aggressiveness) to tune the model for different background noise conditions.
3. **Segment Boundaries**:
   - Every chunk from the WebSocket is analyzed. If silence is detected beyond a threshold, finalize the utterance.

**Test Strategy**:
- **Unit Tests**:
  - Provide short audio clips with known speech boundaries and confirm correct detection.
- **Noise & Overlapping Speech**:
  - Evaluate performance in noisy or overlapping speech scenarios.
- **Integration**:
  - Check correct triggers for finalizing partial transcriptions.

**Developer Docs**:
- **Configuration**: Document the sensitivity thresholds, acceptable pause durations, etc.
- **Performance**: Provide guidelines for adjusting VAD for CPU- vs. GPU-based inference.

---

### 2.4 Whisper STT (`ashtabula/stt.py`, `ashtabula/providers/hugging_face/whisper_sst.py`)

**Purpose**:
- Convert incoming audio chunks to text in near real-time.
- Provide partial/incremental transcriptions for quick feedback.

**Implementation Guide**:
1. **Loading Whisper Model**:
   - Use a Hugging Face pipeline or direct model loading from `models/whisper/whisper`.
2. **Incremental Decoding**:
   - For each chunk, provide partial results. Append or overwrite older partial transcriptions in the session buffer.
3. **Finalization**:
   - On VAD end-of-speech, finalize the transcription for the entire utterance.

**Test Strategy**:
- **Unit Tests**:
  - Confirm correct text output for short test WAV files in `tests/test_wav`.
- **Integration Tests**:
  - Combine with WebSocket + VAD to ensure the end-to-end pipeline works for real-time scenarios.
- **Accuracy Checks**:
  - Evaluate the transcription quality and measure Word Error Rate (WER) on known test sets.

**Developer Docs**:
- **Model Config**: Document how the model is loaded and any custom pipeline parameters (e.g., `language`, `task`).
- **Performance Optimization**: Provide tips on chunk size and model GPU usage.

---

### 2.5 Conversation Manager (`ashtabula/conversation.py`)

**Purpose**:
- Orchestrate the entire pipeline flow: from partial transcription to final LLM response selection.
- Manage comparisons between the final utterance embedding and stored predicted embeddings.

**Implementation Guide**:
1. **Incremental Updates**:
   - For each partial transcription, optionally generate or retrieve partial responses for quick feedback.
2. **Final Utterance Embedding**:
   - On VAD finalization, compute the embedding of the final utterance.
3. **Prediction Matching**:
   - Compare with stored partial predictions via cosine similarity. If a match is found (`distance < K`), use the stored response. Otherwise, generate a new response.
4. **Reset Session**:
   - Once a final response is rendered by TTS, clear the session for the next utterance.

**Test Strategy**:
- **Unit Tests**:
  - Check the logic for computing embeddings and matching predictions.
- **Integration**:
  - Test a multi-utterance conversation to confirm session resets properly and transitions are correct.
- **Stress/Load**:
  - Evaluate system stability under many concurrent users and long utterances.

**Developer Docs**:
- **Embedding Method**: Document which embedding model is used and how to configure it.
- **Matching Criteria**: Include an explanation of the `K` threshold and how it can be tuned.

---

### 2.6 LLM Response Generation (`ashtabula/llm.py`, `ashtabula/providers/hugging_face/hf_llm.py`)

**Purpose**:
- Generate text responses based on the final utterance text or partial predictions.

**Implementation Guide**:
1. **LLM Integration**:
   - Use the Hugging Face Transformers library for Mistral, Parler, or custom models in `models/phi/` or `models/parler/`.
2. **Parallel Inference**:
   - For partial text, you may do a quick generation with smaller context; for final text, use the full context for best results.
3. **Fallback**:
   - If partial generation isn’t validated, the conversation manager triggers a fresh generation on final utterance.

**Test Strategy**:
- **Unit Tests**:
  - Evaluate generation with mock input. Verify structure and length constraints.
- **Integration**:
  - Confirm correct LLM usage within the conversation manager flow.
- **Performance**:
  - Test throughput and latency under concurrency.

**Developer Docs**:
- **Model Config**: Document how to specify different models (Mistral, etc.).
- **Prompt Engineering**: Provide recommended prompts or temperature settings.

---

### 2.7 Text-to-Speech (TTS) (`ashtabula/tts.py`, `ashtabula/providers/hugging_face/speech_t5_tts.py`)

**Purpose**:
- Convert the final textual response into spoken audio.

**Implementation Guide**:
1. **Model Setup**:
   - Load SpeechT5 from `models/speecht5/speecht5`, or use a fallback TTS if configured.
2. **Streaming TTS**:
   - Optionally provide streaming partial results so the user hears the response as it’s generated.
3. **Output Buffer**:
   - Send TTS output back over the WebSocket or a separate audio channel.

**Test Strategy**:
- **Unit Tests**:
  - Generate audio from short strings, then confirm that the output file is non-empty and playable.
- **Integration**:
  - End-to-end test: transcribe audio, generate response, output TTS. 
- **Quality Checks**:
  - Listen for clarity and compare with reference outputs for consistent speed and pitch.

**Developer Docs**:
- **Audio Format**: Document the sample rate, bit depth, etc.
- **Performance Tuning**: Provide best practices for GPU usage or CPU fallback.

---

## 3. End-to-End Sequence Diagram

Below is a simplified sequence diagram showing how an audio input chunk is processed until the TTS output is produced:

articipant User
Participant WS Server
Participant VAD
Participant Whisper STT
Participant Conversation Manager
Participant LLM
Participant TTS

User -> WS Server: Send audio chunk
WS Server -> VAD: Forward chunk
VAD -> WS Server: Return speech status
WS Server -> Whisper STT: Send chunk
Whisper STT -> Conversation Manager: Send partial text
Conversation Manager -> LLM: (Optional) request partial response
LLM -> Conversation Manager: Return partial response
User <- WS Server: (Optional) partial TTS audio

Note over VAD, Conversation Manager: On silence detection, finalize utterance

Conversation Manager -> LLM: Request final response
LLM -> Conversation Manager: Return final text
Conversation Manager -> TTS: Generate speech
TTS -> Conversation Manager: Return audio buffer
WS Server -> User: Stream final response audio

---

## 4. Development & Testing Guidelines

1. **Local Development**:
   - Install dependencies from `requirements.txt` or `pyproject.toml`.
   - Download model files (Whisper, SpeechT5, etc.) into `models/` with `scripts/download_models.py`.
2. **Testing**:
   - Run `pytest` in the `tests/` directory. 
   - Store test audio clips in `test-data/test_wav` for STT validation.
3. **Documentation**:
   - Maintain updated notes in `docs/` folder. 
   - Each module should have docstrings describing public classes/functions.

---

## 5. Conclusion

This **Ashtabula** note captures a high-level design, a step-by-step implementation plan, and a test strategy for all core modules. By following these guidelines and referencing the additional developer docs in `docs/` and `tests/`, you can extend, test, and maintain this real-time conversational AI pipeline effectively.