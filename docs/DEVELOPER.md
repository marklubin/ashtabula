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

### 2.1 WebSocket Server (`ashtabula/websocket.py`) ✅ IMPLEMENTED

**Purpose:** 
- Accept compressed or uncompressed audio streams over WebSocket.
- Chunk audio into configurable time intervals (e.g., 0.5s, 1s).
- Forward these chunks to the STT module.

**Implementation Status:** ✅ Completed
- WebSocket server implemented using the `websockets` library
- Supports both binary audio streaming and JSON-based protocol
- Handles chunking of audio data based on configurable duration
- Provides error handling for malformed requests and corrupted data
- Includes support for interrupting ongoing responses

**Implemented Features:**
1. **WebSocket Connection Setup**: Using Python's `websockets` library
2. **Chunking Logic**: Maintains a buffer for incoming audio bytes, processes when chunk size is reached
3. **JSON Protocol**: Structured message format for complex interactions
4. **Session Management**: Tracks connections with unique session IDs
5. **Error Handling**: Graceful handling of malformed data and protocol errors

**Test Coverage**:
- Basic connection and audio chunking
- JSON protocol and base64 encoding
- Unsupported format handling
- High concurrency with multiple simultaneous clients
- Corrupted chunk recovery
- Chunk size variations
- Text input and conversation
- Interruption handling
- Error handling and recovery

**Documentation**:
- See detailed documentation in `docs/WEBSOCKET.md`
- API reference includes message types and example code
- Configuration options are documented with defaults

---

### 2.2 Session State Management (`ashtabula/buffer.py`, `ashtabula/conversation.py`) ✅ IMPLEMENTED

**Purpose**:
- Maintain incremental transcriptions, predicted sentences, embeddings, and responses for each user session.
- Clear and reset state once the utterance is finalized.

**Implementation Status:** ✅ Completed
- Implemented robust session state management with both ResponseBuffer and SessionBuffer
- Tracks conversational state including partial transcriptions and predictions
- Supports embeddings for better message matching
- Handles stale session cleanup automatically
- Maintains conversation history with configurable limits

**Implemented Features:**
1. **Buffer Structure** (`buffer.py`):
   - SessionBuffer class to track transcriptions, predictions, and embeddings
   - TranscriptionItem and PredictionItem classes with timestamps and metadata
   - Automatic stale session detection and cleanup
2. **Conversation Context** (`conversation.py`):
   - Session management with unique IDs for all conversations 
   - Conversation history tracking with configurable limits
   - Prediction similarity matching for faster responses
   - Embedding storage and retrieval for improved matching accuracy

**Test Coverage**:
- Response buffer operations (add, get, clear)
- Session buffer functionality and stale detection
- Prediction similarity matching
- Session continuity across multiple interactions
- Conversation history tracking
- Interrupt handling and state clearing
- Stale session cleanup
- Embedding integration

**Documentation**:
- Data structures fully documented with type annotations
- Thread safety considerations addressed
- Detailed API documentation for all classes and methods
- Configuration options explained with sensible defaults

---

### 2.3 Voice Activity Detection (VAD) Integration ✅ IMPLEMENTED

**Purpose**:
- Identify the start and end of speech to decide when to finalize transcriptions and trigger response generation.

**Implementation Status:** ✅ Completed
- Implemented a modular VAD system with multiple provider options
- Created pyannote-audio based provider for high-quality detection
- Added a simple energy-based provider as a lightweight alternative
- All providers follow a consistent interface for easy swapping
- Extensive configuration options for different environments

**Implemented Features:**
1. **VAD Models**:
   - PyannoteVADProvider: Neural network-based state-of-the-art VAD
   - SimpleThresholdVADProvider: Lightweight energy-based VAD
2. **Configurable Sensitivity**:
   - Detection thresholds for speech/non-speech
   - Minimum duration settings for speech and silence
   - Tunable parameters for different noise environments
3. **Segment Boundaries**:
   - Accurate detection of speech segment start and end times
   - Real-time streaming with speech/silence tracking
   - Automatic segment management with timestamps

**Test Coverage**:
- Processing of silent and speech audio chunks
- Speech segment detection in mixed audio files
- Streaming audio processing with speech detection
- Continuous speech with small pauses
- Multiple threshold configurations for different sensitivity levels
- Proper initialization and resource management

**Documentation**:
- Detailed documentation in `docs/VAD.md`
- Configuration guidelines for different environments
- Performance considerations for CPU vs. GPU deployment
- Examples of integration with the conversation flow

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

### 2.5 Conversation Manager (`ashtabula/conversation.py`, `ashtabula/conversation_fsm.py`)

**Purpose**:
- Orchestrate the entire pipeline flow: from partial transcription to final LLM response selection.
- Manage comparisons between the final utterance embedding and stored predicted embeddings.
- Implement state machine to manage conversation flow with well-defined states and transitions.

**Implementation Guide**:
1. **State Machine Architecture** (`conversation_fsm.py`):
   - Implement as a Finite State Machine using the `transitions` library
   - Define clear states: IDLE, LISTENING, SPEECH_ACTIVE, PROCESSING_UTTERANCE, GENERATING_RESPONSE, SPEAKING, INTERRUPTED
   - Create transitions between states with appropriate handlers
2. **Incremental Updates**:
   - For each partial transcription, optionally generate or retrieve partial responses for quick feedback.
3. **1-Second Timeslice Processing**:
   - Process Whisper STT output in 1-second chunks
   - Generate predictions for partial utterances to preload responses
4. **Final Utterance Embedding**:
   - On VAD finalization, compute the embedding of the final utterance.
5. **Prediction Matching**:
   - Compare with stored partial predictions via similarity matching. If a match is found (`similarity > threshold`), use the stored response. Otherwise, generate a new response.
6. **Reset Session**:
   - Once a final response is rendered by TTS, clear the session for the next utterance.

**Test Strategy**:
- **Unit Tests**:
  - Check the logic for computing embeddings and matching predictions.
  - Validate state transitions in the FSM
- **Integration**:
  - Test a multi-utterance conversation to confirm session resets properly and transitions are correct.
- **Stress/Load**:
  - Evaluate system stability under many concurrent users and long utterances.

**Developer Docs**:
- **State Machine**: Document the states, transitions, and event handlers in the FSM.
- **Embedding Method**: Document which embedding model is used and how to configure it.
- **Matching Criteria**: Include an explanation of the similarity threshold and how it can be tuned.

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