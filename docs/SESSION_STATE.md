# Ashtabula Session State Management

This document explains the session state management system in Ashtabula, which handles conversation tracking, transcription history, prediction management, and embeddings.

## Overview

The session state management system consists of two main components:

1. **ResponseBuffer**: A simple queue-based buffer for managing AI responses
2. **SessionBuffer**: A more complex buffer for managing session state including transcriptions, predictions, and embeddings

These components work together with the ConversationManager to maintain conversation state across multiple interactions, support prediction matching, and handle cleanup of stale sessions.

## Key Features

- **Persistent Sessions**: Maintain conversation context across multiple audio streams
- **Prediction Matching**: Store and retrieve predictions based on similarity
- **Embedding Support**: Generate embeddings for better text matching
- **Automatic Cleanup**: Clean up stale sessions after configurable timeout
- **Thread Safety**: Safe for concurrent access from multiple tasks

## Classes and Components

### ResponseBuffer

A simple queue-based buffer for AI responses.

```python
buffer = ResponseBuffer()
buffer.add("Response text")
response = buffer.get()  # Returns and removes the first response
buffer.clear()  # Removes all responses
```

### SessionBuffer

Maintains comprehensive session state including:

- Transcription history with timestamps
- Prediction history with source transcriptions
- Embeddings for improved matching
- Session activity tracking

```python
session = SessionBuffer()
session.add_transcription("Hello world", is_final=True)
session.add_prediction("Hello world and everyone", "Hello world")
histories = session.get_transcription_history()
predictions = session.get_prediction_history()
```

### TranscriptionItem & PredictionItem

Data classes for storing transcriptions and predictions with metadata:

- **TranscriptionItem**: Stores text, timestamp, and finalization status
- **PredictionItem**: Stores predicted text, timestamp, source transcription, and optional embedding

### ConversationManager

Orchestrates the conversation flow and manages sessions:

- Creates and maintains sessions with unique IDs
- Tracks conversation history
- Computes similarity between texts
- Generates embeddings for prediction matching
- Performs cleanup of stale sessions

## Configuration Options

The ConversationConfig class provides several configuration options:

```python
config = ConversationConfig(
    prediction_threshold=0.85,      # Minimum similarity for prediction match
    silence_timeout=3.0,            # Seconds of silence before completing sentence
    max_history_length=5,           # Number of previous utterances to keep
    session_timeout=300.0,          # Seconds before a session is considered stale
    max_transcription_history=20,   # Max number of transcription items to keep
    max_prediction_history=10,      # Max number of prediction items to keep
    embed_predictions=True          # Whether to generate embeddings for predictions
)
```

## Usage Examples

### Basic Usage

```python
# Create manager components
llm_provider = MyLLMProvider()
stt_provider = MySTTProvider()
response_buffer = ResponseBuffer()

# Create manager with configuration
manager = ConversationManager(
    llm_provider=llm_provider,
    stt_provider=stt_provider,
    response_buffer=response_buffer,
    config=ConversationConfig(session_timeout=300.0)
)

# Process audio and get responses
async for response in manager.stream_audio("audio.wav"):
    if response.get('is_final'):
        session_id = response.get('session_id')
        print(f"Final response: {response.get('response')}")
        break

# Continue conversation with the same session
async for response in manager.stream_audio("next_audio.wav", session_id=session_id):
    # Process responses
    pass
```

### Handling Interruptions

```python
# Handle user interruption
manager.handle_interrupt()  # Clears buffer and resets state
```

### Manual Session Management

```python
# Create or get a session
session_id, session = manager._get_or_create_session()

# Add data to the session
session.add_transcription("Hello world", is_final=False)
session.add_prediction("Hello world and everyone", "Hello world")

# Reset a session
manager.reset_session(session_id)

# Get a session by ID
session = manager.get_session(session_id)
```

## Implementation Details

### Session ID Generation

Session IDs are generated using UUID4 to ensure uniqueness:

```python
session_id = str(uuid.uuid4())
```

### Prediction Matching

Predictions are matched using similarity measures (default: SequenceMatcher ratio):

```python
similarity = self._calculate_similarity(actual_text, predicted_text)
if similarity >= self.config.prediction_threshold:
    # Use prediction
```

### Stale Session Cleanup

Inactive sessions are automatically removed after the configured timeout:

```python
if (time.time() - session.last_activity_time) > timeout_seconds:
    # Remove session
```

## Thread Safety Considerations

- The SessionBuffer is not inherently thread-safe and should be accessed from a single thread or with proper synchronization
- The ConversationManager handles one session at a time per manager instance
- For multi-user systems, create a separate ConversationManager instance per user or use appropriate locks