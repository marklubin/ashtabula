# Ashtabula WebSocket Server

This document describes the WebSocket server component of the Ashtabula conversational AI pipeline, including its API, configuration options, and usage examples.

## Overview

The WebSocket server is responsible for:

1. Accepting real-time audio streams from clients
2. Chunking audio into configurable time intervals
3. Processing audio chunks through the speech-to-text (STT) system
4. Managing the conversation flow through the AI pipeline
5. Returning AI responses to clients

The server supports both binary audio streaming and a structured JSON protocol for more complex interactions.

## Configuration

The WebSocket server can be configured with the following parameters:

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `host` | Host address to bind to | `"0.0.0.0"` (all interfaces) |
| `port` | Port to listen on | `8765` |
| `chunk_duration` | Duration of audio chunks in seconds | `1.0` |
| `sample_rate` | Expected sample rate of audio in Hz | `16000` |
| `stt_provider` | Speech-to-text provider | `None` |
| `conversation_manager` | Conversation manager | `None` |

## Message Protocol

The WebSocket server supports two main protocols for communication:

1. **Binary audio data** - Raw audio bytes sent directly over the WebSocket
2. **JSON messages** - Structured messages for more complex interactions

### JSON Message Types

#### Client to Server

| Message Type | Description | Required Fields | Optional Fields |
|--------------|-------------|----------------|-----------------|
| `text_input` | Direct text input | `text` | `stream_response` |
| `start_audio_stream` | Begin audio stream | | `format`, `sample_rate` |
| `audio_chunk` | Audio data chunk | `data` (base64) | |
| `end_audio_stream` | End audio stream | | |
| `interrupt` | Interrupt current response | | `new_text` |

#### Server to Client

| Message Type | Description | Fields |
|--------------|-------------|--------|
| `connection_established` | Initial connection welcome | `session_id`, `config` |
| `stream_started` | Audio stream started | |
| `chunk_received` | Audio chunk received | `size`, `duration` |
| `partial_transcription` | Partial STT result | `text`, `is_final` |
| `final_transcription` | Final STT result | `text` |
| `ai_response` | Complete AI response | `text` |
| `ai_response_start` | Start of streaming response | |
| `ai_response_chunk` | Chunk of streaming response | `text` |
| `ai_response_complete` | End of streaming response | |
| `ai_response_interrupted` | Response was interrupted | |
| `stream_ended` | Audio stream ended | |
| `error` | Error occurred | `message` |
| `warning` | Warning | `message` |

## Usage Examples

### Basic Audio Streaming

To stream audio to the server:

1. Connect to the WebSocket
2. Receive the welcome message
3. Send raw audio chunks (e.g., 16-bit PCM at 16kHz)
4. Receive transcription messages
5. Close when done

### Structured JSON Protocol

Here's an example of using the JSON protocol for a complete conversation:

```javascript
// Connect to WebSocket
const socket = new WebSocket("ws://localhost:8765");

// Listen for messages
socket.onmessage = (event) => {
  const message = JSON.parse(event.data);
  console.log("Received:", message);
  
  // Handle different message types
  switch (message.type) {
    case "partial_transcription":
      console.log("Partial:", message.text);
      break;
    case "final_transcription":
      console.log("Final:", message.text);
      break;
    case "ai_response":
      console.log("AI:", message.text);
      break;
    // Handle other message types...
  }
};

// Start audio stream
socket.send(JSON.stringify({
  type: "start_audio_stream",
  format: "wav",
  sample_rate: 16000
}));

// Send audio chunks (base64 encoded)
socket.send(JSON.stringify({
  type: "audio_chunk",
  data: "base64EncodedAudioDataHere..."
}));

// End stream
socket.send(JSON.stringify({
  type: "end_audio_stream"
}));

// Or send text directly
socket.send(JSON.stringify({
  type: "text_input",
  text: "Hello, how are you today?"
}));

// Interrupt an ongoing response
socket.send(JSON.stringify({
  type: "interrupt",
  new_text: "Actually, let me ask about something else..."
}));
```

## Error Handling

The server handles the following error conditions:

1. Invalid JSON messages
2. Missing required fields
3. Unsupported audio formats
4. Corrupted audio data
5. Concurrent connection limits

All errors are logged and most are reported back to the client with an `error` message containing a description of the problem.

## Test Scenarios

The WebSocket server has been tested with the following scenarios:

1. **Basic Connection & Chunking**: A client establishes a WebSocket connection and streams 2 seconds of WAV audio
2. **Unsupported Format Handling**: A client sends audio data in an unsupported format
3. **High Concurrency**: Ten clients connect concurrently, each streaming short audio clips
4. **Corrupted Chunk Recovery**: A client sends partially corrupted data in the middle of the stream
5. **Chunk Size Variations**: The server is configured for different chunk durations
6. **Text Input**: A client sends text input directly instead of audio
7. **Interruption**: A client interrupts an AI response with new input
8. **Error Handling**: A client sends malformed requests and missing fields

## Integration with Other Components

The WebSocket server integrates with:

1. **STT Provider**: For transcribing audio to text
2. **Conversation Manager**: For managing the dialogue state and generating responses
3. **TTS Provider**: For generating speech from text responses (through the conversation manager)

## Future Enhancements

Planned enhancements to the WebSocket server include:

1. Better compression support for audio streams (Opus, FLAC, etc.)
2. Client authentication and session management
3. Improved error recovery for corrupted audio
4. Metrics tracking for performance monitoring
5. WebRTC support for browser-based clients