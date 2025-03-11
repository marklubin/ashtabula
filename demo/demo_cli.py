#!/usr/bin/env python
"""
Demo CLI Application for Ashtabula

This demo application showcases the conversational capabilities of Ashtabula
using a command-line interface that enables speech interaction with the model.
"""

import argparse
import asyncio
import logging
import os
import signal
import sounddevice as sd
import numpy as np
import wave
import tempfile
import time
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List

# Add parent directory to path to import Ashtabula
sys.path.insert(0, str(Path(__file__).parent.parent))

from ashtabula.conversation_fsm import ConversationFSM, ConversationState
from ashtabula.buffer import ResponseBuffer, SessionBuffer
from ashtabula.providers.hugging_face.hf_llm import HuggingFaceLLMProvider
from ashtabula.providers.hugging_face.whisper_sst import HFWhisperSTTProvider, HFWhisperConfig
from ashtabula.providers.hugging_face.speech_t5_tts import SpeechT5TTSProvider

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ashtabula-demo")

# Global variables
running = True
fsm = None
audio_chunks = []
sample_rate = 16000  # Hz
chunk_duration = 1.0  # seconds
chunk_size = int(sample_rate * chunk_duration)


def signal_handler(sig, frame):
    """Handle interrupt signals to immediately exit the application."""
    global running
    print("\n👋 Shutting down immediately!")
    running = False
    # Force exit after a short delay if not shutting down naturally
    # This ensures we don't hang on unresponsive operations
    def force_exit():
        import os, time
        time.sleep(0.5)  # Give a short time for graceful shutdown
        print("Forcing exit...")
        os._exit(0)  # Force exit without cleanup
        
    import threading
    threading.Thread(target=force_exit).start()
    # Also attempt a normal exit
    import sys
    sys.exit(0)


async def process_audio_chunk(chunk: np.ndarray):
    """
    Process an audio chunk with the conversation FSM.
    
    Args:
        chunk: Audio data as numpy array
    """
    global fsm, running
    
    # Skip processing if we're shutting down
    if not running:
        return
    
    # Create a temporary WAV file
    temp_path = None
    try:
        # Save chunk to temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name
        
            # Write the data to the file
            with wave.open(temp_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit audio
                wf.setframerate(sample_rate)
                wf.writeframes((chunk * 32767).astype(np.int16).tobytes())
        
        # Get current state for status updates
        current_state = fsm.get_state()
        
        # Log processing start with condensed output if in the same state
        if current_state == ConversationState.LISTENING.value:
            # Just a dot to show activity without flooding the console
            print(f"\r🎤 Listening... ", end='', flush=True)
        elif current_state == ConversationState.SPEECH_ACTIVE.value:
            # Show we're still processing audio during active speech
            print(f"\r📝 Processing speech... ", end='', flush=True)
        else:
            # Full state for other states
            print(f"\r📊 Processing | State: {current_state}", end='', flush=True)
        
        # Transcribe the audio chunk
        transcription = await fsm.stt.transcribe_file(temp_path)
        
        # Process the transcription with the FSM
        result = await fsm.process_transcription(transcription, False)
        
        # Print partial results if in speech active state
        if result['state'] == ConversationState.SPEECH_ACTIVE.value:
            status_info = f"📝 ACTIVE | Text: {result['text']}"
            
            if 'predicted_completion' in result:
                status_info += f" | 🔮 Prediction: {result['predicted_completion']}"
                
            print(f"\r{status_info}", end='', flush=True)
        elif result['state'] != current_state:
            # Only print state changes to reduce console spam
            print(f"\r🔄 State change: {current_state} → {result['state']}", end='', flush=True)
        
        # Check for silence
        silence_result = await fsm.check_silence()
        if silence_result:
            if silence_result.get('is_final', False):
                # Clear the line for better readability
                print("\n")
                print(f"🗣️ You: {silence_result['text']}")
                print(f"🤖 Assistant: {silence_result['response']}")
                
                # Start audio playback immediately (non-blocking)
                await play_response(silence_result['response'])
                
                # Don't wait for audio to finish - prepare for next interaction right away
                print(f"📢 Ready for next interaction | State: {fsm.get_state()}")
            else:
                # Just a simple status update for non-final silence
                print(f"\r⏱ Silence detected...", end='', flush=True)
            
    except Exception as e:
        logger.error(f"Error processing audio chunk: {e}")
        print(f"\r❌ Error processing audio: {e}", end='', flush=True)
        
    finally:
        # Clean up temporary file
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception as e:
                logger.warning(f"Failed to remove temporary file {temp_path}: {e}")


def audio_callback(indata, frames, time_info, status):
    """
    Callback function for audio stream.
    
    This function is called by the sounddevice InputStream for each audio chunk.
    Note: This must be a synchronous function as it's called directly by sounddevice.
    """
    global audio_chunks
    
    if status:
        print(f"\r⚠️ Stream status: {status}", end='', flush=True)
    
    # Instead of trying to run the coroutine directly, just add the chunk to a queue
    # that the main thread will process
    audio_data = indata.copy().flatten()
    
    # Append to our global audio chunks list - the main loop will process these
    audio_chunks.append(audio_data)
    
    # Simple status update about buffered audio
    print(f"\r🔄 Audio chunk buffered, {len(audio_chunks)} chunks pending", end='', flush=True)


# Global TTS provider to avoid loading model multiple times
_tts_provider = None

def get_tts_provider():
    """Get or initialize the TTS provider (lazy initialization)."""
    global _tts_provider
    if _tts_provider is None:
        # Try to use the local model first
        model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "speecht5")
        if os.path.exists(model_dir):
            _tts_provider = SpeechT5TTSProvider(model_dir=model_dir)
            logger.info(f"Initialized TTS provider with local model")
        else:
            # Fall back to HuggingFace if local model not available
            _tts_provider = SpeechT5TTSProvider()
            logger.info(f"Initialized TTS provider with remote model")
            
        # Configure to suppress attention mask warning
        try:
            import warnings
            # Filter the specific warning about attention mask
            warnings.filterwarnings(
                "ignore", 
                message="The attention mask is not set.*", 
                category=UserWarning
            )
        except Exception as e:
            logger.warning(f"Could not configure warnings filter: {e}")
            
    return _tts_provider

async def play_response(text: str):
    """
    Play the response using TTS.
    
    Args:
        text: Text to convert to speech
    """
    # Schedule the TTS in a separate thread to avoid blocking
    print(f"\n🔊 Generating audio for response...")
    
    # Create a separate thread to run the TTS and audio playback
    # This ensures the audio plays immediately and doesn't block
    def process_tts_and_play():
        try:
            # Get the TTS provider (initialized once and reused)
            tts_provider = get_tts_provider()
                
            # Get a temporary file for the audio output
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                audio_path = tmp.name
                
            # Synthesize speech
            tts_provider.synthesize(text, output_path=audio_path)
            
            # Play audio using sounddevice
            print(f"▶️ Playing audio response...")
            with wave.open(audio_path, 'rb') as wf:
                data = wf.readframes(wf.getnframes())
                audio_data = np.frombuffer(data, dtype=np.int16) / 32767.0
                sd.play(audio_data, wf.getframerate())
                sd.wait()
            
            print(f"✅ Audio playback complete")
            
            # Clean up temporary file
            if os.path.exists(audio_path):
                os.unlink(audio_path)
                
        except Exception as e:
            logger.error(f"Error playing response: {e}")
            print(f"❌ TTS Error: {e}")
            print(f"Response text: {text}")
            
    # Start the TTS processing in a background thread so it doesn't block the main thread
    import threading
    tts_thread = threading.Thread(target=process_tts_and_play)
    tts_thread.daemon = True  # Make thread terminate when main thread exits
    tts_thread.start()
    
    # Return immediately so processing can continue
    return text


async def main_loop():
    """Main application loop for audio processing."""
    global running, fsm
    
    print("🚀 Initializing Ashtabula Conversation Demo...")
    print("⏳ Loading models, please wait...")
    
    # Initialize components with local models if available
    # Check for local phi model
    phi_model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "phi")
    if os.path.exists(phi_model_dir):
        logger.info(f"Using local phi model from {phi_model_dir}")
        print(f"📁 Using local Phi-2 LLM model")
        llm_provider = HuggingFaceLLMProvider("microsoft/phi-2")  # Local model will be used if downloaded
    else:
        logger.info("Local phi model not found, using HuggingFace model")
        print(f"☁️ Using remote Phi-2 LLM model (slower)")
        llm_provider = HuggingFaceLLMProvider("microsoft/phi-2")
    
    # Check for local whisper model
    whisper_config = HFWhisperConfig()
    whisper_model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "whisper")
    if os.path.exists(whisper_model_dir):
        logger.info(f"Using local whisper model from {whisper_model_dir}")
        print(f"📁 Using local Whisper STT model")
        whisper_config.model_path = whisper_model_dir
    else:
        print(f"☁️ Using remote Whisper STT model (slower)")
    
    # Check for SpeechT5 model
    speecht5_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "speecht5")
    if os.path.exists(speecht5_dir):
        print(f"📁 Using local SpeechT5 TTS model")
        # Pre-initialize the TTS provider during startup to avoid first-use delay
        print("⏳ Pre-loading TTS model...")
        get_tts_provider()
        print("✅ TTS model loaded")
    else:
        print(f"☁️ Using remote SpeechT5 TTS model (slower)")
    
    stt_provider = HFWhisperSTTProvider(whisper_config)
    response_buffer = ResponseBuffer()
    session_buffer = SessionBuffer()
    
    # Create FSM instance
    fsm = ConversationFSM(
        llm_provider=llm_provider,
        stt_provider=stt_provider,
        response_buffer=response_buffer,
        session_buffer=session_buffer,
        silence_timeout=2.0,  # 2 seconds of silence to trigger utterance completion
    )
    
    # Log configuration details
    logger.debug(f"Conversation FSM initialized with silence_timeout={fsm.silence_timeout}s")
    logger.debug(f"STT Provider: {type(stt_provider).__name__}")
    logger.debug(f"LLM Provider: {type(llm_provider).__name__}")
    
    # Start in listening state
    fsm.start_listening()
    logger.debug(f"FSM started in state: {fsm.get_state()}")
    
    print("\n✅ Ashtabula Conversation Demo Ready!")
    print("🎤 Speak to interact with the assistant. Live state updates will be shown.")
    print("🔄 Current state: LISTENING | Press Ctrl+C to exit.")
    
    # Set up audio stream
    try:
        # Use a smaller chunk size for more responsive interruption
        chunk_duration = 0.5  # Process in half-second chunks for better responsiveness
        chunk_size = int(sample_rate * chunk_duration)
        
        # Start audio processing in non-blocking mode
        stream = sd.InputStream(
            callback=audio_callback,
            channels=1,
            samplerate=sample_rate,
            blocksize=chunk_size
        )
        stream.start()
        
        # Keep the program running until interrupted
        last_chunk_process_time = time.time()
        max_chunk_queue = 30  # Prevent excessive queue buildup
        
        while running:
            try:
                # Process any queued audio chunks
                if audio_chunks:
                    # Get the oldest chunk (FIFO)
                    chunk = audio_chunks.pop(0)
                    last_chunk_process_time = time.time()
                    
                    # Process it with our FSM
                    await process_audio_chunk(chunk)
                    
                    # If queue is getting too big, trim it to prevent memory issues
                    if len(audio_chunks) > max_chunk_queue:
                        overflow = len(audio_chunks) - max_chunk_queue
                        audio_chunks = audio_chunks[overflow:]  # Keep only the newest chunks
                        print(f"\r⚠️ Audio queue overflow, dropped {overflow} old chunks", end='', flush=True)
                else:
                    # No audio to process, short sleep
                    await asyncio.sleep(0.05)
                    
                    # Periodically check silence (every 100ms) but only if we're not in a backlog
                    if not audio_chunks and (time.time() - last_chunk_process_time) > 0.1:
                        silence_result = await fsm.check_silence()
                        
                        # If we're not getting any audio for too long, maybe the mic is off?
                        if (time.time() - last_chunk_process_time) > 5.0:
                            print(f"\r🎤 Listening... (no audio detected for {int(time.time() - last_chunk_process_time)}s)", 
                                  end='', flush=True)
            
            except asyncio.CancelledError:
                # Handle cancellation gracefully
                break
            except Exception as e:
                # Log but continue processing - don't crash on one bad chunk
                logger.error(f"Error processing chunk: {e}")
                print(f"\r❌ Error processing audio: {e}", end='', flush=True)
                await asyncio.sleep(0.1)
            
            # Check if we need to exit - more responsive to CTRL-C
            if not running:
                break
    
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        logger.error(f"Error in audio stream: {e}")
        print(f"\n❌ Fatal error: {e}")
    finally:
        # Clean up
        try:
            stream.stop()
            stream.close()
        except:
            pass
        print("👋 Goodbye!")


def main():
    """Main entry point for the demo application."""
    parser = argparse.ArgumentParser(description="Ashtabula Conversation Demo CLI")
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose real-time status updates')
    args = parser.parse_args()
    
    # Configure logging based on arguments
    if args.debug:
        # Set root logger to DEBUG level
        logging.getLogger().setLevel(logging.DEBUG)
        # Make sure our own logger is also at DEBUG level
        logging.getLogger("ashtabula-demo").setLevel(logging.DEBUG)
        # Set other loggers to INFO to reduce noise
        logging.getLogger("transitions").setLevel(logging.INFO)
        logging.getLogger("transformers").setLevel(logging.INFO)
        
        print("🐛 Debug logging enabled - detailed information will be shown")
    elif args.verbose:
        # Set our logger to INFO to show more information without full debug noise
        logging.getLogger("ashtabula-demo").setLevel(logging.INFO)
        # Set transition logs to INFO to see state changes
        logging.getLogger("transitions").setLevel(logging.INFO)
        
        print("ℹ️ Verbose logging enabled - state transitions will be shown")
    
    # Set up signal handling for graceful exit
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("⚙️ Starting Ashtabula demo with real-time status updates...")
    
    # Run the main loop
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        print("\n👋 Shutting down gracefully...")
    except Exception as e:
        logger.error(f"Error in main loop: {e}")
        print(f"\n❌ Fatal error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    main()