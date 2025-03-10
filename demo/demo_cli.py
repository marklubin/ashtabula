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
    """Handle interrupt signals to gracefully exit the application."""
    global running
    print("\nShutting down... Please wait.")
    running = False


async def process_audio_chunk(chunk: np.ndarray):
    """
    Process an audio chunk with the conversation FSM.
    
    Args:
        chunk: Audio data as numpy array
    """
    global fsm
    
    # Save chunk to temporary WAV file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_path = temp_file.name
    
    with wave.open(temp_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit audio
        wf.setframerate(sample_rate)
        wf.writeframes((chunk * 32767).astype(np.int16).tobytes())
    
    try:
        # Transcribe the audio chunk
        transcription = await fsm.stt.transcribe_file(temp_path)
        
        # Process the transcription with the FSM
        result = await fsm.process_transcription(transcription, False)
        
        # Print partial results if in speech active state
        if result['state'] == ConversationState.SPEECH_ACTIVE.value and 'predicted_completion' in result:
            print(f"\rPartial: {result['text']} | Predicted: {result['predicted_completion']}", end='', flush=True)
        
        # Check for silence
        silence_result = await fsm.check_silence()
        if silence_result and silence_result.get('is_final', False):
            print(f"\nYou: {silence_result['text']}")
            print(f"Assistant: {silence_result['response']}")
            # Play the response audio
            await play_response(silence_result['response'])
    finally:
        # Clean up temporary file
        os.unlink(temp_path)


async def audio_callback(indata, frames, time_info, status):
    """
    Callback function for audio stream.
    
    This function is called by the sounddevice InputStream for each audio chunk.
    """
    if status:
        print(f"Stream status: {status}")
    
    # Process audio chunk
    audio_data = indata.copy()
    asyncio.create_task(process_audio_chunk(audio_data.flatten()))


async def play_response(text: str):
    """
    Play the response using TTS.
    
    Args:
        text: Text to convert to speech
    """
    try:
        # Try to use the local model first
        model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "speecht5")
        if os.path.exists(model_dir):
            tts_provider = SpeechT5TTSProvider(model_dir=model_dir)
        else:
            # Fall back to HuggingFace if local model not available
            tts_provider = SpeechT5TTSProvider()
            
        # Get a temporary file for the audio output
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            audio_path = tmp.name
            
        # Synthesize speech
        tts_provider.synthesize(text, output_path=audio_path)
        
        # Play audio using sounddevice
        with wave.open(audio_path, 'rb') as wf:
            data = wf.readframes(wf.getnframes())
            audio_data = np.frombuffer(data, dtype=np.int16) / 32767.0
            sd.play(audio_data, wf.getframerate())
            sd.wait()
        
        # Clean up temporary file
        if os.path.exists(audio_path):
            os.unlink(audio_path)
            
    except Exception as e:
        logger.error(f"Error playing response: {e}")
        print(f"(TTS Error - Response: {text})")


async def main_loop():
    """Main application loop for audio processing."""
    global running, fsm
    
    print("Initializing Ashtabula Conversation Demo...")
    print("Loading models, please wait...")
    
    # Initialize components with local models if available
    # Check for local phi model
    phi_model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "phi")
    if os.path.exists(phi_model_dir):
        logger.info(f"Using local phi model from {phi_model_dir}")
        llm_provider = HuggingFaceLLMProvider("microsoft/phi-2")  # Local model will be used if downloaded
    else:
        logger.info("Local phi model not found, using HuggingFace model")
        llm_provider = HuggingFaceLLMProvider("microsoft/phi-2")
    
    # Check for local whisper model
    whisper_config = HFWhisperConfig()
    whisper_model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "whisper")
    if os.path.exists(whisper_model_dir):
        logger.info(f"Using local whisper model from {whisper_model_dir}")
        whisper_config.model_path = whisper_model_dir
    
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
    
    # Start in listening state
    fsm.start_listening()
    
    print("\nAshtabula Conversation Demo Ready!")
    print("Speak to interact with the assistant. Press Ctrl+C to exit.")
    
    # Set up audio stream
    try:
        with sd.InputStream(
            callback=audio_callback,
            channels=1,
            samplerate=sample_rate,
            blocksize=chunk_size
        ):
            # Keep the program running until interrupted
            while running:
                await asyncio.sleep(0.1)
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        logger.error(f"Error in audio stream: {e}")
        print(f"Error: {e}")
    finally:
        print("Goodbye!")


def main():
    """Main entry point for the demo application."""
    parser = argparse.ArgumentParser(description="Ashtabula Conversation Demo CLI")
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Set up signal handling for graceful exit
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run the main loop
    asyncio.run(main_loop())


if __name__ == "__main__":
    main()