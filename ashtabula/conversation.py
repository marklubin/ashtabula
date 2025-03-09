"""
Handles the full AI conversation pipeline, integrating STT, LLM, buffering, and TTS.
"""

class ConversationManager:
    def __init__(self, llm_provider, stt_provider, tts_provider, prediction_threshold=0.8, silence_timeout=3):
        self.llm_provider = llm_provider
        self.stt_provider = stt_provider
        self.tts_provider = tts_provider
        self.prediction_threshold = prediction_threshold
        self.silence_timeout = silence_timeout

    def stream_audio(self, audio_source):
        """Streams audio to STT, predicts user sentence completion, buffers responses, and plays AI output."""
        raise NotImplementedError
