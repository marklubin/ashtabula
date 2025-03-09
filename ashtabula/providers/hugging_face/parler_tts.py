import os
import torch
import soundfile as sf
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

class ParlerTTSProvider:
    def __init__(self, model_dir: str, device: str = "auto", speaker: str = "Thomas", emotion: str = "default"):
        """
        Initialize the Parler-TTS provider.
        """
        self.device = device if device != "auto" else "cuda" if torch.cuda.is_available() else "cpu"
        self.speaker = speaker
        self.emotion = emotion

        # Ensure the local model path is correct
        self.model_dir = os.path.abspath(model_dir)
        if not os.path.exists(self.model_dir):
            raise RuntimeError(f"Parler-TTS model not found at {self.model_dir}")

        self._initialize_model()

    def _initialize_model(self):
        """Load the Parler-TTS model manually from a local directory."""
        print(f"Loading Parler-TTS model from: {self.model_dir}")

        # Load the tokenizer and model from local directory
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, local_files_only=True)
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(self.model_dir, local_files_only=True)

        self.model.to(self.device)
        print(f"✅ Successfully loaded Parler-TTS from {self.model_dir}")

    def synthesize(self, text: str, output_path: str = None) -> str:
        """
        Convert text to speech and save it as a WAV file.
        
        Args:
            text: Text to convert to speech.
            output_path: Optional path to save the audio file.
        
        Returns:
            Path to the generated audio file.
        """
        inputs = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)

        with torch.no_grad():
            output = self.model.generate(input_ids=inputs)
            audio = output.cpu().numpy().squeeze()

        if output_path is None:
            import tempfile
            temp_dir = tempfile.mkdtemp()
            output_path = os.path.join(temp_dir, "output.wav")

        sf.write(output_path, audio, self.model.config.sampling_rate)
        return output_path