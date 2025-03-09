import os
import torch
import soundfile as sf
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan, pipeline
from datasets import load_dataset

import logging

# Configure logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpeechT5TTSProvider:
    """
    Text-to-Speech provider using Microsoft's SpeechT5 model.
    """

    def __init__(self, model_dir: str = None, device: str = "auto"):
        """
        Initialize the SpeechT5 provider.

        Args:
            model_dir: Path to the locally stored SpeechT5 model (optional).
            device: Device to run inference on ('cpu', 'cuda', 'auto').
        """
        self.device = device if device != "auto" else "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load from local directory if provided, otherwise from Hugging Face
        self.model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 
                                        "models", "speecht5", "speecht5")
        self.processor = None
        self.model = None
        self.vocoder = None
        self.speaker_embeddings = None
        self.tts_pipeline = None
        self._initialize_model()

    def _initialize_model(self):
        """Load the SpeechT5 model, processor, and vocoder."""
        try:
            logger.info(f"Loading SpeechT5 model from: {self.model_dir}")

            self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
            self.model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
            self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")


            self.model.to(self.device)
            self.vocoder.to(self.device)

            # Load default speaker embeddings
            logger.info("Loading speaker embeddings dataset...")
            embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
            self.speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0).to(self.device)
        except Exception as e:
            logger.error(f"Failed to load SpeechT5 model: {str(e)}")
            raise RuntimeError(f"Failed to load SpeechT5 model: {str(e)}")

    def synthesize(self, text: str, output_path: str = None) -> str:
        """
        Convert text to speech and save it as a WAV file.

        Args:
            text: Text to convert to speech.
            output_path: Optional path to save the audio file.

        Returns:
            Path to the generated audio file.
        """
        try:
            logger.info(f"Starting synthesis for text: {text}")

            # Ensure correct input format
            inputs = self.processor(text=text, return_tensors="pt").to(self.device)
            logger.info(f"Tokenized input: {inputs}")

            # Extract input_ids explicitly
            input_ids = inputs["input_ids"].to(self.device)
            logger.info(f"Input IDs shape: {input_ids.shape}")


            # Generate speech
            with torch.no_grad():
                logger.info("Generating speech from model...")
                speech = self.model.generate_speech(
                    input_ids,
                    self.speaker_embeddings,
                    vocoder=self.vocoder
                )

            logger.info(f"Generated speech shape: {speech.shape}, min: {speech.min()}, max: {speech.max()}, mean: {speech.mean()}")

            # Save to a temporary file if no output path is provided
            if output_path is None:
                import tempfile
                temp_dir = tempfile.mkdtemp()
                output_path = os.path.join(temp_dir, "output.wav")

            sf.write(output_path, speech.cpu().numpy(), samplerate=16000)  # Increase sample rate for better quality
            logger.info(f"Saved generated audio to {output_path}")

            return output_path

        except Exception as e:
            logger.error(f"Failed to synthesize speech: {str(e)}")
            raise RuntimeError(f"Failed to synthesize speech: {str(e)}")