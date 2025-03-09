"""
Script to download required models from Hugging Face and save them locally.
"""

import os
import logging
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)

def is_model_downloaded(output_dir: str) -> bool:
    """Check if a model has already been downloaded."""
    return os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0

def download_whisper_model(model_size: str = "small", force: bool = False) -> None:
    """Download Whisper model and save locally."""
    model_id = f"openai/whisper-{model_size}"
    output_dir = os.path.join("models", "whisper", "whisper")

    if not force and is_model_downloaded(output_dir):
        logger.info(f"Whisper model already exists at: {output_dir}, skipping download.")
        return

    logger.info(f"Downloading Whisper model: {model_id}")
    ensure_dir(output_dir)

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        use_safetensors=True
    )

    processor.save_pretrained(output_dir)
    model.save_pretrained(output_dir)

    logger.info(f"Whisper model saved to: {output_dir}")

def download_phi_model(force: bool = False) -> None:
    """Download Phi model and save locally."""
    model_id = "microsoft/phi-2"
    output_dir = os.path.join("models", "phi", "phi")

    if not force and is_model_downloaded(output_dir):
        logger.info(f"Phi model already exists at: {output_dir}, skipping download.")
        return

    logger.info(f"Downloading Phi model: {model_id}")
    ensure_dir(output_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        use_safetensors=True
    )

    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir)

    logger.info(f"Phi model saved to: {output_dir}")

def download_speecht5_model(force: bool = False) -> None:
    """Download SpeechT5 model and save locally."""
    model_id = "microsoft/speecht5_tts"
    output_dir = os.path.join("models", "speecht5", "speecht5")

    if not force and is_model_downloaded(output_dir):
        logger.info(f"SpeechT5 model already exists at: {output_dir}, skipping download.")
        return

    logger.info(f"Downloading SpeechT5 model: {model_id}")
    ensure_dir(output_dir)

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        use_safetensors=True
    )

    processor.save_pretrained(output_dir)
    model.save_pretrained(output_dir, safe_serialization=False)
    logger.info(f"SpeechT5 model saved to: {output_dir}")

def main() -> None:
    """Download all required models."""
    parser = argparse.ArgumentParser(description="Download required models.")
    parser.add_argument("--force", action="store_true", help="Force download of models even if they already exist.")
    args = parser.parse_args()

    try:
        ensure_dir("models")

        download_whisper_model(force=args.force)
        download_phi_model(force=args.force)
        download_speecht5_model(force=args.force)

        logger.info("All models downloaded successfully!")

    except Exception as e:
        logger.error(f"Error downloading models: {str(e)}")
        raise

if __name__ == "__main__":
    main()
