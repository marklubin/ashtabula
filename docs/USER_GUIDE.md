# 🛠️ Ashtabula User Guide: Running OpenChat & Hugging Face STT/TTS Locally on Mac (M4)

## **🚀 Overview**
This guide provides instructions to set up and run **OpenChat (LLM)** and **Whisper (STT) + Piper (TTS) via Hugging Face Transformers** **locally on a MacBook Pro M4** with 24GB RAM. This setup eliminates manual model downloads, making installation and usage easier.

---

## **📌 Model Choices for Ashtabula**
We are now using **Hugging Face Transformers API** to load all models dynamically:

| **Model** | **Type** | **Size** | **Why We Chose It?** |
|-----------|---------|---------|----------------------|
| **OpenChat 3.5** | LLM | 7B | Conversational, chat-optimized, expressive |
| **Mistral-7B** | LLM | 7B | Strong reasoning, fast inference |
| **Phi-2** | LLM | 2.7B | Lightweight, best for speed |
| **Whisper (Hugging Face)** | STT | Varies | Auto-downloads, optimized speech recognition |
| **Piper (Hugging Face)** | TTS | Small | Lightweight, efficient speech synthesis |

**Recommendation:** OpenChat 3.5 + Hugging Face Whisper STT + Piper TTS for a **fully local conversational AI experience.**

---

## **🔧 Setting Up OpenChat on macOS (via Hugging Face Transformers)**
### **1️⃣ Install Dependencies**
```bash
pip install transformers torch librosa
```

### **2️⃣ Run OpenChat 3.5 LLM**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "openchat/openchat-3.5"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

def chat_with_ai(user_input):
    prompt = f"<System>
You are a helpful AI assistant.
</System>
<User>
{user_input}
</User>
<Assistant>
"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    output = model.generate(input_ids, max_length=200)
    return tokenizer.decode(output[0], skip_special_tokens=True)

print(chat_with_ai("How do I fix a skipping bike chain?"))
```

---

## **🔧 Setting Up Whisper (STT) via Hugging Face Transformers**
### **1️⃣ Install Whisper Model**
```python
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import torch
import librosa

# Load Whisper model (auto-downloads)
model_id = "openai/whisper-large-v3"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id).to("mps")  # Metal acceleration

# Load and transcribe audio file
audio_path = "your_audio.wav"
audio, sr = librosa.load(audio_path, sr=16000)
inputs = processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to("mps")

with torch.no_grad():
    generated_ids = model.generate(inputs)
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("Transcription:", transcription)
```

---

## **🔧 Setting Up Piper (TTS) via Hugging Face Transformers**
### **1️⃣ Install Piper Model**
```python
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import torch
import soundfile as sf

# Load Piper TTS model
model_id = "rhasspy/piper-voices"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id).to("mps")

# Generate speech
text = "Hello, this is a test."
inputs = processor(text, return_tensors="pt").input_ids.to("mps")

with torch.no_grad():
    audio = model.generate(inputs)

# Save output as a .wav file
sf.write("output.wav", audio.cpu().numpy(), 22050)
print("Generated speech saved as output.wav")
```

---

## **🔥 Optimization Tips for macOS**
- **Use `device="mps"` for Apple Silicon acceleration.**
- **Quantize OpenChat models (`int8` or `fp16`) for lower RAM usage.**
- **Use `16kHz mono audio` for Whisper STT to improve accuracy.**
- **Ensure enough free memory (Close unnecessary apps while running AI).**

---

## **🚀 Next Steps**
- Integrate OpenChat & Hugging Face Whisper/Piper into Ashtabula.
- Test real-time STT → LLM → TTS pipeline.
- Optimize latency for smoother interaction.

🚀 **You’re ready to run local AI-powered conversations on your Mac!**

