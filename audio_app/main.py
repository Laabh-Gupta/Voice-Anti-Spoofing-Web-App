import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from model import BaselineCNN
import io

# --- 1. Initialize FastAPI App and CORS ---
app = FastAPI(title="Voice Anti-Spoofing API (Fast CNN)")
origins = [
    "https://voiceantispoofing.netlify.app",
    "*"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. Load Model ---
MODEL_PATH = "baseline_cnn_finetuned.pth"
device = "cpu"
print("🔍 Checking for model file...")

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"❌ Model not found: {MODEL_PATH}")

print("🧠 Loading model...")
model = BaselineCNN()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()
print("✔ Model loaded successfully (FAST CNN)")

# --- 3. Constants & Preprocessing ---
TARGET_SAMPLE_RATE = 16000
TARGET_LEN_SECS = 4
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 1024
CLASS_NAMES = ["fake", "real"]

def preprocess_audio(audio_bytes: bytes, file_format: str):
    waveform, sample_rate = torchaudio.load(io.BytesIO(audio_bytes), format=file_format)

    if sample_rate != TARGET_SAMPLE_RATE:
        waveform = T.Resample(orig_freq=sample_rate, new_freq=TARGET_SAMPLE_RATE)(waveform)

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    target_len = TARGET_SAMPLE_RATE * TARGET_LEN_SECS
    waveform = waveform[:, :target_len] if waveform.shape[1] > target_len else F.pad(waveform, (0, target_len - waveform.shape[1]))

    mel = T.MelSpectrogram(
        sample_rate=TARGET_SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )(waveform)

    return mel.unsqueeze(0)

@app.get("/")
def health():
    return {"message": "Voice Anti-Spoofing API (FAST CNN) Running!"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    ext = file.filename.split(".")[-1].lower()

    if ext not in ["wav", "mp3"]:
        return {"error": "Please upload a .wav or .mp3 file"}

    try:
        spectrogram = preprocess_audio(audio_bytes, file_format=ext)
        with torch.no_grad():
            logits = model(spectrogram.to(device))
            probs = torch.softmax(logits, dim=1)
            idx = torch.argmax(probs, dim=1).item()
        return {
            "filename": file.filename,
            "predicted_class": CLASS_NAMES[idx],
            "confidence": float(probs[0][idx])
        }
    except Exception as e:
        return {"error": f"Failed to process audio: {e}"}
