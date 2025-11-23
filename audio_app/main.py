import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import os
print("Current Directory:", os.getcwd())
print("Files:", os.listdir())

import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from model import ViTModel
import io
import requests  # <-- NEW for model download

# ==================================================================
# 1. Initialize FastAPI App and CORS
# ==================================================================
app = FastAPI(
    title="Voice Anti-Spoofing API",
    description="An API to detect if an audio file is real or AI-generated."
)

# Allow your frontend (Railway / localhost / etc.)
origins = [
    "*",  # or specific frontend URL
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================================================================
# 2. Model Setup — Download if missing
# ==================================================================
MODEL_PATH = "vit_model.pth"
MODEL_URL = "https://huggingface.co/LaabhGupta/voice-antispoofing-vit/resolve/main/vit_model.pth"

device = "cpu"
print("--- Checking Model ---")

if not os.path.exists(MODEL_PATH):
    print(f"Downloading model from Hugging Face...")
    resp = requests.get(MODEL_URL, stream=True)
    resp.raise_for_status()
    with open(MODEL_PATH, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print("Model downloaded!")

# Global variable, but not loaded yet
model = None

def load_model():
    global model
    if model is None:
        print("Loading model into memory...")
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model_instance = ViTModel()
        model_instance.load_state_dict(state_dict)
        model_instance.to(device)
        model_instance.eval()
        model = model_instance
        print("Model loaded successfully.")
    return model


# ==================================================================
# 3. Define Constants & Preprocessing Function
# ==================================================================
TARGET_SAMPLE_RATE = 16000
TARGET_LEN_SECS = 4
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 1024
CLASS_NAMES = ["fake", "real"]

def preprocess_audio(audio_bytes: bytes):
    waveform, sample_rate = torchaudio.load(io.BytesIO(audio_bytes))

    if sample_rate != TARGET_SAMPLE_RATE:
        resampler = T.Resample(orig_freq=sample_rate, new_freq=TARGET_SAMPLE_RATE)
        waveform = resampler(waveform)

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    target_len = TARGET_SAMPLE_RATE * TARGET_LEN_SECS
    if waveform.shape[1] > target_len:
        waveform = waveform[:, :target_len]
    else:
        waveform = F.pad(waveform, (0, target_len - waveform.shape[1]))

    mel_spectrogram_transform = T.MelSpectrogram(
        sample_rate=TARGET_SAMPLE_RATE, n_fft=N_FFT,
        hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    spectrogram = mel_spectrogram_transform(waveform)
    spectrogram = spectrogram.unsqueeze(0)
    return spectrogram

# ==================================================================
# 4. API Endpoints
# ==================================================================
@app.get("/")
def read_root():
    return {"message": "Voice Anti-Spoofing API is running!"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    audio_bytes = await file.read()

    spectrogram = preprocess_audio(audio_bytes)

    model_instance = load_model()   # <-- model loads only when needed

    with torch.no_grad():
        logits = model_instance(spectrogram.to(device))
        probs = torch.softmax(logits, dim=1)
        idx = torch.argmax(probs, dim=1).item()

    return {
        "filename": file.filename,
        "predicted_class": CLASS_NAMES[idx],
        "confidence": float(probs[0][idx]),
    }

