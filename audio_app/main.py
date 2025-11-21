import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from model import ViTModel # Import the model class from model.py
import io

# ==================================================================
# 1. Initialize FastAPI App and CORS
# ==================================================================
app = FastAPI(title="Voice Anti-Spoofing API", description="An API to detect if an audio file is real or AI-generated.")

# Configure CORS to allow the React frontend (running on localhost:3000) to communicate with the backend.
origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================================================================
# 2. Define Constants and Load The Trained Model
# ==================================================================

# Define constants from your training notebook
TARGET_SAMPLE_RATE = 16000
TARGET_LEN_SECS = 4
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 1024
CLASS_NAMES = ["fake", "real"]

# Set device (CPU is often sufficient and easier for inference)
device = "cpu"

# Load the model at startup to avoid reloading it on every request
print("--- Loading model ---")
model = ViTModel() # Create an empty instance of your model architecture
model.load_state_dict(torch.load("vit_model.pth", map_location=device)) # Load your saved weights
model.to(device)
model.eval() # Set the model to evaluation mode (very important!)
print("--- Model loaded successfully ---")


# ==================================================================
# 3. Define the Preprocessing Function
# ==================================================================
# This function must perform the EXACT same steps as your notebook's AudioDataset

def preprocess_audio(audio_bytes: bytes):
    """Takes raw audio bytes, preprocesses them, and returns a spectrogram tensor."""
    waveform, sample_rate = torchaudio.load(io.BytesIO(audio_bytes))

    # Resample if the sample rate is different from the target
    if sample_rate != TARGET_SAMPLE_RATE:
        resampler = T.Resample(orig_freq=sample_rate, new_freq=TARGET_SAMPLE_RATE)
        waveform = resampler(waveform)
    
    # Mix down to a single (mono) channel if it's stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Pad or truncate the audio to the target length
    target_len = TARGET_SAMPLE_RATE * TARGET_LEN_SECS
    if waveform.shape[1] > target_len:
        waveform = waveform[:, :target_len]
    else:
        waveform = F.pad(waveform, (0, target_len - waveform.shape[1]))
    
    # Create the Mel Spectrogram using the same parameters as in training
    mel_spectrogram_transform = T.MelSpectrogram(
        sample_rate=TARGET_SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    spectrogram = mel_spectrogram_transform(waveform)
    
    # Add a batch dimension for the model (from [channels, height, width] to [1, channels, height, width])
    spectrogram = spectrogram.unsqueeze(0)
    
    return spectrogram


# ==================================================================
# 4. Define the API Endpoints
# ==================================================================

@app.get("/")
def read_root():
    """A root endpoint that provides basic info about the API."""
    return {"message": "Welcome to the Voice Anti-Spoofing API. Navigate to /docs to see the interactive API documentation."}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Accepts an audio file, preprocesses it, runs it through the model,
    and returns the prediction and confidence score.
    """
    # 1. Read the uploaded file's bytes into memory
    audio_bytes = await file.read()
    
    # 2. Preprocess the audio bytes into a spectrogram tensor
    try:
        spectrogram = preprocess_audio(audio_bytes)
    except Exception as e:
        return {"error": f"Failed to process audio file: {str(e)}"}

    # 3. Make a prediction using the loaded model
    with torch.no_grad():
        logits = model(spectrogram.to(device))
        probabilities = torch.softmax(logits, dim=1)
        prediction_idx = torch.argmax(probabilities, dim=1).item()
        
        prediction_class = CLASS_NAMES[prediction_idx]
        confidence = probabilities[0][prediction_idx].item()

    # 4. Return the result in JSON format
    return {
        "filename": file.filename,
        "predicted_class": prediction_class,
        "confidence": confidence
    }