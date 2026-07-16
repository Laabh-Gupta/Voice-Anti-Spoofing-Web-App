# 🎙 Voice Anti-Spoofing Web App  
### Detect AI-Generated / Fake Voices in Real-Time  
🌐 **Live Web App:** https://voiceantispoofing.netlify.app/

---

## 🚀 Project Overview  

This is a **full-stack AI-powered web application** that identifies whether a voice is **real or AI-generated (spoofed)**.  

It is built using a **deep learning model trained on spectrograms** of real and fake audio, and deployed as a fully functional web application.

### 🧠 Core Idea  
The backend loads a **fine-tuned PyTorch model**, takes an audio input from the user, preprocesses it into a spectrogram, and predicts whether the voice is **REAL or FAKE** — in real-time.

🧪 The ML workflow included:
- Comparative model training (CNN, deeper CNN, ViT)  
- Fine-tuning to improve generalization  
- Deploying final optimized model for inference  

---

## 📦 Complete Project Repository

This README belongs to the **Machine Learning Core Project**.
To see the **full web application (frontend + backend deployment)**, check the separate repository below:

🔗 **Full Web App Repository:**  
https://github.com/Laabh-Gupta/Voice_Anti_Spoofing_System

This repo contains:
- React-based frontend (Netlify hosted)
- FastAPI backend (Render hosted)
- Model weights hosted on Hugging Face Hub
- API integration with trained model
- Production-ready deployment setup

---

## 🏗️ Tech Stack  

| Layer | Technology |
|------|-------------|
| **ML / Deep Learning** | PyTorch, Torchaudio, Torchvision |
| **Backend API** | FastAPI + Uvicorn |
| **Frontend** | React.js |
| **Model Hosting** | Hugging Face Hub |
| **Deployment** | **Render (Backend)** + **Netlify (Frontend)** |
| **Others** | Python Multipart, SoundFile |

---

## 🌐 Deployment Details

### 🔹 Model – Hugging Face Hub
1. Trained model weights + architecture code (`model.py`) pushed to [Hugging Face Hub](https://huggingface.co/LaabhGupta/voice-antispoofing)  
2. Backend downloads weights at startup via `huggingface_hub.hf_hub_download()`  
3. Keeps large model files out of the Git repo and deployment builds

### 🔹 Backend (FastAPI) – Render  
1. Set up a FastAPI project  
2. Added `requirements.txt` (CPU-only PyTorch builds to keep deploys lean)  
3. Connected GitHub repo to **Render**  
4. Deployed — Render automatically builds & hosts the API  
5. Retrieved **public backend URL** (used in frontend)

### 🔹 Frontend (React.js) – Netlify  
1. Added backend API URL as an environment variable in Netlify (`REACT_APP_BACKEND_URL`)  
2. Ran `npm run build`  
3. Deployed directly via Netlify GitHub integration  
4. Web App goes live instantly 🚀

---

## 🖥️ Live Demo  
🔗 https://voiceantispoofing.netlify.app/  
Upload any voice → Get **FAKE / REAL** prediction in seconds.

---

## 🧠 Model Training Summary  
From the original ML project:  
- **Three models were trained & compared**:
  1. Baseline CNN  
  2. Deeper CNN  
  3. Vision Transformer (ViT)  
- **Fine-tuning improved performance** significantly  

| Model | Final Test Accuracy |
|-------|---------------------|
| Baseline CNN (Fine-tuned) | **99.51%** |
| Deeper CNN (Fine-tuned) | **99.63%** |
| Vision Transformer (Fine-tuned) | **99.75%** |

---

## 📁 Project Structure

```
VOICE-ANTI-SPOOFING/
│── backend/        # FastAPI + Model
│── frontend/       # React App
│── requirements.txt
│── README.md
```

---

## ⚙️ Requirements

### Python Backend
```bash
pip install "fastapi[all]" uvicorn torch torchaudio torchvision python-multipart soundfile huggingface_hub
```

### React Frontend
```bash
npm install
npm start
```

---

## 📝 License  
Licensed under the **MIT License**.