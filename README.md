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
fileciteturn0file0

---

## 📦 Complete Project Repository

This README belongs to the **Machine Learning Core Project**.
To see the **full web application (frontend + backend deployment)**, check the separate repository below:

🔗 **Full Web App Repository:**  
https://github.com/Laabh-Gupta/Voice-Anti-Spoofing-Web-App

This repo contains:
- React-based frontend (Netlify hosted)
- FastAPI backend (Railway hosted)
- API integration with trained model
- Production-ready deployment setup

---

## 🏗️ Tech Stack  

| Layer | Technology |
|------|-------------|
| **ML / Deep Learning** | PyTorch, Torchaudio, Torchvision |
| **Backend API** | FastAPI + Uvicorn |
| **Frontend** | React.js |
| **Deployment** | **Railway (Backend)** + **Netlify (Frontend)** |
| **Others** | Python Multipart, SoundFile |

---

## 🌐 Deployment Details

### 🔹 Backend (FastAPI) – Railway  
1. Set up a FastAPI project  
2. Added `requirements.txt`  
3. Connected GitHub repo to **Railway**  
4. Deployed — Railway automatically builds & hosts the API  
5. Retrieved **public backend URL** (used in frontend)

### 🔹 Frontend (React.js) – Netlify  
1. Added API URL from Railway in React `.env` file  
2. Ran `npm run build`  
3. Deployed directly via Netlify drag & drop / GitHub deploy  
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
fileciteturn0file0

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
pip install "fastapi[all]" uvicorn torch torchaudio torchvision python-multipart soundfile
```

### React Frontend
```bash
npm install
npm start
```

---

## 📝 License  
Licensed under the **MIT License**.
