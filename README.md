# Hybrid Voice + Text RAG Chatbot

Embeddable conversational widget with text + voice, powered by RAG.

- Frontend: React + TypeScript widget in `frontend/`
- Backend: FastAPI + Azure OpenAI + Azure AI Search + Groq (voice STT) + Edge TTS in `backend/`

## Documentation Index

- Architecture: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- Setup and Build: [docs/SETUP_AND_BUILD.md](docs/SETUP_AND_BUILD.md)
- Deployment: [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)
- API Reference: [docs/API_REFERENCE.md](docs/API_REFERENCE.md)
- Ops Runbook: [docs/RUNBOOK.md](docs/RUNBOOK.md)

## Quick Start

### Backend

```powershell
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
python ingest.py
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```







### Frontend

```powershell
cd frontend
npm install
Copy-Item .env.example .env
npm run dev
```
