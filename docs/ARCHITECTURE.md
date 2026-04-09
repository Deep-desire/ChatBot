# Architecture: Hybrid Voice + Text RAG Chatbot

## 1. System Overview

This project is an embeddable website widget that supports:

- Text chat: user text -> retrieval -> text reply
- Voice chat: user audio -> STT -> retrieval -> TTS -> audio reply

It is split into two deployable services:

- `frontend/`: React + TypeScript floating chat widget
- `backend/`: FastAPI API with retrieval orchestration and voice processing

## 2. Technology Stack

### Frontend
- React + TypeScript (Vite)
- Tailwind CSS for styling
- Browser MediaRecorder API for microphone capture

### Backend
- FastAPI for API endpoints
- Azure OpenAI chat deployment (`gpt-4o`) for generation
- Azure OpenAI embeddings deployment (`text-embedding-3-small`) for vectorization
- Azure AI Search for vector and keyword retrieval
- Groq Whisper (`whisper-large-v3`) for speech-to-text
- Edge TTS (`en-US-AriaNeural`) for text-to-speech

## 3. Data Flow

### 3.1 Knowledge Ingestion Flow

1. `backend/ingest.py` reads `backend/data.txt`
2. Text is chunked (size 1000, overlap 100)
3. Chunks are embedded with Azure OpenAI embeddings
4. Documents are merged/upserted into Azure AI Search index (`AZURE_SEARCH_INDEX_NAME`)

### 3.2 Text Chat Runtime Flow

1. Frontend sends `POST /api/chat/text` with form field `query`
2. Backend performs Azure AI Search vector retrieval (fallback to text/semantic)
3. Prompt + retrieved context sent to Azure OpenAI chat deployment
4. Backend returns JSON: `{ "reply": "..." }`

### 3.3 Voice Chat Runtime Flow

1. Frontend captures microphone audio in `audio/webm`
2. Frontend sends `POST /api/chat/voice` with form field `audio`
3. Groq Whisper transcribes audio to text
4. Backend performs Azure AI Search retrieval
5. Azure OpenAI generates answer text
6. Edge TTS synthesizes MP3 response
7. Backend streams MP3 back and sets metadata headers:
  - `X-Session-Id`
  - `X-User-Query`
  - `X-Bot-Reply`
  - `X-User-Query-Encoded`
  - `X-Bot-Reply-Encoded`
8. Frontend optionally calls `GET /api/chat/last?session_id=...` to fetch full-fidelity text turn data
9. Frontend calls `GET /api/chat/suggestions?session_id=...` to refresh contextual follow-up question chips

## 4. API Surface

- `GET /health`
  - Returns: `{ "status": "ok" }`
- `POST /api/chat/text`
  - Form: `query`
  - Returns: `{ "reply": "..." }`
- `POST /api/chat/voice`
  - Form: `audio` file
  - Returns: `audio/mpeg` stream + transcript headers
- `GET /api/chat/last`
  - Query: `session_id`
  - Returns: latest user query + assistant reply for that session
- `GET /api/chat/suggestions`
  - Query: `session_id`, optional `limit`
  - Returns: context-aware follow-up questions for UI suggestions

## 5. Environment Variables

Backend variables (`backend/.env`):

- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_CHAT_DEPLOYMENT`
- `AZURE_OPENAI_EMBEDDING_DEPLOYMENT`
- `AZURE_SEARCH_ENDPOINT`
- `AZURE_SEARCH_INDEX_NAME`
- `AZURE_SEARCH_API_KEY` (optional when managed identity is used)
- `GROQ_API_KEY`

Frontend variable (`frontend/.env`):

- `VITE_API_BASE_URL` (for example `http://localhost:8000`)

## 6. Deployment Topology

Most common production topology:

- Backend on Render/Railway/Cloud Run (Docker)
- Frontend on Vercel/Netlify/static host
- Frontend points to backend via `VITE_API_BASE_URL`

## 7. Sequence Diagram

```mermaid
sequenceDiagram
    participant U as User
    participant F as Frontend Widget (React)
    participant B as Backend API (FastAPI)
    participant G as Groq Whisper
    participant S as Azure AI Search
    participant A as Azure OpenAI
    participant T as Edge TTS

    alt Text Chat
        U->>F: Type message
        F->>B: POST /api/chat/text (query)
        B->>S: Retrieve top-k context
        S-->>B: Context docs
        B->>A: Prompt + context + query
        A-->>B: Answer
        B-->>F: JSON reply
        F-->>U: Render bot response
    else Voice Chat
        U->>F: Hold mic and speak
        F->>B: POST /api/chat/voice (webm)
        B->>G: Transcribe audio
        G-->>B: Transcript
        B->>S: Retrieve top-k context
        S-->>B: Context docs
        B->>A: Prompt + context + transcript
        A-->>B: Answer text
        B->>T: Synthesize speech
        T-->>B: MP3
        B-->>F: MP3 stream + response headers
        F-->>U: Play audio + show transcript/reply
    end
```

## 8. Operational Notes

- CORS is currently permissive (`allow_origins=["*"]`) for development.
- Restrict origins before production rollout.
- Azure AI Search field names are configurable using env vars (`AZURE_SEARCH_CONTENT_FIELD`, `AZURE_SEARCH_VECTOR_FIELD`, etc.).
- The backend removes temporary uploaded input files after ingestion.
