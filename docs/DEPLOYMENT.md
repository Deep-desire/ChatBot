# Deployment Guide

This guide covers common deployment paths for production.

## 1) Recommended Production Split

- Backend: Render or Railway (Docker service)
- Frontend: Vercel or Netlify (static hosting)

Set frontend `VITE_API_BASE_URL` to the deployed backend URL.

## 2) Backend Deployment (Render/Railway via Docker)

### 2.1 Prepare repository

Ensure repository contains:

- `backend/Dockerfile`
- `backend/requirements.txt`
- `backend/main.py`

### 2.2 Create service

1. Connect GitHub repo in Render/Railway
2. Set root directory to `backend` (or configure equivalent)
3. Platform detects Dockerfile
4. Expose service port `8000`

### 2.3 Configure environment variables

Add these on the platform:

- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_API_VERSION`
- `AZURE_OPENAI_CHAT_DEPLOYMENT`
- `AZURE_OPENAI_EMBEDDING_DEPLOYMENT`
- `AZURE_SEARCH_ENDPOINT`
- `AZURE_SEARCH_INDEX_NAME`
- `AZURE_SEARCH_API_KEY` (optional when managed identity is configured)
- `GROQ_API_KEY`
- `GROQ_TRANSCRIPTION_MODEL`
- `INGEST_API_KEY` (optional but recommended to protect ingestion endpoint)

### 2.4 Verify deployment

- Check `GET /health`
- Run text endpoint test:
  - `POST /api/chat/text` with `query` form field
- Run voice endpoint test with a `.webm` file upload
- Run ingestion endpoint test with a `.pdf` upload (`POST /api/ingest/upload`)

## 3) Frontend Deployment (Vercel/Netlify)

### Build settings

- Root directory: `frontend`
- Build command: `npm run build`
- Output directory: `dist`

### Environment variable

- `VITE_API_BASE_URL=https://your-backend-domain`

### Validate

1. Open deployed frontend URL
2. Confirm text chat works
3. Confirm microphone permission prompt and voice flow

## 4) CORS and Security Hardening (Before Production)

In backend, replace permissive CORS with explicit frontend domains.

Recommended:

- `allow_origins=["https://your-frontend-domain"]`
- Keep `allow_methods` and `allow_headers` only as needed

Also:

- Rotate API keys regularly
- Use platform secret stores (not committed files)
- Add rate limiting/WAF if traffic is public

## 5) Troubleshooting

### 5.1 Backend starts but answers fail

- Check missing/invalid API keys
- Verify Azure AI Search index exists and name matches `AZURE_SEARCH_INDEX_NAME`
- Confirm index contains the configured content/vector fields

### 5.2 Voice endpoint errors

- Ensure incoming mime type is browser-generated `audio/webm`
- Verify `GROQ_API_KEY` and `GROQ_TRANSCRIPTION_MODEL`
- Confirm network egress to Groq is allowed

### 5.3 Empty or poor answers

- Re-run `ingest.py` after updating `data.txt`
- Check that data chunks are present in Azure AI Search index
- Increase retriever `k` in backend if needed

## 6) Optional Single-Host Deploy

You can serve the frontend and backend separately from the same repository, but independent deployments are generally easier to scale and debug.
