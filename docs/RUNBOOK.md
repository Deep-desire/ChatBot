# Ops Runbook

## Routine Workflow

1. Update `backend/data.txt`
2. Activate backend virtual environment
3. Run `python ingest.py`
4. Restart backend if needed
5. Verify with `GET /health`

## Pre-Release Checklist

- Frontend `npm run build` passes
- Backend `python -m py_compile main.py ingest.py` passes
- CORS restricted to production domains
- Environment variables configured in hosting platform
- Smoke tests passed for:
  - Text flow
  - Voice flow

## Incident Quick Checks

### Backend 5xx errors

- Check host logs for stack traces
- Validate external API quotas/limits (Azure OpenAI, Azure AI Search)
- Validate env vars are present

### Voice not transcribing

- Confirm browser sent non-empty audio file
- Verify Groq API key and transcription model
- Check MIME type compatibility (`audio/webm` preferred)

### RAG answers irrelevant

- Re-ingest data
- Ensure right index name
- Confirm retriever settings and prompt instructions

## Rollback Strategy

- Keep last stable deploy in hosting platform
- Roll back service image/build on failure
- Re-validate `GET /health` and one text + one voice request after rollback
