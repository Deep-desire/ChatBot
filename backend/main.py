import os
import re
import shutil
import uuid
import logging
import json
import base64
import warnings
from contextvars import ContextVar, Token
from datetime import datetime, timezone
from collections import deque
from functools import lru_cache
from pathlib import Path
from threading import Lock
from time import time
from typing import Any, AsyncGenerator, Iterable
from urllib.parse import quote, unquote

warnings.filterwarnings(
    "ignore",
    message=r"invalid escape sequence '\\W'",
    category=SyntaxWarning,
)

from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
import edge_tts
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from groq import Groq
from ingestion import ingest_file
from langchain_openai import AzureOpenAIEmbeddings
from openai import AzureOpenAI

load_dotenv()
logger = logging.getLogger(__name__)

app = FastAPI(title="Hybrid Voice + Text RAG Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "X-Session-Id",
        "X-Trace-Id",
        "X-User-Query",
        "X-Bot-Reply",
        "X-User-Query-Encoded",
        "X-Bot-Reply-Encoded",
    ],
)

SUPPORTED_INGEST_EXTENSIONS = {".pdf", ".txt", ".md", ".csv", ".log"}

SERVICE_SUMMARY = (
    "Desire Infoweb provides Microsoft-focused IT services including SharePoint, "
    "Power Apps, Power Automate, Power BI, Office 365, Teams, Dynamics 365, Azure, "
    ".NET, migration, automation, and AI/chatbot solutions."
)

AI_SUMMARY = (
    "Desire Infoweb AI services include Azure OpenAI-based solutions, Teams chatbots, "
    "Copilot-aligned workflows, intelligent automation, and document-grounded chatbot implementations."
)

AI_PROJECTS_SUMMARY = (
    "Some AI project examples from Desire Infoweb include: "
    "(1) a Microsoft Teams chatbot integrated with ChatGPT, and "
    "(2) a document-grounded chatbot using SharePoint/Azure Blob as data sources "
    "to provide responses based on uploaded files."
)

BUDGET_SUMMARY = (
    "Budget depends on scope, integrations, data volume, and deployment model. "
    "For an AI chatbot, we usually start with a discovery session and then share a tailored estimate "
    "with timeline and milestones. If you share your use case, channels (website/Teams/WhatsApp), "
    "and expected users, we can provide a more accurate proposal."
)

DOTNET_SUMMARY = (
    "Desire Infoweb .NET services include custom enterprise application development, "
    "secure and scalable backend systems, workflow and approval systems, and modernization of existing applications."
)

CHATBOT_IMPLEMENTATION_SUMMARY = (
    "For a typical business chatbot project, we usually deliver: "
    "discovery and requirements, data ingestion from documents/web/SharePoint, "
    "RAG-based answer engine, website or Teams chat interface, optional voice support, "
    "testing, and production deployment."
)

CHATBOT_DATA_SOURCE_SUMMARY = (
    "Yes, chatbot data can come from SharePoint. We commonly use SharePoint libraries/sites, "
    "Azure Blob storage, PDFs, Word/Excel files, and website content as knowledge sources. "
    "Then we index that content so answers are grounded in your business data."
)

INDUSTRY_SUMMARY = (
    "Desire Infoweb serves industries such as education, retail/e-commerce, finance, "
    "real estate, travel, healthcare, and logistics/distribution."
)

DEFAULT_FOLLOWUP_QUESTIONS = [
    "What services does Desire Infoweb provide?",
    "What is Desire Infoweb?",
    "What type of AI solutions does Desire create?",
]

_conversation_lock = Lock()
_conversation_store: dict[str, deque[tuple[str, str]]] = {}
_lead_lock = Lock()
_lead_store: dict[str, dict[str, str]] = {}
_graph_token_lock = Lock()
_graph_token: dict[str, float | str] = {"access_token": "", "expires_at": 0.0}
_trace_log_lock = Lock()
_active_trace: ContextVar[dict[str, Any] | None] = ContextVar("active_chat_trace", default=None)


class RetrievalUnavailableError(RuntimeError):
    """Raised when strict retrieval mode requires Azure AI Search context."""


NO_CONTEXT_RESPONSE = (
    "Thank you for your query. At the moment, I'm unable to provide a relevant response "
    "as it falls outside my current scope.\n\n"
    "For further assistance, please contact our support team:\n"
    "- vijay@desireinfoweb.com\n"
    "- hr@desireinfoweb.in\n"
    "- info@desireinfoweb.com\n"
    "- India: +91-8780468807\n"
    "- USA: +1 260 560 2128\n\n"
    "We will be happy to assist you further."
)

OVERLAP_STOPWORDS = {
    "the", "and", "for", "with", "from", "into", "that", "this", "what", "when", "where", "which",
    "about", "your", "you", "have", "has", "are", "was", "were", "will", "would", "could", "should",
    "give", "detail", "details", "please", "need", "want", "tell", "me", "our", "their", "they",
    "first", "day", "procedure", "process", "step", "steps",
}


def _get_required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def _get_required_env_any(names: list[str]) -> str:
    for name in names:
        value = os.getenv(name)
        if value and value.strip():
            return value.strip()
    joined = ", ".join(names)
    raise ValueError(f"Missing required environment variable. Set one of: {joined}")


def _is_env_true(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _is_chat_trace_enabled() -> bool:
    return _is_env_true("CHAT_TRACE_ENABLED", "true")


def _is_chat_trace_console_enabled() -> bool:
    return _is_env_true("CHAT_TRACE_PRINT_CONSOLE", "true")


def _is_chat_trace_include_context() -> bool:
    return _is_env_true("CHAT_TRACE_INCLUDE_CONTEXT", "true")


def _get_chat_trace_log_path() -> Path:
    configured_path = os.getenv("CHAT_TRACE_LOG_PATH", "logs/chat_trace.jsonl").strip() or "logs/chat_trace.jsonl"
    path = Path(configured_path)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parent / path


def _get_chat_trace_clip_chars() -> int:
    raw_value = os.getenv("CHAT_TRACE_CLIP_CHARS", "12000")
    try:
        value = int(raw_value)
    except ValueError:
        value = 12000
    return max(500, min(value, 100000))


def _clip_text(value: str, max_chars: int | None = None) -> str:
    limit = max_chars if max_chars is not None else _get_chat_trace_clip_chars()
    normalized = value.replace("\r\n", "\n").replace("\r", "\n")
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit] + "\n...[truncated]"


def _sanitize_trace_value(value: Any) -> Any:
    if isinstance(value, str):
        return _clip_text(value)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _sanitize_trace_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_trace_value(v) for v in value]
    return _clip_text(str(value))


def _build_trace_record(endpoint: str, raw_query: str, session_id: str | None, *, streaming: bool) -> dict[str, Any] | None:
    if not _is_chat_trace_enabled():
        return None
    return {
        "trace_id": str(uuid.uuid4()),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "started_at_epoch": time(),
        "endpoint": endpoint,
        "streaming": streaming,
        "raw_query": _clip_text(raw_query),
        "input_session_id": (session_id or "").strip(),
        "steps": [],
    }


def _activate_trace(trace: dict[str, Any] | None) -> Token | None:
    if not trace:
        return None
    token = _active_trace.set(trace)
    _trace_step("request.received", endpoint=trace.get("endpoint"), streaming=trace.get("streaming"))
    return token


def _get_active_trace() -> dict[str, Any] | None:
    return _active_trace.get()


def _get_active_trace_id() -> str:
    trace = _get_active_trace()
    if not trace:
        return ""
    return str(trace.get("trace_id") or "")


def _trace_step(step: str, **details: Any) -> None:
    trace = _get_active_trace()
    if not trace:
        return
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "step": step,
    }
    for key, value in details.items():
        payload[key] = _sanitize_trace_value(value)
    trace.setdefault("steps", []).append(payload)


def _persist_trace(trace: dict[str, Any]) -> None:
    output_path = _get_chat_trace_log_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(trace, ensure_ascii=False)
    with _trace_log_lock:
        with output_path.open("a", encoding="utf-8") as handle:
            handle.write(serialized + "\n")


def _finalize_trace(status: str, **summary: Any) -> None:
    trace = _get_active_trace()
    if not trace:
        return
    trace["status"] = status
    trace["ended_at"] = datetime.now(timezone.utc).isoformat()
    trace["duration_ms"] = round((time() - float(trace.get("started_at_epoch", time()))) * 1000, 2)
    trace["summary"] = {key: _sanitize_trace_value(value) for key, value in summary.items()}
    trace.pop("started_at_epoch", None)
    _persist_trace(trace)
    if _is_chat_trace_console_enabled():
        logger.info(
            "chat trace %s status=%s endpoint=%s duration_ms=%s",
            trace.get("trace_id"),
            status,
            trace.get("endpoint"),
            trace.get("duration_ms"),
        )


def _deactivate_trace(token: Token | None) -> None:
    if token is None:
        return
    try:
        _active_trace.reset(token)
    except ValueError:
        # Streaming generators can finalize in a different async context.
        # Fallback to clearing the current context value without failing the request.
        _active_trace.set(None)


def _sanitize_header_value(value: str, *, max_chars: int = 700) -> str:
    normalized = value.replace("\r", " ").replace("\n", " ").strip()
    normalized = normalized[:max_chars]
    return normalized.encode("latin1", "ignore").decode("latin1")


def _encode_header_value(value: str, *, max_chars: int = 2500) -> str:
    normalized = value.replace("\r\n", "\n").replace("\r", "\n").strip()
    normalized = normalized[:max_chars]
    return quote(normalized, safe="")


def _normalize_user_query(query: str) -> str:
    normalized = query.strip()
    replacements = {
        "serivce": "service",
        "serivces": "services",
        "qhat": "what",
        "wht": "what",
        "u": "you",
    }
    words = [replacements.get(token.lower(), token) for token in normalized.split()]
    return " ".join(words)


def _normalize_session_id(session_id: str | None) -> str:
    value = (session_id or "").strip()
    if not value:
        return "default"
    return re.sub(r"[^a-zA-Z0-9_-]", "", value)[:64] or "default"


def _normalize_lead_email(email: str | None) -> str:
    value = (email or "").strip().lower()
    if not value:
        return ""
    if not re.fullmatch(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", value):
        raise HTTPException(status_code=400, detail="Invalid email format")
    return value


def _normalize_lead_name(name: str | None) -> str:
    value = (name or "").strip()
    return re.sub(r"\s+", " ", value)[:120]


def _resolve_lead_identity(session_id: str, email: str | None, name: str | None) -> tuple[str, str]:
    normalized_email = _normalize_lead_email(email)
    normalized_name = _normalize_lead_name(name)

    with _lead_lock:
        if session_id not in _lead_store:
            _lead_store[session_id] = {
                "email": "",
                "name": "",
                "created_at": datetime.now(timezone.utc).isoformat(),
            }

        if normalized_email:
            _lead_store[session_id]["email"] = normalized_email
        if normalized_name:
            _lead_store[session_id]["name"] = normalized_name

        return _lead_store[session_id]["email"], _lead_store[session_id]["name"]


def _build_conversation_transcript(session_id: str) -> str:
    with _conversation_lock:
        history = list(_conversation_store.get(session_id, []))

    lines: list[str] = []
    for user_text, assistant_text in history:
        lines.append(f"User: {user_text}")
        lines.append(f"Assistant: {assistant_text}")

    return "\n".join(lines)


def _get_last_conversation_turn(session_id: str) -> tuple[str, str] | None:
    with _conversation_lock:
        history = _conversation_store.get(session_id)
        if not history:
            return None
        return history[-1]


def _normalize_question_for_compare(question: str) -> str:
    return re.sub(r"\W+", "", question).lower()


def _build_dynamic_followup_questions(session_id: str, limit: int = 3) -> list[str]:
    with _conversation_lock:
        history = list(_conversation_store.get(session_id, []))

    if not history or limit <= 0:
        return []

    suggestions: list[str] = []
    seen: set[str] = set()

    # Use only previous user prompts from the same session history.
    if len(history) <= 1:
        return []

    for user_text, _ in reversed(history[:-1]):
        candidate = user_text.strip()
        if not candidate:
            continue
        key = _normalize_question_for_compare(candidate)
        if not key or key in seen:
            continue
        suggestions.append(candidate)
        seen.add(key)
        if len(suggestions) >= limit:
            break

    return suggestions[:limit]


def _is_sharepoint_sync_enabled() -> bool:
    return os.getenv("ENABLE_SHAREPOINT_SYNC", "false").lower() == "true"


def _is_sharepoint_always_insert_enabled() -> bool:
    return os.getenv("SHAREPOINT_ALWAYS_INSERT", "true").lower() == "true"


def _get_sharepoint_field_names() -> dict[str, str]:
    return {
        "title": os.getenv("SHAREPOINT_FIELD_TITLE", "Title").strip() or "Title",
        "name": os.getenv("SHAREPOINT_FIELD_NAME", "Name").strip() or "Name",
        "email": os.getenv("SHAREPOINT_FIELD_EMAIL", "email").strip() or "email",
        "conversation": os.getenv("SHAREPOINT_FIELD_CONVERSATION", "Conversation").strip() or "Conversation",
    }


def _get_graph_token() -> str:
    if not _is_sharepoint_sync_enabled():
        return ""

    with _graph_token_lock:
        token_value = str(_graph_token.get("access_token", ""))
        expires_at = float(_graph_token.get("expires_at", 0.0) or 0.0)
        if token_value and expires_at - 60 > time():
            return token_value

    tenant_id = _get_required_env("SHAREPOINT_TENANT_ID")
    client_id = _get_required_env("SHAREPOINT_CLIENT_ID")
    client_secret = _get_required_env("SHAREPOINT_CLIENT_SECRET")
    token_url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"

    response = httpx.post(
        token_url,
        data={
            "client_id": client_id,
            "client_secret": client_secret,
            "grant_type": "client_credentials",
            "scope": "https://graph.microsoft.com/.default",
        },
        timeout=15.0,
    )
    response.raise_for_status()
    payload = response.json()
    access_token = payload.get("access_token")
    if not access_token:
        raise ValueError("Failed to obtain Microsoft Graph access token.")

    expires_in = float(payload.get("expires_in", 3600))
    with _graph_token_lock:
        _graph_token["access_token"] = access_token
        _graph_token["expires_at"] = time() + expires_in

    return access_token


def _graph_request(method: str, url: str, **kwargs) -> dict:
    token = _get_graph_token()
    if not token:
        raise ValueError("SharePoint sync is disabled or missing credentials.")

    headers = dict(kwargs.pop("headers", {}))
    headers["Authorization"] = f"Bearer {token}"
    headers["Accept"] = "application/json"

    response = httpx.request(method, url, headers=headers, timeout=20.0, **kwargs)
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as error:
        detail = ""
        try:
            payload = response.json()
            detail = str(payload.get("error", payload))
        except Exception:
            detail = response.text[:500]
        raise ValueError(
            f"Microsoft Graph request failed ({response.status_code}) for {url}. Detail: {detail}"
        ) from error
    if response.status_code == 204:
        return {}
    return response.json()


def _build_sharepoint_fields(lead_name: str, lead_email: str, transcript: str) -> dict[str, str]:
    field_names = _get_sharepoint_field_names()
    title_value = lead_name or lead_email
    return {
        field_names["title"]: title_value,
        field_names["name"]: lead_name,
        field_names["email"]: lead_email,
        field_names["conversation"]: transcript,
    }


def _find_sharepoint_item_id(site_id: str, list_id: str, email_field: str, email_value: str) -> str | None:
    escaped_email = email_value.replace("'", "''")
    url = (
        "https://graph.microsoft.com/v1.0"
        f"/sites/{site_id}/lists/{list_id}/items"
        f"?$expand=fields&$filter=fields/{email_field} eq '{escaped_email}'"
    )
    payload = _graph_request("GET", url)
    items = payload.get("value", [])
    if not items:
        return None
    return str(items[0].get("id") or "") or None


def _upsert_sharepoint_lead(session_id: str) -> None:
    if not _is_sharepoint_sync_enabled():
        return

    with _lead_lock:
        lead = dict(_lead_store.get(session_id, {}))

    if not lead:
        return

    lead_email = lead.get("email", "").strip()
    lead_name = lead.get("name", "").strip()
    if not lead_email or not lead_name:
        return

    transcript = _build_conversation_transcript(session_id)
    if not transcript:
        return

    site_id = _get_required_env("SHAREPOINT_SITE_ID")
    list_id = _get_required_env("SHAREPOINT_LIST_ID")
    fields_payload = _build_sharepoint_fields(lead_name, lead_email, transcript)

    if _is_sharepoint_always_insert_enabled():
        create_url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/lists/{list_id}/items"
        _graph_request("POST", create_url, json={"fields": fields_payload})
        return

    field_names = _get_sharepoint_field_names()
    item_id = _find_sharepoint_item_id(site_id, list_id, field_names["email"], lead_email)

    if item_id:
        update_url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/lists/{list_id}/items/{item_id}/fields"
        _graph_request("PATCH", update_url, json=fields_payload)
        return

    create_url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/lists/{list_id}/items"
    _graph_request("POST", create_url, json={"fields": fields_payload})


async def _sync_sharepoint_lead_safely(session_id: str) -> None:
    try:
        _upsert_sharepoint_lead(session_id)
    except Exception as error:
        logger.warning("SharePoint sync skipped/failed for session %s: %s", session_id, error)


def _direct_company_answer(query: str) -> str | None:
    q = query.lower().strip()
    compact = re.sub(r"[^a-z0-9\s]", "", q)

    if re.fullmatch(r"(hi|hello|hey|hii|hiii|good morning|good afternoon|good evening)", compact):
        return (
            "Hello! Welcome to Desire Infoweb. "
            f"{SERVICE_SUMMARY} "
            "Tell me your requirement and I can suggest the best service approach."
        )

    if any(keyword in compact for keyword in ["what service", "services", "what do you do", "what you do", "what do you provide", "offer"]):
        return (
            "We provide end-to-end Microsoft technology services: "
            "SharePoint and intranet solutions, Power Platform (Power Apps/Automate), "
            "Power BI analytics, Office 365 and Teams implementation, Dynamics 365, Azure, .NET development, "
            "migration, governance, and AI/chatbot solutions."
        )

    if any(
        phrase in compact
        for phrase in [
            "what is desire infoweb",
            "who is desire infoweb",
            "about desire infoweb",
            "tell me about desire infoweb",
        ]
    ):
        return (
            "Desire Infoweb is an IT services company focused on Microsoft technologies and business automation. "
            f"{SERVICE_SUMMARY}"
        )

    if any(keyword in compact for keyword in ["budget", "cost", "pricing", "price", "estimate", "quotation", "quote"]):
        return BUDGET_SUMMARY

    if any(keyword in compact for keyword in ["build ai chatbot", "want to build ai chatbot", "ai chatbot project", "chatbot project"]):
        return (
            "Great choice. We can build an AI chatbot for your website or Microsoft Teams with your business data as context. "
            "Typical scope includes discovery, data ingestion (PDF/web/SharePoint), prompt tuning, voice/text support, testing, and deployment. "
            "If you share your goal and preferred channel, I can suggest the best implementation approach."
        )

    if any(
        keyword in compact
        for keyword in [
            "ever done",
            "done this type",
            "this type of project",
            "done similar",
            "have done",
            "previous chatbot",
            "chatbot past project",
        ]
    ):
        return AI_PROJECTS_SUMMARY

    if any(keyword in compact for keyword in ["normal chatbot", "just chatbot", "simple chatbot", "basic chatbot"]):
        return CHATBOT_IMPLEMENTATION_SUMMARY

    if any(
        keyword in compact
        for keyword in [
            "sharepoint",
            "data source",
            "where data came",
            "data came from",
            "chatbot where data",
            "data from sharepoint",
        ]
    ) and "chatbot" in compact:
        return CHATBOT_DATA_SOURCE_SUMMARY

    if any(keyword in compact for keyword in ["past project", "case study", "ai project", "previous ai", "what this company ai"]):
        return AI_PROJECTS_SUMMARY

    if any(keyword in compact for keyword in [".net", "dotnet", "net service", "what about net"]):
        return DOTNET_SUMMARY

    if any(keyword in compact for keyword in ["industry", "industries", "domain", "sector"]):
        return INDUSTRY_SUMMARY

    if any(keyword in compact for keyword in [" ai", "ai ", "chatbot", "openai", "copilot", "machine learning", "automation"]):
        return AI_SUMMARY

    return None


def _get_embedding_model() -> str:
    return _get_required_env_any(
        [
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
            "AZURE_OPENAI_EMBED_DEPLOYMENT",
            "AZURE_OPENAI_EMBEDDING_MODEL",
        ]
    )


def _get_chat_model() -> str:
    return _get_required_env_any(
        [
            "AZURE_OPENAI_CHAT_DEPLOYMENT",
            "AZURE_OPENAI_DEPLOYMENT",
            "AZURE_OPENAI_CHAT_MODEL",
        ]
    )


def _get_azure_openai_endpoint() -> str:
    return _get_required_env("AZURE_OPENAI_ENDPOINT").rstrip("/")


def _get_azure_openai_api_key() -> str:
    return _get_required_env("AZURE_OPENAI_API_KEY")


def _get_azure_openai_api_version() -> str:
    return os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")


def _get_azure_search_endpoint() -> str:
    return _get_required_env("AZURE_SEARCH_ENDPOINT")


def _get_azure_search_index_name() -> str:
    return _get_required_env("AZURE_SEARCH_INDEX_NAME")


def _get_azure_search_api_key() -> str:
    return os.getenv("AZURE_SEARCH_API_KEY", "").strip()


def _get_azure_search_content_field() -> str:
    return os.getenv("AZURE_SEARCH_CONTENT_FIELD", "content").strip() or "content"


def _get_azure_search_vector_field() -> str:
    return os.getenv("AZURE_SEARCH_VECTOR_FIELD", "contentVector").strip()


def _get_azure_search_top_k() -> int:
    raw_value = os.getenv("AZURE_SEARCH_TOP_K", "5")
    try:
        top_k = int(raw_value)
    except ValueError:
        top_k = 5
    return max(1, min(top_k, 20))


def _get_azure_search_semantic_config() -> str:
    return os.getenv("AZURE_SEARCH_SEMANTIC_CONFIG", "").strip()


def _use_azure_search_semantic() -> bool:
    return _is_env_true("AZURE_SEARCH_USE_SEMANTIC", "false")


def _is_azure_search_required() -> bool:
    return _is_env_true("AZURE_SEARCH_REQUIRED", "false")


def _allow_generic_fallback() -> bool:
    return _is_env_true("ALLOW_GENERIC_FALLBACK", "false")


def _get_transcription_model() -> str:
    return os.getenv("GROQ_TRANSCRIPTION_MODEL", "whisper-large-v3")


def _get_tts_voice() -> str:
    return os.getenv("EDGE_TTS_VOICE", "en-US-AriaNeural")


def _get_max_output_tokens() -> int:
    requested_raw = os.getenv("LLM_MAX_OUTPUT_TOKENS", "1200")
    model_cap_raw = os.getenv("AZURE_OPENAI_MAX_COMPLETION_TOKENS", "16384")

    try:
        requested_tokens = int(requested_raw)
    except ValueError:
        requested_tokens = 1200

    try:
        model_cap = int(model_cap_raw)
    except ValueError:
        model_cap = 16384

    bounded_tokens = max(64, min(requested_tokens, model_cap))
    if bounded_tokens != requested_tokens:
        logger.warning(
            "LLM_MAX_OUTPUT_TOKENS=%s exceeds allowed range; using %s instead.",
            requested_tokens,
            bounded_tokens,
        )

    return bounded_tokens


def _get_llm_temperature() -> float:
    return float(os.getenv("AZURE_OPENAI_TEMPERATURE", "0.1"))


def _get_embedding_similarity_threshold() -> float:
    raw_value = os.getenv(
        "AZURE_SEARCH_SCORE_THRESHOLD",
        os.getenv("EMBEDDING_SIMILARITY_THRESHOLD", "0.2"),
    )
    try:
        threshold = float(raw_value)
    except ValueError:
        threshold = 0.2
    return max(0.0, threshold)


def _get_query_overlap_threshold() -> float:
    raw_value = os.getenv("QUERY_OVERLAP_THRESHOLD", "0.18")
    try:
        threshold = float(raw_value)
    except ValueError:
        threshold = 0.18
    return max(0.0, min(threshold, 1.0))


def _get_query_min_overlap_terms() -> int:
    raw_value = os.getenv("QUERY_MIN_OVERLAP_TERMS", "2")
    try:
        count = int(raw_value)
    except ValueError:
        count = 2
    return max(1, min(count, 20))


def _tokenize_terms(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]{3,}", value.lower())
        if token not in OVERLAP_STOPWORDS
    }


def _compute_query_overlap(query: str, context: str) -> tuple[float, int, int]:
    query_terms = _tokenize_terms(query)
    if not query_terms:
        return 0.0, 0, 0
    context_terms = _tokenize_terms(context[:20000])
    if not context_terms:
        return 0.0, 0, len(query_terms)
    overlap_count = len(query_terms & context_terms)
    return overlap_count / len(query_terms), overlap_count, len(query_terms)


def _get_memory_turns() -> int:
    return int(os.getenv("CONVERSATION_MEMORY_TURNS", "6"))


def _build_model_input(session_id: str, current_query: str) -> str:
    with _conversation_lock:
        history = list(_conversation_store.get(session_id, []))

    if not history:
        return current_query

    history_lines: list[str] = []
    for user_text, assistant_text in history:
        history_lines.append(f"User: {user_text}")
        history_lines.append(f"Assistant: {assistant_text}")

    return (
        "Conversation history:\n"
        + "\n".join(history_lines)
        + "\n\nCurrent user question:\n"
        + current_query
    )


def _sse_event(event: str, data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def _save_conversation_turn(session_id: str, user_text: str, assistant_text: str) -> None:
    with _conversation_lock:
        if session_id not in _conversation_store:
            _conversation_store[session_id] = deque(maxlen=_get_memory_turns())
        _conversation_store[session_id].append((user_text, assistant_text))


system_prompt = (
    "You are Desire Infoweb's professional virtual assistant for an IT services company. "
    "Answer the user's exact question directly and clearly using only company context. "
    "Do not start with generic filler like 'Would you like to know more?'. "
    "Always return the final answer in valid GitHub-flavored Markdown (GFM). "
    "Use clean Markdown structure with short paragraphs and bullet points when useful. "
    "Do not output raw HTML. Do not output JSON unless the user explicitly asks for JSON. "
    "If the user asks about services, provide concrete service categories first. "
    "If the user asks about AI, explain Desire Infoweb AI offerings specifically. "
    "If the user asks about budget/cost, explain that pricing depends on scope and ask for key requirements. "
    "If the user asks about previous projects, provide relevant examples from available context. "
    "For follow-up questions, continue in context and avoid repeating generic summaries. "
    "If you do not know, say that clearly and offer to connect the user with the team. "
    "Keep answers business-focused, friendly, and practical. Prefer complete answers (around 3-8 sentences) when useful.\n\n"
    "Context: {context}"
)


@lru_cache(maxsize=1)
def get_azure_openai_client() -> AzureOpenAI:
    return AzureOpenAI(
        api_version=_get_azure_openai_api_version(),
        azure_endpoint=_get_azure_openai_endpoint(),
        api_key=_get_azure_openai_api_key(),
    )


@lru_cache(maxsize=1)
def get_groq_client() -> Groq:
    return Groq(api_key=_get_required_env("GROQ_API_KEY"))


@lru_cache(maxsize=1)
def get_embeddings_client() -> AzureOpenAIEmbeddings:
    embedding_deployment = _get_embedding_model()
    return AzureOpenAIEmbeddings(
        azure_endpoint=_get_azure_openai_endpoint(),
        api_key=_get_azure_openai_api_key(),
        openai_api_version=_get_azure_openai_api_version(),
        azure_deployment=embedding_deployment,
        model=embedding_deployment,
    )


@lru_cache(maxsize=1)
def get_search_client() -> SearchClient:
    api_key = _get_azure_search_api_key()
    if api_key:
        credential = AzureKeyCredential(api_key)
    else:
        credential = DefaultAzureCredential()

    return SearchClient(
        endpoint=_get_azure_search_endpoint(),
        index_name=_get_azure_search_index_name(),
        credential=credential,
    )


def _extract_content_from_payload(payload: dict) -> str:
    configured_field = _get_azure_search_content_field()
    candidate_fields = [
        configured_field,
        "chunk",
        "content",
        "text",
        "body",
    ]

    seen: set[str] = set()
    for field_name in candidate_fields:
        if not field_name or field_name in seen:
            continue
        seen.add(field_name)
        value = str(payload.get(field_name) or "").strip()
        if value:
            return value

    return ""


def _looks_like_url(value: str) -> bool:
    lowered = value.lower().strip()
    return lowered.startswith("http://") or lowered.startswith("https://")


def _normalize_citation_url(value: str) -> str:
    raw = (value or "").strip().strip('"').strip("'")
    if not raw:
        return ""

    if _looks_like_url(raw):
        return raw.replace(" ", "%20")

    decoded = unquote(raw).strip()
    if _looks_like_url(decoded):
        return decoded.replace(" ", "%20")

    if raw.lower().startswith("www."):
        return f"https://{raw}".replace(" ", "%20")

    if decoded.lower().startswith("www."):
        return f"https://{decoded}".replace(" ", "%20")

    # Handle raw host/path values such as desirechatbotweb.blob.core.windows.net/container/file.pdf
    if re.match(r"^[a-z0-9.-]+\.[a-z]{2,}/", raw, flags=re.IGNORECASE):
        return f"https://{raw}".replace(" ", "%20")

    if re.match(r"^[a-z0-9.-]+\.[a-z]{2,}/", decoded, flags=re.IGNORECASE):
        return f"https://{decoded}".replace(" ", "%20")

    return ""


def _try_decode_base64_to_url(value: str) -> str:
    candidate = (value or "").strip()
    if not candidate:
        return ""

    variants: list[str] = [candidate]
    variants.append(re.sub(r"_pages_\d+$", "", candidate, flags=re.IGNORECASE))
    variants.append(re.sub(r"(?:_)?\d+$", "", candidate))

    # Some IDs embed the encoded URL inside an underscore-delimited token.
    for token in re.split(r"[_|]", candidate):
        token = token.strip()
        if len(token) >= 12:
            variants.append(token)
            variants.append(re.sub(r"(?:_)?\d+$", "", token))

    seen: set[str] = set()
    for item in variants:
        fragment = item.strip()
        if not fragment or fragment in seen:
            continue
        seen.add(fragment)

        padded = fragment + ("=" * (-len(fragment) % 4))
        for decoder in (base64.b64decode, base64.urlsafe_b64decode):
            try:
                decoded = decoder(padded).decode("utf-8", errors="ignore").strip()
            except Exception:
                continue

            normalized = _normalize_citation_url(decoded)
            if normalized:
                return normalized

    return ""


def _decode_parent_id_to_url(parent_id: str) -> str:
    encoded = (parent_id or "").strip()
    if not encoded:
        return ""

    normalized = _normalize_citation_url(encoded)
    if normalized:
        return normalized

    return _try_decode_base64_to_url(encoded)


def _extract_citation_from_payload(payload: dict) -> dict[str, Any]:
    title_candidates = [
        str(payload.get("title") or "").strip(),
        str(payload.get("source") or "").strip(),
        str(payload.get("file_name") or "").strip(),
        str(payload.get("document_name") or "").strip(),
    ]
    title = next((item for item in title_candidates if item), "Source document")

    link_candidates = [
        str(payload.get("url") or "").strip(),
        str(payload.get("source_url") or "").strip(),
        str(payload.get("source") or "").strip(),
        str(payload.get("metadata_storage_path") or "").strip(),
        str(payload.get("parent_id") or "").strip(),
        str(payload.get("id") or "").strip(),
        str(payload.get("chunk_id") or "").strip(),
        _decode_parent_id_to_url(str(payload.get("parent_id") or "").strip()),
    ]

    link = ""
    for item in link_candidates:
        normalized = _normalize_citation_url(item)
        if normalized:
            link = normalized
            break

        decoded = _try_decode_base64_to_url(item)
        if decoded:
            link = decoded
            break

    try:
        score = float(payload.get("@search.score") or 0.0)
    except (TypeError, ValueError):
        score = 0.0

    return {
        "title": title,
        "url": link,
        "id": str(payload.get("chunk_id") or payload.get("id") or "").strip(),
        "score": round(score, 6),
    }


def _extract_context_from_results(results, top_k: int) -> tuple[str, float, list[dict[str, Any]]]:
    context_chunks: list[str] = []
    top_score = 0.0
    citations: list[dict[str, Any]] = []
    seen_citations: set[tuple[str, str]] = set()

    for result in results:
        payload = dict(result)

        try:
            score_value = float(payload.get("@search.score") or 0.0)
        except (TypeError, ValueError):
            score_value = 0.0

        if score_value > top_score:
            top_score = score_value

        page_content = _extract_content_from_payload(payload)
        if not page_content:
            continue

        citation = _extract_citation_from_payload(payload)
        citation_key = (str(citation.get("title") or ""), str(citation.get("url") or ""))
        if citation_key not in seen_citations:
            seen_citations.add(citation_key)
            citations.append(citation)

        context_chunks.append(page_content[:2000])
        if len(context_chunks) >= top_k:
            break

    return "\n\n".join(context_chunks), top_score, citations[:top_k]


def _search_vector_context(query: str, top_k: int) -> tuple[str, float, list[dict[str, Any]]]:
    vector_field = _get_azure_search_vector_field()
    if not vector_field:
        return "", 0.0, []

    query_vector = get_embeddings_client().embed_query(query)
    vector_query = VectorizedQuery(
        vector=query_vector,
        k_nearest_neighbors=top_k,
        fields=vector_field,
    )
    results = get_search_client().search(
        search_text=None,
        vector_queries=[vector_query],
        top=top_k,
    )
    return _extract_context_from_results(results, top_k)


def _search_text_context(query: str, top_k: int) -> tuple[str, float, list[dict[str, Any]]]:
    semantic_config = _get_azure_search_semantic_config()
    if _use_azure_search_semantic() and semantic_config:
        results = get_search_client().search(
            search_text=query,
            top=top_k,
            query_type="semantic",
            semantic_configuration_name=semantic_config,
        )
    else:
        results = get_search_client().search(
            search_text=query,
            top=top_k,
        )

    return _extract_context_from_results(results, top_k)


def _retrieve_context_and_score(query: str) -> tuple[str, float, list[dict[str, Any]]]:
    top_k = _get_azure_search_top_k()
    _trace_step(
        "retrieval.start",
        top_k=top_k,
        vector_field=_get_azure_search_vector_field(),
        content_field=_get_azure_search_content_field(),
    )

    try:
        _trace_step("retrieval.vector.start")
        context, score, citations = _search_vector_context(query, top_k)
        if context:
            _trace_step("retrieval.vector.success", top_score=score, context_chars=len(context))
            _trace_step("retrieval.citations", count=len(citations))
            return context, score, citations
        _trace_step("retrieval.vector.empty")
    except Exception as retriever_error:
        logger.warning("Vector retrieval failed (%s). Falling back to text retrieval.", retriever_error)
        _trace_step("retrieval.vector.error", error=str(retriever_error))

    try:
        _trace_step("retrieval.text.start")
        context, score, citations = _search_text_context(query, top_k)
        _trace_step("retrieval.text.result", top_score=score, context_chars=len(context))
        _trace_step("retrieval.citations", count=len(citations))
        return context, score, citations
    except Exception as retriever_error:
        logger.warning("Text retrieval failed (%s).", retriever_error)
        _trace_step("retrieval.text.error", error=str(retriever_error))
        if _is_azure_search_required():
            raise RetrievalUnavailableError(
                "Azure Search retrieval failed. Verify AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_INDEX_NAME, "
                "network access, and AZURE_SEARCH_API_KEY or managed identity permissions."
            ) from retriever_error
        return "", 0.0, []


def _should_use_embedding_context(normalized_query: str, retrieved_context: str, top_score: float) -> bool:
    if not retrieved_context:
        _trace_step("context.rejected", reason="empty_context")
        return False
    score_threshold = _get_embedding_similarity_threshold()
    if top_score < score_threshold:
        _trace_step(
            "context.rejected",
            reason="score_below_threshold",
            top_score=top_score,
            score_threshold=score_threshold,
        )
        return False

    overlap_ratio, overlap_count, query_term_count = _compute_query_overlap(normalized_query, retrieved_context)
    overlap_threshold = _get_query_overlap_threshold()
    min_overlap_terms = _get_query_min_overlap_terms()

    if query_term_count == 0:
        _trace_step(
            "context.rejected",
            reason="query_has_no_meaningful_terms",
            min_overlap_terms=min_overlap_terms,
            query_term_count=query_term_count,
        )
        return False

    required_overlap_terms = min(min_overlap_terms, query_term_count)
    if overlap_count < required_overlap_terms:
        _trace_step(
            "context.rejected",
            reason="query_overlap_terms_below_minimum",
            overlap_count=overlap_count,
            min_overlap_terms=required_overlap_terms,
            configured_min_overlap_terms=min_overlap_terms,
            query_term_count=query_term_count,
        )
        return False

    if overlap_ratio < overlap_threshold:
        _trace_step(
            "context.rejected",
            reason="query_overlap_below_threshold",
            overlap_ratio=round(overlap_ratio, 4),
            overlap_threshold=overlap_threshold,
            overlap_count=overlap_count,
            query_term_count=query_term_count,
        )
        return False

    _trace_step(
        "context.accepted",
        top_score=top_score,
        score_threshold=score_threshold,
        overlap_ratio=round(overlap_ratio, 4),
        overlap_threshold=overlap_threshold,
        overlap_count=overlap_count,
        min_overlap_terms=required_overlap_terms,
        configured_min_overlap_terms=min_overlap_terms,
        context_chars=len(retrieved_context),
    )
    return True


def _raise_if_strict_without_context(use_embedding_context: bool, retrieved_context: str, top_score: float) -> None:
    if not _is_azure_search_required():
        return
    if use_embedding_context:
        return

    if retrieved_context.strip():
        _trace_step(
            "context.strict_mode",
            decision="return_no_context_response",
            reason="context_present_but_not_relevant_enough",
            top_score=top_score,
        )
        return

    raise RetrievalUnavailableError(
        "Azure Search strict mode is enabled, but no relevant indexed context was retrieved. "
        "Check ingestion data, AZURE_SEARCH_CONTENT_FIELD/AZURE_SEARCH_VECTOR_FIELD mappings, "
        "and AZURE_SEARCH_SCORE_THRESHOLD."
    )


def _no_context_response() -> str:
    return os.getenv("NO_CONTEXT_RESPONSE", NO_CONTEXT_RESPONSE).strip() or NO_CONTEXT_RESPONSE


def _should_attach_citations(answer: str, normalized_query: str, citations: list[dict[str, Any]]) -> bool:
    if not citations:
        return False

    normalized_answer = answer.strip()
    if normalized_answer == _no_context_response().strip():
        return False

    return True


def _build_retrieved_context(query: str) -> str:
    context, _, _ = _retrieve_context_and_score(query)
    return context


def _build_runtime_issue_message(error: Exception) -> str:
    error_text = str(error).strip()
    lowered = error_text.lower()

    if error_text.startswith("Missing required environment variable"):
        missing_name = error_text.split(":", 1)[-1].strip() if ":" in error_text else error_text
        return (
            "Server configuration issue detected. "
            f"{missing_name}."
        )

    if "429" in lowered or "rate limit" in lowered:
        return "The AI service is temporarily rate-limited. Please try again in a moment."

    if "timeout" in lowered or "timed out" in lowered:
        return "The AI service timed out. Please try again."

    if "401" in lowered or "unauthorized" in lowered or "forbidden" in lowered:
        return "Authentication with an upstream AI service failed. Please verify deployed credentials."

    return "Temporary answer generation issue encountered. A safe fallback response was returned."


def _generate_completion_with_context(model_input: str, retrieved_context: str) -> str:
    _trace_step(
        "llm.completion.request",
        model=_get_chat_model(),
        context_chars=len(retrieved_context),
    )
    completion = get_azure_openai_client().chat.completions.create(
        model=_get_chat_model(),
        messages=[
            {"role": "system", "content": system_prompt.format(context=retrieved_context)},
            {"role": "user", "content": model_input},
        ],
        temperature=_get_llm_temperature(),
        max_tokens=_get_max_output_tokens(),
        stream=False,
    )

    if not completion.choices:
        raise ValueError("Completion response returned no choices.")

    message = completion.choices[0].message
    answer = str(getattr(message, "content", "") or "").strip()
    if not answer:
        raise ValueError("Completion response returned an empty answer.")

    if _is_chat_trace_include_context():
        _trace_step("llm.completion.success", answer=answer, answer_chars=len(answer))
    else:
        _trace_step("llm.completion.success", answer_chars=len(answer))

    return answer


def _stream_answer_tokens(
    model_input: str,
    normalized_query: str,
    retrieved_context: str | None = None,
    top_score: float | None = None,
) -> Iterable[str]:
    if retrieved_context is None or top_score is None:
        retrieved_context, top_score, _ = _retrieve_context_and_score(normalized_query)
    use_embedding_context = _should_use_embedding_context(normalized_query, retrieved_context, top_score)
    _raise_if_strict_without_context(use_embedding_context, retrieved_context, top_score)

    if not use_embedding_context:
        _trace_step("answer.no_context", source="no_context_response")
        yield _no_context_response()
        return

    try:
        _trace_step("llm.stream.request", model=_get_chat_model(), context_chars=len(retrieved_context))
        completion_stream = get_azure_openai_client().chat.completions.create(
            model=_get_chat_model(),
            messages=[
                {"role": "system", "content": system_prompt.format(context=retrieved_context)},
                {"role": "user", "content": model_input},
            ],
            temperature=_get_llm_temperature(),
            max_tokens=_get_max_output_tokens(),
            stream=True,
        )

        has_streamed_content = False
        streamed_token_count = 0
        for chunk in completion_stream:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta
            token = getattr(delta, "content", None) if delta else None
            if not token:
                continue

            has_streamed_content = True
            streamed_token_count += 1
            yield token

        if not has_streamed_content:
            raise ValueError("Streaming response returned no content.")
        _trace_step("llm.stream.success", token_count=streamed_token_count)
    except Exception as stream_error:
        logger.warning(
            "Streaming generation failed (%s). Falling back to non-streaming answer.",
            stream_error,
        )
        _trace_step("llm.stream.error", error=str(stream_error))
        fallback_answer = _generate_answer(
            model_input,
            normalized_query,
            retrieved_context=retrieved_context,
            top_score=top_score,
        )
        if fallback_answer:
            _trace_step("llm.stream.fallback_answer", answer_chars=len(fallback_answer))
            yield fallback_answer


def _generate_answer(
    model_input: str,
    normalized_query: str,
    retrieved_context: str | None = None,
    top_score: float | None = None,
) -> str:
    if retrieved_context is None or top_score is None:
        retrieved_context, top_score, _ = _retrieve_context_and_score(normalized_query)

    use_embedding_context = _should_use_embedding_context(normalized_query, retrieved_context, top_score)
    _raise_if_strict_without_context(use_embedding_context, retrieved_context, top_score)

    if use_embedding_context:
        try:
            _trace_step("answer.path", mode="context_completion")
            return _generate_completion_with_context(model_input, retrieved_context)
        except Exception as completion_error:
            logger.warning(
                "Context-grounded completion failed (%s).",
                completion_error,
            )
            _trace_step("answer.context_completion.error", error=str(completion_error))
            _trace_step("answer.no_context", source="context_completion_failed")
            return _no_context_response()

    _trace_step("answer.no_context", source="context_not_relevant")
    return _no_context_response()


def _required_backend_env_checks() -> list[tuple[str, list[str]]]:
    return [
        ("AZURE_OPENAI_ENDPOINT", ["AZURE_OPENAI_ENDPOINT"]),
        ("AZURE_OPENAI_API_KEY", ["AZURE_OPENAI_API_KEY"]),
        (
            "AZURE_OPENAI_CHAT_DEPLOYMENT",
            ["AZURE_OPENAI_CHAT_DEPLOYMENT", "AZURE_OPENAI_DEPLOYMENT", "AZURE_OPENAI_CHAT_MODEL"],
        ),
        (
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
            ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "AZURE_OPENAI_EMBED_DEPLOYMENT", "AZURE_OPENAI_EMBEDDING_MODEL"],
        ),
        ("AZURE_SEARCH_ENDPOINT", ["AZURE_SEARCH_ENDPOINT"]),
        ("AZURE_SEARCH_INDEX_NAME", ["AZURE_SEARCH_INDEX_NAME"]),
    ]


def _missing_backend_env_summary() -> list[dict[str, Any]]:
    missing: list[dict[str, Any]] = []
    for required_name, accepted_names in _required_backend_env_checks():
        is_present = any((os.getenv(name) or "").strip() for name in accepted_names)
        if not is_present:
            missing.append(
                {
                    "required": required_name,
                    "accepted": accepted_names,
                }
            )

    if _is_sharepoint_sync_enabled():
        for name in [
            "SHAREPOINT_TENANT_ID",
            "SHAREPOINT_CLIENT_ID",
            "SHAREPOINT_CLIENT_SECRET",
            "SHAREPOINT_SITE_ID",
            "SHAREPOINT_LIST_ID",
        ]:
            if not (os.getenv(name) or "").strip():
                missing.append({"required": name, "accepted": [name]})

    return missing


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/health/config")
async def health_config() -> dict:
    missing = _missing_backend_env_summary()
    return {
        "status": "ok" if not missing else "degraded",
        "missing_count": len(missing),
        "missing": missing,
        "sharepoint_sync_enabled": _is_sharepoint_sync_enabled(),
        "effective": {
            "azure_openai_endpoint": _get_azure_openai_endpoint() if (os.getenv("AZURE_OPENAI_ENDPOINT") or "").strip() else "",
            "azure_openai_chat_deployment": next(
                (
                    (os.getenv(name) or "").strip()
                    for name in ["AZURE_OPENAI_CHAT_DEPLOYMENT", "AZURE_OPENAI_DEPLOYMENT", "AZURE_OPENAI_CHAT_MODEL"]
                    if (os.getenv(name) or "").strip()
                ),
                "",
            ),
            "azure_openai_embedding_deployment": next(
                (
                    (os.getenv(name) or "").strip()
                    for name in ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "AZURE_OPENAI_EMBED_DEPLOYMENT", "AZURE_OPENAI_EMBEDDING_MODEL"]
                    if (os.getenv(name) or "").strip()
                ),
                "",
            ),
            "azure_search_endpoint": (os.getenv("AZURE_SEARCH_ENDPOINT") or "").strip(),
            "azure_search_index_name": (os.getenv("AZURE_SEARCH_INDEX_NAME") or "").strip(),
            "query_overlap_threshold": _get_query_overlap_threshold(),
            "query_min_overlap_terms": _get_query_min_overlap_terms(),
            "azure_search_score_threshold": _get_embedding_similarity_threshold(),
            "azure_search_required": _is_azure_search_required(),
            "allow_generic_fallback": _allow_generic_fallback(),
        },
    }


@app.post("/api/chat/text")
async def text_chat(
    query: str = Form(...),
    session_id: str | None = Form(default=None),
    lead_email: str | None = Form(default=None),
    lead_name: str | None = Form(default=None),
) -> dict:
    trace = _build_trace_record("/api/chat/text", query, session_id, streaming=False)
    trace_token = _activate_trace(trace)
    try:
        normalized_query = _normalize_user_query(query)
        _trace_step("request.normalized", normalized_query=normalized_query)

        effective_session_id = _normalize_session_id(session_id)
        _trace_step("session.resolved", effective_session_id=effective_session_id)
        current_lead_email, current_lead_name = _resolve_lead_identity(
            effective_session_id,
            lead_email,
            lead_name,
        )
        _trace_step(
            "lead.resolved",
            has_email=bool(current_lead_email),
            has_name=bool(current_lead_name),
        )

        model_input = _build_model_input(effective_session_id, normalized_query)
        retrieved_context, top_score, citations = _retrieve_context_and_score(normalized_query)
        answer = _generate_answer(
            model_input,
            normalized_query,
            retrieved_context=retrieved_context,
            top_score=top_score,
        )
        response_citations = citations if _should_attach_citations(answer, normalized_query, citations) else []
        _save_conversation_turn(effective_session_id, normalized_query, answer)
        await _sync_sharepoint_lead_safely(effective_session_id)

        trace_id = _get_active_trace_id()
        response_payload = {
            "reply": answer,
            "session_id": effective_session_id,
            "lead": {
                "email": current_lead_email,
                "name": current_lead_name,
            },
            "citations": response_citations,
        }
        if trace_id:
            response_payload["trace_id"] = trace_id

        _trace_step("response.ready", status_code=200, answer_chars=len(answer))
        if _is_chat_trace_include_context():
            _finalize_trace("success", status_code=200, reply=answer)
        else:
            _finalize_trace("success", status_code=200, answer_chars=len(answer))
        return response_payload
    except RetrievalUnavailableError as error:
        logger.warning("Text chat retrieval unavailable: %s", error)
        _trace_step("response.error", status_code=503, error=str(error))
        _finalize_trace("retrieval_unavailable", status_code=503, error=str(error))
        raise HTTPException(status_code=503, detail=str(error)) from error
    except Exception as error:
        logger.exception("Text chat pipeline failed")
        runtime_issue = _build_runtime_issue_message(error)
        fallback_answer = _no_context_response()
        trace_id = _get_active_trace_id()
        _trace_step("response.error", status_code=500, error=str(error))
        _trace_step("response.degraded", issue=runtime_issue)
        if _is_chat_trace_include_context():
            _finalize_trace("degraded", status_code=200, issue=runtime_issue, reply=fallback_answer)
        else:
            _finalize_trace("degraded", status_code=200, issue=runtime_issue, answer_chars=len(fallback_answer))

        degraded_payload = {
            "reply": fallback_answer,
            "session_id": _normalize_session_id(session_id),
            "lead": {
                "email": "",
                "name": "",
            },
            "citations": [],
        }
        if trace_id:
            degraded_payload["trace_id"] = trace_id
        return degraded_payload
    finally:
        _deactivate_trace(trace_token)


@app.post("/api/chat/text/stream")
async def text_chat_stream(
    query: str = Form(...),
    session_id: str | None = Form(default=None),
    lead_email: str | None = Form(default=None),
    lead_name: str | None = Form(default=None),
) -> StreamingResponse:
    trace = _build_trace_record("/api/chat/text/stream", query, session_id, streaming=True)
    trace_token = _activate_trace(trace)
    try:
        normalized_query = _normalize_user_query(query)
        _trace_step("request.normalized", normalized_query=normalized_query)
        effective_session_id = _normalize_session_id(session_id)
        _trace_step("session.resolved", effective_session_id=effective_session_id)
        current_lead_email, current_lead_name = _resolve_lead_identity(
            effective_session_id,
            lead_email,
            lead_name,
        )
        _trace_step(
            "lead.resolved",
            has_email=bool(current_lead_email),
            has_name=bool(current_lead_name),
        )
        model_input = _build_model_input(effective_session_id, normalized_query)
        trace_id = _get_active_trace_id()
    except HTTPException as error:
        _trace_step("response.error", status_code=error.status_code, error=str(error.detail), stream=True)
        _finalize_trace("failed", status_code=error.status_code, error=str(error.detail), stream=True)
        _deactivate_trace(trace_token)
        raise
    except Exception as error:
        _trace_step("response.error", status_code=500, error=str(error), stream=True)
        _finalize_trace("failed", status_code=500, error=str(error), stream=True)
        _deactivate_trace(trace_token)
        raise

    async def event_generator() -> AsyncGenerator[str, None]:
        answer_parts: list[str] = []
        citations: list[dict[str, Any]] = []
        response_citations: list[dict[str, Any]] = []

        try:
            retrieved_context, top_score, citations = _retrieve_context_and_score(normalized_query)
            for token in _stream_answer_tokens(
                model_input,
                normalized_query,
                retrieved_context=retrieved_context,
                top_score=top_score,
            ):
                if not token:
                    continue
                answer_parts.append(token)
                yield _sse_event("token", {"token": token})

            final_answer = "".join(answer_parts).strip()
            if not final_answer:
                final_answer = _generate_answer(
                    model_input,
                    normalized_query,
                    retrieved_context=retrieved_context,
                    top_score=top_score,
                )
                if final_answer:
                    yield _sse_event("token", {"token": final_answer})

            response_citations = citations if _should_attach_citations(final_answer, normalized_query, citations) else []

            _save_conversation_turn(effective_session_id, normalized_query, final_answer)
            await _sync_sharepoint_lead_safely(effective_session_id)

            yield _sse_event(
                "done",
                {
                    "reply": final_answer,
                    "session_id": effective_session_id,
                    "trace_id": trace_id,
                    "lead": {
                        "email": current_lead_email,
                        "name": current_lead_name,
                    },
                    "citations": response_citations,
                    "suggestions": _build_dynamic_followup_questions(effective_session_id, 3),
                },
            )
            _trace_step("response.ready", status_code=200, answer_chars=len(final_answer), stream=True)
            if _is_chat_trace_include_context():
                _finalize_trace("success", status_code=200, reply=final_answer, stream=True)
            else:
                _finalize_trace("success", status_code=200, answer_chars=len(final_answer), stream=True)
        except RetrievalUnavailableError as error:
            logger.warning("Text chat streaming retrieval unavailable: %s", error)
            yield _sse_event(
                "error",
                {
                    "message": str(error),
                    "error_type": type(error).__name__,
                    "trace_id": trace_id,
                },
            )
            _trace_step("response.error", status_code=503, error=str(error), stream=True)
            _finalize_trace("retrieval_unavailable", status_code=503, error=str(error), stream=True)
        except Exception as error:
            logger.exception("Text chat streaming pipeline failed")
            runtime_issue = _build_runtime_issue_message(error)
            fallback_answer = _no_context_response()
            yield _sse_event(
                "done",
                {
                    "reply": fallback_answer,
                    "session_id": effective_session_id,
                    "trace_id": trace_id,
                    "lead": {
                        "email": current_lead_email,
                        "name": current_lead_name,
                    },
                    "citations": [],
                    "suggestions": _build_dynamic_followup_questions(effective_session_id, 3),
                },
            )
            _trace_step("response.error", status_code=500, error=str(error), stream=True)
            _trace_step("response.degraded", issue=runtime_issue, stream=True)
            if _is_chat_trace_include_context():
                _finalize_trace("degraded", status_code=200, issue=runtime_issue, reply=fallback_answer, stream=True)
            else:
                _finalize_trace("degraded", status_code=200, issue=runtime_issue, answer_chars=len(fallback_answer), stream=True)
        finally:
            trace_state = _get_active_trace()
            if trace_state and not trace_state.get("status"):
                _finalize_trace("cancelled", status_code=499, stream=True)
            _deactivate_trace(trace_token)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Trace-Id": trace_id,
        },
    )


@app.post("/api/ingest/upload")
async def ingest_upload(
    file: UploadFile = File(...),
    x_ingest_key: str | None = Header(default=None),
) -> dict:
    configured_ingest_key = os.getenv("INGEST_API_KEY")
    if configured_ingest_key and x_ingest_key != configured_ingest_key:
        raise HTTPException(status_code=401, detail="Invalid ingestion API key")

    original_name = file.filename or "upload"
    extension = Path(original_name).suffix.lower()
    if extension not in SUPPORTED_INGEST_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Allowed: .pdf, .txt, .md, .csv, .log",
        )

    temp_file_path = f"ingest_{uuid.uuid4()}_{original_name}"

    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = ingest_file(temp_file_path, source_name=original_name)
        return {
            "status": "success",
            "message": "File ingested successfully",
            **result,
        }
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@app.post("/api/chat/voice")
async def voice_chat(
    audio: UploadFile = File(...),
    x_session_id: str | None = Header(default=None),
    x_lead_email: str | None = Header(default=None),
    x_lead_name: str | None = Header(default=None),
) -> Response:
    input_filename = audio.filename or "recording.webm"

    try:
        groq_client = get_groq_client()
        audio_bytes = await audio.read()

        transcription_model = _get_transcription_model()
        try:
            transcription = groq_client.audio.transcriptions.create(
                file=(input_filename, audio_bytes),
                model=transcription_model,
                prompt="The user is asking a question.",
                response_format="json",
            )
        except Exception as primary_error:
            should_retry_with_turbo = (
                "GROQ_TRANSCRIPTION_MODEL" not in os.environ
                and transcription_model != "whisper-large-v3-turbo"
            )

            if not should_retry_with_turbo:
                raise

            logger.warning(
                "Primary transcription model failed (%s). Retrying with whisper-large-v3-turbo.",
                primary_error,
            )
            transcription = groq_client.audio.transcriptions.create(
                file=(input_filename, audio_bytes),
                model="whisper-large-v3-turbo",
                prompt="The user is asking a question.",
                response_format="json",
            )

        user_text = _normalize_user_query((transcription.text or "").strip())
        if not user_text:
            raise HTTPException(status_code=400, detail="Could not transcribe user audio.")

        effective_session_id = _normalize_session_id(x_session_id)
        _resolve_lead_identity(
            effective_session_id,
            x_lead_email,
            x_lead_name,
        )

        model_input = _build_model_input(effective_session_id, user_text)
        bot_reply_text = _generate_answer(model_input, user_text)
        _save_conversation_turn(effective_session_id, user_text, bot_reply_text)
        await _sync_sharepoint_lead_safely(effective_session_id)

        communicate = edge_tts.Communicate(bot_reply_text, _get_tts_voice())
        output_audio_bytes = bytearray()
        async for chunk in communicate.stream():
            if chunk.get("type") == "audio":
                output_audio_bytes.extend(chunk.get("data", b""))

        return Response(
            content=bytes(output_audio_bytes),
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": "inline; filename=reply.mp3",
                "X-Session-Id": effective_session_id,
                "X-User-Query": _sanitize_header_value(user_text),
                "X-Bot-Reply": _sanitize_header_value(bot_reply_text),
                "X-User-Query-Encoded": _encode_header_value(user_text),
                "X-Bot-Reply-Encoded": _encode_header_value(bot_reply_text),
            },
        )
    except HTTPException:
        raise
    except Exception as error:
        logger.exception("Voice pipeline failed")
        raise HTTPException(
            status_code=500,
            detail=(
                "Voice pipeline failed: "
                f"{type(error).__name__}: {error}"
            ),
        ) from error
    finally:
        await audio.close()


@app.get("/api/chat/last")
async def get_last_chat_turn(session_id: str) -> dict:
    effective_session_id = _normalize_session_id(session_id)
    last_turn = _get_last_conversation_turn(effective_session_id)
    if not last_turn:
        raise HTTPException(status_code=404, detail="No conversation found for session_id")

    with _lead_lock:
        lead_data = dict(_lead_store.get(effective_session_id, {}))

    user_text, bot_reply_text = last_turn

    return {
        "session_id": effective_session_id,
        "user_query": user_text,
        "reply": bot_reply_text,
        "lead": {
            "email": lead_data.get("email", ""),
            "name": lead_data.get("name", ""),
        },
    }


@app.get("/api/chat/suggestions")
async def get_chat_suggestions(session_id: str, limit: int = 3) -> dict:
    effective_session_id = _normalize_session_id(session_id)
    bounded_limit = max(1, min(limit, 6))
    suggestions = _build_dynamic_followup_questions(effective_session_id, bounded_limit)
    return {
        "session_id": effective_session_id,
        "suggestions": suggestions,
    }
