import os
import re
import shutil
import uuid
import logging
import json
import base64
import warnings
from difflib import get_close_matches
from contextvars import ContextVar, Token
from datetime import datetime, timezone
from collections import deque
from functools import lru_cache
from pathlib import Path
from threading import Lock
from time import time
from typing import Any, AsyncGenerator, Callable, Iterable
from urllib.parse import parse_qs, quote, unquote, urlparse

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
from ingestion import ingest_blob_container, ingest_file
from langchain_openai import AzureOpenAIEmbeddings
from openai import AzureOpenAI

load_dotenv()
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

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
    "please", "need", "want", "tell", "me", "our", "their", "they",
    "first", "day", "procedure", "process", "step", "steps", "chat",
}

VIDEO_MATCH_STOPWORDS = {
    "the", "and", "for", "with", "from", "into", "that", "this", "what", "when", "where",
    "which", "about", "your", "you", "have", "has", "are", "was", "were", "will", "would",
    "could", "should", "please", "need", "want", "tell", "me", "our", "their", "they",
    "http", "https", "www", "watch", "video", "youtube", "youtu", "vimeo", "com", "channel",
    "project", "portal",
}

SOURCE_MATCH_STOPWORDS = {
    "the", "and", "for", "with", "from", "into", "that", "this", "what", "when", "where", "which",
    "about", "your", "you", "have", "has", "are", "was", "were", "will", "would", "could", "should",
    "please", "need", "want", "tell", "me", "our", "their", "they", "give", "show", "list", "all",
    "detail", "details", "desire", "infoweb", "desireinfoweb", "kindly",
}


def _get_required_env(name: str) -> str:
    value = _sanitize_env_value(os.getenv(name) or "")
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def _get_required_env_any(names: list[str]) -> str:
    for name in names:
        value = _sanitize_env_value(os.getenv(name) or "")
        if value:
            return value
    joined = ", ".join(names)
    raise ValueError(f"Missing required environment variable. Set one of: {joined}")


def _sanitize_env_value(value: str) -> str:
    cleaned = (value or "")
    cleaned = cleaned.replace("\r", "").replace("\n", "")
    cleaned = cleaned.replace("\\r", "").replace("\\n", "")
    return cleaned.strip()


def _is_env_true(name: str, default: str = "false") -> bool:
    return _sanitize_env_value(os.getenv(name, default)).lower() in {"1", "true", "yes", "on"}


def _use_direct_faq_answers() -> bool:
    # Retrieval-first mode is the default in production.
    return _is_env_true("USE_DIRECT_FAQ_ANSWERS", "false")


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


def _get_chat_process_log_path() -> Path:
    configured_path = os.getenv("CHAT_PROCESS_LOG_PATH", "logs/chat_process_last10.json").strip() or "logs/chat_process_last10.json"
    path = Path(configured_path)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parent / path


def _get_chat_process_log_limit() -> int:
    raw_value = os.getenv("CHAT_PROCESS_LOG_LIMIT", "10")
    try:
        limit = int(raw_value)
    except ValueError:
        limit = 10
    return max(1, min(limit, 200))


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


def _trace_pipeline_stage(stage: str, **details: Any) -> None:
    _trace_step(f"pipeline.{stage}", **details)


def _persist_trace(trace: dict[str, Any]) -> None:
    output_path = _get_chat_trace_log_path()
    serialized = json.dumps(trace, ensure_ascii=False)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with _trace_log_lock:
            with output_path.open("a", encoding="utf-8") as handle:
                handle.write(serialized + "\n")
    except Exception as error:
        logger.warning("Trace persistence skipped (path=%s): %s", output_path, error)


def _extract_pipeline_process(trace: dict[str, Any]) -> dict[str, Any]:
    process: dict[str, Any] = {
        "user_request": {},
        "ai_request": {},
        "ai_search_response": {},
        "ai_search_json_response": {},
        "ai_search_blob_response": {},
        "ai_selection": {},
        "ai_openai_response": {},
        "display_response": {},
    }

    for step in trace.get("steps", []):
        step_name = str(step.get("step") or "")
        if not step_name.startswith("pipeline."):
            continue

        stage = step_name.split(".", 1)[1]
        details = {
            key: value
            for key, value in step.items()
            if key not in {"step", "ts"}
        }
        details["ts"] = step.get("ts")
        process[stage] = details

    # Keep alias for user naming preference in ops discussions.
    process["ai_research_response"] = process.get("ai_search_response", {})
    return process


def _build_process_log_entry(trace: dict[str, Any]) -> dict[str, Any]:
    return {
        "trace_id": trace.get("trace_id"),
        "started_at": trace.get("started_at"),
        "ended_at": trace.get("ended_at"),
        "endpoint": trace.get("endpoint"),
        "streaming": bool(trace.get("streaming")),
        "status": trace.get("status"),
        "duration_ms": trace.get("duration_ms"),
        "session_id": trace.get("input_session_id") or "",
        "user_query": trace.get("raw_query") or "",
        "process": _extract_pipeline_process(trace),
        "summary": trace.get("summary", {}),
    }


def _persist_process_log(trace: dict[str, Any]) -> None:
    output_path = _get_chat_process_log_path()
    limit = _get_chat_process_log_limit()
    entry = _build_process_log_entry(trace)

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with _trace_log_lock:
            existing: list[dict[str, Any]] = []
            if output_path.exists():
                try:
                    existing_payload = json.loads(output_path.read_text(encoding="utf-8"))
                    if isinstance(existing_payload, list):
                        existing = [item for item in existing_payload if isinstance(item, dict)]
                except Exception:
                    existing = []

            existing.append(entry)
            trimmed = existing[-limit:]
            output_path.write_text(
                json.dumps(trimmed, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
    except Exception as error:
        logger.warning("Process log persistence skipped (path=%s): %s", output_path, error)


def _load_recent_process_logs(limit: int | None = None) -> list[dict[str, Any]]:
    output_path = _get_chat_process_log_path()
    if not output_path.exists():
        return []

    resolved_limit = limit if limit is not None else _get_chat_process_log_limit()
    resolved_limit = max(1, min(resolved_limit, 200))

    try:
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    if not isinstance(payload, list):
        return []

    entries = [item for item in payload if isinstance(item, dict)]
    return entries[-resolved_limit:]


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
    _persist_process_log(trace)
    if _is_chat_trace_console_enabled():
        try:
            logger.info(
                "chat trace %s status=%s endpoint=%s duration_ms=%s",
                trace.get("trace_id"),
                status,
                trace.get("endpoint"),
                trace.get("duration_ms"),
            )
        except Exception:
            pass


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


def _get_query_spell_similarity_cutoff() -> float:
    raw = os.getenv("QUERY_SPELL_SIMILARITY_CUTOFF", "0.88")
    try:
        return min(0.98, max(0.80, float(raw)))
    except (TypeError, ValueError):
        return 0.88


def _match_token_case(original: str, replacement: str) -> str:
    if not replacement:
        return original
    if original.isupper():
        return replacement.upper()
    if original[0].isupper() and original[1:].islower():
        return replacement.capitalize()
    return replacement


@lru_cache(maxsize=1)
def _get_query_spell_vocabulary() -> tuple[str, ...]:
    seed_text = " ".join(
        [
            SERVICE_SUMMARY,
            AI_SUMMARY,
            AI_PROJECTS_SUMMARY,
            BUDGET_SUMMARY,
            DOTNET_SUMMARY,
            CHATBOT_IMPLEMENTATION_SUMMARY,
            CHATBOT_DATA_SOURCE_SUMMARY,
            INDUSTRY_SUMMARY,
            "Desire Infoweb ai projects solutions services organization chart fluentify interviewer"
            " sharepoint azure teams chatbot voice architecture compliance delivered completed",
        ]
    )
    words = {
        token
        for token in re.findall(r"[a-z]+", seed_text.lower())
        if len(token) >= 3
    }
    words.update(
        {
            "desire",
            "infoweb",
            "organization",
            "chart",
            "projects",
            "project",
            "solutions",
            "solution",
            "services",
            "service",
            "fluentify",
            "interviewer",
            "portfolio",
            "details",
            "detail",
            "give",
            "what",
            "show",
            "list",
            "all",
            "about",
            "learning",
            "management",
            "portal",
            "system",
            "lms",
            "audit",
            "quick",
            "links",
            "dynamics",
            "module",
            "modules",
            "sharepoint",
            "new",
            "joinee",
        }
    )
    return tuple(sorted(words))


def _fuzzy_spell_correct_token(token: str) -> str:
    vocabulary = _get_query_spell_vocabulary()
    lowered = token.lower()
    if len(lowered) < 4 or lowered in vocabulary:
        return token

    match = get_close_matches(lowered, vocabulary, n=1, cutoff=_get_query_spell_similarity_cutoff())
    if not match:
        return token
    return _match_token_case(token, match[0])


def _normalize_similarity_text(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", value.lower())).strip()


@lru_cache(maxsize=1)
def _get_semantic_topic_aliases() -> dict[str, tuple[str, ...]]:
    return {
        "project management portal": (
            "project management portal",
            "project management",
            "pm portal",
            "project portal",
        ),
        "learning management system": (
            "learning management system",
            "learning management portal",
            "learning portal",
            "lms",
        ),
        "audit system": (
            "audit system",
            "audit portal",
        ),
        "organization chart": (
            "organization chart",
            "org chart",
            "organogram",
        ),
        "quick links": (
            "quick links",
            "quick link",
            "shortcut links",
        ),
        "new joinee": (
            "new joinee",
            "new employee",
            "onboarding profile",
        ),
    }


def _infer_canonical_topic_from_query(query: str) -> str:
    normalized_query = _normalize_similarity_text(query)
    if not normalized_query:
        return ""

    query_terms = {token for token in normalized_query.split() if len(token) >= 3}
    if not query_terms:
        return ""

    best_topic = ""
    best_score = 0
    query_term_list = list(query_terms)

    for topic, aliases in _get_semantic_topic_aliases().items():
        topic_score = 0
        for alias in aliases:
            alias_norm = _normalize_similarity_text(alias)
            if not alias_norm:
                continue

            alias_terms = [token for token in alias_norm.split() if len(token) >= 3]
            exact_overlap = len(set(alias_terms).intersection(query_terms))

            fuzzy_overlap = 0
            for term in alias_terms:
                if term in query_terms:
                    continue
                if get_close_matches(term, query_term_list, n=1, cutoff=0.83):
                    fuzzy_overlap += 1

            phrase_bonus = 4 if alias_norm in normalized_query else 0
            alias_score = (exact_overlap * 4) + (fuzzy_overlap * 2) + phrase_bonus
            topic_score = max(topic_score, alias_score)

        if topic_score > best_score:
            best_score = topic_score
            best_topic = topic

    if best_score < 6:
        return ""
    return best_topic


def _normalize_user_query(query: str) -> str:
    normalized = query.strip()
    replacements = {
        "serivce": "service",
        "serivces": "services",
        "servies": "services",
        "servics": "services",
        "fluetify": "fluentify",
        "fluintify": "fluentify",
        "fluntify": "fluentify",
        "orgration": "organization",
        "orgaration": "organization",
        "ogration": "organization",
        "orgnization": "organization",
        "organzation": "organization",
        "oragnization": "organization",
        "organziation": "organization",
        "organiztion": "organization",
        "poratl": "portal",
        "portel": "portal",
        "mangement": "management",
        "managment": "management",
        "eproject": "project",
        "char": "chart",
        "chatr": "chart",
        "porject": "project",
        "porjects": "projects",
        "projet": "project",
        "projets": "projects",
        "prject": "project",
        "prjects": "projects",
        "detial": "detail",
        "detials": "details",
        "deatil": "detail",
        "deatils": "details",
        "soltion": "solution",
        "soltions": "solutions",
        "compnay": "company",
        "qhat": "what",
        "wht": "what",
        "leaning": "learning",
        "learing": "learning",
        "learnig": "learning",
        "managment": "management",
        "managemnt": "management",
        "manageent": "management",
        "mangment": "management",
        "u": "you",
    }

    def _normalize_match(match: re.Match[str]) -> str:
        token = match.group(0)
        replacement = replacements.get(token.lower())
        if replacement:
            return _match_token_case(token, replacement)
        return _fuzzy_spell_correct_token(token)

    normalized_query = re.sub(r"[A-Za-z]+", _normalize_match, normalized)

    # Convert assistant-style follow-up prompts into actionable user intents.
    lowered = normalized_query.lower().strip()
    transformed = normalized_query
    match = re.match(r"^do you want (?:a |an )?(.+?)\?*$", lowered)
    if match:
        transformed = f"give me {match.group(1).strip()}"
    else:
        match = re.match(r"^would you like (?:a |an )?(.+?)\?*$", lowered)
        if match:
            transformed = f"give me {match.group(1).strip()}"
        else:
            match = re.match(r"^should i share (.+?)\?*$", lowered)
            if match:
                transformed = f"share {match.group(1).strip()}"

    # Fix common product typo/phrase variants seen in user queries.
    transformed = re.sub(r"\borganization\s+chat\b", "organization chart", transformed, flags=re.IGNORECASE)

    lowered_transformed = transformed.lower().strip()
    asks_ai_catalog_intent = (
        any(phrase in lowered_transformed for phrase in ["ai project", "ai projects", "ai solution", "ai solutions"])
        and any(token in lowered_transformed for token in ["all", "list", "show", "example", "examples", "portfolio", "detail", "details", "more"])
        and not any(token in lowered_transformed for token in ["fluentify", "interviewer", "compliance", "voice workflow", "organization chart"])
    )
    if asks_ai_catalog_intent:
        transformed = "what ai solutions has desire infoweb delivered"

    inferred_topic = _infer_canonical_topic_from_query(transformed)
    if inferred_topic:
        normalized_topic = _normalize_similarity_text(inferred_topic)
        normalized_transformed = _normalize_similarity_text(transformed)
        if normalized_topic not in normalized_transformed:
            transformed = f"{transformed} {inferred_topic}"

    transformed = re.sub(r"\s+", " ", transformed).strip()
    return transformed


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


def _looks_like_prompt_phrase(value: str) -> bool:
    lowered = value.strip().lower()
    return lowered.startswith(
        (
            "can you",
            "could you",
            "what ",
            "which ",
            "how ",
            "tell me",
            "give me",
            "share ",
            "please ",
        )
    )


def _get_suggestion_max_chars() -> int:
    raw_value = os.getenv("SUGGESTION_MAX_CHARS", "84")
    try:
        length = int(raw_value)
    except ValueError:
        length = 84
    return max(40, min(length, 160))


def _truncate_question_to_limit(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value

    candidate = value[: max_chars - 1].rstrip()
    if " " in candidate:
        maybe_word_boundary = candidate.rsplit(" ", 1)[0].rstrip()
        if len(maybe_word_boundary) >= 28:
            candidate = maybe_word_boundary
    return candidate


def _sanitize_followup_question(candidate: str) -> str:
    value = re.sub(r"\s+", " ", (candidate or "").strip())
    value = re.sub(r"^[\-\*\d\.)\s]+", "", value).strip()
    if not value:
        return ""
    max_chars = _get_suggestion_max_chars()
    value = _truncate_question_to_limit(value, max_chars)
    if value.endswith(":"):
        value = value[:-1].strip()
    if not value.endswith("?"):
        value = value.rstrip(".") + "?"
    if len(value) > max_chars:
        value = value[: max_chars - 1].rstrip(" .?!") + "?"
    if len(value) < 16 or len(value) > max_chars:
        return ""
    lowered = value.lower()
    if "for can you" in lowered or "for what " in lowered:
        return ""
    if lowered.count("?") > 1:
        return ""
    if "thank you for your query" in lowered:
        return ""
    if "support team" in lowered:
        return ""
    if "here is a concise answer from indexed documents about" in lowered:
        return ""
    if re.search(r"\b(summary|details?) of [^?]*\b(has|have)\b[^?]*\buse cases?\b", lowered):
        return ""
    trailing_fragment = lowered.rstrip("?").strip()
    if trailing_fragment:
        trailing_word = trailing_fragment.split()[-1]
        if trailing_word in {"to", "for", "with", "about", "of", "in", "on", "at", "from", "and", "or", "the", "a", "an"}:
            return ""
        if re.search(r"\b(to enhance|to improve|to optimize|to support)$", trailing_fragment):
            return ""
    return value


def _is_no_context_like_answer(answer: str) -> bool:
    text = (answer or "").strip().lower()
    if not text:
        return False
    if text.startswith("thank you for your query"):
        return True
    if "falls outside my current scope" in text:
        return True
    if "for further assistance, please contact our support team" in text:
        return True
    canonical = _no_context_response().strip().lower()
    return text == canonical


def _extract_focus_topic_from_query(query: str) -> str:
    cleaned = re.sub(r"[^a-z0-9\s]", " ", (query or "").lower())
    tokens = [token for token in re.split(r"\s+", cleaned) if token]
    stop = {
        "give", "me", "all", "about", "details", "detail", "tell", "show", "project", "projects",
        "please", "desire", "infoweb", "the", "a", "an", "to", "for", "of", "in", "on", "with",
        "can", "you", "i", "want", "need", "list", "what", "use", "uses", "case", "cases",
        "summary", "concise", "should", "would", "like", "share", "has", "have", "had", "deliver",
        "delivered", "delivers", "solution", "solutions", "does", "did", "provide", "provides",
        "mainly", "serve", "serves", "served", "based", "type",
    }
    topical = [token for token in tokens if len(token) >= 3 and token not in stop]
    if not topical:
        return ""
    return " ".join(topical[:4]).strip()


def _build_query_anchored_followups(latest_user_query: str, limit: int = 3) -> list[str]:
    topic = _extract_focus_topic_from_query(latest_user_query)
    if not topic:
        return []

    raw_candidates = [
        f"Give me a concise summary of {topic} use cases",
        f"Share implementation details for {topic}",
        f"Give me tech stack and business impact for {topic}",
        f"Compare major {topic} projects and outcomes",
    ]

    suggestions: list[str] = []
    seen: set[str] = set()
    for candidate in raw_candidates:
        sanitized = _sanitize_followup_question(candidate)
        key = _normalize_question_for_compare(sanitized)
        if not sanitized or not key or key in seen:
            continue
        suggestions.append(sanitized)
        seen.add(key)
        if len(suggestions) >= limit:
            break

    return suggestions


def _extract_questions_from_llm_payload(raw_content: str) -> list[str]:
    text = (raw_content or "").strip()
    if not text:
        return []

    attempts: list[str] = [text]
    json_like_match = re.search(r"\{[\s\S]*\}|\[[\s\S]*\]", text)
    if json_like_match:
        attempts.append(json_like_match.group(0))

    for payload in attempts:
        try:
            parsed = json.loads(payload)
        except Exception:
            continue

        if isinstance(parsed, dict):
            questions = parsed.get("questions")
            if isinstance(questions, list):
                return [str(item) for item in questions if isinstance(item, str)]
        if isinstance(parsed, list):
            return [str(item) for item in parsed if isinstance(item, str)]

    line_candidates = []
    for line in text.splitlines():
        cleaned = re.sub(r"^[\-\*\d\.)\s]+", "", line).strip()
        if cleaned:
            line_candidates.append(cleaned)
    return line_candidates


def _generate_followups_with_llm(
    history: list[tuple[str, str]],
    asked_question_keys: set[str],
    limit: int,
) -> list[str]:
    history_turns = max(1, min(int(os.getenv("SUGGESTION_HISTORY_TURNS", "4")), 8))
    recent_turns = history[-history_turns:]
    transcript_lines: list[str] = []
    for user_text, assistant_text in recent_turns:
        transcript_lines.append(f"User: {_clip_text(user_text, 220)}")
        transcript_lines.append(f"Assistant: {_clip_text(assistant_text, 420)}")

    prompt = (
        "Generate exactly 3 concise, high-quality follow-up questions for the user based on the full conversation. "
        "Rules: avoid repeating any previously asked user question, avoid generic fillers, avoid malformed nested phrasing, "
        "and keep each question specific to the discussed topics. Return strict JSON: {\"questions\":[\"q1\",\"q2\",\"q3\"]}.\n\n"
        "Conversation:\n"
        + "\n".join(transcript_lines)
    )

    try:
        completion = get_azure_openai_client().chat.completions.create(
            model=_get_chat_model(),
            messages=[
                {
                    "role": "system",
                    "content": "You generate business chatbot follow-up questions from conversation context.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=_get_suggestion_max_tokens(),
            stream=False,
        )
        if not completion.choices:
            return []
        raw_content = str(getattr(completion.choices[0].message, "content", "") or "")
    except Exception as error:
        _trace_step("suggestions.llm.error", error=str(error))
        return []

    parsed_questions = _extract_questions_from_llm_payload(raw_content)
    finalized: list[str] = []
    seen: set[str] = set()
    for candidate in parsed_questions:
        sanitized = _sanitize_followup_question(candidate)
        if not sanitized:
            continue
        key = _normalize_question_for_compare(sanitized)
        if not key or key in seen or key in asked_question_keys:
            continue
        finalized.append(sanitized)
        seen.add(key)
        if len(finalized) >= limit:
            break

    return finalized[:limit]


def _get_suggestion_max_tokens() -> int:
    raw_value = os.getenv("SUGGESTION_MAX_TOKENS", "380")
    try:
        value = int(raw_value)
    except ValueError:
        value = 380
    return max(120, min(value, 900))


def _extract_topic_seeds_from_history(history: list[tuple[str, str]], max_topics: int = 3) -> list[str]:
    seeds: list[str] = []
    seen: set[str] = set()

    for user_text, assistant_text in reversed(history):
        assistant_candidates: list[str] = []
        if not _is_no_context_like_answer(assistant_text):
            assistant_candidates.extend(re.findall(r"(?m)^#{1,6}\s+(.+)$", assistant_text or ""))
            assistant_candidates.extend(re.findall(r"\*\*([^*\n]{4,90})\*\*", assistant_text or ""))

        first_sentence = re.split(r"[\n\.\!\?]", (assistant_text or "").strip(), maxsplit=1)[0].strip()
        if 12 <= len(first_sentence) <= 100 and not _is_no_context_like_answer(first_sentence):
            assistant_candidates.append(first_sentence)

        user_candidate = re.sub(r"\s+", " ", (user_text or "").strip()).rstrip("?.!,;:")
        if 12 <= len(user_candidate) <= 100 and not _looks_like_prompt_phrase(user_candidate):
            assistant_candidates.append(user_candidate)

        for candidate in assistant_candidates:
            cleaned = re.sub(r"\s+", " ", candidate.strip()).rstrip("?.!,;:")
            if len(cleaned) < 8 or _looks_like_prompt_phrase(cleaned):
                continue
            lowered_cleaned = cleaned.lower()
            if lowered_cleaned.startswith("here is a concise answer from indexed documents about"):
                continue
            if lowered_cleaned.startswith("voice-based projects overview"):
                continue
            key = _normalize_question_for_compare(cleaned)
            if not key or key in seen:
                continue
            seeds.append(cleaned)
            seen.add(key)
            if len(seeds) >= max_topics:
                return seeds

    return seeds


def _build_dynamic_followup_questions(session_id: str, limit: int = 3) -> list[str]:
    with _conversation_lock:
        history = list(_conversation_store.get(session_id, []))

    if not history or limit <= 0:
        return []

    suggestions: list[str] = []
    seen_suggestions: set[str] = set()
    low_signal_queries = {
        "hi",
        "hii",
        "hiii",
        "hello",
        "hey",
        "ok",
        "okay",
        "thanks",
        "thankyou",
        "thank you",
    }

    asked_question_keys = {
        _normalize_question_for_compare(user_text)
        for user_text, _ in history
        if user_text and user_text.strip()
    }

    latest_user_query = history[-1][0] if history else ""
    latest_assistant_reply = history[-1][1] if history else ""

    lower_latest_user_query = latest_user_query.lower().strip()

    is_meta_followup_query = lower_latest_user_query.startswith(
        (
            "do you want",
            "would you like",
            "should i share",
        )
    )

    if is_meta_followup_query:
        for previous_user_text, _ in reversed(history[:-1]):
            candidate = (previous_user_text or "").strip()
            lowered = candidate.lower()
            if not candidate:
                continue
            if lowered.startswith(("do you want", "would you like", "should i share")):
                continue
            latest_user_query = candidate
            break

    anchored_candidates = _build_query_anchored_followups(latest_user_query, limit)
    if anchored_candidates:
        for candidate in anchored_candidates:
            key = _normalize_question_for_compare(candidate)
            if not key or key in seen_suggestions or key in asked_question_keys:
                continue
            suggestions.append(candidate)
            seen_suggestions.add(key)
            if len(suggestions) >= limit:
                return suggestions[:limit]

    if _is_no_context_like_answer(latest_assistant_reply):
        # Do not let fallback text drive follow-up generation quality.
        return suggestions[:limit]

    # LLM-based generation uses full recent conversation (user + assistant).
    if _is_env_true("SUGGESTIONS_USE_LLM", "true"):
        llm_suggestions = _generate_followups_with_llm(history, asked_question_keys, limit)
        for candidate in llm_suggestions:
            key = _normalize_question_for_compare(candidate)
            if not key or key in seen_suggestions or key in asked_question_keys:
                continue
            suggestions.append(candidate)
            seen_suggestions.add(key)
            if len(suggestions) >= limit:
                return suggestions[:limit]

    topic_seeds = _extract_topic_seeds_from_history(history, max_topics=3)

    if not topic_seeds:
        # Keep suggestions retrieval-driven and avoid static FAQ-style prompts.
        return suggestions[:limit]

    for topic_seed in topic_seeds:
        trimmed_topic = topic_seed[:90].rstrip()
        generated_candidates = [
            f"Can you share implementation details for {trimmed_topic}?",
            f"What technologies and architecture were used in {trimmed_topic}?",
            f"What business outcomes were achieved with {trimmed_topic}?",
            f"Can you share similar projects related to {trimmed_topic}?",
            f"What would be the timeline and cost range for a project like {trimmed_topic}?",
        ]

        for candidate in generated_candidates:
            sanitized = _sanitize_followup_question(candidate)
            key = _normalize_question_for_compare(sanitized)
            if not sanitized or not key or key in seen_suggestions or key in asked_question_keys:
                continue
            normalized_candidate = sanitized.lower().strip("?")
            if normalized_candidate in low_signal_queries:
                continue
            suggestions.append(sanitized)
            seen_suggestions.add(key)
            if len(suggestions) >= limit:
                return suggestions[:limit]

    return suggestions[:limit]


def _is_sharepoint_sync_enabled() -> bool:
    return os.getenv("ENABLE_SHAREPOINT_SYNC", "false").lower() == "true"


def _is_sharepoint_always_insert_enabled() -> bool:
    return os.getenv("SHAREPOINT_ALWAYS_INSERT", "false").lower() == "true"


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
    if not _use_direct_faq_answers():
        return None

    q = query.lower().strip()
    compact = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", q)).strip()
    compact_no_space = compact.replace(" ", "")
    logger.debug("Direct answer check: query='%s', compact='%s'", q, compact)

    asks_voice_catalog = (
        ("voice based projects" in compact)
        or ("voice based project" in compact and any(token in compact for token in ["all", "list"]))
        or (
            "voice" in compact
            and "projects" in compact
            and any(token in compact for token in ["all", "list", "complete", "full"])
        )
    )

    if asks_voice_catalog:
        logger.debug("Direct answer matched: voice projects catalog")
        return (
            "## Voice-Based Projects by Desire Infoweb\n\n"
            "### 1. AI Interviewer Pro - Voice-Based Candidate Screening\n"
            "- **Challenge**: Manual first-round candidate screening is slow and inconsistent.\n"
            "- **Solution**: AI-driven voice interviews with dynamic question generation and contextual response analysis.\n"
            "- **Impact**: Faster shortlisting, better screening consistency, and reduced time-to-hire.\n"
            "- **Technologies**: Gemini/GPT models, Speech-to-Text, TypeScript, React, cloud-native backend.\n\n"
            "### 2. Fluentify AI - Voice Communication Assessment\n"
            "- **Challenge**: Teams need objective spoken communication improvement at scale.\n"
            "- **Solution**: Voice and pronunciation assessment integrated with Microsoft Teams, with detailed feedback on fluency, grammar, and vocabulary.\n"
            "- **Impact**: Measurable improvement in communication quality with personalized recommendations.\n"
            "- **Technologies**: Azure Pronunciation Assessment, Azure OpenAI, Microsoft Teams integration, SharePoint/Azure storage.\n\n"
            "### 3. Voice Workflow Assistant\n"
            "- **Challenge**: Repetitive workflow actions reduce productivity.\n"
            "- **Solution**: Voice-triggered workflow assistant for task routing, reminders, and action execution.\n"
            "- **Impact**: Lower operational overhead and faster day-to-day execution.\n"
            "- **Technologies**: NLP pipelines, workflow orchestration, cloud APIs, secure enterprise integrations.\n\n"
            "### 4. Voice-Enabled Compliance Monitoring\n"
            "- **Challenge**: Real-time compliance enforcement in operations is difficult with manual monitoring alone.\n"
            "- **Solution**: AI-assisted voice-enabled monitoring and alerts for policy/safety adherence.\n"
            "- **Impact**: Better compliance visibility and faster corrective actions.\n"
            "- **Technologies**: AI monitoring stack, real-time processing services, analytics dashboards.\n\n"
            "If you want, I can give a deep technical breakdown project-by-project (architecture, deployment, and implementation flow)."
        )

    asks_ai_project_catalog = (
        compact in {"ai projects", "ai project", "all ai projects", "all ai solutions"}
        or (
            any(phrase in compact for phrase in ["ai project", "ai projects", "ai solution", "ai solutions"])
            and any(token in compact for token in ["all", "list", "show", "examples", "example", "portfolio", "delivered", "completed"])
        )
    )

    if asks_ai_project_catalog:
        logger.debug("Direct answer matched: AI project catalog")
        return (
            "Desire Infoweb has delivered multiple AI solutions across enterprise use cases:\n\n"
            "1. Fluentify AI for voice communication assessment with pronunciation scoring and coaching feedback.\n"
            "2. AI Interviewer Pro for voice-based candidate screening with automated evaluation support.\n"
            "3. Teams AI assistants integrated with business workflows for employee support and knowledge lookup.\n"
            "4. Document-grounded chatbots using SharePoint/Azure storage to answer from company documents.\n\n"
            "If you want, I can provide detailed architecture, technology stack, and outcomes for each project."
        )

    if re.match(r"^(hi+|hello+|hey+|good morning|good afternoon|good evening)\b", compact) and len(compact.split()) <= 4:
        logger.debug("Direct answer matched: greeting")
        return (
            "Hello! Welcome to Desire Infoweb. "
            f"{SERVICE_SUMMARY} "
            "Tell me your requirement and I can suggest the best service approach."
        )

    if any(
        phrase in compact
        for phrase in [
            "what ai solutions has desire infoweb delivered",
            "what type of ai projects has desire infoweb completed",
            "ai projects delivered",
            "ai solutions delivered",
        ]
    ):
        logger.debug("Direct answer matched: delivered AI solutions query")
        return (
            "Desire Infoweb has delivered multiple AI solutions across enterprise use cases:\n\n"
            "1. Fluentify AI for voice communication assessment with pronunciation scoring and coaching feedback.\n"
            "2. AI Interviewer Pro for voice-based candidate screening with automated evaluation support.\n"
            "3. Teams AI assistants integrated with business workflows for employee support and knowledge lookup.\n"
            "4. Document-grounded chatbots using SharePoint/Azure storage to answer from company documents.\n\n"
            "If you want, I can provide a project-wise deep dive with architecture, tech stack, and outcomes."
        )

    if any(keyword in compact for keyword in ["what service", "services", "what do you do", "what you do", "what do you provide", "offer"]):
        logger.debug("Direct answer matched: services query")
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
    ) or (
        ("desire infoweb" in compact or "desireinfoweb" in compact_no_space)
        and re.search(r"\b(details?|about|information|profile|overview)\b", compact)
    ):
        logger.debug("Direct answer matched: company info query")
        return (
            "Desire Infoweb is an IT services company focused on Microsoft technologies and business automation. "
            f"{SERVICE_SUMMARY}"
        )

    if any(keyword in compact for keyword in ["budget", "cost", "pricing", "price", "estimate", "quotation", "quote"]):
        logger.debug("Direct answer matched: budget query")
        return BUDGET_SUMMARY

    if any(keyword in compact for keyword in ["build ai chatbot", "want to build ai chatbot", "ai chatbot project", "chatbot project"]):
        logger.debug("Direct answer matched: chatbot build query")
        return (
            "Great choice. We can build an AI chatbot for your website or Microsoft Teams with your business data as context. "
            "Typical scope includes discovery, data ingestion (PDF/web/SharePoint), prompt tuning, voice/text support, testing, and deployment. "
            "If you share your goal and preferred channel, I can suggest the best implementation approach."
        )

    if any(keyword in compact for keyword in ["normal chatbot", "just chatbot", "simple chatbot", "basic chatbot"]):
        logger.debug("Direct answer matched: basic chatbot query")
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
        logger.debug("Direct answer matched: chatbot data source query")
        return CHATBOT_DATA_SOURCE_SUMMARY

    if any(keyword in compact for keyword in [".net", "dotnet", "net service", "what about net"]):
        logger.debug("Direct answer matched: .NET query")
        return DOTNET_SUMMARY

    if any(keyword in compact for keyword in ["industry", "industries", "domain", "sector"]):
        logger.debug("Direct answer matched: industry query")
        return INDUSTRY_SUMMARY

    if compact in {
        "ai",
        "ai services",
        "ai solutions",
        "what is ai",
        "what ai services",
        "what ai solutions",
        "tell me about ai services",
        "tell me about ai solutions",
    }:
        logger.debug("Direct answer matched: generic AI query")
        return AI_SUMMARY

    logger.debug("No direct answer match, will use RAG retrieval")
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
    return _sanitize_env_value(os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"))


def _get_azure_search_endpoint() -> str:
    return _get_required_env("AZURE_SEARCH_ENDPOINT")


def _get_azure_search_index_name() -> str:
    return _get_required_env("AZURE_SEARCH_INDEX_NAME")


def _get_azure_search_api_key() -> str:
    return _sanitize_env_value(os.getenv("AZURE_SEARCH_API_KEY", ""))


def _get_azure_search_content_field() -> str:
    return _sanitize_env_value(os.getenv("AZURE_SEARCH_CONTENT_FIELD", "content")) or "content"


def _get_azure_search_vector_field() -> str:
    return _sanitize_env_value(os.getenv("AZURE_SEARCH_VECTOR_FIELD", "contentVector"))


def _get_azure_search_top_k() -> int:
    raw_value = os.getenv("AZURE_SEARCH_TOP_K", "15")
    try:
        top_k = int(raw_value)
    except ValueError:
        top_k = 15
    return max(1, min(top_k, 20))


def _get_azure_search_excluded_sources() -> set[str]:
    raw_value = _sanitize_env_value(os.getenv("AZURE_SEARCH_EXCLUDE_SOURCES", ""))
    if not raw_value:
        return set()
    configured = {item.strip().lower() for item in raw_value.split(",") if item.strip()}
    return configured


def _get_knowledge_base_dir() -> str:
    return _sanitize_env_value(os.getenv("KNOWLEDGE_BASE_DIR", "knowledge")) or "knowledge"


def _get_retrieval_priority_scan_factor() -> int:
    raw_value = _sanitize_env_value(os.getenv("AZURE_SEARCH_PRIORITY_SCAN_FACTOR", "8"))
    try:
        factor = int(raw_value)
    except ValueError:
        factor = 8
    return max(1, min(factor, 20))


def _get_retrieval_priority_scan_top(top_k: int) -> int:
    factor = _get_retrieval_priority_scan_factor()
    return max(top_k, min(100, top_k * factor))


def _normalize_source_key(value: str) -> str:
    candidate = (value or "").strip().replace("\\", "/")
    if not candidate:
        return ""
    return candidate.rsplit("/", 1)[-1].lower()


def _get_configured_priority_sources() -> set[str]:
    raw_value = _sanitize_env_value(os.getenv("AZURE_SEARCH_PRIORITY_SOURCES", ""))
    if not raw_value:
        return set()

    configured: set[str] = set()
    for item in raw_value.split(","):
        normalized = _normalize_source_key(item)
        if normalized:
            configured.add(normalized)
    return configured


def _get_local_knowledge_sources() -> set[str]:
    knowledge_dir = (Path(__file__).resolve().parent / _get_knowledge_base_dir()).resolve()
    if not knowledge_dir.exists() or not knowledge_dir.is_dir():
        return set()

    supported_extensions = SUPPORTED_INGEST_EXTENSIONS.union({".json"})
    detected: set[str] = set()

    for file_path in knowledge_dir.rglob("*"):
        if not file_path.is_file():
            continue
        suffix = file_path.suffix.lower()
        if suffix and suffix not in supported_extensions:
            continue

        normalized_name = _normalize_source_key(file_path.name)
        if normalized_name:
            detected.add(normalized_name)

    return detected


def _get_priority_knowledge_sources() -> set[str]:
    return _get_configured_priority_sources().union(_get_local_knowledge_sources())


def _blob_only_sources_enabled() -> bool:
    return _is_env_true("AZURE_BLOB_ONLY_SOURCES", "false")


def _get_azure_blob_base_url() -> str:
    direct = _sanitize_env_value(os.getenv("AZURE_BLOB_ACCOUNT_URL", "")).rstrip("/")
    if direct:
        return direct.lower()

    connection_string = _sanitize_env_value(os.getenv("AZURE_BLOB_CONNECTION_STRING", ""))
    if not connection_string:
        return ""

    fields: dict[str, str] = {}
    for fragment in connection_string.split(";"):
        if "=" not in fragment:
            continue
        key, value = fragment.split("=", 1)
        fields[key.strip().lower()] = value.strip()

    blob_endpoint = fields.get("blobendpoint", "").rstrip("/")
    if blob_endpoint:
        return blob_endpoint.lower()

    account_name = fields.get("accountname", "")
    if not account_name:
        return ""
    protocol = fields.get("defaultendpointsprotocol", "https") or "https"
    suffix = fields.get("endpointsuffix", "core.windows.net") or "core.windows.net"
    return f"{protocol}://{account_name}.blob.{suffix}".rstrip("/").lower()


def _get_azure_search_semantic_config() -> str:
    return _sanitize_env_value(os.getenv("AZURE_SEARCH_SEMANTIC_CONFIG", ""))


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
    requested_raw = os.getenv("LLM_MAX_OUTPUT_TOKENS", "52000")
    model_cap_raw = os.getenv("AZURE_OPENAI_MAX_COMPLETION_TOKENS", "16384")

    try:
        requested_tokens = int(requested_raw)
    except ValueError:
        requested_tokens = 1200

    try:
        model_cap = int(model_cap_raw)
    except ValueError:
        model_cap = 16384

    bounded_tokens = max(64, min(requested_tokens, model_cap, 16384))
    if bounded_tokens != requested_tokens:
        logger.warning(
            "LLM_MAX_OUTPUT_TOKENS=%s exceeds allowed range; using %s instead.",
            requested_tokens,
            bounded_tokens,
        )

    return bounded_tokens


def _get_retry_output_tokens() -> int:
    raw_value = os.getenv("LLM_RETRY_OUTPUT_TOKENS", "8000")
    try:
        value = int(raw_value)
    except ValueError:
        value = 700
    return max(64, min(value, _get_max_output_tokens()))


def _is_token_limit_error(error: Exception) -> bool:
    message = str(error).lower()
    return any(
        marker in message
        for marker in [
            "maximum context length",
            "context_length_exceeded",
            "too many tokens",
            "max tokens",
            "reduce your prompt",
            "max_completion_tokens",
        ]
    )


def _extract_current_query_from_model_input(model_input: str) -> str:
    marker = "Current user question:\n"
    if marker in model_input:
        return model_input.split(marker, 1)[1].strip()
    return model_input.strip()


def _get_max_completion_segments() -> int:
    raw_value = os.getenv("LLM_MAX_CONTINUATION_SEGMENTS", "24")
    try:
        value = int(raw_value)
    except ValueError:
        value = 24
    return max(1, min(value, 40))


def _is_length_finish_reason(finish_reason: Any) -> bool:
    return str(finish_reason or "").strip().lower() == "length"


def _build_continuation_prompt(original_query: str, partial_answer: str) -> str:
    tail = _clip_text(partial_answer[-1400:], 1400)
    return (
        "Continue the same answer from exactly where it stopped. "
        "Do not repeat earlier content, do not restart headings, and keep markdown formatting consistent.\n\n"
        f"Original user question:\n{original_query}\n\n"
        "Already generated answer (tail):\n"
        f"{tail}"
    )


def _looks_abruptly_truncated(answer: str) -> bool:
    text = (answer or "").rstrip()
    if not text:
        return True
    if text.endswith((".", "!", "?", "`", "```")):
        return False
    if text.endswith((":", "-", "*", "#")):
        return True
    # If last line is a heading or list lead-in, treat as truncated.
    last_line = text.splitlines()[-1].strip()
    if not last_line:
        return False
    if re.match(r"^#{1,6}\s+", last_line):
        return True
    if re.match(r"^[-*]\s*$", last_line):
        return True
    return True


def _build_closing_completion_prompt(original_query: str, partial_answer: str) -> str:
    tail = _clip_text(partial_answer[-1200:], 1200)
    return (
        "Finish the response cleanly from where it stopped in 2-4 concise lines. "
        "Do not repeat previous text and do not start over.\n\n"
        f"Original user question:\n{original_query}\n\n"
        "Current partial answer tail:\n"
        f"{tail}"
    )


def _create_chat_completion(
    *,
    retrieved_context: str,
    user_content: str,
    max_tokens: int,
    stream: bool,
) -> Any:
    messages = [
        {"role": "system", "content": system_prompt.format(context=retrieved_context)},
        {"role": "user", "content": user_content},
    ]

    reduced_user_input = _extract_current_query_from_model_input(user_content)
    token_attempts: list[int] = []
    for candidate in [
        max_tokens,
        min(_get_retry_output_tokens(), max_tokens),
        min(4096, max_tokens),
        min(2048, max_tokens),
        min(1024, max_tokens),
    ]:
        if candidate >= 64 and candidate not in token_attempts:
            token_attempts.append(candidate)

    last_error: Exception | None = None
    for idx, attempt_tokens in enumerate(token_attempts):
        attempt_messages = messages if idx == 0 else [
            {"role": "system", "content": system_prompt.format(context=retrieved_context)},
            {"role": "user", "content": reduced_user_input},
        ]
        try:
            if idx > 0:
                _trace_step(
                    "llm.request.retry_low_tokens",
                    reason="token_limit",
                    stream=stream,
                    retry_max_tokens=attempt_tokens,
                    reduced_input_chars=len(reduced_user_input),
                    attempt=idx + 1,
                )
                logger.warning(
                    "LLM retry attempt=%s stream=%s max_tokens=%s",
                    idx + 1,
                    stream,
                    attempt_tokens,
                )
            return get_azure_openai_client().chat.completions.create(
                model=_get_chat_model(),
                messages=attempt_messages,
                temperature=_get_llm_temperature(),
                max_tokens=attempt_tokens,
                stream=stream,
            )
        except Exception as error:
            last_error = error
            if _is_token_limit_error(error):
                continue
            raise

    if last_error is not None:
        raise last_error
    raise ValueError("LLM completion failed without an error payload.")


def _get_llm_temperature() -> float:
    return float(os.getenv("AZURE_OPENAI_TEMPERATURE", "0.1"))


def _get_embedding_similarity_threshold() -> float:
    raw_value = os.getenv(
        "AZURE_SEARCH_SCORE_THRESHOLD",
        os.getenv("EMBEDDING_SIMILARITY_THRESHOLD", "0.12"),
    )
    try:
        threshold = float(raw_value)
    except ValueError:
        threshold = 0.12
    return max(0.0, threshold)


def _get_query_overlap_threshold() -> float:
    raw_value = os.getenv("QUERY_OVERLAP_THRESHOLD", "0.08")
    try:
        threshold = float(raw_value)
    except ValueError:
        threshold = 0.08
    return max(0.0, min(threshold, 1.0))


def _get_query_min_overlap_terms() -> int:
    raw_value = os.getenv("QUERY_MIN_OVERLAP_TERMS", "1")
    try:
        count = int(raw_value)
    except ValueError:
        count = 1
    return max(1, min(count, 20))


def _get_strong_match_score_threshold() -> float:
    raw_value = os.getenv("RAG_STRONG_MATCH_SCORE_THRESHOLD", "0.55")
    try:
        threshold = float(raw_value)
    except ValueError:
        threshold = 0.55
    return max(0.0, min(threshold, 2.0))


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


def _normalize_retrieval_score(score: float, retrieval_mode: str) -> float:
    safe_score = max(0.0, float(score or 0.0))
    mode = (retrieval_mode or "").strip().lower()
    if mode == "text":
        # Text search scores are not bounded to [0, 1]; compress for fair cross-mode ranking.
        return safe_score / (safe_score + 2.0)
    return max(0.0, min(safe_score, 1.0))


def _get_memory_turns() -> int:
    return int(os.getenv("CONVERSATION_MEMORY_TURNS", "6"))


def _build_response_style_instruction(current_query: str) -> str:
    query = current_query.lower().strip()

    detailed_markers = {
        "detailed",
        "in depth",
        "in-depth",
        "full detail",
        "full details",
        "step by step",
        "architecture",
        "implementation",
        "deep dive",
        "technical details",
    }
    wants_detailed = any(marker in query for marker in detailed_markers)

    asks_broad_list = (
        ("all" in query and "project" in query)
        or "list all" in query
        or "all voice based" in query
        or "all ai projects" in query
    )

    asks_voice_projects = (
        "voice project" in query
        or "voice-based project" in query
        or ("voice" in query and "project" in query)
    )

    if wants_detailed:
        return (
            "Response style: Provide a well-structured but focused answer in markdown. "
            "Avoid unnecessary repetition. Use concise sections and practical points."
        )

    if asks_broad_list:
        return (
            "Response style: Return a complete structured markdown answer. Include up to 6 projects found in context. "
            "For each project include: Challenge, Solution, Impact, and Technologies in 1-2 bullets each. "
            "Avoid truncation and finish with a short summary line."
        )

    if asks_voice_projects:
        return (
            "Response style: Provide a detailed voice-project overview in markdown. Include all relevant voice projects "
            "from context and for each include Challenge, Solution, Impact, and Technologies. "
            "Do not stop after one project unless only one exists in context."
        )

    return (
        "Response style: Keep the answer concise and useful in 4-8 lines, focused on the exact question. "
        "Avoid long paragraphs unless explicitly requested."
    )


def _build_model_input(session_id: str, current_query: str) -> str:
    with _conversation_lock:
        history = list(_conversation_store.get(session_id, []))

    style_instruction = _build_response_style_instruction(current_query)

    if not history:
        return (
            f"{style_instruction}\n\n"
            "Current user question:\n"
            + current_query
        )

    history_lines: list[str] = []
    for user_text, assistant_text in history:
        history_lines.append(f"User: {user_text}")
        history_lines.append(f"Assistant: {assistant_text}")

    return (
        f"{style_instruction}\n\n"
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
    "Do not generate or mention an 'about us' page or link if it does not exist on the original site. "
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


def _repair_malformed_document_url(url: str) -> str:
    normalized = (url or "").strip()
    if not normalized:
        return ""

    try:
        parsed = urlparse(normalized)
    except Exception:
        parsed = None

    if parsed:
        repaired_path = re.sub(
            r"(\.(?:pdf|doc|docx|txt|md|csv|xls|xlsx|ppt|pptx|html|htm|json))\d+$",
            r"\1",
            parsed.path or "",
            flags=re.IGNORECASE,
        )
        if repaired_path != (parsed.path or ""):
            return parsed._replace(path=repaired_path).geturl()
        return normalized

    return re.sub(
        r"(\.(?:pdf|doc|docx|txt|md|csv|xls|xlsx|ppt|pptx|html|htm|json))\d+$",
        r"\1",
        normalized,
        flags=re.IGNORECASE,
    )


def _normalize_citation_url(value: str) -> str:
    raw = (value or "").strip().strip('"').strip("'")
    if not raw:
        return ""

    # Block 'about us' page links if not present in the original site
    about_us_patterns = [
        "/about-us", "/aboutus", "/about_us", "/about", "about-us", "aboutus", "about_us", "about"
    ]
    for pattern in about_us_patterns:
        if pattern in raw.lower():
            # If the about us page does not exist, block the link
            return ""
    
    
    if _looks_like_url(raw):
        return _repair_malformed_document_url(raw.replace(" ", "%20"))

    decoded = unquote(raw).strip()
    if _looks_like_url(decoded):
        return _repair_malformed_document_url(decoded.replace(" ", "%20"))

    if raw.lower().startswith("www."):
        return _repair_malformed_document_url(f"https://{raw}".replace(" ", "%20"))

    if decoded.lower().startswith("www."):
        return _repair_malformed_document_url(f"https://{decoded}".replace(" ", "%20"))

    # Handle raw host/path values such as desirechatbotweb.blob.core.windows.net/container/file.pdf
    if re.match(r"^[a-z0-9.-]+\.[a-z]{2,}/", raw, flags=re.IGNORECASE):
        return _repair_malformed_document_url(f"https://{raw}".replace(" ", "%20"))

    if re.match(r"^[a-z0-9.-]+\.[a-z]{2,}/", decoded, flags=re.IGNORECASE):
        return _repair_malformed_document_url(f"https://{decoded}".replace(" ", "%20"))

    return ""


def _is_blob_source_url(value: str) -> bool:
    normalized = _normalize_citation_url(value)
    if not normalized:
        return False

    lowered = normalized.lower()
    if ".blob.core.windows.net/" in lowered or ".blob.storage.azure.net/" in lowered:
        return True

    blob_base = _get_azure_blob_base_url()
    if blob_base and lowered.startswith(f"{blob_base}/"):
        return True

    return False


def _looks_like_document_file_name(value: str) -> bool:
    candidate = (value or "").strip().lower().replace("\\", "/")
    if not candidate:
        return False
    name = candidate.rsplit("/", 1)[-1]
    return bool(re.search(r"\.(pdf|doc|docx|txt|md|csv|xls|xlsx|ppt|pptx|html|htm|json)$", name))


def _build_blob_url_from_source_path(value: str) -> str:
    source = (value or "").strip()
    if not source:
        return ""

    normalized = _normalize_citation_url(source)
    if normalized:
        return normalized

    blob_base = _get_azure_blob_base_url().rstrip("/")
    if not blob_base:
        return ""

    normalized_path = source.replace("\\", "/").strip("/")
    if "/" not in normalized_path:
        return ""

    container, blob_path = normalized_path.split("/", 1)
    container = container.strip()
    blob_path = blob_path.strip()
    if not container or not blob_path:
        return ""

    return f"{blob_base}/{quote(container, safe='-_.~')}/{quote(blob_path, safe='/-_.~')}"


def _is_blob_source_payload(payload: dict) -> bool:
    candidates = [
        str(payload.get("url") or "").strip(),
        str(payload.get("source_url") or "").strip(),
        str(payload.get("metadata_storage_path") or "").strip(),
        str(payload.get("source") or "").strip(),
        str(payload.get("file_name") or "").strip(),
        str(payload.get("document_name") or "").strip(),
        str(payload.get("id") or "").strip(),
        str(payload.get("chunk_id") or "").strip(),
        str(payload.get("parent_id") or "").strip(),
        _decode_parent_id_to_url(str(payload.get("parent_id") or "").strip()),
    ]

    for item in candidates:
        if not item:
            continue
        if _is_blob_source_url(item):
            return True
        if _build_blob_url_from_source_path(item):
            return True
        decoded = _try_decode_base64_to_url(item)
        if decoded and _is_blob_source_url(decoded):
            return True

    # Some Azure AI Search indexers only keep filename-level source metadata.
    # Treat document-like source names as blob-backed (data.txt is filtered elsewhere).
    source_candidates = [
        str(payload.get("source") or "").strip(),
        str(payload.get("file_name") or "").strip(),
        str(payload.get("document_name") or "").strip(),
    ]
    for item in source_candidates:
        if _looks_like_document_file_name(item):
            return True

    return False


def _extract_urls_from_text(value: str) -> list[str]:
    if not value:
        return []

    pattern = re.compile(r"https?://[^\s<>'\"]+", flags=re.IGNORECASE)
    urls: list[str] = []
    for match in pattern.finditer(value):
        candidate = match.group(0).strip().rstrip(".,;:!?)]}'\"")
        if candidate:
            urls.append(candidate)
    return urls


def _is_video_candidate_url(url: str) -> bool:
    lowered = (url or "").lower()
    if not lowered:
        return False

    known_hosts = [
        "youtube.com",
        "youtu.be",
        "vimeo.com",
        "loom.com",
        "streamable.com",
        "dailymotion.com",
        "vidyard.com",
        "wistia.com",
    ]
    if any(host in lowered for host in known_hosts):
        return True

    return bool(re.search(r"\.(mp4|webm|mov|m3u8)(\?|$)", lowered))


def _to_video_embed_url(url: str) -> str:
    normalized = _normalize_citation_url(url)
    if not normalized:
        return ""

    try:
        parsed = urlparse(normalized)
    except Exception:
        return ""

    host = (parsed.netloc or "").lower().split(":", 1)[0]
    path = (parsed.path or "").strip()

    if host == "youtu.be":
        video_id = path.strip("/")
        if video_id:
            return f"https://www.youtube.com/embed/{video_id}"

    if "youtube.com" in host:
        if path == "/watch":
            video_id = parse_qs(parsed.query or "").get("v", [""])[0].strip()
            if video_id:
                return f"https://www.youtube.com/embed/{video_id}"
        if path.startswith("/shorts/"):
            video_id = path.split("/", 3)[2].strip()
            if video_id:
                return f"https://www.youtube.com/embed/{video_id}"
        if path.startswith("/embed/"):
            return normalized

    if "vimeo.com" in host:
        if host.startswith("player.") and path.startswith("/video/"):
            return normalized
        match = re.search(r"/(\d+)", path)
        if match:
            return f"https://player.vimeo.com/video/{match.group(1)}"

    if re.search(r"\.(mp4|webm|mov|m3u8)(\?|$)", normalized.lower()):
        return normalized

    return ""


def _tokenize_video_match_terms(value: str) -> set[str]:
    if not value:
        return set()
    return {
        token
        for token in re.findall(r"[a-z0-9]{3,}", value.lower())
        if token and token not in VIDEO_MATCH_STOPWORDS
    }


def _tokenize_source_match_terms(value: str) -> list[str]:
    if not value:
        return []

    ordered_terms: list[str] = []
    seen: set[str] = set()
    for token in re.findall(r"[a-z0-9]{3,}", value.lower()):
        if token in SOURCE_MATCH_STOPWORDS or token in seen:
            continue
        seen.add(token)
        ordered_terms.append(token)
    return ordered_terms


def _score_query_source_alignment(query_text: str, candidate_text: str) -> int:
    query_terms = _tokenize_source_match_terms(query_text)
    if not query_terms:
        return 0

    candidate_terms = set(_tokenize_source_match_terms(candidate_text))
    if not candidate_terms:
        return 0

    overlap_count = len(candidate_terms.intersection(query_terms))
    normalized_candidate = re.sub(r"[^a-z0-9]+", " ", candidate_text.lower()).strip()

    bigram_hits = 0
    for index in range(len(query_terms) - 1):
        bigram = f"{query_terms[index]} {query_terms[index + 1]}"
        if bigram and bigram in normalized_candidate:
            bigram_hits += 1

    first_term_bonus = 2 if query_terms and query_terms[0] in candidate_terms else 0

    score = (overlap_count * 5) + (bigram_hits * 6) + first_term_bonus

    if len(query_terms) >= 3 and overlap_count <= 1 and bigram_hits == 0:
        score -= 4

    return score


def _is_company_profile_query(query_text: str) -> bool:
    lowered = re.sub(r"\s+", " ", (query_text or "").lower()).strip()
    if not lowered:
        return False

    mentions_company = (
        "desire infoweb" in lowered
        or "desireinfoweb" in lowered
        or "company" in lowered
    )
    if not mentions_company:
        return False

    profile_terms = [
        "vision",
        "mission",
        "about",
        "overview",
        "profile",
        "corporate",
        "who is",
        "what is",
    ]
    return any(term in lowered for term in profile_terms)


def _score_citation_preference(
    citation: dict[str, Any],
    *,
    normalized_query: str,
    answer_text: str,
) -> int:
    title_value = str(citation.get("title") or "").strip()
    url_value = str(citation.get("url") or "").strip()
    snippet_value = str(citation.get("context_snippet") or "").strip()
    match_text = f"{title_value} {url_value} {snippet_value}".strip()

    query_alignment = _score_query_source_alignment(normalized_query, match_text) if normalized_query else 0
    answer_alignment = _score_query_source_alignment(answer_text, match_text) if answer_text else 0
    score = max(query_alignment, answer_alignment)

    if _is_company_profile_query(normalized_query):
        lowered_match = match_text.lower()
        if any(token in lowered_match for token in ["corporate profile", "company profile", "about-us", "about desire", "about our company"]):
            score += 14
        if "calendar" in lowered_match and "calendar" not in normalized_query.lower():
            score -= 10

    return score


def _fetch_company_profile_citation_hint(normalized_query: str) -> list[dict[str, Any]]:
    if not _is_company_profile_query(normalized_query):
        return []

    focused_query = f"{normalized_query} desire infoweb corporate profile about us mission vision"
    focused_top_k = max(3, min(_get_azure_search_top_k(), 8))
    focused_scan_top = _get_retrieval_priority_scan_top(focused_top_k)

    hinted_citations: list[dict[str, Any]] = []
    seen_keys: set[str] = set()

    def _collect(citation_items: list[dict[str, Any]], retrieval_mode: str) -> None:
        for raw in citation_items:
            citation = dict(raw or {})
            normalized_url = _normalize_citation_url(str(citation.get("url") or ""))
            if normalized_url:
                citation["url"] = normalized_url

            path_name = ""
            if normalized_url:
                parsed = urlparse(normalized_url)
                path_name = unquote((parsed.path or "").strip("/")).rsplit("/", 1)[-1].strip().lower()

            key = (
                str(citation.get("url") or "").strip().lower()
                or path_name
                or str(citation.get("id") or "").strip().lower()
                or str(citation.get("title") or "").strip().lower()
            )
            if not key or key in seen_keys:
                continue

            match_text = (
                f"{str(citation.get('title') or '')} {str(citation.get('url') or '')} "
                f"{str(citation.get('context_snippet') or '')}"
            ).lower()
            if not any(token in match_text for token in ["corporate", "profile", "about-us", "about desire", "company"]):
                continue

            seen_keys.add(key)
            citation.setdefault("retrieval_mode", retrieval_mode)
            citation.setdefault("entry_rank", 0.0)
            hinted_citations.append(citation)

    try:
        _, _, text_citations = _search_text_context(
            focused_query,
            focused_top_k,
            scan_top=focused_scan_top,
        )
        _collect(text_citations, "text")
    except Exception as error:
        _trace_step("citations.company_profile_hint.text_error", error=str(error))

    try:
        _, _, vector_citations = _search_vector_context(
            focused_query,
            focused_top_k,
            scan_top=focused_scan_top,
        )
        _collect(vector_citations, "vector")
    except Exception as error:
        _trace_step("citations.company_profile_hint.vector_error", error=str(error))

    return hinted_citations


def _select_best_url_from_context_for_query(query_text: str, context_text: str) -> str:
    if not context_text:
        return ""

    matches = list(re.finditer(r"https?://[^\s<>'\"]+", context_text, flags=re.IGNORECASE))
    if not matches:
        return ""

    candidates: list[tuple[int, int, int, str]] = []
    seen_urls: set[str] = set()
    for index, match in enumerate(matches):
        raw_url = match.group(0).strip().rstrip(".,;:!?)]}'\"")
        normalized_url = _normalize_citation_url(raw_url)
        if not normalized_url:
            continue

        url_key = normalized_url.lower()
        if url_key in seen_urls:
            continue
        seen_urls.add(url_key)

        window_start = max(0, match.start() - 180)
        window_end = min(len(context_text), match.end() + 20)
        local_window = context_text[window_start:window_end]

        alignment_score = _score_query_source_alignment(query_text, f"{normalized_url} {local_window}")
        product_bonus = 1 if "/products/" in url_key else 0

        candidates.append((alignment_score, product_bonus, -index, normalized_url))

    if not candidates:
        return ""

    candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    best_score, _, _, best_url = candidates[0]
    if best_score <= 0:
        # Fall back to the first valid URL if query alignment is weak.
        return sorted(candidates, key=lambda item: item[2], reverse=True)[0][3]
    return best_url


def _guess_video_title_from_context(context_text: str, raw_url: str, normalized_url: str) -> str:
    lines = [line.strip() for line in context_text.splitlines() if line.strip()]
    url_variants = {
        raw_url,
        normalized_url,
        unquote(normalized_url),
    }

    for index, line in enumerate(lines):
        if not any(variant and variant in line for variant in url_variants):
            continue

        cleaned_line = re.sub(r"https?://[^\s<>'\"]+", "", line).strip(" -:|\t")
        if cleaned_line and not re.fullmatch(r"[A-Za-z0-9_-]{8,}", cleaned_line):
            return cleaned_line

        if index > 0:
            previous_line = lines[index - 1].strip(" -:|\t")
            if previous_line and not re.fullmatch(r"[A-Za-z0-9_-]{8,}", previous_line):
                return previous_line
        break

    fallback_name = unquote(normalized_url.split("?", 1)[0]).rstrip("/").rsplit("/", 1)[-1].strip()
    fallback_name = fallback_name.replace("-", " ").replace("_", " ").strip()
    if fallback_name and not re.fullmatch(r"[A-Za-z0-9]{8,}", fallback_name.replace(" ", "")):
        return fallback_name.title()

    if "youtube" in normalized_url or "youtu.be" in normalized_url:
        return "YouTube video"
    if "vimeo" in normalized_url:
        return "Vimeo video"
    return "Video source"


def _extract_video_sources_from_context(context_text: str, query_text: str, limit: int = 1) -> list[dict[str, str]]:
    if not context_text:
        return []

    resolved_limit = max(1, min(limit, 6))
    query_tokens = _tokenize_video_match_terms(query_text)
    seen_embed_urls: set[str] = set()
    scored_videos: list[tuple[int, int, dict[str, str]]] = []
    url_matches = list(re.finditer(r"https?://[^\s<>'\"]+", context_text, flags=re.IGNORECASE))

    for position, match in enumerate(url_matches):
        raw_url = match.group(0).strip().rstrip(".,;:!?)]}'\"")
        normalized_url = _normalize_citation_url(raw_url)
        if not normalized_url or not _is_video_candidate_url(normalized_url):
            continue

        embed_url = _to_video_embed_url(normalized_url)
        if not embed_url:
            continue

        embed_key = embed_url.lower()
        if embed_key in seen_embed_urls:
            continue
        seen_embed_urls.add(embed_key)

        title = _guess_video_title_from_context(context_text, raw_url, normalized_url)
        local_start = max(0, match.start() - 140)
        local_pre_context = context_text[local_start:match.start()]
        local_pre_context = re.sub(r"https?://[^\s<>'\"]+", " ", local_pre_context, flags=re.IGNORECASE)
        local_pre_context = re.sub(r"\s+", " ", local_pre_context).strip()
        local_pre_context = local_pre_context[-90:]
        text_for_match = f"{normalized_url} {local_pre_context}"
        match_score = 1
        if query_tokens:
            match_score = _score_query_source_alignment(query_text, text_for_match)
            min_required_score = 7 if len(query_tokens) >= 2 else 1
            if match_score < min_required_score:
                continue

        scored_videos.append(
            (
                -match_score,
                position,
                {
                    "title": title,
                    "url": normalized_url,
                    "embed_url": embed_url,
                },
            )
        )

    if not scored_videos:
        return []

    scored_videos.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in scored_videos[:resolved_limit]]


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


def _decode_source_token_from_document_id(value: str) -> str:
    candidate = (value or "").strip()
    if not candidate:
        return ""

    match = re.search(r"(?:^|_)s_([A-Za-z0-9_-]+)_i_\d+_u_[0-9a-fA-F]{16,}$", candidate)
    if not match:
        return ""

    token = match.group(1)
    padded = token + ("=" * (-len(token) % 4))
    try:
        decoded = base64.urlsafe_b64decode(padded).decode("utf-8", errors="ignore").strip()
    except Exception:
        return ""

    return decoded.replace("\\", "/")


def _extract_citation_from_payload(
    payload: dict,
    *,
    query_text: str = "",
    content_hint: str = "",
) -> dict[str, Any]:
    source_value = str(payload.get("source") or "").strip()
    if not source_value:
        source_value = _extract_source_name_from_payload(payload)

    title_candidates = [
        str(payload.get("title") or "").strip(),
        source_value,
        str(payload.get("file_name") or "").strip(),
        str(payload.get("document_name") or "").strip(),
    ]
    title = next((item for item in title_candidates if item), "")

    link_candidates = [
        str(payload.get("url") or "").strip(),
        str(payload.get("source_url") or "").strip(),
        _build_blob_url_from_source_path(source_value),
        source_value,
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

    content_text = content_hint or _extract_content_from_payload(payload)

    if query_text and content_text:
        query_matched_url = _select_best_url_from_context_for_query(query_text, content_text)
        if query_matched_url:
            if not link:
                link = query_matched_url
            else:
                parsed_link = urlparse(link)
                parsed_query_link = urlparse(query_matched_url)
                current_is_document_source = _is_blob_source_url(link) or _looks_like_document_file_name(parsed_link.path or "")
                query_match_is_document_source = _is_blob_source_url(query_matched_url) or _looks_like_document_file_name(parsed_query_link.path or "")

                current_alignment = _score_query_source_alignment(query_text, f"{title} {link}")
                matched_alignment = _score_query_source_alignment(query_text, f"{title} {query_matched_url}")

                # Keep reliable source metadata links (blob/doc URLs) unless the context URL is clearly better aligned.
                if not current_is_document_source and matched_alignment >= current_alignment:
                    link = query_matched_url
                elif query_match_is_document_source and matched_alignment > current_alignment:
                    link = query_matched_url

    if not link:
        candidate_urls = _extract_urls_from_text(content_text)
        for item in candidate_urls:
            normalized = _normalize_citation_url(item)
            if not normalized:
                continue
            parsed = urlparse(normalized)
            if _is_blob_source_url(normalized) or _looks_like_document_file_name(parsed.path or ""):
                link = normalized
                break

        if not link:
            for item in candidate_urls:
                normalized = _normalize_citation_url(item)
                if normalized and "desireinfoweb.com/" in normalized.lower():
                    link = normalized
                    break

    if not title:
        if source_value:
            title = source_value.replace("\\", "/").rsplit("/", 1)[-1]
        elif link:
            title = link
        else:
            title = "Source document"
    elif title.lower() == "source document" and link:
        title = link

    if link and title.lower() in {"servicecatalog.txt", "servicecatalog.json"}:
        parsed_link = urlparse(link)
        path_name = unquote((parsed_link.path or "").strip("/")).rsplit("/", 1)[-1].strip()
        if path_name:
            title = path_name
        else:
            title = link

    if _looks_like_url(title):
        normalized_title_url = _normalize_citation_url(title)
        if normalized_title_url:
            parsed_title = urlparse(normalized_title_url)
            path_name = unquote((parsed_title.path or "").strip("/")).rsplit("/", 1)[-1].strip()
            if path_name:
                title = path_name

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


def _extract_source_name_from_payload(payload: dict[str, Any]) -> str:
    direct_candidates = [
        str(payload.get("source") or "").strip(),
        str(payload.get("file_name") or "").strip(),
        str(payload.get("document_name") or "").strip(),
    ]
    for item in direct_candidates:
        if item:
            return item.replace("\\", "/")

    id_candidates = [
        str(payload.get("chunk_id") or "").strip(),
        str(payload.get("id") or "").strip(),
        str(payload.get("parent_id") or "").strip(),
    ]
    for item in id_candidates:
        if not item:
            continue

        decoded_source = _decode_source_token_from_document_id(item)
        if decoded_source:
            return decoded_source

        if "::" in item:
            return item.split("::", 1)[0].replace("\\", "/")

        normalized_url = _normalize_citation_url(item)
        if normalized_url:
            parsed = urlparse(normalized_url)
            normalized_path = (parsed.path or "").strip("/")
            if normalized_path:
                return normalized_path.replace("\\", "/")

        decoded_url = _try_decode_base64_to_url(item)
        if decoded_url:
            parsed = urlparse(decoded_url)
            normalized_path = (parsed.path or "").strip("/")
            if normalized_path:
                return normalized_path.replace("\\", "/")

    return ""


def _is_knowledge_source_payload(payload: dict[str, Any], knowledge_sources: set[str]) -> bool:
    if not knowledge_sources:
        return False

    source_name = _extract_source_name_from_payload(payload)
    if not source_name:
        return False

    source_key = _normalize_source_key(source_name)
    if source_key and source_key in knowledge_sources:
        return True

    full_key = source_name.replace("\\", "/").strip().lower()
    if full_key and full_key in knowledge_sources:
        return True

    return False


def _is_blob_fallback_payload(payload: dict[str, Any], knowledge_sources: set[str]) -> bool:
    if _is_knowledge_source_payload(payload, knowledge_sources):
        return False

    if _is_blob_source_payload(payload):
        return True

    source_name = _extract_source_name_from_payload(payload)
    return bool(source_name and _looks_like_document_file_name(source_name))


def _extract_context_from_results(
    results,
    query: str,
    top_k: int,
    retrieval_mode: str,
    payload_filter: Callable[[dict[str, Any]], bool] | None = None,
) -> tuple[str, float, list[dict[str, Any]]]:
    scored_entries: list[dict[str, Any]] = []
    query_terms = _tokenize_terms(query)
    excluded_sources = _get_azure_search_excluded_sources()

    for result in results:
        payload = dict(result)

        if payload_filter and not payload_filter(payload):
            continue

        if _blob_only_sources_enabled() and not _is_blob_source_payload(payload):
            continue

        if excluded_sources:
            source_value = str(
                payload.get("source")
                or payload.get("file_name")
                or payload.get("document_name")
                or ""
            ).strip().lower()
            source_name = source_value.replace("\\", "/").rsplit("/", 1)[-1] if source_value else ""
            if source_value in excluded_sources or source_name in excluded_sources:
                continue

        try:
            score_value = float(payload.get("@search.score") or 0.0)
        except (TypeError, ValueError):
            score_value = 0.0

        page_content = _extract_content_from_payload(payload)
        if not page_content:
            continue

        citation = _extract_citation_from_payload(
            payload,
            query_text=query,
            content_hint=page_content,
        )
        overlap_ratio, overlap_count, query_term_count = _compute_query_overlap(query, page_content)
        normalized_score = _normalize_retrieval_score(score_value, retrieval_mode)

        rank = (overlap_ratio * 0.70) + (normalized_score * 0.30)
        if query_terms and overlap_count == 0:
            rank -= 0.22
        if query_term_count >= 2 and overlap_count < 2:
            rank -= 0.08

        citation = dict(citation or {})
        citation["context_snippet"] = re.sub(r"\s+", " ", page_content).strip()[:1800]
        citation["overlap_count"] = overlap_count
        citation["overlap_ratio"] = round(overlap_ratio, 6)
        citation["entry_rank"] = round(rank, 6)
        citation["retrieval_mode"] = retrieval_mode

        scored_entries.append(
            {
                "content": page_content,
                "score": score_value,
                "normalized_score": normalized_score,
                "overlap_ratio": overlap_ratio,
                "overlap_count": overlap_count,
                "rank": rank,
                "citation": citation,
            }
        )

    if not scored_entries:
        return "", 0.0, []

    scored_entries.sort(
        key=lambda item: (
            float(item.get("rank") or 0.0),
            int(item.get("overlap_count") or 0),
            float(item.get("normalized_score") or 0.0),
            float(item.get("score") or 0.0),
        ),
        reverse=True,
    )

    selected_entries = scored_entries[:top_k]
    context_chunks = [str(item.get("content") or "")[:2000] for item in selected_entries if str(item.get("content") or "")]
    top_score = max(float(item.get("score") or 0.0) for item in selected_entries)

    citations: list[dict[str, Any]] = []
    seen_citations: set[str] = set()
    for item in selected_entries:
        citation = dict(item.get("citation") or {})
        citation_url = _normalize_citation_url(str(citation.get("url") or ""))
        if citation_url:
            citation["url"] = citation_url

        path_name = ""
        if citation_url:
            parsed_url = urlparse(citation_url)
            path_name = unquote((parsed_url.path or "").strip("/")).rsplit("/", 1)[-1].strip().lower()

        key = (
            str(citation.get("url") or "").strip().lower()
            or path_name
            or str(citation.get("id") or "").strip().lower()
            or str(citation.get("title") or "").strip().lower()
        )
        if not key or key in seen_citations:
            continue
        seen_citations.add(key)
        citations.append(citation)

    return "\n\n".join(context_chunks), top_score, citations


def _search_vector_context(
    query: str,
    top_k: int,
    payload_filter: Callable[[dict[str, Any]], bool] | None = None,
    scan_top: int | None = None,
) -> tuple[str, float, list[dict[str, Any]]]:
    vector_field = _get_azure_search_vector_field()
    if not vector_field:
        logger.warning("Vector field not configured, skipping vector search.")
        return "", 0.0, []

    resolved_scan_top = max(top_k, scan_top or top_k)

    logger.info("Vector search: generating embeddings for query (len=%d)...", len(query))
    query_vector = get_embeddings_client().embed_query(query)
    logger.info("Vector search: embedding generated (dim=%d)", len(query_vector))
    
    vector_query = VectorizedQuery(
        vector=query_vector,
        k_nearest_neighbors=top_k,
        fields=vector_field,
    )
    logger.info("Vector search: executing search against field '%s' with top_k=%d...", vector_field, top_k)
    results = get_search_client().search(
        search_text=None,
        vector_queries=[vector_query],
        top=resolved_scan_top,
    )
    context, score, citations = _extract_context_from_results(
        results,
        query,
        top_k,
        retrieval_mode="vector",
        payload_filter=payload_filter,
    )
    logger.info(
        "Vector search: completed - scan_top=%d retrieved=%d chars top_score=%.4f citations=%d",
        resolved_scan_top,
        len(context),
        score,
        len(citations),
    )
    return context, score, citations


def _search_text_context(
    query: str,
    top_k: int,
    payload_filter: Callable[[dict[str, Any]], bool] | None = None,
    scan_top: int | None = None,
) -> tuple[str, float, list[dict[str, Any]]]:
    semantic_config = _get_azure_search_semantic_config()
    resolved_scan_top = max(top_k, scan_top or top_k)
    logger.info("Text search: executing with top_k=%d, semantic_enabled=%s", top_k, _use_azure_search_semantic() and bool(semantic_config))
    
    if _use_azure_search_semantic() and semantic_config:
        logger.info("Text search: using semantic config '%s'", semantic_config)
        results = get_search_client().search(
            search_text=query,
            top=resolved_scan_top,
            query_type="semantic",
            semantic_configuration_name=semantic_config,
        )
    else:
        logger.info("Text search: using basic keyword search")
        results = get_search_client().search(
            search_text=query,
            top=resolved_scan_top,
        )

    context, score, citations = _extract_context_from_results(
        results,
        query,
        top_k,
        retrieval_mode="text",
        payload_filter=payload_filter,
    )
    logger.info(
        "Text search: completed - scan_top=%d retrieved=%d chars top_score=%.4f citations=%d",
        resolved_scan_top,
        len(context),
        score,
        len(citations),
    )
    return context, score, citations


def _build_retrieval_candidate(
    query: str,
    source_scope: str,
    retrieval_mode: str,
    context: str,
    score: float,
    citations: list[dict[str, Any]],
) -> dict[str, Any]:
    overlap_ratio, overlap_count, query_term_count = _compute_query_overlap(query, context)
    normalized_score = _normalize_retrieval_score(score, retrieval_mode)
    score_threshold = _get_embedding_similarity_threshold()
    overlap_threshold = _get_query_overlap_threshold()
    strong_match_threshold = _get_strong_match_score_threshold()

    if query_term_count == 0:
        meets_overlap = True
    else:
        required_overlap_terms = min(_get_query_min_overlap_terms(), query_term_count)
        strong_match = normalized_score >= strong_match_threshold
        # For strong semantic hits, allow single-term lexical overlap.
        # This avoids false negatives on natural phrasing like
        # "give me detail about fluentify ai" where only the topic token overlaps.
        if query_term_count >= 2 and not strong_match:
            required_overlap_terms = max(required_overlap_terms, 2)
        meets_overlap = overlap_count >= required_overlap_terms and (
            overlap_ratio >= overlap_threshold or strong_match
        )

    meets_score = normalized_score >= score_threshold

    # Composite rank balances similarity score and lexical overlap.
    rank = (normalized_score * 0.62) + (overlap_ratio * 0.38)
    if source_scope == "knowledge":
        rank += 0.10
    elif source_scope == "blob_fallback":
        rank -= 0.06
    if query_term_count >= 2 and overlap_count < 2:
        rank -= 0.12
    if not meets_score:
        rank -= 0.2
    if not meets_overlap:
        rank -= 0.35
    if not citations:
        rank -= 0.05

    return {
        "source_scope": source_scope,
        "retrieval_mode": retrieval_mode,
        "context": context,
        "score": score,
        "normalized_score": normalized_score,
        "citations": citations,
        "overlap_ratio": overlap_ratio,
        "overlap_count": overlap_count,
        "query_term_count": query_term_count,
        "rank": rank,
        "meets_score": meets_score,
        "meets_overlap": meets_overlap,
    }


def _select_best_retrieval_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not candidates:
        return None

    return max(
        candidates,
        key=lambda item: (
            1 if bool(item.get("meets_overlap")) else 0,
            1 if bool(item.get("meets_score")) else 0,
            int(item.get("overlap_count") or 0),
            float(item.get("overlap_ratio") or 0.0),
            float(item.get("rank") or 0.0),
            float(item.get("normalized_score") or 0.0),
            float(item.get("score") or 0.0),
            len(str(item.get("context") or "")),
            1 if str(item.get("source_scope") or "") == "knowledge" else 0,
        ),
    )


def _build_candidate_log_payload(candidate: dict[str, Any] | None, *, source_scope: str) -> dict[str, Any]:
    if not candidate:
        return {
            "source_scope": source_scope,
            "found": False,
            "context_chars": 0,
            "context_preview": "",
            "citations_count": 0,
        }

    context_text = str(candidate.get("context") or "")
    context_preview = re.sub(r"\s+", " ", context_text).strip()[:600]
    citations = list(candidate.get("citations") or [])

    return {
        "source_scope": source_scope,
        "found": bool(context_text),
        "retrieval_mode": str(candidate.get("retrieval_mode") or ""),
        "score": round(float(candidate.get("score") or 0.0), 6),
        "normalized_score": round(float(candidate.get("normalized_score") or 0.0), 6),
        "rank": round(float(candidate.get("rank") or 0.0), 6),
        "overlap_ratio": round(float(candidate.get("overlap_ratio") or 0.0), 6),
        "overlap_count": int(candidate.get("overlap_count") or 0),
        "query_term_count": int(candidate.get("query_term_count") or 0),
        "meets_score": bool(candidate.get("meets_score")),
        "meets_overlap": bool(candidate.get("meets_overlap")),
        "context_chars": len(context_text),
        "context_preview": context_preview,
        "citations_count": len(citations),
        "citation": citations[0] if citations else None,
    }


def _collect_candidate_citations(
    candidates: list[dict[str, Any]],
    selected_candidate: dict[str, Any] | None,
    *,
    max_items: int = 40,
) -> list[dict[str, Any]]:
    ordered_candidates: list[dict[str, Any]] = []
    if selected_candidate:
        ordered_candidates.append(selected_candidate)
    ordered_candidates.extend(
        candidate
        for candidate in candidates
        if selected_candidate is None or candidate is not selected_candidate
    )

    merged: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for candidate in ordered_candidates:
        candidate_rank = float(candidate.get("rank") or 0.0)
        candidate_scope = str(candidate.get("source_scope") or "")
        candidate_mode = str(candidate.get("retrieval_mode") or "")

        for raw_citation in list(candidate.get("citations") or []):
            citation = dict(raw_citation or {})
            normalized_url = _normalize_citation_url(str(citation.get("url") or ""))
            if normalized_url:
                citation["url"] = normalized_url

            path_name = ""
            if normalized_url:
                parsed_url = urlparse(normalized_url)
                path_name = unquote((parsed_url.path or "").strip("/")).rsplit("/", 1)[-1].strip().lower()

            identity_key = (
                str(citation.get("url") or "").strip().lower()
                or path_name
                or str(citation.get("id") or "").strip().lower()
                or str(citation.get("title") or "").strip().lower()
            )
            if not identity_key or identity_key in seen_keys:
                continue

            seen_keys.add(identity_key)
            citation.setdefault("entry_rank", round(candidate_rank, 6))
            citation.setdefault("source_scope", candidate_scope)
            citation.setdefault("retrieval_mode", candidate_mode)
            merged.append(citation)
            if len(merged) >= max_items:
                return merged

    return merged


def _retrieve_context_and_score(query: str) -> tuple[str, float, list[dict[str, Any]]]:
    top_k = _get_azure_search_top_k()
    priority_sources = _get_priority_knowledge_sources()
    priority_scan_top = _get_retrieval_priority_scan_top(top_k) if priority_sources else top_k

    _trace_step(
        "retrieval.start",
        top_k=top_k,
        vector_field=_get_azure_search_vector_field(),
        content_field=_get_azure_search_content_field(),
        priority_sources_count=len(priority_sources),
        priority_scan_top=priority_scan_top,
    )

    knowledge_filter = (
        (lambda payload: _is_knowledge_source_payload(payload, priority_sources))
        if priority_sources
        else None
    )
    blob_fallback_filter = (
        (lambda payload: _is_blob_fallback_payload(payload, priority_sources))
        if priority_sources
        else None
    )

    scopes: list[tuple[str, Callable[[dict[str, Any]], bool] | None]]
    if priority_sources:
        scopes = [
            ("knowledge", knowledge_filter),
            ("blob_fallback", blob_fallback_filter),
        ]
    else:
        scopes = [("all", None)]

    candidates: list[dict[str, Any]] = []
    retrieval_errors: list[Exception] = []

    for scope_name, scope_filter in scopes:
        try:
            _trace_step("retrieval.vector.start", scope=scope_name)
            context, score, citations = _search_vector_context(
                query,
                top_k,
                payload_filter=scope_filter,
                scan_top=priority_scan_top,
            )
            if context:
                candidate = _build_retrieval_candidate(
                    query,
                    scope_name,
                    "vector",
                    context,
                    score,
                    citations,
                )
                candidates.append(candidate)
                _trace_step(
                    "retrieval.vector.success",
                    scope=scope_name,
                    top_score=score,
                    context_chars=len(context),
                    rank=round(float(candidate.get("rank") or 0.0), 6),
                    overlap_ratio=round(float(candidate.get("overlap_ratio") or 0.0), 6),
                )
            else:
                _trace_step("retrieval.vector.empty", scope=scope_name)
        except Exception as retriever_error:
            logger.warning("Vector retrieval failed for scope=%s (%s).", scope_name, retriever_error)
            _trace_step("retrieval.vector.error", scope=scope_name, error=str(retriever_error))
            retrieval_errors.append(retriever_error)

        try:
            _trace_step("retrieval.text.start", scope=scope_name)
            context, score, citations = _search_text_context(
                query,
                top_k,
                payload_filter=scope_filter,
                scan_top=priority_scan_top,
            )
            if context:
                candidate = _build_retrieval_candidate(
                    query,
                    scope_name,
                    "text",
                    context,
                    score,
                    citations,
                )
                candidates.append(candidate)
                _trace_step(
                    "retrieval.text.result",
                    scope=scope_name,
                    top_score=score,
                    context_chars=len(context),
                    rank=round(float(candidate.get("rank") or 0.0), 6),
                    overlap_ratio=round(float(candidate.get("overlap_ratio") or 0.0), 6),
                )
            else:
                _trace_step("retrieval.text.empty", scope=scope_name)
        except Exception as retriever_error:
            logger.warning("Text retrieval failed for scope=%s (%s).", scope_name, retriever_error)
            _trace_step("retrieval.text.error", scope=scope_name, error=str(retriever_error))
            retrieval_errors.append(retriever_error)

    if priority_sources:
        knowledge_best = _select_best_retrieval_candidate(
            [candidate for candidate in candidates if str(candidate.get("source_scope") or "") == "knowledge"]
        )
        blob_best = _select_best_retrieval_candidate(
            [candidate for candidate in candidates if str(candidate.get("source_scope") or "") == "blob_fallback"]
        )

        _trace_pipeline_stage(
            "ai_search_json_response",
            **_build_candidate_log_payload(knowledge_best, source_scope="knowledge"),
        )
        _trace_pipeline_stage(
            "ai_search_blob_response",
            **_build_candidate_log_payload(blob_best, source_scope="blob_fallback"),
        )

    selected = _select_best_retrieval_candidate(candidates)
    if selected:
        _trace_pipeline_stage(
            "ai_selection",
            selected_source_scope=selected.get("source_scope"),
            selected_retrieval_mode=selected.get("retrieval_mode"),
            selected_score=round(float(selected.get("score") or 0.0), 6),
            selected_rank=round(float(selected.get("rank") or 0.0), 6),
            selected_overlap_ratio=round(float(selected.get("overlap_ratio") or 0.0), 6),
            candidate_count=len(candidates),
        )
        _trace_step(
            "retrieval.selection",
            candidate_count=len(candidates),
            selected_scope=selected.get("source_scope"),
            selected_mode=selected.get("retrieval_mode"),
            selected_score=selected.get("score"),
            selected_rank=round(float(selected.get("rank") or 0.0), 6),
            selected_overlap_ratio=round(float(selected.get("overlap_ratio") or 0.0), 6),
            candidates=[
                {
                    "scope": candidate.get("source_scope"),
                    "mode": candidate.get("retrieval_mode"),
                    "score": round(float(candidate.get("score") or 0.0), 6),
                    "rank": round(float(candidate.get("rank") or 0.0), 6),
                    "overlap_ratio": round(float(candidate.get("overlap_ratio") or 0.0), 6),
                }
                for candidate in candidates
            ],
        )
        selected_citations = _collect_candidate_citations(candidates, selected)
        _trace_step(
            "retrieval.citations",
            count=len(selected_citations),
            primary=selected_citations[0] if selected_citations else None,
        )
        return (
            str(selected.get("context") or ""),
            float(selected.get("score") or 0.0),
            selected_citations,
        )

    if retrieval_errors and _is_azure_search_required():
        first_error = retrieval_errors[0]
        raise RetrievalUnavailableError(
            "Azure Search retrieval failed. Verify AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_INDEX_NAME, "
            "network access, and AZURE_SEARCH_API_KEY or managed identity permissions."
        ) from first_error

    _trace_step("retrieval.empty", candidate_count=0)
    return "", 0.0, []


def _should_use_embedding_context(normalized_query: str, retrieved_context: str, top_score: float) -> bool:
    logger.info("Context evaluation: checking if retrieved context should be used...")
    logger.info("  - retrieved_context length: %d chars", len(retrieved_context))
    logger.info("  - top_score: %.6f", top_score)
    
    if not retrieved_context:
        _trace_step("context.rejected", reason="empty_context")
        logger.warning("Context rejected: no retrieved context")
        return False
    
    score_threshold = _get_embedding_similarity_threshold()
    logger.info("  - score_threshold: %.6f", score_threshold)
    
    if top_score < score_threshold:
        _trace_step(
            "context.rejected",
            reason="score_below_threshold",
            top_score=top_score,
            score_threshold=score_threshold,
        )
        logger.warning("Context rejected: top_score (%.4f) below threshold (%.4f)", top_score, score_threshold)
        return False

    overlap_ratio, overlap_count, query_term_count = _compute_query_overlap(normalized_query, retrieved_context)
    overlap_threshold = _get_query_overlap_threshold()
    min_overlap_terms = _get_query_min_overlap_terms()
    strong_match_score_threshold = _get_strong_match_score_threshold()
    normalized_top_score = _normalize_retrieval_score(top_score, "text" if top_score > 1.0 else "vector")
    strong_match = normalized_top_score >= strong_match_score_threshold

    logger.info("  - query terms: %d (overlap: %d/%.2f%%)", query_term_count, overlap_count, overlap_ratio * 100)
    logger.info(
        "  - thresholds: overlap_ratio=%.2f, overlap_terms=%d, strong_match_score=%.4f (normalized=%.4f)",
        overlap_threshold,
        min_overlap_terms,
        strong_match_score_threshold,
        normalized_top_score,
    )

    if query_term_count == 0:
        _trace_step(
            "context.rejected",
            reason="query_has_no_meaningful_terms",
            min_overlap_terms=min_overlap_terms,
            query_term_count=query_term_count,
        )
        logger.warning("Context rejected: query has no meaningful terms")
        return False

    required_overlap_terms = min(min_overlap_terms, query_term_count)
    if query_term_count >= 2 and not strong_match:
        required_overlap_terms = max(required_overlap_terms, 2)
    if overlap_count < required_overlap_terms:
        logger.warning("Context rejected: overlap_count (%d) < required_overlap_terms (%d)", overlap_count, required_overlap_terms)
        _trace_step(
            "context.rejected",
            reason="query_overlap_terms_below_minimum",
            overlap_count=overlap_count,
            min_overlap_terms=required_overlap_terms,
            configured_min_overlap_terms=min_overlap_terms,
            query_term_count=query_term_count,
            normalized_top_score=round(normalized_top_score, 4),
            strong_match=strong_match,
        )
        return False

    if overlap_ratio < overlap_threshold and not strong_match:
        logger.warning("Context rejected: overlap_ratio (%.4f) < threshold (%.4f)", overlap_ratio, overlap_threshold)
        _trace_step(
            "context.rejected",
            reason="query_overlap_below_threshold",
            overlap_ratio=round(overlap_ratio, 4),
            overlap_threshold=overlap_threshold,
            overlap_count=overlap_count,
            query_term_count=query_term_count,
            normalized_top_score=round(normalized_top_score, 4),
            strong_match=strong_match,
        )
        return False

    logger.info("Context ACCEPTED: all checks passed (score=%.4f, overlap=%.2f%%/%d terms)", top_score, overlap_ratio * 100, overlap_count)
    _trace_step(
        "context.accepted",
        top_score=top_score,
        score_threshold=score_threshold,
        overlap_ratio=round(overlap_ratio, 4),
        overlap_threshold=overlap_threshold,
        overlap_count=overlap_count,
        min_overlap_terms=required_overlap_terms,
        configured_min_overlap_terms=min_overlap_terms,
        normalized_top_score=round(normalized_top_score, 4),
        strong_match=strong_match,
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


def _build_context_grounded_fallback_answer(normalized_query: str, retrieved_context: str) -> str:
    context = (retrieved_context or "").strip()
    if not context:
        return _no_context_response()

    lowered_query = (normalized_query or "").lower()
    normalized_context = re.sub(r"\n\s*\n", "\n", context)
    context_lc = normalized_context.lower()

    asks_voice_catalog = (
        ("voice based projects" in lowered_query)
        or ("voice based project" in lowered_query and any(token in lowered_query for token in ["all", "list"]))
        or (
            "voice" in lowered_query
            and "projects" in lowered_query
            and any(token in lowered_query for token in ["all", "list", "complete", "full"])
        )
    )

    if asks_voice_catalog:
        def _clean_field_text(value: str) -> str:
            cleaned = re.sub(r"\s+", " ", (value or "")).strip(" -•\t")
            cleaned = re.sub(r"\s*[●•]\s*", ". ", cleaned)
            cleaned = re.sub(r"^[\)\]\.,:;\-/]+", "", cleaned).strip()
            cleaned = re.sub(r"\bProject\s*\d+\s*:\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(
                r"\b(Client|Industry|Department|Technology Focus Area|Architecture|Deployment|Databases)\s*:\s*[^:]{0,220}",
                "",
                cleaned,
                flags=re.IGNORECASE,
            )
            cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" ,;:-")
            return cleaned

        def _compact_field_text(value: str, max_sentences: int = 2, max_chars: int = 340) -> str:
            cleaned = _clean_field_text(value)
            if not cleaned:
                return ""
            parts = [part.strip() for part in re.split(r"(?<=[\.!?])\s+", cleaned) if part.strip()]
            if parts:
                cleaned = " ".join(parts[:max_sentences]).strip()
            if len(cleaned) > max_chars:
                truncated = cleaned[: max_chars - 1].rstrip()
                if " " in truncated:
                    truncated = truncated.rsplit(" ", 1)[0].rstrip()
                cleaned = truncated.rstrip(" ,;:-") + "."
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            if cleaned.lower().endswith(("our", "and", "or", "to", "for", "with")):
                cleaned = cleaned.rsplit(" ", 1)[0].rstrip(" ,;:-") + "."
            return cleaned

        def _extract_labeled_field(source_text: str, label: str, next_labels: list[str]) -> str:
            match = re.search(rf"\b{label}\s*:\s*", source_text, flags=re.IGNORECASE)
            if not match:
                return ""
            tail = source_text[match.end():]
            end_idx = len(tail)
            for next_label in next_labels:
                next_match = re.search(rf"\b{next_label}\s*:\s*", tail, flags=re.IGNORECASE)
                if next_match and next_match.start() < end_idx:
                    end_idx = next_match.start()
            return _clean_field_text(tail[:end_idx])

        def _extract_project_excerpt(source_text: str, marker: str, window: int = 1100) -> str:
            lowered = source_text.lower()
            idx = lowered.find(marker.lower())
            if idx < 0:
                return ""
            start = max(0, idx - 140)
            end = min(len(source_text), idx + window)
            snippet = source_text[start:end]
            snippet = re.sub(r"\s+", " ", snippet).strip()
            return snippet

        def _extract_sentences(snippet: str, max_sentences: int = 4) -> list[str]:
            parts = [part.strip() for part in re.split(r"(?<=[\.!?])\s+", snippet) if part.strip()]
            cleaned: list[str] = []
            for part in parts:
                normalized_part = _clean_field_text(part)
                if len(normalized_part) < 30:
                    continue
                if "thank you for your query" in normalized_part.lower():
                    continue
                cleaned.append(normalized_part)
                if len(cleaned) >= max_sentences:
                    break
            return cleaned

        def _extract_tech_line(snippet: str) -> str:
            tech_keywords = [
                "azure", "openai", "gemini", "teams", "typescript", "react", "python", "speech", "stt",
                "tts", "sharepoint", "durable functions", "blob", "semantic search", "vectorization",
            ]
            matches: list[str] = []
            lowered = snippet.lower()
            for kw in tech_keywords:
                if kw in lowered:
                    matches.append(kw.title())
            if matches:
                deduped: list[str] = []
                seen: set[str] = set()
                for item in matches:
                    key = item.lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    deduped.append(item)
                return ", ".join(deduped[:8])
            return "Not explicitly listed in the retrieved excerpt."

        project_specs = [
            ("AI Interviewer Pro", ["ai interviewer pro", "candidate screening"]),
            ("Fluentify AI", ["fluentify", "pronunciation assessment"]),
            ("Voice Workflow Assistant", ["voice workflow", "voice-driven workflow"]),
            ("Compliance Voice Monitoring", ["compliance monitoring", "voice-enabled system"]),
        ]

        sections: list[str] = []
        for project_name, markers in project_specs:
            excerpt = ""
            for marker in markers:
                excerpt = _extract_project_excerpt(normalized_context, marker)
                if excerpt:
                    break
            if not excerpt:
                continue

            sentences = _extract_sentences(excerpt, max_sentences=4)
            if not sentences:
                continue

            challenge = _extract_labeled_field(excerpt, "Challenge", ["Solution", "Impact", "Technologies"])
            solution = _extract_labeled_field(excerpt, "Solution", ["Impact", "Technologies", "Challenge"])
            impact = _extract_labeled_field(excerpt, "Impact", ["Technologies", "Challenge", "Solution"])
            tech_line = _extract_labeled_field(excerpt, "Technologies", ["Challenge", "Solution", "Impact"])

            if not challenge:
                challenge = sentences[0]
            if not solution:
                solution = sentences[1] if len(sentences) > 1 else "Detailed implementation is available in the project context."
            if not impact:
                impact = sentences[2] if len(sentences) > 2 else "Business impact is described in the project documentation."
            if not tech_line:
                tech_line = _extract_tech_line(excerpt)

            challenge = _compact_field_text(challenge, max_sentences=2, max_chars=320)
            solution = _compact_field_text(solution, max_sentences=2, max_chars=360)
            impact = _compact_field_text(impact, max_sentences=2, max_chars=300)
            tech_line = _compact_field_text(tech_line, max_sentences=1, max_chars=180)

            if not challenge:
                challenge = "Project challenge details are available in the indexed project documentation."
            if not solution:
                solution = "Project implementation details are available in the indexed project documentation."
            if not impact:
                impact = "Project outcomes are available in the indexed project documentation."
            if not tech_line:
                tech_line = "Technologies are listed in the indexed project documentation."

            sections.append(
                "\n".join(
                    [
                        f"### {project_name}",
                        f"- **Challenge**: {challenge}",
                        f"- **Solution**: {solution}",
                        f"- **Impact**: {impact}",
                        f"- **Technologies**: {tech_line}",
                    ]
                )
            )

        if sections:
            return (
                "Voice-Based Projects Overview\n\n"
                "Below are the detailed overviews of Desire Infoweb's voice-based projects, including challenge, solution, impact, and technologies:\n\n"
                + "\n\n".join(sections[:4])
            )

    asks_project_catalog = (
        ("project" in lowered_query or "projects" in lowered_query)
        and any(token in lowered_query for token in ["all", "list", "based", "catalog", "overview"])
    )

    if asks_project_catalog:
        def _clean_project_text(value: str) -> str:
            cleaned = re.sub(r"\s+", " ", (value or "")).strip(" -•\t")
            cleaned = re.sub(r"\s*[●•]\s*", ". ", cleaned)
            cleaned = re.sub(r"\bProject\s*\d+\s*:\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" ,;:-")
            if cleaned and cleaned[-1] not in ".!?":
                cleaned += "."
            return cleaned

        def _compact_project_text(value: str, max_sentences: int = 2, max_chars: int = 320) -> str:
            cleaned = _clean_project_text(value)
            if not cleaned:
                return ""
            parts = [part.strip() for part in re.split(r"(?<=[\.!?])\s+", cleaned) if part.strip()]
            if parts:
                cleaned = " ".join(parts[:max_sentences]).strip()
            if len(cleaned) > max_chars:
                trimmed = cleaned[: max_chars - 1].rstrip()
                if " " in trimmed:
                    trimmed = trimmed.rsplit(" ", 1)[0].rstrip()
                cleaned = trimmed.rstrip(" ,;:-") + "."
            if cleaned.lower().endswith(("and.", "or.", "to.", "for.", "with.")):
                cleaned = cleaned.rsplit(" ", 1)[0].rstrip(" ,;:-") + "."
            return cleaned

        def _extract_between_labels(block_text: str, labels: list[str], next_labels: list[str]) -> str:
            match = None
            for label in labels:
                match = re.search(rf"\b{label}\s*:\s*", block_text, flags=re.IGNORECASE)
                if match:
                    break
            if not match:
                return ""

            tail = block_text[match.end():]
            end_idx = len(tail)
            for next_label in next_labels:
                next_match = re.search(rf"\b{next_label}\s*:\s*", tail, flags=re.IGNORECASE)
                if next_match and next_match.start() < end_idx:
                    end_idx = next_match.start()
            return _compact_project_text(tail[:end_idx])

        focus_terms = {
            term
            for term in _tokenize_terms(lowered_query)
            if term not in {"project", "projects", "based", "list", "all", "overview", "catalog", "give"}
        }
        project_matches = list(re.finditer(r"Project\s*\d+\s*:\s*([^\n]+)", normalized_context, flags=re.IGNORECASE))
        project_sections: list[str] = []

        for idx, project_match in enumerate(project_matches):
            start = project_match.start()
            end = project_matches[idx + 1].start() if idx + 1 < len(project_matches) else min(len(normalized_context), start + 2000)
            block = normalized_context[start:end]
            block_lower = block.lower()
            if focus_terms and not any(term in block_lower for term in focus_terms):
                continue

            title_raw = _clean_project_text(project_match.group(1))
            title = re.split(
                r"\b(Client|Industry|Department|Technology Focus Area|Architecture|Deployment|Databases|Challenge|Solution|Impact|Technologies)\s*:",
                title_raw,
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0].strip(" ,;:-")
            if not title:
                title = "Project"

            challenge = _extract_between_labels(block, ["Challenge"], ["Our Solution", "Solution", "Impact", "Technologies", "Software & Frameworks"])
            solution = _extract_between_labels(block, ["Our Solution", "Solution"], ["Impact", "Technologies", "Software & Frameworks", "Challenge"])
            impact = _extract_between_labels(block, ["Impact"], ["Technologies", "Software & Frameworks", "Challenge", "Solution"])
            technologies = _extract_between_labels(block, ["Technologies", "Software & Frameworks"], ["Challenge", "Solution", "Impact"])

            if not challenge:
                challenge = "Challenge details are available in indexed documentation."
            if not solution:
                solution = "Implementation details are available in indexed documentation."
            if not impact:
                impact = "Business outcomes are described in indexed documentation."
            if not technologies:
                technologies = "Technology stack is documented in indexed content."

            project_sections.append(
                "\n".join(
                    [
                        f"### {title}",
                        f"- **Challenge**: {challenge}",
                        f"- **Solution**: {solution}",
                        f"- **Impact**: {impact}",
                        f"- **Technologies**: {technologies}",
                    ]
                )
            )

            if len(project_sections) >= 4:
                break

        if project_sections:
            return (
                "Project Overview\n\n"
                f"Here are project-wise highlights for **{normalized_query.strip()}** based on indexed documents:\n\n"
                + "\n\n".join(project_sections)
            )

    query_terms = _tokenize_terms(normalized_query)

    def _is_complete_sentence(text: str) -> bool:
        cleaned = (text or "").strip()
        if not cleaned:
            return False
        tail = cleaned.lower().rstrip(".?!").strip()
        if not tail:
            return False
        tail_word = tail.split()[-1]
        if tail_word in {"and", "or", "to", "for", "with", "about", "of", "in", "on", "at", "from", "the", "a", "an"}:
            return False
        return True

    def _looks_like_metadata_noise(text: str) -> bool:
        lowered = (text or "").lower()
        label_hits = len(
            re.findall(
                r"\b(client|industry|department|technology focus area|architecture|deployment|databases|project\s*\d+)\b",
                lowered,
            )
        )
        digit_count = len(re.findall(r"\d", lowered))
        alpha_count = len(re.findall(r"[a-z]", lowered))
        digit_ratio = (digit_count / max(1, alpha_count + digit_count))
        if label_hits >= 2:
            return True
        if "project" in lowered and digit_count >= 3:
            return True
        if digit_ratio > 0.22:
            return True
        return False

    def _normalize_sentence(text: str) -> str:
        cleaned = re.sub(r"\s+", " ", (text or "")).strip(" -•\t")
        cleaned = cleaned.replace("●", " ").replace("•", " ")
        cleaned = re.sub(
            r"\b(Client|Industry|Department|Technology Focus Area|Architecture|Deployment|Databases)\s*:\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" ,;:-")
        if not cleaned:
            return ""
        if cleaned[-1] not in ".!?":
            cleaned = cleaned + "."
        return cleaned

    normalized_for_sentences = normalized_context.replace("●", ". ").replace("•", ". ")
    normalized_for_sentences = re.sub(r"\n+", " ", normalized_for_sentences)
    raw_sentences = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", normalized_for_sentences) if segment.strip()]

    scored: list[tuple[int, str]] = []
    seen: set[str] = set()
    for raw_sentence in raw_sentences:
        sentence = _normalize_sentence(raw_sentence)
        if not sentence:
            continue
        if len(sentence) < 38 or len(sentence) > 320:
            continue
        lowered_sentence = sentence.lower()
        if "thank you for your query" in lowered_sentence or "for further assistance" in lowered_sentence:
            continue
        if _looks_like_metadata_noise(sentence):
            continue
        if not _is_complete_sentence(sentence):
            continue

        key = re.sub(r"\W+", "", lowered_sentence)
        if not key or key in seen:
            continue
        seen.add(key)

        overlap = 0
        if query_terms:
            line_terms = _tokenize_terms(sentence)
            overlap = len(line_terms & query_terms)
        scored.append((overlap, sentence))

    scored.sort(key=lambda item: (-item[0], -len(item[1])))

    max_items = 6 if ("project" in lowered_query and any(token in lowered_query for token in ["all", "list", "based"])) else 4
    highlights = [line for _, line in scored[:max_items]]
    if not highlights:
        return _no_context_response()

    bullets = "\n".join(f"- {item}" for item in highlights)
    return (
        f"Here is a concise answer from indexed documents about **{normalized_query.strip()}**:\n\n"
        f"{bullets}\n\n"
        "If you want, I can provide a cleaner project-wise summary next."
    )


def _generate_compact_context_summary(normalized_query: str, retrieved_context: str) -> str:
    context = (retrieved_context or "").strip()
    if not context:
        return ""

    query_terms = _tokenize_terms(normalized_query)

    def _looks_like_metadata_noise(text: str) -> bool:
        lowered = (text or "").lower()
        label_hits = len(
            re.findall(
                r"\b(client|industry|department|technology focus area|architecture|deployment|databases|project\s*\d+)\b",
                lowered,
            )
        )
        digit_count = len(re.findall(r"\d", lowered))
        alpha_count = len(re.findall(r"[a-z]", lowered))
        digit_ratio = (digit_count / max(1, alpha_count + digit_count))
        if label_hits >= 2:
            return True
        if "project" in lowered and digit_count >= 3:
            return True
        if digit_ratio > 0.22:
            return True
        return False

    normalized_for_sentences = context.replace("●", ". ").replace("•", ". ")
    normalized_for_sentences = re.sub(r"\n+", " ", normalized_for_sentences)
    raw_sentences = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", normalized_for_sentences) if segment.strip()]

    ranked: list[tuple[int, str]] = []
    seen: set[str] = set()
    for raw_sentence in raw_sentences:
        sentence = re.sub(r"\s+", " ", raw_sentence).strip(" -•\t")
        if len(sentence) < 42 or len(sentence) > 280:
            continue
        lowered_sentence = sentence.lower()
        if "thank you for your query" in lowered_sentence or "for further assistance" in lowered_sentence:
            continue
        if _looks_like_metadata_noise(sentence):
            continue

        key = re.sub(r"\W+", "", lowered_sentence)
        if not key or key in seen:
            continue
        seen.add(key)

        overlap = 0
        if query_terms:
            overlap = len(_tokenize_terms(sentence) & query_terms)
        ranked.append((overlap, sentence))

    ranked.sort(key=lambda item: (-item[0], -len(item[1])))
    selected = [sentence for _, sentence in ranked[:6]]
    if not selected:
        return ""

    compact_context = "\n".join(f"- {sentence}" for sentence in selected)
    prompt = (
        "Create a clean, complete markdown answer using only the context excerpts. "
        "Do not copy raw metadata, IDs, or fragmented phrases. "
        "Keep it concise and readable with 3-6 bullets when appropriate.\n\n"
        f"User question:\n{normalized_query}\n\n"
        f"Context excerpts:\n{compact_context}"
    )

    try:
        completion = get_azure_openai_client().chat.completions.create(
            model=_get_chat_model(),
            messages=[
                {
                    "role": "system",
                    "content": "You are a RAG response cleaner. Return only grounded, complete, readable markdown.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=min(520, _get_retry_output_tokens()),
            stream=False,
        )
        if not completion.choices:
            return ""
        answer = str(getattr(completion.choices[0].message, "content", "") or "").strip()
        if not answer or _is_no_context_like_answer(answer):
            return ""
        if len(answer) < 80:
            return ""
        return answer
    except Exception as error:
        _trace_step("answer.compact_context_summary.error", error=str(error))
        return ""


def _should_attach_citations(answer: str, normalized_query: str, citations: list[dict[str, Any]]) -> bool:
    if not citations:
        return False

    normalized_answer = answer.strip()
    if normalized_answer == _no_context_response().strip():
        return False

    return True


def _get_citation_max_items() -> int:
    raw_value = _sanitize_env_value(os.getenv("CITATION_MAX_ITEMS", "1"))
    try:
        value = int(raw_value)
    except ValueError:
        value = 1
    return max(1, min(value, 5))


def _select_response_citations(
    citations: list[dict[str, Any]],
    limit: int | None = None,
    normalized_query: str = "",
    answer_text: str = "",
) -> list[dict[str, Any]]:
    if not citations:
        return []

    resolved_limit = _get_citation_max_items() if limit is None else max(1, min(limit, 5))
    selected: list[dict[str, Any]] = []
    seen_urls: set[str] = set()

    normalized_answer = re.sub(r"\s+", " ", (answer_text or "")).strip()
    citations_pool = [dict(item or {}) for item in citations]

    if _is_company_profile_query(normalized_query):
        hint_citations = _fetch_company_profile_citation_hint(normalized_query)
        if hint_citations:
            citations_pool = hint_citations + citations_pool
            _trace_step(
                "citations.company_profile_hint",
                count=len(hint_citations),
                primary=hint_citations[0] if hint_citations else None,
            )

    ranked_citations: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    for raw_item in citations_pool:
        citation = dict(raw_item or {})

        normalized_url = _normalize_citation_url(str(citation.get("url") or ""))
        if normalized_url:
            citation["url"] = normalized_url

        title_value = str(citation.get("title") or "").strip()
        snippet_value = str(citation.get("context_snippet") or "").strip()
        match_text = f"{title_value} {str(citation.get('url') or '')} {snippet_value}".strip()

        query_alignment = _score_query_source_alignment(normalized_query, match_text) if normalized_query else 0
        answer_alignment = _score_query_source_alignment(normalized_answer, match_text) if normalized_answer else 0
        intent_alignment = _score_citation_preference(
            citation,
            normalized_query=normalized_query,
            answer_text=normalized_answer,
        )
        best_alignment = max(query_alignment, answer_alignment, intent_alignment)

        overlap_count = int(citation.get("overlap_count") or 0)
        overlap_ratio = float(citation.get("overlap_ratio") or 0.0)
        entry_rank = float(citation.get("entry_rank") or 0.0)
        retrieval_score = float(citation.get("score") or 0.0)

        ranked_citations.append(
            (
                (
                    1 if str(citation.get("url") or "").strip() else 0,
                    best_alignment,
                    intent_alignment,
                    query_alignment,
                    answer_alignment,
                    overlap_count,
                    overlap_ratio,
                    entry_rank,
                    retrieval_score,
                ),
                citation,
            )
        )

    ranked_citations.sort(key=lambda item: item[0], reverse=True)
    ordered_citations = [item[1] for item in ranked_citations]

    for citation in ordered_citations:
        normalized_url = _normalize_citation_url(str(citation.get("url") or ""))
        if normalized_url:
            citation["url"] = normalized_url

        url_value = str(citation.get("url") or "").strip().lower()
        if url_value:
            if url_value in seen_urls:
                continue
            seen_urls.add(url_value)

        title_value = str(citation.get("title") or "").strip()
        if (not title_value or _looks_like_url(title_value)) and url_value:
            parsed_url = urlparse(str(citation.get("url") or ""))
            path_name = unquote((parsed_url.path or "").strip("/")).rsplit("/", 1)[-1].strip()
            if path_name:
                citation["title"] = path_name

        selected.append(
            {
                "title": str(citation.get("title") or "").strip(),
                "url": str(citation.get("url") or "").strip(),
                "id": str(citation.get("id") or "").strip(),
                "score": float(citation.get("score") or 0.0),
            }
        )
        if len(selected) >= resolved_limit:
            break

    return selected


def _extract_response_videos(normalized_query: str, retrieved_context: str, limit: int = 1) -> list[dict[str, str]]:
    primary_videos = _extract_video_sources_from_context(retrieved_context, normalized_query, limit=limit)
    if primary_videos:
        return primary_videos

    top_k = max(1, min(_get_azure_search_top_k(), 3))
    priority_sources = _get_priority_knowledge_sources()
    scan_top = _get_retrieval_priority_scan_top(top_k) if priority_sources else top_k

    if priority_sources:
        blob_filter: Callable[[dict[str, Any]], bool] | None = (
            lambda payload: _is_blob_fallback_payload(payload, priority_sources)
        )
    else:
        blob_filter = _is_blob_source_payload

    fallback_contexts: list[str] = []
    try:
        blob_vector_context, _, _ = _search_vector_context(
            normalized_query,
            top_k,
            payload_filter=blob_filter,
            scan_top=scan_top,
        )
        if blob_vector_context:
            fallback_contexts.append(blob_vector_context)
    except Exception as error:
        _trace_step("videos.blob_vector.error", error=str(error))

    try:
        blob_text_context, _, _ = _search_text_context(
            normalized_query,
            top_k,
            payload_filter=blob_filter,
            scan_top=scan_top,
        )
        if blob_text_context:
            fallback_contexts.append(blob_text_context)
    except Exception as error:
        _trace_step("videos.blob_text.error", error=str(error))

    if not fallback_contexts:
        return []

    merged_context = "\n\n".join(fallback_contexts)
    merged_videos = _extract_video_sources_from_context(merged_context, normalized_query, limit=limit)
    if merged_videos:
        return merged_videos

    # Final pass: run a video-focused retrieval query to fetch chunks that
    # explicitly contain demo/watch URLs when regular product retrieval omits them.
    focused_query = f"{normalized_query} youtube video demo"
    focused_top_k = max(3, min(8, _get_azure_search_top_k()))
    focused_scan_top = _get_retrieval_priority_scan_top(focused_top_k)
    focused_contexts: list[str] = []

    try:
        focused_text_context, _, _ = _search_text_context(
            focused_query,
            focused_top_k,
            payload_filter=blob_filter,
            scan_top=focused_scan_top,
        )
        if focused_text_context:
            focused_contexts.append(focused_text_context)
    except Exception as error:
        _trace_step("videos.focused_text.error", error=str(error))

    try:
        focused_vector_context, _, _ = _search_vector_context(
            focused_query,
            focused_top_k,
            payload_filter=blob_filter,
            scan_top=focused_scan_top,
        )
        if focused_vector_context:
            focused_contexts.append(focused_vector_context)
    except Exception as error:
        _trace_step("videos.focused_vector.error", error=str(error))

    if not focused_contexts:
        return []

    focused_merged_context = "\n\n".join(focused_contexts)
    return _extract_video_sources_from_context(focused_merged_context, normalized_query, limit=limit)


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
            f"{missing_name}. Check deployed environment variables and /health/config."
        )

    if "429" in lowered or "rate limit" in lowered:
        return "The AI service is temporarily rate-limited. Please try again in a moment."

    if "timeout" in lowered or "timed out" in lowered:
        return "The AI service timed out. Please try again."

    if "401" in lowered or "unauthorized" in lowered or "forbidden" in lowered:
        return "Authentication with an upstream AI service failed. Please verify deployed credentials."

    return "Temporary answer generation issue encountered. A safe fallback response was returned."


def _is_missing_required_env_error(error: Exception) -> bool:
    return str(error).strip().startswith("Missing required environment variable")


def _generate_completion_with_context(model_input: str, retrieved_context: str) -> str:
    _trace_step(
        "llm.completion.request",
        model=_get_chat_model(),
        context_chars=len(retrieved_context),
    )
    original_query = _extract_current_query_from_model_input(model_input)
    current_prompt = model_input
    answer_parts: list[str] = []
    segment_count = 0
    max_segments = _get_max_completion_segments()

    reached_segment_cap_with_length_finish = False
    for segment_index in range(max_segments):
        max_tokens = _get_max_output_tokens() if segment_index == 0 else _get_retry_output_tokens()
        completion = _create_chat_completion(
            retrieved_context=retrieved_context,
            user_content=current_prompt,
            max_tokens=max_tokens,
            stream=False,
        )

        if not completion.choices:
            raise ValueError("Completion response returned no choices.")

        choice = completion.choices[0]
        message = choice.message
        segment_text = str(getattr(message, "content", "") or "").strip()
        if segment_text:
            answer_parts.append(segment_text)
            segment_count += 1

        finish_reason = getattr(choice, "finish_reason", None)
        if not _is_length_finish_reason(finish_reason):
            break

        if segment_index + 1 >= max_segments:
            _trace_step(
                "llm.completion.truncated_after_max_segments",
                max_segments=max_segments,
                current_answer_chars=len("\n\n".join(answer_parts).strip()),
            )
            reached_segment_cap_with_length_finish = True
            break

        partial_answer = "\n\n".join(answer_parts).strip()
        current_prompt = _build_continuation_prompt(original_query, partial_answer)
        _trace_step(
            "llm.completion.continuation",
            segment=segment_index + 2,
            max_segments=max_segments,
            answer_chars=len(partial_answer),
        )

    answer = "\n\n".join(answer_parts).strip()

    if reached_segment_cap_with_length_finish and _looks_abruptly_truncated(answer):
        try:
            closure_completion = _create_chat_completion(
                retrieved_context=retrieved_context,
                user_content=_build_closing_completion_prompt(original_query, answer),
                max_tokens=_get_retry_output_tokens(),
                stream=False,
            )
            if closure_completion.choices:
                closure_text = str(getattr(closure_completion.choices[0].message, "content", "") or "").strip()
                if closure_text:
                    answer = f"{answer}\n\n{closure_text}".strip()
                    _trace_step("llm.completion.closure_appended", closure_chars=len(closure_text))
        except Exception as closure_error:
            logger.warning("Completion closure append failed (%s).", closure_error)
            _trace_step("llm.completion.closure_error", error=str(closure_error))

    if not answer:
        raise ValueError("Completion response returned an empty answer.")

    if _is_chat_trace_include_context():
        _trace_step("llm.completion.success", answer=answer, answer_chars=len(answer), segments=segment_count)
    else:
        _trace_step("llm.completion.success", answer_chars=len(answer), segments=segment_count)

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
        original_query = _extract_current_query_from_model_input(model_input)
        current_prompt = model_input
        max_segments = _get_max_completion_segments()
        has_streamed_content = False
        streamed_token_count = 0
        streamed_segments = 0
        streamed_answer_parts: list[str] = []
        reached_segment_cap_with_length_finish = False

        for segment_index in range(max_segments):
            max_tokens = _get_max_output_tokens() if segment_index == 0 else _get_retry_output_tokens()
            completion_stream = _create_chat_completion(
                retrieved_context=retrieved_context,
                user_content=current_prompt,
                max_tokens=max_tokens,
                stream=True,
            )

            segment_had_content = False
            segment_finish_reason: Any = None
            for chunk in completion_stream:
                if not chunk.choices:
                    continue

                choice = chunk.choices[0]
                if getattr(choice, "finish_reason", None):
                    segment_finish_reason = choice.finish_reason

                delta = choice.delta
                token = getattr(delta, "content", None) if delta else None
                if not token:
                    continue

                has_streamed_content = True
                segment_had_content = True
                streamed_token_count += 1
                streamed_answer_parts.append(token)
                yield token

            if segment_had_content:
                streamed_segments += 1

            if not _is_length_finish_reason(segment_finish_reason):
                break

            if segment_index + 1 >= max_segments:
                _trace_step(
                    "llm.stream.truncated_after_max_segments",
                    max_segments=max_segments,
                    token_count=streamed_token_count,
                )
                reached_segment_cap_with_length_finish = True
                break

            partial_answer = "".join(streamed_answer_parts)
            current_prompt = _build_continuation_prompt(original_query, partial_answer)
            _trace_step(
                "llm.stream.continuation",
                segment=segment_index + 2,
                max_segments=max_segments,
                token_count=streamed_token_count,
            )

        if reached_segment_cap_with_length_finish:
            partial_answer = "".join(streamed_answer_parts)
            if _looks_abruptly_truncated(partial_answer):
                try:
                    closure_completion = _create_chat_completion(
                        retrieved_context=retrieved_context,
                        user_content=_build_closing_completion_prompt(original_query, partial_answer),
                        max_tokens=_get_retry_output_tokens(),
                        stream=False,
                    )
                    if closure_completion.choices:
                        closure_text = str(getattr(closure_completion.choices[0].message, "content", "") or "").strip()
                        if closure_text:
                            streamed_answer_parts.append("\n\n")
                            streamed_answer_parts.append(closure_text)
                            streamed_token_count += 1
                            yield "\n\n" + closure_text
                            _trace_step("llm.stream.closure_appended", closure_chars=len(closure_text))
                except Exception as closure_error:
                    logger.warning("Streaming closure append failed (%s).", closure_error)
                    _trace_step("llm.stream.closure_error", error=str(closure_error))

        if not has_streamed_content:
            raise ValueError("Streaming response returned no content.")
        _trace_step("llm.stream.success", token_count=streamed_token_count, segments=streamed_segments)
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
            compact_summary = _generate_compact_context_summary(normalized_query, retrieved_context)
            if compact_summary:
                _trace_step("answer.compact_context_summary", answer_chars=len(compact_summary))
                return compact_summary
            fallback_answer = _build_context_grounded_fallback_answer(normalized_query, retrieved_context)
            if fallback_answer.strip() != _no_context_response().strip():
                _trace_step("answer.context_fallback", source="context_completion_failed")
                return fallback_answer
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
            "use_direct_faq_answers": _use_direct_faq_answers(),
            "sharepoint_always_insert": _is_sharepoint_always_insert_enabled(),
        },
    }


@app.get("/api/chat/logs/recent")
async def chat_recent_logs(limit: int = 10) -> dict:
    resolved_limit = max(1, min(limit, 200))
    logs = _load_recent_process_logs(resolved_limit)
    return {
        "limit": resolved_limit,
        "count": len(logs),
        "log_path": str(_get_chat_process_log_path()),
        "logs": logs,
    }


@app.get("/debug/search")
async def debug_search(q: str = "") -> dict:
    """Direct Azure Search test endpoint for debugging retrieval issues."""
    if not q or len(q.strip()) < 2:
        return {
            "error": "Query too short. Use ?q=your-test-query",
            "example": "/debug/search?q=fluentify",
        }

    logger.info("DEBUG: Direct search test for query='%s'", q)
    
    results_vector = {"success": False, "error": None, "results": []}
    results_text = {"success": False, "error": None, "results": []}
    
    try:
        logger.info("DEBUG: Attempting vector search...")
        context_v, score_v, citations_v = _search_vector_context(q, 3)
        results_vector = {
            "success": True,
            "context_length": len(context_v),
            "top_score": score_v,
            "citations_count": len(citations_v),
            "context_preview": context_v[:300] if context_v else None,
            "citations": citations_v,
        }
    except Exception as e:
        logger.warning("DEBUG: Vector search failed: %s", e)
        results_vector["error"] = str(e)
    
    try:
        logger.info("DEBUG: Attempting text search...")
        context_t, score_t, citations_t = _search_text_context(q, 3)
        results_text = {
            "success": True,
            "context_length": len(context_t),
            "top_score": score_t,
            "citations_count": len(citations_t),
            "context_preview": context_t[:300] if context_t else None,
            "citations": citations_t,
        }
    except Exception as e:
        logger.warning("DEBUG: Text search failed: %s", e)
        results_text["error"] = str(e)
    
    return {
        "query": q,
        "config": {
            "azure_search_endpoint": (os.getenv("AZURE_SEARCH_ENDPOINT") or "").strip(),
            "azure_search_index_name": (os.getenv("AZURE_SEARCH_INDEX_NAME") or "").strip(),
            "azure_search_vector_field": _get_azure_search_vector_field(),
            "azure_search_content_field": _get_azure_search_content_field(),
            "score_threshold": _get_embedding_similarity_threshold(),
            "overlap_threshold": _get_query_overlap_threshold(),
        },
        "vector_search": results_vector,
        "text_search": results_text,
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
        _trace_pipeline_stage("user_request", query=normalized_query, endpoint="/api/chat/text", stream=False)

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

        direct_answer = _direct_company_answer(normalized_query)
        if direct_answer:
            _trace_pipeline_stage("ai_request", mode="direct_answer", model_input_chars=0)
            _trace_pipeline_stage(
                "ai_search_response",
                source="direct_answer",
                context_chars=0,
                top_score=0.0,
                citations_count=0,
            )
            _save_conversation_turn(effective_session_id, normalized_query, direct_answer)
            await _sync_sharepoint_lead_safely(effective_session_id)

            trace_id = _get_active_trace_id()
            response_payload = {
                "reply": direct_answer,
                "session_id": effective_session_id,
                "lead": {
                    "email": current_lead_email,
                    "name": current_lead_name,
                },
                "citations": [],
                "videos": [],
            }
            if trace_id:
                response_payload["trace_id"] = trace_id

            _trace_pipeline_stage("ai_openai_response", source="direct_answer", answer_chars=len(direct_answer))
            _trace_pipeline_stage(
                "display_response",
                status_code=200,
                answer_chars=len(direct_answer),
                citations_count=0,
                stream=False,
            )
            _trace_step("response.ready", status_code=200, answer_chars=len(direct_answer), source="direct_answer")
            if _is_chat_trace_include_context():
                _finalize_trace("success", status_code=200, reply=direct_answer)
            else:
                _finalize_trace("success", status_code=200, answer_chars=len(direct_answer))
            return response_payload

        model_input = _build_model_input(effective_session_id, normalized_query)
        _trace_pipeline_stage(
            "ai_request",
            mode="rag",
            model=_get_chat_model(),
            model_input_chars=len(model_input),
        )
        retrieved_context, top_score, citations = _retrieve_context_and_score(normalized_query)
        response_videos = _extract_response_videos(normalized_query, retrieved_context, limit=1)
        _trace_pipeline_stage(
            "ai_search_response",
            source="azure_ai_search",
            context_chars=len(retrieved_context),
            top_score=top_score,
            citations_count=len(citations),
            videos_count=len(response_videos),
            citation=citations[0] if citations else None,
        )
        answer = _generate_answer(
            model_input,
            normalized_query,
            retrieved_context=retrieved_context,
            top_score=top_score,
        )
        _trace_pipeline_stage("ai_openai_response", source="azure_openai", answer_chars=len(answer))
        response_citations = (
            _select_response_citations(citations, normalized_query=normalized_query, answer_text=answer)
            if _should_attach_citations(answer, normalized_query, citations)
            else []
        )
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
            "videos": response_videos,
        }
        if trace_id:
            response_payload["trace_id"] = trace_id

        _trace_pipeline_stage(
            "display_response",
            status_code=200,
            answer_chars=len(answer),
            citations_count=len(response_citations),
            videos_count=len(response_videos),
            stream=False,
        )
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
        if _is_missing_required_env_error(error):
            _trace_step("response.error", status_code=503, error=str(error))
            _trace_step("response.failed", issue=runtime_issue)
            _finalize_trace("config_error", status_code=503, error=runtime_issue)
            raise HTTPException(status_code=503, detail=runtime_issue) from error

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
            "videos": [],
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
        _trace_pipeline_stage("user_request", query=normalized_query, endpoint="/api/chat/text/stream", stream=True)
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

        direct_answer = _direct_company_answer(normalized_query)
        if direct_answer:
            async def direct_event_generator() -> AsyncGenerator[str, None]:
                try:
                    _trace_pipeline_stage("ai_request", mode="direct_answer", model_input_chars=0, stream=True)
                    _trace_pipeline_stage(
                        "ai_search_response",
                        source="direct_answer",
                        context_chars=0,
                        top_score=0.0,
                        citations_count=0,
                        stream=True,
                    )
                    _save_conversation_turn(effective_session_id, normalized_query, direct_answer)
                    await _sync_sharepoint_lead_safely(effective_session_id)
                    _trace_pipeline_stage(
                        "ai_openai_response",
                        source="direct_answer",
                        answer_chars=len(direct_answer),
                        stream=True,
                    )
                    _trace_pipeline_stage(
                        "display_response",
                        status_code=200,
                        answer_chars=len(direct_answer),
                        citations_count=0,
                        stream=True,
                    )
                    yield _sse_event(
                        "done",
                        {
                            "reply": direct_answer,
                            "session_id": effective_session_id,
                            "trace_id": _get_active_trace_id(),
                            "lead": {
                                "email": current_lead_email,
                                "name": current_lead_name,
                            },
                            "citations": [],
                            "videos": [],
                            "suggestions": _build_dynamic_followup_questions(effective_session_id, 3),
                        },
                    )
                    _trace_step("response.ready", status_code=200, answer_chars=len(direct_answer), stream=True, source="direct_answer")
                    if _is_chat_trace_include_context():
                        _finalize_trace("success", status_code=200, reply=direct_answer, stream=True)
                    else:
                        _finalize_trace("success", status_code=200, answer_chars=len(direct_answer), stream=True)
                finally:
                    trace_state = _get_active_trace()
                    if trace_state and not trace_state.get("status"):
                        _finalize_trace("cancelled", status_code=499, stream=True)
                    _deactivate_trace(trace_token)

            return StreamingResponse(
                direct_event_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                    "X-Trace-Id": _get_active_trace_id(),
                },
            )

        model_input = _build_model_input(effective_session_id, normalized_query)
        _trace_pipeline_stage(
            "ai_request",
            mode="rag",
            model=_get_chat_model(),
            model_input_chars=len(model_input),
            stream=True,
        )
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
        response_videos: list[dict[str, str]] = []

        try:
            retrieved_context, top_score, citations = _retrieve_context_and_score(normalized_query)
            response_videos = _extract_response_videos(normalized_query, retrieved_context, limit=1)
            _trace_pipeline_stage(
                "ai_search_response",
                source="azure_ai_search",
                context_chars=len(retrieved_context),
                top_score=top_score,
                citations_count=len(citations),
                videos_count=len(response_videos),
                citation=citations[0] if citations else None,
                stream=True,
            )
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

            response_citations = (
                _select_response_citations(citations, normalized_query=normalized_query, answer_text=final_answer)
                if _should_attach_citations(final_answer, normalized_query, citations)
                else []
            )
            _trace_pipeline_stage(
                "ai_openai_response",
                source="azure_openai",
                answer_chars=len(final_answer),
                stream=True,
            )

            _save_conversation_turn(effective_session_id, normalized_query, final_answer)
            await _sync_sharepoint_lead_safely(effective_session_id)

            _trace_pipeline_stage(
                "display_response",
                status_code=200,
                answer_chars=len(final_answer),
                citations_count=len(response_citations),
                stream=True,
            )
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
                    "videos": response_videos,
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
            if _is_missing_required_env_error(error):
                yield _sse_event(
                    "error",
                    {
                        "message": runtime_issue,
                        "error_type": type(error).__name__,
                        "trace_id": trace_id,
                    },
                )
                _trace_step("response.error", status_code=503, error=str(error), stream=True)
                _trace_step("response.failed", issue=runtime_issue, stream=True)
                _finalize_trace("config_error", status_code=503, error=runtime_issue, stream=True)
                return

            fallback_answer = _no_context_response()
            _trace_pipeline_stage(
                "ai_openai_response",
                source="fallback",
                answer_chars=len(fallback_answer),
                stream=True,
            )
            _trace_pipeline_stage(
                "display_response",
                status_code=200,
                answer_chars=len(fallback_answer),
                citations_count=0,
                stream=True,
            )
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
                    "videos": [],
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


@app.post("/api/ingest/blob")
async def ingest_blob(
    container: str | None = Form(default=None),
    prefix: str | None = Form(default=None),
    max_files: int | None = Form(default=None),
    x_ingest_key: str | None = Header(default=None),
) -> dict:
    configured_ingest_key = os.getenv("INGEST_API_KEY")
    if configured_ingest_key and x_ingest_key != configured_ingest_key:
        raise HTTPException(status_code=401, detail="Invalid ingestion API key")

    try:
        result = ingest_blob_container(
            container_name=container,
            prefix=prefix,
            max_files=max_files,
        )
        return {
            "status": "success",
            "message": "Blob documents ingested successfully",
            **result,
        }
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        logger.exception("Blob ingestion failed")
        raise HTTPException(status_code=500, detail=f"Blob ingestion failed: {error}") from error


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
