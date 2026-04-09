import json
import os
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from io import BytesIO
from typing import Any

import azure.functions as func
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from langchain_openai import AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

# Optional CORS override for deployed function apps.
ALLOWED_ORIGIN = os.getenv("ALLOWED_ORIGIN", "*")
MANIFEST_PREFIX = "manifest::"


def _get_required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def _get_embedding_model() -> str:
    return _get_required_env("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")


def _get_azure_openai_endpoint() -> str:
    return _get_required_env("AZURE_OPENAI_ENDPOINT")


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


def _get_search_id_field() -> str:
    return os.getenv("AZURE_SEARCH_ID_FIELD", "id").strip() or "id"


def _get_search_content_field() -> str:
    return os.getenv("AZURE_SEARCH_CONTENT_FIELD", "content").strip() or "content"


def _get_search_vector_field() -> str:
    return os.getenv("AZURE_SEARCH_VECTOR_FIELD", "contentVector").strip()


def _get_search_file_id_field() -> str:
    return os.getenv("AZURE_SEARCH_FILE_ID_FIELD", "file_id").strip() or "file_id"


def _get_search_file_name_field() -> str:
    return os.getenv("AZURE_SEARCH_FILE_NAME_FIELD", "file_name").strip() or "file_name"


def _get_search_chunk_index_field() -> str:
    return os.getenv("AZURE_SEARCH_CHUNK_INDEX_FIELD", "chunk_index").strip() or "chunk_index"


def _get_search_chunk_count_field() -> str:
    return os.getenv("AZURE_SEARCH_CHUNK_COUNT_FIELD", "chunk_count").strip() or "chunk_count"


def _get_search_uploaded_at_field() -> str:
    return os.getenv("AZURE_SEARCH_UPLOADED_AT_FIELD", "uploaded_at").strip() or "uploaded_at"


def _get_search_record_type_field() -> str:
    return os.getenv("AZURE_SEARCH_RECORD_TYPE_FIELD", "record_type").strip() or "record_type"


def _json_response(payload: dict[str, Any], status_code: int = 200) -> func.HttpResponse:
    return func.HttpResponse(
        body=json.dumps(payload),
        status_code=status_code,
        mimetype="application/json",
        headers={
            "Access-Control-Allow-Origin": ALLOWED_ORIGIN,
            "Access-Control-Allow-Methods": "GET,POST,PUT,DELETE,OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type,x-file-name",
        },
    )


def _options_response() -> func.HttpResponse:
    return func.HttpResponse(
        status_code=204,
        headers={
            "Access-Control-Allow-Origin": ALLOWED_ORIGIN,
            "Access-Control-Allow-Methods": "GET,POST,PUT,DELETE,OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type,x-file-name",
        },
    )


@lru_cache(maxsize=1)
def _create_embeddings_client() -> AzureOpenAIEmbeddings:
    deployment = _get_embedding_model()
    return AzureOpenAIEmbeddings(
        azure_endpoint=_get_azure_openai_endpoint(),
        api_key=_get_azure_openai_api_key(),
        openai_api_version=_get_azure_openai_api_version(),
        azure_deployment=deployment,
        model=deployment,
    )


@lru_cache(maxsize=1)
def _create_search_client() -> SearchClient:
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


def _extract_pdf_text(file_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(file_bytes))
    pages_text: list[str] = []
    for page in reader.pages:
        pages_text.append((page.extract_text() or "").strip())
    return "\n\n".join(text for text in pages_text if text)


def _chunk_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=120)
    chunks = [chunk.strip() for chunk in splitter.split_text(text)]
    return [chunk for chunk in chunks if chunk]


def _coerce_filename(request: func.HttpRequest, explicit_name: str | None = None) -> str:
    file_name = (explicit_name or "").strip()
    if not file_name:
        file_name = (request.params.get("file_name") or "").strip()
    if not file_name:
        file_name = (request.headers.get("x-file-name") or "").strip()
    if not file_name:
        raise ValueError("Missing file name. Provide query param file_name or x-file-name header.")
    if not file_name.lower().endswith(".pdf"):
        raise ValueError("Only PDF files are supported.")
    return file_name


def _read_pdf_bytes(request: func.HttpRequest) -> bytes:
    body = request.get_body() or b""
    if not body:
        raise ValueError("Request body is empty. Send the PDF bytes in the body.")
    return body


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _build_manifest_key(file_id: str) -> str:
    return f"{MANIFEST_PREFIX}{file_id}"


def _upsert_file(
    search_client: SearchClient,
    embeddings: AzureOpenAIEmbeddings,
    file_id: str,
    file_name: str,
    pdf_bytes: bytes,
) -> dict[str, Any]:
    text = _extract_pdf_text(pdf_bytes)
    if not text.strip():
        raise ValueError("Could not extract readable text from PDF.")

    chunks = _chunk_text(text)
    if not chunks:
        raise ValueError("No valid text chunks produced from PDF.")

    key_field = _get_search_id_field()
    content_field = _get_search_content_field()
    vector_field = _get_search_vector_field()
    file_id_field = _get_search_file_id_field()
    file_name_field = _get_search_file_name_field()
    chunk_index_field = _get_search_chunk_index_field()
    chunk_count_field = _get_search_chunk_count_field()
    uploaded_at_field = _get_search_uploaded_at_field()
    record_type_field = _get_search_record_type_field()

    vectors = embeddings.embed_documents(chunks) if vector_field else []
    timestamp = datetime.now(timezone.utc).isoformat()

    chunk_documents: list[dict[str, Any]] = []
    for idx, chunk in enumerate(chunks):
        document: dict[str, Any] = {
            key_field: f"{file_id}::chunk::{idx}",
            content_field: chunk,
            file_id_field: file_id,
            file_name_field: file_name,
            chunk_index_field: idx,
            uploaded_at_field: timestamp,
            record_type_field: "chunk",
        }
        if vector_field:
            document[vector_field] = vectors[idx]
        chunk_documents.append(document)

    manifest_document: dict[str, Any] = {
        key_field: _build_manifest_key(file_id),
        content_field: f"Manifest for {file_name}",
        file_id_field: file_id,
        file_name_field: file_name,
        chunk_count_field: len(chunk_documents),
        uploaded_at_field: timestamp,
        record_type_field: "manifest",
    }
    if vector_field:
        manifest_document[vector_field] = embeddings.embed_query(
            f"manifest {file_id} {file_name} {timestamp}"
        )

    search_client.merge_or_upload_documents(chunk_documents + [manifest_document])

    return {
        "file_id": file_id,
        "file_name": file_name,
        "chunk_count": len(chunk_documents),
        "uploaded_at": timestamp,
    }


def _list_manifest_records(search_client: SearchClient) -> list[dict[str, Any]]:
    key_field = _get_search_id_field()
    file_id_field = _get_search_file_id_field()
    file_name_field = _get_search_file_name_field()
    chunk_count_field = _get_search_chunk_count_field()
    uploaded_at_field = _get_search_uploaded_at_field()
    record_type_field = _get_search_record_type_field()

    select_fields = [
        key_field,
        file_id_field,
        file_name_field,
        chunk_count_field,
        uploaded_at_field,
        record_type_field,
    ]

    items: list[dict[str, Any]] = []

    try:
        results = search_client.search(
            search_text="*",
            filter=f"{record_type_field} eq 'manifest'",
            top=1000,
            select=select_fields,
        )
    except Exception:
        results = search_client.search(search_text="*", top=1000, select=select_fields)

    for result in results:
        payload = dict(result)
        record_id = str(payload.get(key_field) or "")
        if not record_id.startswith(MANIFEST_PREFIX):
            continue

        file_id = str(payload.get(file_id_field) or "") or record_id[len(MANIFEST_PREFIX) :]
        items.append(
            {
                "file_id": file_id,
                "file_name": str(payload.get(file_name_field) or ""),
                "chunk_count": _safe_int(payload.get(chunk_count_field)),
                "uploaded_at": str(payload.get(uploaded_at_field) or ""),
            }
        )

    items.sort(key=lambda item: item.get("uploaded_at") or "", reverse=True)
    return items


def _get_manifest_record(search_client: SearchClient, file_id: str) -> dict[str, Any] | None:
    key_field = _get_search_id_field()

    try:
        record = dict(search_client.get_document(key=_build_manifest_key(file_id)))
    except Exception:
        return None

    if not record:
        return None

    if str(record.get(key_field) or "") != _build_manifest_key(file_id):
        return None

    return record


def _delete_file_chunks(search_client: SearchClient, file_id: str, chunk_count: int) -> None:
    key_field = _get_search_id_field()

    delete_documents = [{key_field: f"{file_id}::chunk::{idx}"} for idx in range(max(chunk_count, 0))]
    delete_documents.append({key_field: _build_manifest_key(file_id)})

    search_client.delete_documents(delete_documents)


@app.route(route="files", methods=["GET", "POST", "OPTIONS"])
def files(req: func.HttpRequest) -> func.HttpResponse:
    if req.method == "OPTIONS":
        return _options_response()

    try:
        search_client = _create_search_client()
        embeddings = _create_embeddings_client()

        if req.method == "GET":
            return _json_response({"files": _list_manifest_records(search_client)})

        file_name = _coerce_filename(req)
        pdf_bytes = _read_pdf_bytes(req)
        file_id = uuid.uuid4().hex
        result = _upsert_file(search_client, embeddings, file_id, file_name, pdf_bytes)

        return _json_response({"message": "File uploaded and indexed.", "file": result}, status_code=201)

    except ValueError as error:
        return _json_response({"error": str(error)}, status_code=400)
    except Exception as error:
        return _json_response({"error": f"Server error: {type(error).__name__}: {error}"}, status_code=500)


@app.route(route="files/{file_id}", methods=["PUT", "DELETE", "OPTIONS"])
def file_item(req: func.HttpRequest) -> func.HttpResponse:
    if req.method == "OPTIONS":
        return _options_response()

    file_id = (req.route_params.get("file_id") or "").strip()
    if not file_id:
        return _json_response({"error": "Missing file_id in route."}, status_code=400)

    try:
        search_client = _create_search_client()
        existing = _get_manifest_record(search_client, file_id)
        if not existing:
            return _json_response({"error": "File not found."}, status_code=404)

        file_name_field = _get_search_file_name_field()
        chunk_count_field = _get_search_chunk_count_field()
        existing_file_name = str(existing.get(file_name_field) or "").strip()
        existing_chunk_count = _safe_int(existing.get(chunk_count_field))

        if req.method == "DELETE":
            _delete_file_chunks(search_client, file_id, existing_chunk_count)
            return _json_response({"message": "File deleted successfully.", "file_id": file_id})

        embeddings = _create_embeddings_client()
        file_name = _coerce_filename(req, explicit_name=existing_file_name)

        # Allow changing file name on update if caller sends one.
        override_name = (req.params.get("file_name") or req.headers.get("x-file-name") or "").strip()
        if override_name:
            if not override_name.lower().endswith(".pdf"):
                raise ValueError("Only PDF files are supported.")
            file_name = override_name

        pdf_bytes = _read_pdf_bytes(req)

        _delete_file_chunks(search_client, file_id, existing_chunk_count)
        result = _upsert_file(search_client, embeddings, file_id, file_name, pdf_bytes)

        return _json_response({"message": "File updated and re-indexed.", "file": result})

    except ValueError as error:
        return _json_response({"error": str(error)}, status_code=400)
    except Exception as error:
        return _json_response({"error": f"Server error: {type(error).__name__}: {error}"}, status_code=500)
