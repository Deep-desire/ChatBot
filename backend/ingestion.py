import os
import uuid
from functools import lru_cache
from pathlib import Path
from datetime import datetime, timezone

from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_openai import AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


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


def _get_azure_search_id_field() -> str:
    return os.getenv("AZURE_SEARCH_ID_FIELD", "id").strip() or "id"


def _get_azure_search_content_field() -> str:
    return os.getenv("AZURE_SEARCH_CONTENT_FIELD", "content").strip() or "content"


def _get_azure_search_vector_field() -> str:
    return os.getenv("AZURE_SEARCH_VECTOR_FIELD", "contentVector").strip()


def _get_azure_search_source_field() -> str:
    return os.getenv("AZURE_SEARCH_SOURCE_FIELD", "").strip()


@lru_cache(maxsize=1)
def _get_search_client() -> SearchClient:
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


def _load_documents(file_path: str):
    extension = Path(file_path).suffix.lower()

    if extension == ".pdf":
        return PyPDFLoader(file_path).load()

    if extension in {".txt", ".md", ".csv", ".log"}:
        return TextLoader(file_path, encoding="utf-8").load()

    raise ValueError("Unsupported file type. Allowed: .pdf, .txt, .md, .csv, .log")


def ingest_file(file_path: str, source_name: str | None = None) -> dict:
    documents = _load_documents(file_path)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)

    resolved_source = source_name or Path(file_path).name
    for chunk in chunks:
        chunk.metadata = {**chunk.metadata, "source": resolved_source}

    embedding_deployment = _get_embedding_model()
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=_get_azure_openai_endpoint(),
        api_key=_get_azure_openai_api_key(),
        openai_api_version=_get_azure_openai_api_version(),
        azure_deployment=embedding_deployment,
        model=embedding_deployment,
    )
    index_name = _get_azure_search_index_name()
    id_field = _get_azure_search_id_field()
    content_field = _get_azure_search_content_field()
    vector_field = _get_azure_search_vector_field()
    source_field = _get_azure_search_source_field()

    chunk_texts = [str(chunk.page_content or "").strip() for chunk in chunks]
    chunk_texts = [chunk_text for chunk_text in chunk_texts if chunk_text]
    if not chunk_texts:
        raise ValueError("No text chunks were produced from the source document.")

    vectors = embeddings.embed_documents(chunk_texts) if vector_field else []
    timestamp = datetime.now(timezone.utc).isoformat()

    upload_documents: list[dict] = []
    for idx, chunk_text in enumerate(chunk_texts):
        document = {
            id_field: f"{resolved_source}::{idx}::{uuid.uuid4().hex}",
            content_field: chunk_text,
        }

        if source_field:
            document[source_field] = resolved_source
        if vector_field:
            document[vector_field] = vectors[idx]

        upload_documents.append(document)

    search_client = _get_search_client()
    batch_size = 100
    for start in range(0, len(upload_documents), batch_size):
        search_client.merge_or_upload_documents(upload_documents[start : start + batch_size])

    return {
        "source": resolved_source,
        "chunks": len(upload_documents),
        "index": index_name,
        "indexed_at": timestamp,
    }
