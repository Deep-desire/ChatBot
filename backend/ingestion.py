import argparse
import os
import uuid
import tempfile
import base64
from functools import lru_cache
from pathlib import Path
from datetime import datetime, timezone

from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_openai import AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


SUPPORTED_INGEST_EXTENSIONS = {".pdf", ".txt", ".md", ".csv", ".log"}


def _build_safe_document_key(source_name: str, chunk_index: int) -> str:
    normalized_source = (source_name or "source").strip().replace("\\", "/") or "source"
    source_token = base64.urlsafe_b64encode(normalized_source.encode("utf-8")).decode("ascii").rstrip("=")
    if not source_token:
        source_token = "source"
    return f"s_{source_token}_i_{chunk_index}_u_{uuid.uuid4().hex}"


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


def _get_azure_search_source_url_field() -> str:
    return os.getenv("AZURE_SEARCH_SOURCE_URL_FIELD", "").strip()


def _get_azure_blob_connection_string() -> str:
    return os.getenv("AZURE_BLOB_CONNECTION_STRING", "").strip()


def _get_azure_blob_account_url() -> str:
    return os.getenv("AZURE_BLOB_ACCOUNT_URL", "").strip()


def _get_azure_blob_account_key() -> str:
    return os.getenv("AZURE_BLOB_ACCOUNT_KEY", "").strip()


def _get_azure_blob_container_name() -> str:
    return os.getenv("AZURE_BLOB_CONTAINER", "").strip()


def _get_azure_blob_prefix() -> str:
    return os.getenv("AZURE_BLOB_PREFIX", "").strip()


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

    if extension in SUPPORTED_INGEST_EXTENSIONS:
        return TextLoader(file_path, encoding="utf-8").load()

    raise ValueError("Unsupported file type. Allowed: .pdf, .txt, .md, .csv, .log")


def ingest_file(file_path: str, source_name: str | None = None, source_url: str | None = None) -> dict:
    documents = _load_documents(file_path)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)

    resolved_source = source_name or Path(file_path).name
    resolved_source_url = (source_url or "").strip()
    for chunk in chunks:
        metadata = {**chunk.metadata, "source": resolved_source}
        if resolved_source_url:
            metadata["source_url"] = resolved_source_url
        chunk.metadata = metadata

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
    source_url_field = _get_azure_search_source_url_field()

    chunk_texts = [str(chunk.page_content or "").strip() for chunk in chunks]
    chunk_texts = [chunk_text for chunk_text in chunk_texts if chunk_text]
    if not chunk_texts:
        raise ValueError("No text chunks were produced from the source document.")

    vectors = embeddings.embed_documents(chunk_texts) if vector_field else []
    timestamp = datetime.now(timezone.utc).isoformat()

    upload_documents: list[dict] = []
    for idx, chunk_text in enumerate(chunk_texts):
        document = {
            id_field: _build_safe_document_key(resolved_source, idx),
            content_field: chunk_text,
        }

        if source_field:
            document[source_field] = resolved_source
        if source_url_field and resolved_source_url:
            document[source_url_field] = resolved_source_url
        if vector_field:
            document[vector_field] = vectors[idx]

        upload_documents.append(document)

    search_client = _get_search_client()
    batch_size = 100
    for start in range(0, len(upload_documents), batch_size):
        search_client.merge_or_upload_documents(upload_documents[start : start + batch_size])

    return {
        "source": resolved_source,
        "source_url": resolved_source_url,
        "chunks": len(upload_documents),
        "index": index_name,
        "indexed_at": timestamp,
    }


def _create_blob_service_client():
    try:
        from azure.storage.blob import BlobServiceClient
    except Exception as error:
        raise ValueError(
            "Azure Blob ingestion requires the 'azure-storage-blob' package. Install it with: pip install azure-storage-blob"
        ) from error

    connection_string = _get_azure_blob_connection_string()
    if connection_string:
        return BlobServiceClient.from_connection_string(connection_string)

    account_url = _get_azure_blob_account_url()
    if not account_url:
        raise ValueError(
            "Set AZURE_BLOB_CONNECTION_STRING or AZURE_BLOB_ACCOUNT_URL for blob ingestion."
        )

    account_key = _get_azure_blob_account_key()
    credential = account_key if account_key else DefaultAzureCredential()
    return BlobServiceClient(account_url=account_url, credential=credential)


def _build_blob_source_url(blob_client) -> str:
    try:
        return str(blob_client.url or "").strip()
    except Exception:
        return ""


def ingest_blob_container(
    container_name: str | None = None,
    prefix: str | None = None,
    max_files: int | None = None,
) -> dict:
    resolved_container = (container_name or _get_azure_blob_container_name() or "").strip()
    if not resolved_container:
        raise ValueError("Missing AZURE_BLOB_CONTAINER (or pass container_name).")

    resolved_prefix = (prefix if prefix is not None else _get_azure_blob_prefix()).strip()
    blob_service_client = _create_blob_service_client()
    container_client = blob_service_client.get_container_client(resolved_container)

    ingested_count = 0
    skipped_count = 0
    chunks_total = 0
    scanned_count = 0
    ingested_sources: list[str] = []
    skipped_sources: list[str] = []

    for blob in container_client.list_blobs(name_starts_with=resolved_prefix):
        blob_name = str(getattr(blob, "name", "") or "").strip()
        if not blob_name or blob_name.endswith("/"):
            continue

        scanned_count += 1
        extension = Path(blob_name).suffix.lower()
        if extension not in SUPPORTED_INGEST_EXTENSIONS:
            skipped_count += 1
            skipped_sources.append(blob_name)
            continue

        blob_client = container_client.get_blob_client(blob_name)
        blob_url = _build_blob_source_url(blob_client)
        temp_path = ""

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as temp_handle:
                temp_path = temp_handle.name
                temp_handle.write(blob_client.download_blob().readall())

            result = ingest_file(
                temp_path,
                source_name=blob_name,
                source_url=blob_url,
            )
            ingested_count += 1
            chunks_total += int(result.get("chunks") or 0)
            ingested_sources.append(blob_name)
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

        if max_files is not None and ingested_count >= max(0, max_files):
            break

    return {
        "container": resolved_container,
        "prefix": resolved_prefix,
        "scanned": scanned_count,
        "ingested_files": ingested_count,
        "skipped_files": skipped_count,
        "chunks": chunks_total,
        "index": _get_azure_search_index_name(),
        "indexed_at": datetime.now(timezone.utc).isoformat(),
        "sources": ingested_sources,
        "skipped_sources": skipped_sources,
    }


def _print_local_ingest_result(file_path: str) -> None:
    result = ingest_file(file_path)
    print(
        f"Ingestion complete! source={result['source']} chunks={result['chunks']} index={result['index']}"
    )


def _print_blob_ingest_result(container: str | None = None, prefix: str | None = None, max_files: int | None = None) -> None:
    result = ingest_blob_container(container_name=container, prefix=prefix, max_files=max_files)
    print(
        "Blob ingestion complete! "
        f"container={result['container']} ingested_files={result['ingested_files']} "
        f"skipped_files={result['skipped_files']} chunks={result['chunks']} index={result['index']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest local file or Azure Blob documents into Azure AI Search")
    parser.add_argument("--file", default="data.txt", help="Local file path for manual ingestion")
    parser.add_argument("--blob", action="store_true", help="Ingest from Azure Blob container")
    parser.add_argument("--container", default="", help="Blob container name (overrides AZURE_BLOB_CONTAINER)")
    parser.add_argument("--prefix", default="", help="Blob prefix/path filter")
    parser.add_argument("--max-files", type=int, default=None, help="Maximum number of blob files to ingest")
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        # Allow execution even when python-dotenv is not available.
        pass

    env_container = (os.getenv("AZURE_BLOB_CONTAINER") or "").strip()
    use_blob_mode = bool(args.blob or args.container or env_container)

    if use_blob_mode:
        _print_blob_ingest_result(
            container=args.container or None,
            prefix=args.prefix or None,
            max_files=args.max_files,
        )
    else:
        _print_local_ingest_result(args.file)


if __name__ == "__main__":
    main()
