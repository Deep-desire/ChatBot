import json
from dataclasses import dataclass
from typing import Any, Callable

from fastapi.testclient import TestClient

import main


@dataclass
class QAResult:
    name: str
    passed: bool
    detail: str = ""


class PatchAttr:
    def __init__(self, target: Any, attr_name: str, replacement: Any):
        self.target = target
        self.attr_name = attr_name
        self.replacement = replacement
        self.original = None

    def __enter__(self):
        self.original = getattr(self.target, self.attr_name)
        setattr(self.target, self.attr_name, self.replacement)
        return self

    def __exit__(self, exc_type, exc, tb):
        setattr(self.target, self.attr_name, self.original)


def _parse_sse_done_payload(raw_body: str) -> dict[str, Any]:
    body = raw_body.replace("\r", "")
    chunks = [chunk for chunk in body.split("\n\n") if chunk.strip()]
    done_payload: dict[str, Any] = {}

    for chunk in chunks:
        event_name = "message"
        data_lines: list[str] = []
        for line in chunk.split("\n"):
            if line.startswith("event:"):
                event_name = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].strip())

        if event_name == "done" and data_lines:
            done_payload = json.loads("\n".join(data_lines))

    return done_payload


def _run_check(name: str, fn: Callable[[], tuple[bool, str]]) -> QAResult:
    try:
        passed, detail = fn()
        return QAResult(name=name, passed=passed, detail=detail)
    except Exception as error:
        return QAResult(name=name, passed=False, detail=f"Exception: {type(error).__name__}: {error}")


def run_suite() -> list[QAResult]:
    client = TestClient(main.app)
    results: list[QAResult] = []

    results.append(
        _run_check(
            "GET /health",
            lambda: (
                (resp := client.get("/health")).status_code == 200 and resp.json().get("status") == "ok",
                f"status={resp.status_code}, body={resp.json()}",
            ),
        )
    )

    results.append(
        _run_check(
            "GET /health/config",
            lambda: (
                (resp := client.get("/health/config")).status_code == 200 and isinstance(resp.json(), dict),
                f"status={resp.status_code}, keys={list(resp.json().keys())}",
            ),
        )
    )

    results.append(
        _run_check(
            "POST /api/chat/text (RAG-first response)",
            lambda: (
                (resp := client.post(
                    "/api/chat/text",
                    data={"query": "what is desire infoweb", "session_id": "qa-direct"},
                )).status_code == 200
                and bool(str(resp.json().get("reply", "")).strip()),
                f"status={resp.status_code}, citations={len(resp.json().get('citations', []))}",
            ),
        )
    )

    def _check_text_and_stream_with_citations() -> tuple[bool, str]:
        fake_citations = [{"title": "Fluentify Source", "url": "https://example.com/fluentify.pdf", "id": "1", "score": 0.91}]

        def fake_retrieve_context_and_score(_: str):
            return "fluentify context", 0.91, fake_citations

        def fake_generate_answer(*_args, **_kwargs):
            return "Fluentify detailed answer"

        def fake_stream_answer_tokens(*_args, **_kwargs):
            yield "Fluentify "
            yield "detailed "
            yield "answer"

        async def fake_sync_sharepoint_lead_safely(_session_id: str) -> None:
            return None

        with PatchAttr(main, "_retrieve_context_and_score", fake_retrieve_context_and_score), PatchAttr(
            main, "_generate_answer", fake_generate_answer
        ), PatchAttr(main, "_stream_answer_tokens", fake_stream_answer_tokens), PatchAttr(
            main, "_sync_sharepoint_lead_safely", fake_sync_sharepoint_lead_safely
        ):
            text_resp = client.post(
                "/api/chat/text",
                data={"query": "give me detail about the fluentify ai project", "session_id": "qa-cite"},
            )
            stream_resp = client.post(
                "/api/chat/text/stream",
                data={"query": "give me detail about the fluentify ai project", "session_id": "qa-cite"},
            )

        text_ok = text_resp.status_code == 200 and len(text_resp.json().get("citations", [])) == 1
        done_payload = _parse_sse_done_payload(stream_resp.text)
        stream_ok = stream_resp.status_code == 200 and len(done_payload.get("citations", [])) == 1
        return text_ok and stream_ok, (
            f"text_status={text_resp.status_code}, text_citations={len(text_resp.json().get('citations', []))}, "
            f"stream_status={stream_resp.status_code}, stream_citations={len(done_payload.get('citations', []))}"
        )

    results.append(_run_check("POST /api/chat/text + /api/chat/text/stream (citation flow)", _check_text_and_stream_with_citations))

    def _check_strong_match_overlap_gate() -> tuple[bool, str]:
        query = "give me detail about the fluentify ai"
        context = (
            "Fluentify AI offers a comprehensive solution for enhancing language proficiency "
            "using AI-powered assessments and communication feedback."
        )

        candidate = main._build_retrieval_candidate(
            query=query,
            source_scope="knowledge",
            retrieval_mode="text",
            context=context,
            score=7.4,
            citations=[{"id": "fluentify-source"}],
        )
        should_use_context = main._should_use_embedding_context(query, context, 7.4)

        ok = bool(candidate.get("meets_overlap")) and should_use_context
        return ok, (
            f"meets_overlap={candidate.get('meets_overlap')}, "
            f"overlap_count={candidate.get('overlap_count')}, "
            f"query_terms={candidate.get('query_term_count')}, "
            f"should_use_context={should_use_context}"
        )

    results.append(_run_check("Strong semantic match keeps Fluentify context", _check_strong_match_overlap_gate))

    def _check_product_citation_alignment() -> tuple[bool, str]:
        citations = [
            {
                "title": "project-management-portal",
                "url": "https://desireinfoweb.com/products/project-management-portal",
                "id": "pm-1",
                "score": 0.82,
            },
            {
                "title": "learning-management-system",
                "url": "https://desireinfoweb.com/products/learning-management-system",
                "id": "lms-1",
                "score": 0.78,
            },
        ]

        lms_selected = main._select_response_citations(
            citations,
            limit=1,
            normalized_query="learning management portal",
        )
        pm_selected = main._select_response_citations(
            citations,
            limit=1,
            normalized_query="project management portal",
        )

        lms_ok = bool(lms_selected) and "learning-management-system" in str(lms_selected[0].get("url") or "")
        pm_ok = bool(pm_selected) and "project-management-portal" in str(pm_selected[0].get("url") or "")

        return lms_ok and pm_ok, (
            f"lms_url={lms_selected[0].get('url') if lms_selected else ''}, "
            f"pm_url={pm_selected[0].get('url') if pm_selected else ''}"
        )

    results.append(_run_check("Product query returns aligned citation", _check_product_citation_alignment))

    def _check_product_video_alignment() -> tuple[bool, str]:
        context = (
            "Project Management Portal walkthrough https://youtu.be/pmportal123 "
            "Learning Management System (LMS) demo https://youtu.be/lmsdemo456"
        )

        pm_videos = main._extract_video_sources_from_context(context, "project management portal", limit=1)
        lms_videos = main._extract_video_sources_from_context(context, "learning management portal", limit=1)

        pm_ok = bool(pm_videos) and "pmportal123" in str(pm_videos[0].get("url") or "")
        lms_ok = bool(lms_videos) and "lmsdemo456" in str(lms_videos[0].get("url") or "")

        return pm_ok and lms_ok, (
            f"pm_video={pm_videos[0].get('url') if pm_videos else ''}, "
            f"lms_video={lms_videos[0].get('url') if lms_videos else ''}"
        )

    results.append(_run_check("Product query returns aligned video", _check_product_video_alignment))

    def _check_project_video_fallback_for_typo_query() -> tuple[bool, str]:
        def fake_search_text_context(
            query: str,
            _top_k: int,
            payload_filter: Callable[[dict[str, Any]], bool] | None = None,
            scan_top: int | None = None,
        ) -> tuple[str, float, list[dict[str, Any]]]:
            _ = payload_filter, scan_top
            if "youtube video demo" in query.lower():
                context = (
                    "Project Management Portal YouTube URL: https://youtu.be/uue2gmznRD0 "
                    "Learning Management System demo: https://youtu.be/bfTZ2YKs2rY"
                )
                return context, 10.0, []
            return "project details without video url", 2.0, []

        def fake_search_vector_context(
            _query: str,
            _top_k: int,
            payload_filter: Callable[[dict[str, Any]], bool] | None = None,
            scan_top: int | None = None,
        ) -> tuple[str, float, list[dict[str, Any]]]:
            _ = payload_filter, scan_top
            return "", 0.0, []

        with PatchAttr(main, "_search_text_context", fake_search_text_context), PatchAttr(
            main, "_search_vector_context", fake_search_vector_context
        ):
            videos = main._extract_response_videos(
                "project management poratl?",
                "project details without video url",
                limit=1,
            )

        ok = bool(videos) and "uue2gmznRD0" in str(videos[0].get("url") or "")
        return ok, f"videos={videos}"

    results.append(_run_check("Typo query gets project video fallback", _check_project_video_fallback_for_typo_query))

    def _check_semantic_intent_normalization_with_typos() -> tuple[bool, str]:
        learning_typo = main._normalize_user_query("give me deatil about the leaning managment poratl")
        learning_clean = main._normalize_user_query("give me detail about the learning management portal")
        project_typo = main._normalize_user_query("show me deatil of eproject managment poratl")
        project_clean = main._normalize_user_query("show me detail of project management portal")

        learning_ok = (
            "learning management system" in learning_typo.lower()
            and "learning management system" in learning_clean.lower()
        )
        project_ok = (
            "project management portal" in project_typo.lower()
            and "project management portal" in project_clean.lower()
        )

        return learning_ok and project_ok, (
            f"learning_typo={learning_typo}; learning_clean={learning_clean}; "
            f"project_typo={project_typo}; project_clean={project_clean}"
        )

    results.append(_run_check("Meaning-first normalization is typo robust", _check_semantic_intent_normalization_with_typos))

    results.append(
        _run_check(
            "GET /api/chat/last",
            lambda: (
                (resp := client.get("/api/chat/last", params={"session_id": "qa-cite"})).status_code == 200
                and bool(resp.json().get("reply")),
                f"status={resp.status_code}",
            ),
        )
    )

    results.append(
        _run_check(
            "GET /api/chat/suggestions",
            lambda: (
                (resp := client.get("/api/chat/suggestions", params={"session_id": "qa-cite", "limit": 3})).status_code == 200
                and isinstance(resp.json().get("suggestions"), list)
                and all(len(str(question)) <= 84 for question in resp.json().get("suggestions", [])),
                f"status={resp.status_code}, suggestions={len(resp.json().get('suggestions', []))}",
            ),
        )
    )

    def _check_ingest_upload() -> tuple[bool, str]:
        def fake_ingest_file(_path: str, source_name: str | None = None):
            return {
                "source": source_name or "uploaded.txt",
                "chunks": 3,
                "index": "test-index",
                "indexed_at": "2026-04-10T00:00:00+00:00",
            }

        with PatchAttr(main, "ingest_file", fake_ingest_file):
            success_resp = client.post(
                "/api/ingest/upload",
                files={"file": ("uploaded.txt", b"hello world", "text/plain")},
            )

        unsupported_resp = client.post(
            "/api/ingest/upload",
            files={"file": ("bad.exe", b"hello", "application/octet-stream")},
        )

        ok = (
            success_resp.status_code == 200
            and unsupported_resp.status_code == 400
            and success_resp.json().get("status") == "success"
        )
        return ok, (
            f"success_status={success_resp.status_code}, "
            f"unsupported_status={unsupported_resp.status_code}"
        )

    results.append(_run_check("POST /api/ingest/upload", _check_ingest_upload))

    def _check_voice_endpoint() -> tuple[bool, str]:
        class FakeTranscription:
            text = "voice query"

        class FakeGroqAudioTranscriptions:
            @staticmethod
            def create(**_kwargs):
                return FakeTranscription()

        class FakeGroqAudio:
            transcriptions = FakeGroqAudioTranscriptions()

        class FakeGroqClient:
            audio = FakeGroqAudio()

        class FakeCommunicate:
            def __init__(self, _text: str, _voice: str):
                pass

            async def stream(self):
                yield {"type": "audio", "data": b"FAKEAUDIO"}

        def fake_get_groq_client():
            return FakeGroqClient()

        def fake_generate_answer(*_args, **_kwargs):
            return "voice answer"

        async def fake_sync_sharepoint_lead_safely(_session_id: str) -> None:
            return None

        with PatchAttr(main, "get_groq_client", fake_get_groq_client), PatchAttr(
            main, "_generate_answer", fake_generate_answer
        ), PatchAttr(main.edge_tts, "Communicate", FakeCommunicate), PatchAttr(
            main, "_sync_sharepoint_lead_safely", fake_sync_sharepoint_lead_safely
        ):
            resp = client.post(
                "/api/chat/voice",
                files={"audio": ("voice.webm", b"abc", "audio/webm")},
                headers={
                    "X-Session-Id": "qa-voice",
                    "X-Lead-Email": "qa@example.com",
                    "X-Lead-Name": "QA User",
                },
            )

        ok = (
            resp.status_code == 200
            and resp.headers.get("content-type", "").startswith("audio/mpeg")
            and bool(resp.headers.get("X-User-Query"))
            and bool(resp.headers.get("X-Bot-Reply"))
        )
        return ok, f"status={resp.status_code}, content_type={resp.headers.get('content-type', '')}"

    results.append(_run_check("POST /api/chat/voice", _check_voice_endpoint))

    return results


if __name__ == "__main__":
    qa_results = run_suite()
    passed_count = sum(1 for result in qa_results if result.passed)
    total_count = len(qa_results)

    for result in qa_results:
        status_text = "PASS" if result.passed else "FAIL"
        print(f"[{status_text}] {result.name} :: {result.detail}")

    print(f"\nSummary: {passed_count}/{total_count} checks passed")

    if passed_count != total_count:
        raise SystemExit(1)
