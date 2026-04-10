import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASE_URL = "https://visit-to-lead.vercel.app/backend"
LEAD_EMAIL = "qa@example.com"
LEAD_NAME = "QA"


def _post_form(path: str, payload: dict[str, str]) -> dict:
    body = urllib.parse.urlencode(payload).encode("utf-8")
    request = urllib.request.Request(
        url=f"{BASE_URL}{path}",
        data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        raw = response.read().decode("utf-8", errors="replace")
    return json.loads(raw)


def _get_json(path: str, params: dict[str, str]) -> dict:
    query = urllib.parse.urlencode(params)
    url = f"{BASE_URL}{path}?{query}"
    request = urllib.request.Request(url=url, method="GET")
    with urllib.request.urlopen(request, timeout=60) as response:
        raw = response.read().decode("utf-8", errors="replace")
    return json.loads(raw)


def _post_stream_raw(path: str, payload: dict[str, str]) -> str:
    body = urllib.parse.urlencode(payload).encode("utf-8")
    request = urllib.request.Request(
        url=f"{BASE_URL}{path}",
        data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read().decode("utf-8", errors="replace")


def _new_session(prefix: str) -> str:
    return f"{prefix}_{int(time.time() * 1000)}"


def _safe_preview(text: str, chars: int = 260) -> str:
    return text[:chars]


def run_cycle() -> dict:
    health_config = _get_json("/health/config", {})

    report: dict[str, object] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": "production",
        "base_url": BASE_URL,
        "health_config": {
            "status": health_config.get("status"),
            "missing_count": health_config.get("missing_count"),
            "effective": health_config.get("effective", {}),
        },
    }

    core_session = _new_session("qa_core")
    q1 = "What is Desire Infoweb?"
    q2 = "What AI solutions has Desire Infoweb delivered?"

    r1 = _post_form(
        "/api/chat/text",
        {
            "query": q1,
            "session_id": core_session,
            "lead_email": LEAD_EMAIL,
            "lead_name": LEAD_NAME,
        },
    )
    s1 = _get_json("/api/chat/suggestions", {"session_id": core_session, "limit": "3"})

    r2 = _post_form(
        "/api/chat/text",
        {
            "query": q2,
            "session_id": core_session,
            "lead_email": LEAD_EMAIL,
            "lead_name": LEAD_NAME,
        },
    )
    s2 = _get_json("/api/chat/suggestions", {"session_id": core_session, "limit": "3"})

    q1_reply = str(r1.get("reply", ""))
    q2_reply = str(r2.get("reply", ""))
    s2_joined = " | ".join(str(item) for item in s2.get("suggestions", []))
    q1_citations = r1.get("citations") if isinstance(r1.get("citations"), list) else []
    q2_citations = r2.get("citations") if isinstance(r2.get("citations"), list) else []
    q2_is_fallback = q2_reply.lower().startswith("thank you for your query")

    report["post_deploy_core_flow"] = {
        "session": core_session,
        "q1": {
            "query": q1,
            "chars": len(q1_reply),
            "preview": _safe_preview(q1_reply),
            "citations_count": len(q1_citations),
            "suggestions": s1.get("suggestions", []),
        },
        "q2": {
            "query": q2,
            "chars": len(q2_reply),
            "preview": _safe_preview(q2_reply),
            "citations_count": len(q2_citations),
            "suggestions": s2.get("suggestions", []),
        },
        "checks": {
            "q2_not_equal_q1": q2_reply != q1_reply,
            "q2_not_fallback": not q2_is_fallback,
            "q2_detailed": len(q2_reply) >= 350,
            "q1_has_citations": len(q1_citations) > 0,
            "q2_has_citations": len(q2_citations) > 0,
            "no_malformed_suggestion": "summary of solutions has delivered use cases" not in s2_joined.lower(),
        },
    }

    matrix_queries = [
        "what is fluentify ai",
        "give me all the voice based projects",
        "give me all the voice based projects in detail",
        "what services does desire infoweb provide",
        "which industries does desire infoweb mainly serve",
        "give me detail about ai interviewer pro",
        "what is compliance voice monitoring project",
        "can you give architecture details for fluentify ai",
        "tell me about teams ai assistant project",
        "what budget needed for ai chatbot",
    ]

    matrix_results: list[dict[str, object]] = []
    for query in matrix_queries:
        session_id = _new_session("qa_mx")
        response = _post_form(
            "/api/chat/text",
            {
                "query": query,
                "session_id": session_id,
                "lead_email": LEAD_EMAIL,
                "lead_name": LEAD_NAME,
            },
        )
        reply = str(response.get("reply", ""))
        citations = response.get("citations") if isinstance(response.get("citations"), list) else []
        matrix_results.append(
            {
                "query": query,
                "session": session_id,
                "chars": len(reply),
                "fallback": reply.lower().startswith("thank you for your query"),
                "citations_count": len(citations),
                "preview": _safe_preview(reply),
            }
        )

    report["prompt_matrix"] = matrix_results

    stream_session_1 = _new_session("qa_stream")
    stream_text_1 = _post_stream_raw(
        "/api/chat/text/stream",
        {
            "query": "What AI solutions has Desire Infoweb delivered?",
            "session_id": stream_session_1,
            "lead_email": LEAD_EMAIL,
            "lead_name": LEAD_NAME,
        },
    )

    stream_session_2 = _new_session("qa_stream")
    stream_text_2 = _post_stream_raw(
        "/api/chat/text/stream",
        {
            "query": "what is fluentify ai",
            "session_id": stream_session_2,
            "lead_email": LEAD_EMAIL,
            "lead_name": LEAD_NAME,
        },
    )

    report["stream_checks"] = {
        "ai_solutions_prompt": {
            "session": stream_session_1,
            "has_done": "event: done" in stream_text_1,
            "has_error": "event: error" in stream_text_1,
            "token_events": stream_text_1.count("event: token"),
        },
        "fluentify_prompt": {
            "session": stream_session_2,
            "has_done": "event: done" in stream_text_2,
            "has_error": "event: error" in stream_text_2,
            "token_events": stream_text_2.count("event: token"),
        },
    }

    char_counts = [int(item["chars"]) for item in matrix_results]
    fallback_count = sum(1 for item in matrix_results if item["fallback"])

    report["summary"] = {
        "matrix_total": len(matrix_results),
        "matrix_fallback_count": fallback_count,
        "matrix_with_citations": sum(1 for item in matrix_results if int(item.get("citations_count", 0)) > 0),
        "matrix_min_chars": min(char_counts) if char_counts else 0,
        "matrix_max_chars": max(char_counts) if char_counts else 0,
    }

    return report


def main() -> int:
    try:
        report = run_cycle()
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        print(f"HTTPError {error.code}: {detail}")
        return 1
    except Exception as error:
        print(f"QA cycle failed: {type(error).__name__}: {error}")
        return 1

    logs_dir = Path(__file__).resolve().parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    filename = f"qa_full_cycle_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_path = logs_dir / filename
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(str(output_path))
    print(json.dumps(report["post_deploy_core_flow"]["checks"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
