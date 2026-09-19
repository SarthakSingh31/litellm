import asyncio
import json
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Final
from uuid import uuid4

import httpx
import pytest
import respx
from fastapi import HTTPException
from pydantic import TypeAdapter

import litellm
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.internal_call_metadata import responses_compaction_history_metadata
from litellm.litellm_core_utils.litellm_logging import StandardLoggingPayloadSetup
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.llms.anthropic.experimental_pass_through.context_management.constants import (
    COMPACT_SAME_AS_REQUEST,
    COMPACT_SUMMARY_MODEL_SETTING_KEY,
)
from litellm.proxy import proxy_server
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.hooks.proxy_track_cost_callback import _get_budget_reservation_from_metadata
from litellm.proxy.spend_tracking.budget_reservation import reconcile_budget_reservation
from litellm.responses.compaction import (
    COMPACTION_CHILD_KEY,
    CompactionSession,
    _count,
    finish_compaction,
    prepare_compaction,
)
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.llms.openai import ResponsesAPIResponse

MODEL: Final = "openai/test-compaction-model"
API_BASE: Final = "https://compaction.example/v1"
MODEL_INFO: Final = {"max_input_tokens": 20000, "max_output_tokens": 4096, "supports_native_compaction": False}
HISTORY: Final = [
    {"role": "user", "content": "Earlier task " * 100},
    {"role": "assistant", "content": "Earlier answer " * 100},
    {"role": "user", "content": "Continue the current task"},
]
SUMMARY_USAGE: Final = {
    "input_tokens": 80000,
    "output_tokens": 800,
    "total_tokens": 80800,
    "input_tokens_details": {"cached_tokens": 20000, "cache_write_tokens": 5000},
    "output_tokens_details": {"reasoning_tokens": 300},
}
MAIN_USAGE: Final = {
    "input_tokens": 12000,
    "output_tokens": 1600,
    "total_tokens": 13600,
    "input_tokens_details": {"cached_tokens": 4000},
    "output_tokens_details": {"reasoning_tokens": 600},
}


def response(text: str = "<summary>Prior task and progress</summary>", summary: bool = True) -> ResponsesAPIResponse:
    return ResponsesAPIResponse(
        id="resp_summary" if summary else "resp_main",
        created_at=1,
        object="response",
        status="completed",
        model=MODEL.removeprefix("openai/"),
        output=[
            {
                "type": "message",
                "id": "msg_summary" if summary else "msg_main",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        usage=SUMMARY_USAGE if summary else MAIN_USAGE,
    )


def request(**overrides: object) -> Mapping[str, object]:
    return {
        "model": MODEL,
        "input": HISTORY,
        "model_info": MODEL_INFO,
        "api_base": API_BASE,
        "api_key": "test-key",
        "custom_llm_provider": "openai",
        "context_management": [{"type": "compaction", "compact_threshold": 1}],
        **overrides,
    }


class Summary:
    def __init__(self, result: ResponsesAPIResponse | None = None) -> None:
        self.result = result if result is not None else response()
        self.calls: tuple[Mapping[str, object], ...] = ()

    async def __call__(self, value: Mapping[str, object]) -> ResponsesAPIResponse:
        self.calls = (*self.calls, value)
        return self.result


@pytest.fixture(autouse=True)
def settings(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(proxy_server.general_settings, COMPACT_SUMMARY_MODEL_SETTING_KEY, COMPACT_SAME_AS_REQUEST)
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "num_retries", 0)
    monkeypatch.setitem(
        litellm.model_cost,
        "test-compaction-model",
        {
            **MODEL_INFO,
            "litellm_provider": "openai",
            "mode": "responses",
            "supports_native_streaming": True,
            "input_cost_per_token": 0,
            "output_cost_per_token": 0,
        },
    )
    litellm.in_memory_llm_clients_cache.flush_cache()
    yield
    litellm.in_memory_llm_clients_cache.flush_cache()


@pytest.mark.asyncio
async def test_prepare_pins_model_preserves_context_and_bills_reported_usage() -> None:
    summary: Final = Summary()
    session: Final = CompactionSession()
    source: Final = request(
        instructions="Keep the task constraints", tools=[{"type": "function", "name": "execute", "parameters": {}}]
    )
    prepared: Final = await prepare_compaction(source, session, summary)
    assert len(summary.calls) == 1
    child: Final = summary.calls[0]
    assert (child["model"], child["api_key"], child["api_base"]) == (
        source["model"],
        source["api_key"],
        source["api_base"],
    )
    assert child[COMPACTION_CHILD_KEY] is True
    assert child["stream"] is False
    assert "tools" not in child and "context_management" not in child
    assert "Tool definitions" in str(child["input"])
    assert prepared["instructions"] == source["instructions"]
    assert "Continue the current task" in str(prepared["input"])
    assert "Earlier answer Earlier answer" not in str(prepared["input"])
    assert prepared["context_management"] is None
    assert session.locked
    final: Final = finish_compaction(response("answer", summary=False), session)
    assert isinstance(final, ResponsesAPIResponse)
    assert final.id == "resp_main"
    assert final.output[0]["type"] == "compaction"
    assert final.usage.input_tokens == 92000
    assert final.usage.output_tokens == 2400
    assert final.usage.total_tokens == 94400
    assert final.usage.input_tokens_details.cached_tokens == 24000
    assert final.usage.input_tokens_details.cache_write_tokens == 5000
    assert final.usage.output_tokens_details.reasoning_tokens == 900


@pytest.mark.asyncio
@pytest.mark.parametrize("bucket", ["metadata", "litellm_metadata"])
@pytest.mark.parametrize("auth_object", [False, True])
async def test_summary_does_not_finalize_the_main_budget_reservation(bucket: str, auth_object: bool) -> None:
    reservation: Final = {"reserved_cost": 1.0}
    auth: Final = (
        UserAPIKeyAuth(api_key="test-key", team_id="test-team", budget_reservation=reservation)
        if auth_object
        else {"api_key": "test-key", "team_id": "test-team", "budget_reservation": reservation}
    )
    metadata: Final = {
        "user_api_key_budget_reservation": reservation,
        "user_api_key_auth": auth,
        "user_api_key_team_id": "test-team",
    }
    summary: Final = Summary()
    prepared: Final = await prepare_compaction(request(**{bucket: metadata}), CompactionSession(), summary)
    child_metadata: Final = summary.calls[0][bucket]
    assert _get_budget_reservation_from_metadata(child_metadata) is None
    assert child_metadata["user_api_key_team_id"] == "test-team"
    assert _get_budget_reservation_from_metadata(prepared[bucket]) == reservation
    assert _get_budget_reservation_from_metadata(metadata) == reservation


@pytest.mark.asyncio
@pytest.mark.parametrize("offset, expected", [(1, 0), (0, 1), (-1, 1)])
async def test_threshold_boundary_counts_instructions_and_tools(offset: int, expected: int) -> None:
    source: Final = request(
        instructions="System context " * 100,
        tools=[{"type": "function", "name": "tool", "description": "Important definition " * 80}],
    )
    tokens: Final = _count(source, HISTORY)
    assert tokens > _count(request(), HISTORY)
    summary: Final = Summary()
    await prepare_compaction(
        {**source, "context_management": [{"type": "compaction", "compact_threshold": tokens + offset}]},
        CompactionSession(),
        summary,
    )
    assert len(summary.calls) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "overrides",
    [
        {"context_management": None},
        {"model_info": {**MODEL_INFO, "supports_native_compaction": True}},
        {"context_management": [{"type": "compaction", "compact_threshold": 10000}]},
        {"context_management": [{"type": "compaction"}]},
    ],
)
async def test_native_missing_and_below_threshold_are_usage_noops(overrides: dict[str, object]) -> None:
    summary: Final = Summary()
    session: Final = CompactionSession()
    await prepare_compaction(request(**overrides), session, summary)
    main: Final = response("main", False)
    assert finish_compaction(main, session) is main
    assert not summary.calls


@pytest.mark.asyncio
async def test_retries_reuse_summary_but_cannot_change_deployment() -> None:
    summary: Final = Summary()
    session: Final = CompactionSession()
    first: Final = await prepare_compaction(request(), session, summary)
    second: Final = await prepare_compaction(request(), session, summary)
    assert first == second
    assert len(summary.calls) == 1
    with pytest.raises(litellm.BadRequestError, match="Cannot change"):
        await prepare_compaction(request(api_base="https://different.example"), session, summary)


@pytest.mark.asyncio
async def test_extraction_failure_preserves_context_and_measured_consumption() -> None:
    summary: Final = Summary(response("No summary tags"))
    session: Final = CompactionSession()
    prepared: Final = await prepare_compaction(request(), session, summary)
    assert prepared["input"] == HISTORY
    assert session.accounting.item is None
    assert session.accounting.usages[0].total_tokens == 80800
    assert session.locked


@pytest.mark.asyncio
async def test_missing_usage_fails_visibly_and_keeps_pin() -> None:
    summary: Final = Summary(response().model_copy(update={"usage": None}))
    session: Final = CompactionSession()
    with pytest.raises(litellm.BadRequestError, match="did not report usage"):
        await prepare_compaction(request(), session, summary)
    assert session.locked


@pytest.mark.asyncio
async def test_summary_cache_hit_has_artifact_without_an_extra_usage_contributor() -> None:
    cached: Final = response()
    cached._hidden_params["cache_hit"] = True
    session: Final = CompactionSession()
    await prepare_compaction(request(), session, Summary(cached))
    assert session.accounting.item is not None
    assert not session.accounting.usages


@pytest.mark.asyncio
async def test_replay_without_new_compaction_drops_old_history_and_does_not_rebill() -> None:
    first: Final = CompactionSession()
    await prepare_compaction(request(), first, Summary())
    second: Final = CompactionSession()
    child: Final = Summary()
    prepared: Final = await prepare_compaction(
        request(
            input=[
                *HISTORY,
                first.accounting.item.model_dump(),
                {"role": "assistant", "content": "first answer"},
                {"role": "user", "content": "next task"},
            ],
            context_management=None,
        ),
        second,
        child,
    )
    assert "Earlier answer Earlier answer" not in str(prepared["input"])
    assert "Continue the current task" in str(prepared["input"])
    assert "first answer" in str(prepared["input"])
    assert "next task" in str(prepared["input"])
    assert not child.calls and not second.accounting.usages


@pytest.mark.asyncio
@pytest.mark.parametrize("setting", [None, "explicit-summary-alias"])
async def test_existing_configuration_is_not_reinterpreted(
    setting: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(proxy_server.general_settings, COMPACT_SUMMARY_MODEL_SETTING_KEY, setting)
    summary: Final = Summary()
    prepared: Final = await prepare_compaction(request(), CompactionSession(), summary)
    assert prepared["context_management"] == request()["context_management"]
    assert not summary.calls


@pytest.mark.asyncio
@pytest.mark.parametrize("setting", [False, "", 3])
async def test_invalid_configuration_fails_before_execution(setting: object, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(proxy_server.general_settings, COMPACT_SUMMARY_MODEL_SETTING_KEY, setting)
    summary: Final = Summary()
    with pytest.raises(litellm.BadRequestError, match=COMPACT_SUMMARY_MODEL_SETTING_KEY):
        await prepare_compaction(request(), CompactionSession(), summary)
    assert not summary.calls


@pytest.mark.asyncio
async def test_native_route_returns_serialized_aggregate_with_separate_operation_logs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    end_user: Final = f"compaction-test-{uuid4()}"

    class Recorder(CustomLogger):
        def __init__(self) -> None:
            self.usages: tuple[tuple[int, int], ...] = ()

        async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
            if kwargs.get("user") == end_user:
                self.usages = (*self.usages, (response_obj.usage.prompt_tokens, response_obj.usage.completion_tokens))

    recorder: Final = Recorder()
    monkeypatch.setattr(litellm, "callbacks", [recorder])

    with respx.mock as backend:
        route: Final = backend.post(f"{API_BASE}/responses").mock(
            side_effect=[
                httpx.Response(200, json=response().model_dump(exclude_none=True)),
                httpx.Response(200, json=response("main", False).model_dump(exclude_none=True)),
            ]
        )
        result: Final = await litellm.aresponses(**request(user=end_user))
        assert isinstance(result, ResponsesAPIResponse)
        serialized: Final = json.loads(result.model_dump_json(exclude_none=True))
        assert serialized["usage"]["total_tokens"] == 94400
        assert serialized["output"][0]["type"] == "compaction"
        assert serialized["output"][1]["content"][0]["text"] == "main"
        assert len(route.calls) == 2
        child_body: Final = json.loads(route.calls[0].request.content)
        main_body: Final = json.loads(route.calls[1].request.content)
        assert child_body["model"] == main_body["model"]
        assert "context_management" not in main_body
        assert "context_management" not in child_body
        assert "tools" not in child_body
        await asyncio.sleep(0)
        from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

        await asyncio.wait_for(GLOBAL_LOGGING_WORKER.flush(), timeout=10)
        assert sorted(recorder.usages) == [(12000, 1600), (80000, 800)]


@pytest.mark.asyncio
async def test_native_stream_serialization_and_next_turn_replay() -> None:
    main: Final = response("main", False).model_dump(exclude_none=True)
    events: Final = [
        {
            "type": "response.created",
            "sequence_number": 0,
            "response": {**main, "status": "in_progress", "output": [], "usage": None},
        },
        {"type": "response.output_item.added", "sequence_number": 1, "output_index": 0, "item": main["output"][0]},
        {
            "type": "response.output_text.delta",
            "sequence_number": 2,
            "output_index": 0,
            "content_index": 0,
            "item_id": "msg_main",
            "delta": "main",
        },
        {"type": "response.output_item.done", "sequence_number": 3, "output_index": 0, "item": main["output"][0]},
        {"type": "response.completed", "sequence_number": 4, "response": main},
    ]
    with respx.mock as backend:
        route: Final = backend.post(f"{API_BASE}/responses").mock(
            side_effect=[
                httpx.Response(200, json=response().model_dump(exclude_none=True)),
                httpx.Response(
                    200,
                    text="".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events),
                    headers={"content-type": "text/event-stream"},
                ),
                httpx.Response(200, json=main),
            ]
        )
        stream: Final = await litellm.aresponses(**request(stream=True))
        emitted: Final = [json.loads(event.model_dump_json(exclude_none=True)) async for event in stream]
        terminal: Final = emitted[-1]["response"]
        artifact: Final = terminal["output"][0]
        assert terminal["usage"]["total_tokens"] == 94400
        assert terminal["usage"]["input_tokens_details"]["cache_write_tokens"] == 5000
        assert terminal["output"][1]["id"] == "msg_main"
        assert artifact["type"] == "compaction"
        artifact_events: Final = [event for event in emitted if event.get("item", {}).get("type") == "compaction"]
        assert [event["type"] for event in artifact_events] == [
            "response.output_item.added",
            "response.output_item.done",
        ]
        assert all(event["item"] == artifact and event["output_index"] == 0 for event in artifact_events)
        assert next(event for event in emitted if event["type"] == "response.output_text.delta")["output_index"] == 1
        assert [event["sequence_number"] for event in emitted] == sorted(
            {event["sequence_number"] for event in emitted}
        )
        await litellm.aresponses(
            **request(
                input=[*HISTORY, *terminal["output"], {"role": "user", "content": "next task"}],
                context_management=[{"type": "compaction", "compact_threshold": 10000}],
            )
        )
        assert len(route.calls) == 3
        followup: Final = json.loads(route.calls[-1].request.content)
        assert "Earlier answer Earlier answer" not in str(followup["input"])
        assert "next task" in str(followup["input"])
        assert "Prior task and progress" in str(followup["input"])


@pytest.mark.asyncio
async def test_chat_bridge_uses_same_backend_for_summary_and_main() -> None:
    def chat_body(summary: bool) -> Mapping[str, object]:
        usage: Final = SUMMARY_USAGE if summary else MAIN_USAGE
        return {
            "id": "chat_summary" if summary else "chat_main",
            "object": "chat.completion",
            "created": 1,
            "model": "test-compaction-model",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "<summary>progress</summary>" if summary else "main"},
                }
            ],
            "usage": {
                "prompt_tokens": usage["input_tokens"],
                "completion_tokens": usage["output_tokens"],
                "total_tokens": usage["total_tokens"],
                "prompt_tokens_details": usage["input_tokens_details"],
                "completion_tokens_details": usage["output_tokens_details"],
            },
        }

    with respx.mock as backend:
        route: Final = backend.post(f"{API_BASE}/chat/completions").mock(
            side_effect=[
                httpx.Response(200, json=chat_body(True)),
                httpx.Response(200, json=chat_body(False)),
            ]
        )
        result: Final = await litellm.aresponses(
            **request(model="custom_openai/test-compaction-model", custom_llm_provider="custom_openai")
        )
        serialized: Final = json.loads(result.model_dump_json(exclude_none=True))
        assert serialized["usage"]["input_tokens"] == 92000
        assert serialized["usage"]["total_tokens"] == 94400
        assert serialized["usage"]["input_tokens_details"]["cache_write_tokens"] == 5000
        assert serialized["usage"]["output_tokens_details"]["reasoning_tokens"] == 900
        assert serialized["output"][0]["type"] == "compaction"
        assert len(route.calls) == 2
        assert all(call.request.headers["authorization"] == "Bearer test-key" for call in route.calls)


def test_sync_nonstream_matches_async_usage() -> None:
    with respx.mock as backend:
        backend.post(f"{API_BASE}/responses").mock(
            side_effect=[
                httpx.Response(200, json=response().model_dump(exclude_none=True)),
                httpx.Response(200, json=response("main", False).model_dump(exclude_none=True)),
            ]
        )
        result: Final = litellm.responses(**request())
        assert json.loads(result.model_dump_json(exclude_none=True))["usage"]["total_tokens"] == 94400


@pytest.mark.asyncio
async def test_small_window_reduces_summary_output_budget_without_dropping_input() -> None:
    summary: Final = Summary()
    prepared: Final = await prepare_compaction(
        request(model_info={**MODEL_INFO, "max_input_tokens": 2048, "max_output_tokens": 2048}),
        CompactionSession(),
        summary,
    )
    assert 0 < summary.calls[0]["max_output_tokens"] < 2048
    assert "Earlier answer Earlier answer" in str(summary.calls[0]["input"])
    assert "Continue the current task" in str(prepared["input"])


@pytest.mark.asyncio
async def test_oversize_summary_is_rejected_without_provider_work() -> None:
    summary: Final = Summary()
    with pytest.raises(litellm.BadRequestError, match="exceeds"):
        await prepare_compaction(
            request(model_info={**MODEL_INFO, "max_input_tokens": 20}), CompactionSession(), summary
        )
    assert not summary.calls


class _ProxySpendRecorder(CustomLogger):
    def __init__(self) -> None:
        self.usages: tuple[tuple[int, int], ...] = ()
        self.failure_usages: tuple[tuple[int, int], ...] = ()

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        usage: Final = response_obj.usage
        counters: Final = usage if isinstance(usage, dict) else usage.model_dump()
        self.usages = (*self.usages, (counters["prompt_tokens"], counters["completion_tokens"]))

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        usage: Final = kwargs["standard_logging_object"]
        self.failure_usages = (*self.failure_usages, (usage["prompt_tokens"], usage["completion_tokens"]))


class _BufferingCompactionGuardrail(CustomGuardrail):
    use_native_lifecycle_hooks = True

    def __init__(self, blocked: bool, recorder: _ProxySpendRecorder) -> None:
        super().__init__(guardrail_name="compaction-probe", event_hook=GuardrailEventHooks.post_call, default_on=True)
        self.blocked = blocked
        self.recorder = recorder
        self.deferred_armed = False
        self.usage_before_release: tuple[tuple[int, int], ...] = ()

    async def async_post_call_streaming_iterator_hook(self, user_api_key_dict, response, request_data):
        logging_obj: Final = request_data["litellm_logging_obj"]
        self.deferred_armed = getattr(logging_obj, "_on_deferred_stream_complete", None) is not None
        buffered: Final = [item async for item in response]
        await asyncio.sleep(0)
        await asyncio.wait_for(GLOBAL_LOGGING_WORKER.flush(), timeout=10)
        self.usage_before_release = self.recorder.usages
        if self.blocked:
            raise HTTPException(status_code=403, detail="compaction test guardrail blocked the output")
        for item in buffered:
            yield item


def _proxy_provider_responses(bridge: bool) -> tuple[httpx.Response, httpx.Response]:
    main: Final = response("main", False).model_dump(exclude_none=True)
    if not bridge:
        events: Final = (
            {
                "type": "response.created",
                "sequence_number": 0,
                "response": {**main, "status": "in_progress", "output": [], "usage": None},
            },
            {"type": "response.output_item.added", "sequence_number": 1, "output_index": 0, "item": main["output"][0]},
            {
                "type": "response.output_text.delta",
                "sequence_number": 2,
                "output_index": 0,
                "content_index": 0,
                "item_id": "msg_main",
                "delta": "main",
            },
            {"type": "response.output_item.done", "sequence_number": 3, "output_index": 0, "item": main["output"][0]},
            {"type": "response.completed", "sequence_number": 4, "response": main},
        )
        return (
            httpx.Response(200, json=response().model_dump(exclude_none=True)),
            httpx.Response(
                200,
                text="".join(f"data: {json.dumps(event)}\n\n" for event in events),
                headers={"content-type": "text/event-stream"},
            ),
        )
    summary: Final = response()
    summary_usage: Final = {
        "prompt_tokens": SUMMARY_USAGE["input_tokens"],
        "completion_tokens": SUMMARY_USAGE["output_tokens"],
        "total_tokens": SUMMARY_USAGE["total_tokens"],
        "prompt_tokens_details": SUMMARY_USAGE["input_tokens_details"],
        "completion_tokens_details": SUMMARY_USAGE["output_tokens_details"],
    }
    main_usage: Final = {
        "prompt_tokens": MAIN_USAGE["input_tokens"],
        "completion_tokens": MAIN_USAGE["output_tokens"],
        "total_tokens": MAIN_USAGE["total_tokens"],
        "prompt_tokens_details": MAIN_USAGE["input_tokens_details"],
        "completion_tokens_details": MAIN_USAGE["output_tokens_details"],
    }
    chat_base: Final = {"id": "chat_main", "created": 1, "model": "test-compaction-model"}
    chunks: Final = (
        {
            **chat_base,
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": "main"}, "finish_reason": None}],
        },
        {
            **chat_base,
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": main_usage,
        },
    )
    return (
        httpx.Response(
            200,
            json={
                **chat_base,
                "id": "chat_summary",
                "object": "chat.completion",
                "usage": summary_usage,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": summary.output_text},
                    }
                ],
            },
        ),
        httpx.Response(
            200,
            text="".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n",
            headers={"content-type": "text/event-stream"},
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("blocked", [False, True], ids=["guardrail-pass", "guardrail-block"])
@pytest.mark.parametrize("bridge", [False, True], ids=["native", "chat-bridge"])
async def test_proxy_sse_compaction_keeps_deferred_spend_operation_local(
    monkeypatch: pytest.MonkeyPatch,
    blocked: bool,
    bridge: bool,
) -> None:
    recorder: Final = _ProxySpendRecorder()
    guardrail: Final = _BufferingCompactionGuardrail(blocked, recorder)
    monkeypatch.setattr(litellm, "callbacks", [recorder, guardrail])
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "compaction-alias",
                "litellm_params": {
                    "model": MODEL,
                    "api_key": "test-key",
                    "api_base": API_BASE,
                    "use_chat_completions_api": bridge,
                },
                "model_info": {**MODEL_INFO, "id": "compaction-deployment"},
            }
        ],
        num_retries=0,
    )
    monkeypatch.setattr(proxy_server, "llm_router", router)
    monkeypatch.setitem(
        proxy_server.app.dependency_overrides,
        user_api_key_auth,
        lambda: UserAPIKeyAuth(
            api_key="compaction-test-key",
            user_id="compaction-test-user",
            models=["compaction-alias"],
            request_route="/v1/responses",
        ),
    )
    with respx.mock as backend:
        route: Final = backend.post(f"{API_BASE}/{'chat/completions' if bridge else 'responses'}").mock(
            side_effect=_proxy_provider_responses(bridge),
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=proxy_server.app),
            base_url="http://testserver",
        ) as client:
            outward: Final = await client.post(
                "/v1/responses",
                json={
                    "model": "compaction-alias",
                    "stream": True,
                    "input": HISTORY,
                    "context_management": [{"type": "compaction", "compact_threshold": 1}],
                },
            )
        assert outward.status_code == 200, outward.text
        events: Final = tuple(
            json.loads(line.removeprefix("data: "))
            for line in outward.text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        )
        await asyncio.sleep(0)
        await asyncio.wait_for(GLOBAL_LOGGING_WORKER.flush(), timeout=10)
        assert guardrail.deferred_armed
        assert guardrail.usage_before_release == ((80000, 800),)
        billable: Final = tuple(usage for usage in (*recorder.usages, *recorder.failure_usages) if any(usage))
        assert sorted(billable) == [(12000, 1600), (80000, 800)]
        assert len(route.calls) == 2
        child_body, main_body = (json.loads(call.request.content) for call in route.calls)
        assert child_body["model"] == main_body["model"]
        assert "tools" not in child_body and "context_management" not in child_body
        assert all(call.request.headers["authorization"] == "Bearer test-key" for call in route.calls)
        if blocked:
            assert [event["type"] for event in events] == ["response.failed"]
            return
        terminal: Final = events[-1]["response"]
        assert events[-1]["type"] == "response.completed"
        assert terminal["usage"]["input_tokens"] == 92000
        assert terminal["usage"]["output_tokens"] == 2400
        assert terminal["usage"]["total_tokens"] == 94400
        assert terminal["usage"]["input_tokens_details"]["cache_write_tokens"] == 5000
        assert terminal["usage"]["input_tokens_details"]["cached_tokens"] == 24000
        assert terminal["usage"]["output_tokens_details"]["reasoning_tokens"] == 900
        assert "cost" not in terminal["usage"]
        artifact: Final = terminal["output"][0]
        artifact_events: Final = tuple(event for event in events if event.get("item", {}).get("type") == "compaction")
        assert [event["type"] for event in artifact_events] == [
            "response.output_item.added",
            "response.output_item.done",
        ]
        assert all(event["item"] == artifact and event["output_index"] == 0 for event in artifact_events)
        assert next(event for event in events if event["type"] == "response.output_text.delta")["output_index"] == 1
        assert events[0]["response"]["id"] == terminal["id"]


@pytest.mark.asyncio
@pytest.mark.parametrize("connection", ["declared-native", "azure-token", "azure-token-provider"])
async def test_previous_response_native_compaction_preserves_transport_and_auth(
    monkeypatch: pytest.MonkeyPatch,
    connection: str,
) -> None:
    azure: Final = connection.startswith("azure")
    api_base: Final = "https://compaction.openai.azure.com" if azure else API_BASE
    responses_url: Final = f"{api_base}/{'openai/' if azure else ''}responses"
    token: Final = "test-entra-token" if azure else "test-key"
    credentials: Final = (
        {"api_key": None, "azure_ad_token": token}
        if connection == "azure-token"
        else {"api_key": None, "azure_ad_token_provider": lambda: token}
        if azure
        else {"api_key": token}
    )
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    with respx.mock as backend:
        history_route: Final = backend.get(f"{responses_url}/resp_previous/input_items").respond(
            200,
            json={
                "object": "list",
                "data": [{"type": "message", "role": "user", "content": "Persisted native user task"}],
                "has_more": False,
            },
        )
        previous_route: Final = backend.get(f"{responses_url}/resp_previous").respond(
            200,
            json=response("Persisted native assistant answer", False).model_dump(exclude_none=True),
        )
        generation_route: Final = backend.post(responses_url).mock(
            side_effect=[
                httpx.Response(200, json=response().model_dump(exclude_none=True)),
                httpx.Response(200, json=response("main", False).model_dump(exclude_none=True)),
            ]
        )
        result: Final = await litellm.aresponses(
            **request(
                model="azure/test-compaction-model" if azure else "custom_openai/test-compaction-model",
                custom_llm_provider="azure" if azure else "custom_openai",
                api_base=api_base,
                api_version="2025-04-01-preview" if azure else None,
                model_info={**MODEL_INFO, "supported_endpoints": ["/v1/responses"]},
                previous_response_id="resp_previous",
                input=[{"role": "user", "content": "Continue current task"}],
                **credentials,
            )
        )
        assert history_route.call_count == previous_route.call_count == 1
        assert generation_route.call_count == 2
        assert all(call.request.headers["authorization"] == f"Bearer {token}" for call in backend.calls)
        summary_body, main_body = (json.loads(call.request.content) for call in generation_route.calls)
        assert "Persisted native user task" in str(summary_body["input"])
        assert "Persisted native assistant answer" in str(summary_body["input"])
        assert "Persisted native user task" not in str(main_body["input"])
        assert "Continue current task" in str(main_body["input"])
        assert "previous_response_id" not in main_body
        assert result.usage.total_tokens == 94400
        assert result.output[0]["type"] == "compaction"


@pytest.mark.asyncio
async def test_previous_response_chat_prefix_compaction_uses_stored_chat_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ReplayDatabase:
        def __init__(self) -> None:
            self.requests: tuple[str, ...] = ()

        async def query_raw(self, query: str, response_id: str) -> list[dict[str, object]]:
            self.requests = (*self.requests, response_id)
            return [
                {
                    "request_id": "chat_previous",
                    "session_id": "stored-chat-session",
                    "proxy_server_request": {"input": "Persisted chat user task"},
                    "response": {
                        "id": "chat_previous",
                        "object": "chat.completion",
                        "created": 1,
                        "model": "test-compaction-model",
                        "choices": [
                            {
                                "index": 0,
                                "finish_reason": "stop",
                                "message": {
                                    "role": "assistant",
                                    "content": "Persisted chat assistant answer",
                                },
                            }
                        ],
                    },
                }
            ]

    database: Final = ReplayDatabase()
    monkeypatch.setattr(proxy_server, "prisma_client", SimpleNamespace(db=database))
    with respx.mock as backend:
        summary_response: Final = _proxy_provider_responses(True)[0]
        main_response: Final = {
            **summary_response.json(),
            "id": "chat_main",
            "usage": {
                "prompt_tokens": MAIN_USAGE["input_tokens"],
                "completion_tokens": MAIN_USAGE["output_tokens"],
                "total_tokens": MAIN_USAGE["total_tokens"],
            },
        }
        generation_route: Final = backend.post(f"{API_BASE}/chat/completions").mock(
            side_effect=[summary_response, httpx.Response(200, json=main_response)]
        )
        result: Final = await litellm.aresponses(
            **request(
                model="openai/chat_completions/test-compaction-model",
                previous_response_id="chat_previous",
                input=[{"role": "user", "content": "Continue current task"}],
            )
        )
        assert database.requests == ("chat_previous",)
        assert len(backend.calls) == generation_route.call_count == 2
        summary_body, main_body = (json.loads(call.request.content) for call in generation_route.calls)
        assert "Persisted chat user task" in str(summary_body["messages"])
        assert "Persisted chat assistant answer" in str(summary_body["messages"])
        assert "Persisted chat user task" not in str(main_body["messages"])
        assert "Continue current task" in str(main_body["messages"])
        assert summary_body["model"] == main_body["model"] == "test-compaction-model"
        assert result.output[0]["type"] == "compaction"
        assert result.usage.total_tokens == 94400


@pytest.mark.asyncio
@pytest.mark.parametrize("bucket", ["metadata", "litellm_metadata", "both"])
async def test_background_history_reads_preserve_callbacks_and_only_main_consumes_reservation(
    monkeypatch: pytest.MonkeyPatch,
    bucket: str,
) -> None:
    monkeypatch.setitem(
        litellm.model_cost,
        "test-compaction-model",
        {
            **litellm.model_cost["test-compaction-model"],
            "input_cost_per_token": 0.000001,
            "output_cost_per_token": 0.000002,
        },
    )
    reservation: Final = {"reserved_cost": 1.0, "entries": [], "finalized": False}
    parent: Final = {
        "user_api_key_budget_reservation": reservation,
        "user_api_key_auth": UserAPIKeyAuth(api_key="test-key", budget_reservation=reservation),
    }

    class Recorder:
        def __init__(self) -> None:
            self.rows: tuple[tuple[str, int, int, float, bool, bool], ...] = ()

        async def record(self, kwargs, response_obj, start_time, end_time):
            logged: Final = kwargs["standard_logging_object"]
            metadata: Final = StandardLoggingPayloadSetup.merge_litellm_metadata(kwargs["litellm_params"])
            found: Final = _get_budget_reservation_from_metadata(metadata)
            self.rows = (
                *self.rows,
                (
                    kwargs["call_type"],
                    logged["prompt_tokens"],
                    logged["completion_tokens"],
                    logged["response_cost"],
                    found is not None,
                    reservation["finalized"],
                ),
            )
            if found is not None:
                await reconcile_budget_reservation(found, logged["response_cost"])

    recorder: Final = Recorder()
    old: Final = {
        **response("Persisted answer", False).model_dump(exclude_none=True),
        "background": True,
        "usage": {"input_tokens": 111000, "output_tokens": 2220, "total_tokens": 113220},
    }
    with respx.mock as backend:
        pages: Final = backend.get(f"{API_BASE}/responses/resp_old/input_items").mock(
            side_effect=[
                httpx.Response(
                    200,
                    json={
                        "data": [{"role": "user", "content": "Persisted task"}],
                        "has_more": True,
                        "last_id": "first-page-last-item",
                    },
                ),
                httpx.Response(
                    200,
                    json={
                        "data": [{"role": "assistant", "content": "Persisted intermediate progress"}],
                        "has_more": False,
                    },
                ),
            ]
        )
        backend.get(f"{API_BASE}/responses/resp_old").respond(200, json=old)
        backend.post(f"{API_BASE}/responses").mock(
            side_effect=[
                httpx.Response(200, json=response().model_dump(exclude_none=True)),
                httpx.Response(200, json=response("main", False).model_dump(exclude_none=True)),
            ]
        )
        result: Final = await litellm.aresponses(
            **request(
                previous_response_id="resp_old",
                input=[{"role": "user", "content": "Current task"}],
                success_callback=[recorder.record],
                **{name: parent for name in ("metadata", "litellm_metadata") if bucket in (name, "both")},
            )
        )
        await asyncio.sleep(0.1)
        await asyncio.wait_for(GLOBAL_LOGGING_WORKER.flush(), timeout=10)
        assert result.usage.total_tokens == 94400
        assert pages.call_count == 2
        assert pages.calls[1].request.url.params["after"] == "first-page-last-item"
        assert len(recorder.rows) == 5
        history: Final = tuple(row for row in recorder.rows if row[0] in ("alist_input_items", "aget_responses"))
        assert len(history) == 3
        assert all(row[1:5] == (0, 0, 0.0, False) for row in history)
        inference: Final = sorted((row for row in recorder.rows if row[0] == "aresponses"), key=lambda row: row[1])
        assert len(inference) == 2
        assert inference[0][1:3] == (12000, 1600)
        assert inference[1][1:3] == (80000, 800)
        assert all(row[3] > 0 for row in inference)
        main_has_reservation: Final = bucket != "metadata"
        assert inference[0][4:] == (main_has_reservation, False)
        assert inference[1][4] is False
        assert reservation["finalized"] is main_has_reservation


@pytest.mark.asyncio
async def test_public_background_response_read_with_serialized_history_marker_remains_billable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        litellm.model_cost,
        "test-compaction-model",
        {
            **litellm.model_cost["test-compaction-model"],
            "input_cost_per_token": 0.000001,
            "output_cost_per_token": 0.000002,
        },
    )
    adapter: Final = TypeAdapter(dict[str, object])
    spoof: Final = adapter.validate_json(adapter.dump_json(responses_compaction_history_metadata({"user": "test"})))

    class Recorder:
        def __init__(self) -> None:
            self.rows: tuple[tuple[int, int, float], ...] = ()

        async def record(self, kwargs, response_obj, start_time, end_time):
            logged: Final = kwargs["standard_logging_object"]
            self.rows = (*self.rows, (logged["prompt_tokens"], logged["completion_tokens"], logged["response_cost"]))

    recorder: Final = Recorder()
    with respx.mock as backend:
        backend.get(f"{API_BASE}/responses/resp_old").respond(
            200,
            json={
                **response("Persisted answer", False).model_dump(exclude_none=True),
                "background": True,
            },
        )
        await litellm.aget_responses(
            response_id="resp_old",
            api_base=API_BASE,
            api_key="test-key",
            custom_llm_provider="openai",
            litellm_metadata=spoof,
            success_callback=[recorder.record],
        )
        await asyncio.sleep(0.1)
        await asyncio.wait_for(GLOBAL_LOGGING_WORKER.flush(), timeout=10)
        assert len(recorder.rows) == 1
        assert recorder.rows[0][:2] == (12000, 1600)
        assert recorder.rows[0][2] > 0
