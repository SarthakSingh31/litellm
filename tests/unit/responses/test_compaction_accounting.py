import asyncio
import json
from typing import Final

import httpx
import pytest

from litellm.responses.compaction_accounting import CompactionAccounting, CompactionResponsesStream
from litellm.responses.streaming_iterator import BaseResponsesAPIStreamingIterator
from litellm.types.llms.base import BaseLiteLLMOpenAIResponseObject
from litellm.types.llms.openai import (
    GenericEvent,
    InputTokensDetails,
    OutputItemAddedEvent,
    OutputItemDoneEvent,
    OutputTextDeltaEvent,
    OutputTokensDetails,
    ResponseAPIUsage,
    ResponseCompletedEvent,
    ResponseCreatedEvent,
    ResponseFailedEvent,
    ResponseIncompleteEvent,
    ResponseInProgressEvent,
    ResponsesAPIResponse,
    ResponsesAPIStreamEvents,
    ResponsesAPIStreamingResponse,
)
from litellm.types.responses.main import CompactionOutputItem, GenericResponseOutputItem, OutputText


def _usage(
    input_tokens: int, output_tokens: int, cached: int = 0, writes: int = 0, reasoning: int = 0
) -> ResponseAPIUsage:
    return ResponseAPIUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        input_tokens_details=InputTokensDetails.model_validate({"cached_tokens": cached, "cache_write_tokens": writes}),
        output_tokens_details=OutputTokensDetails(reasoning_tokens=reasoning),
        cost=0.5,
    )


def _item() -> CompactionOutputItem:
    return CompactionOutputItem(type="compaction", id="cmp_summary", encrypted_content="private-summary-envelope")


def _response(usage: ResponseAPIUsage | None, status: str = "completed") -> ResponsesAPIResponse:
    return ResponsesAPIResponse(
        id="resp_main",
        created_at=1,
        object="response",
        model="same-model",
        status=status,
        output=[
            GenericResponseOutputItem(
                type="message",
                id="msg_main",
                role="assistant",
                status="completed",
                content=[OutputText(type="output_text", text="answer", annotations=[])],
            )
        ],
        usage=usage,
    )


def _accounting() -> CompactionAccounting:
    return CompactionAccounting(usages=(_usage(80000, 800, 20000, 5000, 300),), item=_item())


def _assert_fixture(response: ResponsesAPIResponse) -> None:
    payload: Final = json.loads(response.model_dump_json(exclude_none=True))
    assert payload["id"] == "resp_main"
    assert payload["usage"]["input_tokens"] == 92000
    assert payload["usage"]["output_tokens"] == 2400
    assert payload["usage"]["total_tokens"] == 94400
    assert payload["usage"]["input_tokens_details"]["cached_tokens"] == 24000
    assert payload["usage"]["input_tokens_details"]["cache_write_tokens"] == 5000
    assert payload["usage"]["output_tokens_details"]["reasoning_tokens"] == 900
    assert payload["usage"].get("cost") is None
    assert payload["output"][0] == _item().model_dump()
    assert payload["output"][1]["id"] == "msg_main"
    assert "_compaction_accounted_" not in response.model_dump_json()


def test_exact_aggregation_idempotence_and_logging_snapshot_isolation() -> None:
    main: Final = _response(_usage(12000, 1600, 4000, 0, 600))
    main._hidden_params = {"additional_headers": {"x-provider": "original"}}
    before: Final = main.model_dump_json()
    accounting: Final = _accounting()
    outward: Final = accounting.finalize(main)
    repeated: Final = accounting.finalize(outward)
    _assert_fixture(outward)
    _assert_fixture(repeated)
    _assert_fixture(accounting.finalize(main))
    assert outward is not main and repeated is not outward
    assert outward.usage is not main.usage
    assert main.model_dump_json() == before
    outward._hidden_params["additional_headers"]["x-provider"] = "changed"
    assert main._hidden_params["additional_headers"]["x-provider"] == "original"
    assert accounting.usages[0].input_tokens == 80000
    assert main.usage is not None and main.usage.cost == 0.5


def test_multiple_summaries_and_missing_optional_details() -> None:
    main: Final = _response(ResponseAPIUsage(input_tokens=10, output_tokens=2, total_tokens=999))
    accounting: Final = CompactionAccounting(usages=(_usage(20, 3, 5, 1, 1), _usage(30, 4, 6, 2, 2)))
    result: Final = accounting.finalize(main)
    assert result.usage is not None
    assert result.usage.model_dump(exclude_none=True) == {
        "input_tokens": 60,
        "output_tokens": 9,
        "total_tokens": 69,
        "input_tokens_details": {"cached_tokens": 11, "cache_write_tokens": 3},
        "output_tokens_details": {"reasoning_tokens": 3},
    }
    assert len(result.output) == 1


def test_native_sdk_output_serializes_after_inserting_compaction() -> None:
    native: Final = ResponsesAPIResponse.model_validate_json(
        _response(_usage(12000, 1600, 4000, 0, 600)).model_dump_json()
    )
    before: Final = native.model_dump_json()
    _assert_fixture(_accounting().finalize(native))
    assert native.model_dump_json() == before


def test_native_noop_and_replayed_item_do_not_add_usage() -> None:
    native: Final = _response(_usage(5, 3))
    native.output = [_item(), *native.output]
    assert CompactionAccounting().finalize(native) is native
    replayed: Final = CompactionAccounting(item=_item()).finalize(_response(_usage(5, 3)))
    assert replayed.usage == native.usage
    assert replayed.model_dump()["output"] == native.model_dump()["output"]


@pytest.mark.parametrize("status", ["completed", "incomplete", "failed"])
def test_missing_main_usage_reports_known_summary_lower_bound(status: str, caplog: pytest.LogCaptureFixture) -> None:
    main: Final = _response(None, status)
    accounting: Final = _accounting()
    result: Final = accounting.finalize(main)
    assert result.usage is not None
    assert result.usage.input_tokens == 80000
    assert result.usage.output_tokens == 800
    assert result.usage.total_tokens == 80800
    assert result.usage.input_tokens_details == accounting.usages[0].input_tokens_details
    assert result.usage.output_tokens_details == accounting.usages[0].output_tokens_details
    assert result.usage.cost is None
    assert accounting.finalize(result).usage == result.usage
    assert main.usage is None
    assert result.model_dump()["output"][0] == _item().model_dump()
    assert main.output[0] != _item()
    assert "main usage unavailable; reported aggregate is a lower bound" in caplog.text


class _Upstream(BaseResponsesAPIStreamingIterator):
    def __init__(self, events: tuple[ResponsesAPIStreamingResponse, ...], failure: BaseException | None = None) -> None:
        self.events = events
        self.index = 0
        self.failure = failure
        self.logged: tuple[ResponsesAPIResponse, ...] = ()
        self.closed = 0
        self.async_closed = 0
        self._hidden_params = {"model_id": "deployment", "additional_headers": {"x-provider": "original"}}
        self.model = "same-model"

    def __iter__(self) -> "_Upstream":
        return self

    def __aiter__(self) -> "_Upstream":
        return self

    def __next__(self) -> ResponsesAPIStreamingResponse:
        if self.index == len(self.events):
            if self.failure is not None:
                raise self.failure
            raise StopIteration
        event: Final = self.events[self.index]
        self.index += 1
        if isinstance(event, (ResponseCompletedEvent, ResponseIncompleteEvent, ResponseFailedEvent)):
            self.logged = (*self.logged, event.response)
        return event

    async def __anext__(self) -> ResponsesAPIStreamingResponse:
        await asyncio.sleep(0)
        try:
            return next(self)
        except StopIteration:
            raise StopAsyncIteration from None

    def close(self) -> None:
        self.closed += 1

    async def aclose(self) -> None:
        self.async_closed += 1


def _events(status: str = "completed") -> tuple[ResponsesAPIStreamingResponse, ...]:
    response: Final = _response(_usage(12000, 1600, 4000, 0, 600), status)
    startup: Final = response.model_copy(update={"output": [], "usage": None, "status": "in_progress"})
    item: Final = BaseLiteLLMOpenAIResponseObject.model_validate(response.model_dump()["output"][0])
    terminal_class: Final = {
        "completed": ResponseCompletedEvent,
        "incomplete": ResponseIncompleteEvent,
        "failed": ResponseFailedEvent,
    }[status]
    return (
        ResponseCreatedEvent.model_validate({"type": "response.created", "response": startup, "sequence_number": 0}),
        ResponseInProgressEvent.model_validate(
            {"type": "response.in_progress", "response": startup, "sequence_number": 1}
        ),
        OutputItemAddedEvent.model_validate(
            {"type": "response.output_item.added", "output_index": 0, "item": item, "sequence_number": 2}
        ),
        OutputTextDeltaEvent.model_validate(
            {
                "type": "response.output_text.delta",
                "output_index": 0,
                "item_id": "msg_main",
                "content_index": 0,
                "delta": "answer",
                "sequence_number": 3,
            }
        ),
        OutputItemDoneEvent.model_validate(
            {"type": "response.output_item.done", "output_index": 0, "item": item, "sequence_number": 4}
        ),
        terminal_class.model_validate({"type": f"response.{status}", "response": response, "sequence_number": 5}),
    )


def _assert_stream(events: tuple[ResponsesAPIStreamingResponse, ...], upstream: _Upstream) -> None:
    wire: Final = tuple(type(event).model_validate_json(event.model_dump_json()) for event in events)
    payloads: Final = tuple(json.loads(event.model_dump_json(exclude_none=True)) for event in wire)
    assert tuple(payload["sequence_number"] for payload in payloads) == tuple(range(8))
    assert tuple(payload["type"] for payload in payloads[:4]) == (
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.output_item.done",
    )
    assert payloads[2]["item"] == payloads[3]["item"] == payloads[-1]["response"]["output"][0]
    assert payloads[2]["output_index"] == payloads[3]["output_index"] == 0
    assert tuple(payload["output_index"] for payload in payloads[4:7]) == (1, 1, 1)
    assert payloads[4]["item"]["id"] == payloads[5]["item_id"] == payloads[6]["item"]["id"] == "msg_main"
    assert payloads[-1]["response"]["output"][1]["id"] == "msg_main"
    assert isinstance(wire[-1], (ResponseCompletedEvent, ResponseIncompleteEvent, ResponseFailedEvent))
    _assert_fixture(wire[-1].response)
    assert len(upstream.logged) == 1
    assert upstream.logged[0].usage is not None and upstream.logged[0].usage.input_tokens == 12000
    assert len(upstream.logged[0].output) == 1


@pytest.mark.parametrize("status", ["completed", "incomplete", "failed"])
def test_sync_standard_events_and_all_terminal_statuses(status: str) -> None:
    upstream: Final = _Upstream(_events(status))
    before: Final = tuple(event.model_dump_json() for event in upstream.events)
    stream: Final = CompactionResponsesStream(upstream, _accounting())
    assert isinstance(stream, BaseResponsesAPIStreamingIterator)
    assert stream._hidden_params is upstream._hidden_params
    assert stream.model == upstream.model
    _assert_stream(tuple(stream), upstream)
    assert tuple(event.model_dump_json() for event in upstream.events) == before
    stream.close()
    assert upstream.closed == 1


@pytest.mark.asyncio
async def test_async_stream_and_concurrent_request_isolation() -> None:
    upstream: Final = _Upstream(_events())
    stream: Final = CompactionResponsesStream(upstream, _accounting())
    other: Final = CompactionResponsesStream(_Upstream(_events()), CompactionAccounting(usages=(_usage(1, 1),)))

    async def collect(source: CompactionResponsesStream) -> tuple[ResponsesAPIStreamingResponse, ...]:
        return tuple([event async for event in source])

    first, second = await asyncio.gather(collect(stream), collect(other))
    _assert_stream(first, upstream)
    assert isinstance(second[-1], ResponseCompletedEvent)
    assert second[-1].response.usage is not None and second[-1].response.usage.input_tokens == 12001
    assert len(second[-1].response.output) == 1
    await stream.aclose()
    assert upstream.async_closed == 1


def test_noop_stream_and_repeated_terminal_processing() -> None:
    events: Final = _events()
    assert tuple(CompactionResponsesStream(_Upstream(events), CompactionAccounting())) == events
    repeated: Final = tuple(CompactionResponsesStream(_Upstream((*events, events[-1])), _accounting()))
    assert isinstance(repeated[-1], ResponseCompletedEvent) and isinstance(repeated[-2], ResponseCompletedEvent)
    _assert_fixture(repeated[-1].response)
    assert repeated[-1].response == repeated[-2].response


def test_close_before_queued_terminal_is_delivered_warns(caplog: pytest.LogCaptureFixture) -> None:
    upstream: Final = _Upstream((_events()[-1],))
    stream: Final = CompactionResponsesStream(upstream, _accounting())
    assert next(stream).type == ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED
    stream.close()
    assert tuple(stream) == ()
    assert upstream.closed == 1
    assert "without a terminal response" in caplog.text


@pytest.mark.parametrize("failure", [None, httpx.ReadError("disconnected")])
def test_disconnect_never_fabricates_terminal_and_closes(
    failure: BaseException | None, caplog: pytest.LogCaptureFixture
) -> None:
    upstream: Final = _Upstream(_events()[:2], failure)
    stream: Final = CompactionResponsesStream(upstream, _accounting())
    assert next(stream).type == ResponsesAPIStreamEvents.RESPONSE_CREATED
    assert next(stream).type == ResponsesAPIStreamEvents.RESPONSE_IN_PROGRESS
    with pytest.raises(StopIteration if failure is None else httpx.ReadError):
        next(stream)
    assert upstream.closed == 1
    assert not upstream.logged
    assert "without a terminal response" in caplog.text


@pytest.mark.asyncio
async def test_cancellation_closes_without_terminal(caplog: pytest.LogCaptureFixture) -> None:
    upstream: Final = _Upstream((), asyncio.CancelledError())
    stream: Final = CompactionResponsesStream(upstream, _accounting())
    with pytest.raises(asyncio.CancelledError):
        await anext(stream)
    assert upstream.async_closed == 1
    assert not upstream.logged
    assert "without a terminal response" in caplog.text


def test_unknown_standard_events_shift_output_index_and_missing_sequence() -> None:
    event: Final = GenericEvent(type="response.custom.delta", output_index=3, delta="payload")
    result: Final = tuple(CompactionResponsesStream(_Upstream((event,)), CompactionAccounting(item=_item())))
    assert result[-1].model_dump() == {
        "type": "response.custom.delta",
        "output_index": 4,
        "delta": "payload",
        "sequence_number": 2,
    }
