from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final, Protocol, runtime_checkable
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from litellm._logging import verbose_logger
from litellm.responses.streaming_iterator import BaseResponsesAPIStreamingIterator
from litellm.types.llms.base import BaseLiteLLMOpenAIResponseObject
from litellm.types.llms.openai import (
    InputTokensDetails,
    OutputItemAddedEvent,
    OutputItemDoneEvent,
    OutputTokensDetails,
    ResponseAPIUsage,
    ResponseCompletedEvent,
    ResponseFailedEvent,
    ResponseIncompleteEvent,
    ResponsesAPIResponse,
    ResponsesAPIStreamEvents,
    ResponsesAPIStreamingResponse,
)
from litellm.types.responses.main import CompactionOutputItem

_HIDDEN_PARAMS_FIELD: Final = "_hidden_params"


def _detail_tokens(details: InputTokensDetails | OutputTokensDetails | None, name: str) -> int:
    value: Final[object] = getattr(details, name, None)
    return value if isinstance(value, int) else 0


def _aggregate(contributors: tuple[ResponseAPIUsage, ...]) -> ResponseAPIUsage:
    input_tokens: Final = sum(usage.input_tokens for usage in contributors)
    output_tokens: Final = sum(usage.output_tokens for usage in contributors)
    return ResponseAPIUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        input_tokens_details=InputTokensDetails.model_validate(
            MappingProxyType(
                {
                    name: sum(_detail_tokens(usage.input_tokens_details, name) for usage in contributors)
                    for name in ("cached_tokens", "cache_write_tokens")
                }
            )
        ),
        output_tokens_details=OutputTokensDetails(
            reasoning_tokens=sum(
                _detail_tokens(usage.output_tokens_details, "reasoning_tokens") for usage in contributors
            )
        ),
    )


class _ResponseState(BaseModel):
    model_config = ConfigDict(frozen=True, from_attributes=True)
    output: tuple[object, ...]
    hidden: Mapping[str, object] = Field(alias=_HIDDEN_PARAMS_FIELD)


@dataclass(frozen=True, slots=True)
class CompactionAccounting:
    usages: tuple[ResponseAPIUsage, ...] = ()
    item: CompactionOutputItem | None = None
    _marker: str = field(
        default_factory=lambda: f"_compaction_accounted_{uuid4().hex}", init=False, repr=False, compare=False
    )

    def finalize(self, response: ResponsesAPIResponse) -> ResponsesAPIResponse:
        if not self.usages and self.item is None:
            return response
        outward: Final = response.model_copy(deep=True)
        state: Final = _ResponseState.model_validate(outward)
        if state.hidden.get(self._marker):
            return outward
        if self.usages and outward.usage is None:
            verbose_logger.warning(
                "Compaction main usage unavailable; reported aggregate is a lower bound (status=%s)",
                response.status,
            )
        if self.usages:
            outward.usage = _aggregate((*((outward.usage,) if outward.usage is not None else ()), *self.usages))
        output: Final = [  # mutable-ok: Responses output is a list field
            *((self.item.model_dump(),) if self.item is not None else ()),
            *state.output,
        ]
        result: Final = outward.model_copy(update=MappingProxyType({"output": output}))
        metadata: Final = {**state.hidden, self._marker: True}  # mutable-ok: existing metadata consumers require a dict
        setattr(result, _HIDDEN_PARAMS_FIELD, metadata)
        return result


@runtime_checkable
class _SyncStream(Protocol):
    def __next__(self) -> ResponsesAPIStreamingResponse: ...


@runtime_checkable
class _AsyncStream(Protocol):
    async def __anext__(self) -> ResponsesAPIStreamingResponse: ...


@runtime_checkable
class _Closable(Protocol):
    def close(self) -> None: ...


@runtime_checkable
class _AsyncClosable(Protocol):
    async def aclose(self) -> None: ...


class CompactionResponsesStream(BaseResponsesAPIStreamingIterator):
    def __init__(self, upstream: BaseResponsesAPIStreamingIterator, accounting: CompactionAccounting) -> None:
        self._upstream = upstream
        self._accounting = accounting
        self._pending: tuple[ResponsesAPIStreamingResponse, ...] = ()
        self._inserted = False
        self._last_sequence = -1
        self._saw_terminal = False
        self._closed = False

    def __getattr__(self, name: str) -> object:
        value: Final[object] = getattr(self._upstream, name, None)
        if value is None and not hasattr(self._upstream, name):
            raise AttributeError(name)
        return value

    def __iter__(self) -> Iterator[ResponsesAPIStreamingResponse]:
        return self

    def __aiter__(self) -> AsyncIterator[ResponsesAPIStreamingResponse]:
        return self

    def __next__(self) -> ResponsesAPIStreamingResponse:
        if self._closed:
            raise StopIteration
        if self._pending:
            return self._pop_pending()
        try:
            if not isinstance(self._upstream, _SyncStream):
                raise TypeError("Upstream does not support synchronous iteration")
            event: Final = next(self._upstream)
            self._pending = self._process(event)
        except BaseException:
            self.close()
            raise
        return self._pop_pending()

    async def __anext__(self) -> ResponsesAPIStreamingResponse:
        if self._closed:
            raise StopAsyncIteration
        if self._pending:
            return self._pop_pending()
        try:
            if not isinstance(self._upstream, _AsyncStream):
                raise TypeError("Upstream does not support asynchronous iteration")
            event: Final = await anext(self._upstream)
            self._pending = self._process(event)
        except BaseException:
            await self.aclose()
            raise
        return self._pop_pending()

    def _pop_pending(self) -> ResponsesAPIStreamingResponse:
        event: Final = self._pending[0]
        self._pending = self._pending[1:]
        if isinstance(event, (ResponseCompletedEvent, ResponseIncompleteEvent, ResponseFailedEvent)):
            self._saw_terminal = True
            self.completed_response = event
        return event

    def _numbered(
        self, event: ResponsesAPIStreamingResponse, *, synthetic: bool = False
    ) -> ResponsesAPIStreamingResponse:
        source_sequence: Final[object] = getattr(event, "sequence_number", None)
        sequence: Final = max(
            self._last_sequence + 1,
            source_sequence + (2 if self._inserted and not synthetic else 0)
            if isinstance(source_sequence, int)
            else self._last_sequence + 1,
        )
        self._last_sequence = sequence
        output_index: Final[object] = getattr(event, "output_index", None)
        return event.model_copy(
            deep=True,
            update=MappingProxyType(
                {
                    "sequence_number": sequence,
                    **(
                        MappingProxyType({"output_index": output_index + 1})
                        if isinstance(output_index, int) and not synthetic
                        else MappingProxyType({})
                    ),
                }
            ),
        )

    def _process(self, event: ResponsesAPIStreamingResponse) -> tuple[ResponsesAPIStreamingResponse, ...]:
        item: Final = self._accounting.item
        kind: Final[object] = getattr(event, "type", None)
        inject: Final = (
            item is not None
            and not self._inserted
            and kind
            not in (
                ResponsesAPIStreamEvents.RESPONSE_CREATED,
                ResponsesAPIStreamEvents.RESPONSE_IN_PROGRESS,
            )
        )
        self._inserted = self._inserted or inject
        prefix: Final = (
            (
                self._numbered(
                    OutputItemAddedEvent(
                        type=ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED,
                        output_index=0,
                        item=BaseLiteLLMOpenAIResponseObject.model_validate(item.model_dump()),
                    ),
                    synthetic=True,
                ),
                self._numbered(
                    OutputItemDoneEvent(
                        type=ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE,
                        output_index=0,
                        item=BaseLiteLLMOpenAIResponseObject.model_validate(item.model_dump()),
                    ),
                    synthetic=True,
                ),
            )
            if inject and item is not None
            else ()
        )
        outward: Final = self._numbered(event) if item is not None else event
        if isinstance(outward, (ResponseCompletedEvent, ResponseIncompleteEvent, ResponseFailedEvent)):
            response: Final = self._accounting.finalize(outward.response)
            terminal: Final = (
                outward
                if response is outward.response
                else outward.model_copy(update=MappingProxyType({"response": response}))
            )
            return (*prefix, terminal)
        return (*prefix, outward)

    def _finish(self) -> None:
        self._closed = True
        if self._accounting.usages and not self._saw_terminal:
            verbose_logger.warning(
                "Compaction summary usage was incurred but the stream closed without a terminal response; "
                "outward usage settlement cannot be guaranteed"
            )

    def close(self) -> None:
        if self._closed:
            return
        self._finish()
        target: Final = (
            self._upstream if isinstance(self._upstream, _Closable) else getattr(self._upstream, "response", None)
        )
        if isinstance(target, _Closable):
            target.close()

    async def aclose(self) -> None:
        if self._closed:
            return
        self._finish()
        target: Final = (
            self._upstream if isinstance(self._upstream, _AsyncClosable) else getattr(self._upstream, "response", None)
        )
        if isinstance(target, _AsyncClosable):
            await target.aclose()
        elif isinstance(self._upstream, _Closable):
            self._upstream.close()
