import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial
from types import MappingProxyType
from typing import (
    Final,
    Protocol,
    TypeAlias,
    cast,  # noqa: TID251  # narrowing the serialized output of the SDK-validated input adapter
)

from pydantic import BaseModel, ConfigDict, Field, StrictInt, TypeAdapter, ValidationError

import litellm
from litellm._logging import verbose_logger
from litellm.litellm_core_utils.internal_call_metadata import (
    forwarded_internal_call_metadata,
    responses_compaction_history_metadata,
)
from litellm.llms.anthropic.experimental_pass_through.context_management.constants import (
    COMPACT_DEFAULT_INSTRUCTIONS,
    COMPACT_NO_TOOL_CALLS_SUFFIX,
    COMPACT_SAME_AS_REQUEST,
    COMPACT_SUMMARY_MAX_TOKENS,
    COMPACT_SUMMARY_MAX_TOKENS_SETTING_KEY,
    COMPACT_SUMMARY_MODEL_SETTING_KEY,
    COMPACT_SUMMARY_TIMEOUT_SECONDS,
)
from litellm.llms.anthropic.experimental_pass_through.context_management.editors.compact import (
    _extract_summary_text,  # pyright: ignore[reportPrivateUsage]  # reuse the existing summary parser
)
from litellm.responses.compaction_accounting import CompactionAccounting, CompactionResponsesStream
from litellm.responses.compaction_history import (
    create_compaction_item,
    has_gateway_compaction,
    replay_compaction_input,
    split_compaction_input,
)
from litellm.responses.litellm_completion_transformation.transformation import LiteLLMCompletionResponsesConfig
from litellm.responses.streaming_iterator import BaseResponsesAPIStreamingIterator
from litellm.types.llms.openai import (
    AllMessageValues,
    ResponseInputParam,
    ResponsesAPIOptionalRequestParams,
    ResponsesAPIResponse,
)
from litellm.types.utils import RESPONSES_COMPACTION_CALL_ORIGIN

COMPACTION_SESSION_KEY: Final = "_responses_compaction_session"
COMPACTION_CHILD_KEY: Final = "_responses_compaction_child"
NATIVE_COMPACTION_CAPABILITY: Final = "supports_native_compaction"
COMPACTION_TYPE: Final = "compaction"
_INPUT_ADAPTER: Final = TypeAdapter(ResponseInputParam)
_MAPPING_ADAPTER: Final = TypeAdapter(Mapping[str, object])
_ENTRIES_ADAPTER: Final = TypeAdapter(tuple[object, ...])
_OPTIONS_ADAPTER: Final = TypeAdapter(ResponsesAPIOptionalRequestParams)
_MESSAGES_ADAPTER: Final = TypeAdapter(list[AllMessageValues])
_SUMMARY_OMIT: Final = frozenset(
    (
        COMPACTION_SESSION_KEY,
        *(
            "input messages context_management previous_response_id tools tool_choice parallel_tool_calls max_tool_calls "
            "text text_format response_format stream stream_options background max_tokens max_completion_tokens "
            "max_output_tokens litellm_logging_obj litellm_call_id litellm_trace_id prompt_id prompt prompt_variables "
            "prompt_label prompt_version original_function num_retries fallbacks context_window_fallbacks "
            "content_policy_fallbacks proxy_server_request aresponses _async_prompt_merged_params functions function_call"
        ).split(),
    )
)
_HISTORY_OMIT: Final = (
    _SUMMARY_OMIT
    | frozenset(ResponsesAPIOptionalRequestParams.__annotations__)
    | frozenset(("model", "extra_body", "use_chat_completions_api", COMPACTION_CHILD_KEY))
)
_AMBIGUOUS_EXTRA: Final = frozenset(
    "model input messages instructions tools functions function_call tool_choice parallel_tool_calls "
    "context_management previous_response_id prompt prompt_id".split()
)
ResponsesResult: TypeAlias = ResponsesAPIResponse | BaseResponsesAPIStreamingIterator


class SummaryExecutor(Protocol):
    async def __call__(self, request: Mapping[str, object], /) -> ResponsesResult: ...


class _CompactionEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    type: str
    compact_threshold: StrictInt | None = Field(default=None, gt=0)


class _ProviderContext(BaseModel):
    model_config = ConfigDict(frozen=True)
    api_base: str | None = None
    base_url: str | None = None
    custom_llm_provider: str | None = None


class _HistoryPage(BaseModel):
    model_config = ConfigDict(frozen=True)
    data: ResponseInputParam
    has_more: bool = False
    last_id: str | None = None


class _ResponseMetadata(BaseModel):
    model_config = ConfigDict(frozen=True, from_attributes=True)
    hidden: Mapping[str, object] = Field(alias="_hidden_params")


@dataclass
class CompactionSession:
    locked: bool = False
    deployment_id: str | None = None
    accounting: CompactionAccounting = field(default_factory=CompactionAccounting)
    prepared: Mapping[str, object] | None = None
    identity: tuple[object, ...] | None = None
    defer_summary_lock: bool = False

    def restore(self, request: Mapping[str, object]) -> Mapping[str, object] | None:
        if self.locked and self.identity != _identity(request):
            raise ValueError("Cannot change model/provider/deployment after gateway compaction has executed")
        if self.prepared is None or (not self.locked and self.identity != _identity(request)):
            return None
        _validate_preparation(request)
        return MappingProxyType({**request, **self.prepared})

    def remember(self, request: Mapping[str, object], prepared: Mapping[str, object]) -> None:
        self.identity = _identity(request)
        self.prepared = prepared

    def lock(self, request: Mapping[str, object]) -> None:
        self.identity = _identity(request)
        self.locked = True

    def begin_summary(self, request: Mapping[str, object]) -> None:
        self.identity = _identity(request)
        if not self.defer_summary_lock:
            self.lock(request)

    def record(self, accounting: CompactionAccounting) -> None:
        self.accounting = accounting


@dataclass(frozen=True, slots=True)
class _EffectiveContext:
    input: ResponseInputParam
    trace_id: str | None = None


def _settings() -> Mapping[str, object]:
    proxy_server: Final = sys.modules.get("litellm.proxy.proxy_server")
    return _mapping(getattr(proxy_server, "general_settings", None))


def _mapping(value: object) -> Mapping[str, object]:
    return _MAPPING_ADAPTER.validate_python(value) if isinstance(value, Mapping) else MappingProxyType({})


def _without(request: Mapping[str, object], excluded: frozenset[str]) -> Mapping[str, object]:
    return MappingProxyType({key: value for key, value in request.items() if key not in excluded})


def _same_model_enabled() -> bool:
    selected: Final = _settings().get(COMPACT_SUMMARY_MODEL_SETTING_KEY)
    if selected is not None and (not isinstance(selected, str) or not selected.strip()):
        raise ValueError(f"{COMPACT_SUMMARY_MODEL_SETTING_KEY} must be a model alias or {COMPACT_SAME_AS_REQUEST}")
    return selected == COMPACT_SAME_AS_REQUEST


def _input(value: object) -> ResponseInputParam:
    validated: Final = _INPUT_ADAPTER.validate_python(
        (MappingProxyType({"role": "user", "content": value}),) if isinstance(value, str) else value
    )
    return cast(  # cast-ok: SDK validation establishes the schema; JSON-mode serialization materializes its lazy Iterable fields
        ResponseInputParam, _INPUT_ADAPTER.dump_python(validated, mode="json")
    )


def _model(request: Mapping[str, object]) -> str:
    value: Final = request.get("model")
    if not isinstance(value, str):
        raise ValueError("Responses compaction requires a selected model")  # noqa: TRY004  # request validation maps to BadRequestError
    return value


def _identity(request: Mapping[str, object]) -> tuple[object, ...]:
    return tuple(request.get(key) for key in ("model", "custom_llm_provider", "api_base", "base_url", "api_key"))


def _model_limits(request: Mapping[str, object]) -> tuple[int, int]:
    configured: Final = request.get("model_info")
    info: Final = _mapping(configured)
    provider: Final = request.get("custom_llm_provider")
    catalog: Final[Mapping[str, object]] = (
        MappingProxyType({})
        if info.get("max_input_tokens") and info.get("max_output_tokens")
        else _catalog_limits(_model(request), provider if isinstance(provider, str) else None)
    )
    input_limit: Final = info.get("max_input_tokens") or catalog.get("max_input_tokens")
    output_limit: Final = info.get("max_output_tokens") or catalog.get("max_output_tokens")
    if not isinstance(input_limit, int) or input_limit <= 0 or not isinstance(output_limit, int) or output_limit <= 0:
        raise ValueError("Responses compaction requires known positive max_input_tokens and max_output_tokens")
    return input_limit, output_limit


def _catalog_limits(model: str, provider: str | None) -> Mapping[str, object]:
    try:
        return _MAPPING_ADAPTER.validate_python(litellm.get_model_info(model, custom_llm_provider=provider))
    except Exception as exc:
        raise ValueError(
            "Responses compaction requires known model limits; configure model_info.max_input_tokens and max_output_tokens"
        ) from exc


def _token_count(model: str, *, messages: Sequence[object] | None = None, text: str | None = None) -> int:
    return litellm.token_counter(model=model, messages=messages, text=text)  # pyright: ignore[reportUnknownMemberType]  # unused legacy tokenizer parameters are untyped


def _count(request: Mapping[str, object], items: ResponseInputParam) -> int:
    options: Final = _OPTIONS_ADAPTER.validate_python(MappingProxyType({"instructions": request.get("instructions")}))
    messages: Final = LiteLLMCompletionResponsesConfig.transform_responses_api_input_to_messages(  # pyright: ignore[reportUnknownMemberType]  # legacy helper also accepts an untyped dict
        input=items, responses_api_request=options, replay_reasoning=True
    )
    tool_context: Final = _tool_context(request)
    return _token_count(_model(request), messages=messages) + (
        _token_count(_model(request), text=tool_context) if tool_context else 0
    )


def _tool_context(request: Mapping[str, object]) -> str:
    tools: Final = request.get("tools")
    functions: Final = request.get("functions")
    definitions: Final = (tools, functions) if tools and functions else tools or functions
    return json.dumps(definitions, separators=(",", ":")) if definitions else ""


def _child_callbacks(request: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType(
        {
            key: list(  # mutable-ok: SDK callback setup consumes each operation's list in place
                _ENTRIES_ADAPTER.validate_python(request[key])
            )
            for key in ("success_callback", "failure_callback")
            if isinstance(request.get(key), list)
        }
    )


def _history_connection(request: Mapping[str, object]) -> Mapping[str, object]:
    metadata: Final = MappingProxyType(
        {**_mapping(request.get("metadata")), **_mapping(request.get("litellm_metadata"))}
    )
    return MappingProxyType(
        {
            **_without(request, _HISTORY_OMIT),
            **_child_callbacks(request),
            "litellm_metadata": responses_compaction_history_metadata(metadata),
        }
    )


async def _native_history(
    request: Mapping[str, object], response_id: str, after: str | None = None
) -> ResponseInputParam:
    connection: Final = _history_connection(request)
    page: Final = _HistoryPage.model_validate(
        await partial(litellm.alist_input_items, **connection)(  # pyright: ignore[reportUnknownMemberType]  # native history endpoint returns an untyped dict, validated immediately
            response_id=response_id, order="asc", limit=100, after=after
        )
    )
    items: Final = _input(page.data)
    if not page.has_more:
        return items
    last_id: Final = page.last_id
    if not isinstance(last_id, str) or last_id == after:
        raise ValueError("Cannot restore complete Responses history: invalid pagination cursor")
    return _input((*items, *await _native_history(request, response_id, last_id)))


def _historical_input_item(item: object) -> object:
    fields: Final = _mapping(item)
    content: Final = fields.get("content")
    if fields.get("type") != "message" or fields.get("role") != "assistant" or not isinstance(content, (list, tuple)):
        return item
    parts: Final = _ENTRIES_ADAPTER.validate_python(content)
    return MappingProxyType(
        {
            **fields,
            "content": tuple(
                MappingProxyType({**part, "type": "input_text"})
                if (part := _mapping(content_part)).get("type") == "output_text"
                else content_part
                for content_part in parts
            ),
        }
    )


async def _effective_input(request: Mapping[str, object]) -> _EffectiveContext:
    current: Final = _input(request.get("input"))
    previous: Final = request.get("previous_response_id")
    if not isinstance(previous, str) or not previous:
        return _EffectiveContext(_input(replay_compaction_input(current)))
    from litellm.responses.main import (
        _will_bridge_to_chat_completions,  # pyright: ignore[reportPrivateUsage]  # share dispatch resolution without a startup import cycle
    )

    routing: Final = _ProviderContext.model_validate(request)
    provider: Final = routing.custom_llm_provider or litellm.get_llm_provider(_model(request))[1]
    api_base: Final = routing.api_base or routing.base_url
    bridged: Final = _will_bridge_to_chat_completions(
        _model(request), provider, request.get("use_chat_completions_api") is True, request.get("model_info"), api_base
    )
    if not bridged:
        history_request: Final = MappingProxyType({**request, "custom_llm_provider": provider})
        history: Final = await _native_history(history_request, previous)
        response: Final = await partial(
            litellm.aget_responses,  # pyright: ignore[reportUnknownMemberType]  # legacy endpoint has untyped provider kwargs
            **_history_connection(history_request),
        )(response_id=previous)
        payload: Final = _MAPPING_ADAPTER.validate_python(response.model_dump(exclude_none=True))
        output: Final = _input(payload["output"])
        return _EffectiveContext(_input(replay_compaction_input(_input((*history, *output, *current)))))
    from litellm.completion_extras.litellm_responses_transformation.transformation import (
        LiteLLMResponsesTransformationHandler,
    )
    from litellm.responses.litellm_completion_transformation.session_handler import ResponsesSessionHandler

    session: Final = await ResponsesSessionHandler.get_chat_completion_message_history_for_previous_response_id(
        previous, compaction_ancestry_only=True
    )
    messages: Final = session.get("messages")
    if not messages:
        raise ValueError(
            "Cannot compact previous_response_id without its complete stored history; replay the input instead"
        )
    normalized: Final = _MESSAGES_ADAPTER.validate_python(
        tuple(
            message.model_dump(exclude_none=True) if isinstance(message, BaseModel) else message for message in messages
        )
    )
    history_items, history_instructions = (
        LiteLLMResponsesTransformationHandler().convert_chat_completion_messages_to_responses_api(normalized)
    )
    instruction_items: Final = (
        (MappingProxyType({"role": "system", "content": history_instructions}),) if history_instructions else ()
    )
    history_input: Final = tuple(_historical_input_item(item) for item in history_items)
    return _EffectiveContext(
        _input(replay_compaction_input(_input((*instruction_items, *history_input, *current)))),
        session.get("litellm_session_id"),
    )


def _summary_request(
    request: Mapping[str, object], prefix: ResponseInputParam, max_tokens: int
) -> Mapping[str, object]:
    tool_context: Final = _tool_context(request)
    prompt: Final = (
        COMPACT_DEFAULT_INSTRUCTIONS
        + COMPACT_NO_TOOL_CALLS_SUFFIX
        + ("\nTool definitions (context only):\n" + tool_context if tool_context else "")
    )
    nested: Final = MappingProxyType(
        {
            key: forwarded_internal_call_metadata(
                _without(_mapping(request.get(key)), _SUMMARY_OMIT), RESPONSES_COMPACTION_CALL_ORIGIN
            )
            if key != "extra_body"
            else dict(  # mutable-ok: provider extra_body requires a concrete JSON object
                _without(_mapping(request.get(key)), _SUMMARY_OMIT)
            )
            for key in ("extra_body", "litellm_metadata", "metadata")
            if isinstance(request.get(key), Mapping)
        }
    )
    return MappingProxyType(
        {
            **_without(request, _SUMMARY_OMIT),
            **nested,
            **_child_callbacks(request),
            "input": _input((*prefix, MappingProxyType({"role": "user", "content": prompt}))),
            "stream": False,
            "max_output_tokens": max_tokens,
            "timeout": COMPACT_SUMMARY_TIMEOUT_SECONDS,
            COMPACTION_CHILD_KEY: True,
        }
    )


def _validate_preparation(request: Mapping[str, object]) -> None:
    if request.get("prompt_id") or request.get("prompt"):
        raise ValueError("Gateway compaction requires an expanded prompt; resolve stored prompts before this request")
    ambiguous: Final = _AMBIGUOUS_EXTRA.intersection(_mapping(request.get("extra_body")))
    if ambiguous:
        raise ValueError(
            "Gateway compaction does not support inference overrides in extra_body: " + ", ".join(sorted(ambiguous))
        )


async def _prepare_compaction(
    request: Mapping[str, object], session: CompactionSession, execute: SummaryExecutor
) -> Mapping[str, object]:
    if request.get(COMPACTION_CHILD_KEY) is True:
        return request
    restored: Final = session.restore(request)
    if restored is not None:
        return restored
    raw_entries: Final = request.get("context_management")
    entries: Final = _ENTRIES_ADAPTER.validate_python(raw_entries) if isinstance(raw_entries, list) else ()
    requested: Final = tuple(entry for entry in entries if _mapping(entry).get("type") == COMPACTION_TYPE)
    raw_input: Final = request.get("input")
    replay_request: Final = (
        MappingProxyType({**request, "input": _input(replay_compaction_input(_input(raw_input)))})
        if has_gateway_compaction(raw_input)
        else request
    )
    if not requested or not _same_model_enabled():
        return replay_request
    info: Final = _mapping(request.get("model_info"))
    capability: Final = info.get(NATIVE_COMPACTION_CAPABILITY)
    if capability is not None and not isinstance(capability, bool):
        raise ValueError(f"model_info.{NATIVE_COMPACTION_CAPABILITY} must be a boolean")
    if capability is True:
        return replay_request
    if len(requested) != 1:
        raise ValueError("At most one gateway compaction entry is supported")
    entry: Final = _CompactionEntry.model_validate(requested[0])
    _validate_preparation(request)
    input_limit, output_limit = _model_limits(request)
    context: Final = await _effective_input(request)
    effective: Final = context.input
    remaining_entries: Final = [  # mutable-ok: context_management is a JSON list
        value for value in entries if value is not requested[0]
    ]
    prepared: Final = MappingProxyType(
        {
            "input": effective,
            "previous_response_id": None,
            "context_management": remaining_entries or None,
            **(
                MappingProxyType({"litellm_trace_id": context.trace_id})
                if context.trace_id is not None
                else MappingProxyType({})
            ),
        }
    )
    threshold: Final = entry.compact_threshold if entry.compact_threshold is not None else input_limit * 9 // 10
    if _count(request, effective) < threshold:
        session.remember(request, prepared)
        return MappingProxyType({**request, **prepared})
    prefix, retained = split_compaction_input(effective)
    if not prefix:
        raise ValueError("No completed history is available to compact while preserving the active user task")
    configured_budget: Final = _settings().get(COMPACT_SUMMARY_MAX_TOKENS_SETTING_KEY, COMPACT_SUMMARY_MAX_TOKENS)
    if type(configured_budget) is not int or configured_budget <= 0:
        raise ValueError(f"{COMPACT_SUMMARY_MAX_TOKENS_SETTING_KEY} must be a positive integer")
    instruction_items: Final = tuple(item for item in retained if item.get("role") in ("system", "developer"))
    desired_budget: Final = min(configured_budget, output_limit)
    candidate: Final = _summary_request(request, _input((*instruction_items, *prefix)), desired_budget)
    available: Final = input_limit - _count(candidate, _input(candidate["input"]))
    if available <= 0:
        raise ValueError("Compaction summary prompt exceeds the selected model context window")
    summary_request: Final = MappingProxyType({**candidate, "max_output_tokens": min(desired_budget, available)})
    session.begin_summary(request)
    summary_response: Final = await execute(summary_request)
    if not isinstance(summary_response, ResponsesAPIResponse):
        raise ValueError("Compaction summary must return a non-streaming Responses response")  # noqa: TRY004  # response validation maps to BadRequestError
    if summary_response.usage is None:
        verbose_logger.error("Responses compaction executed without reported usage; complete billing is unavailable")
        raise ValueError("Compaction summary did not report usage; refusing incomplete aggregate billing")
    metadata: Final = _ResponseMetadata.model_validate(summary_response)
    measured: Final = () if metadata.hidden.get("cache_hit") else (summary_response.usage.model_copy(deep=True),)
    session.record(CompactionAccounting(usages=measured))
    summary: Final = (
        _extract_summary_text(summary_response.output_text) if summary_response.status == "completed" else None
    )
    if not summary:
        verbose_logger.warning("Responses compaction summary extraction failed; retaining context and measured usage")
        session.remember(request, prepared)
        return MappingProxyType({**request, **prepared})
    item: Final = create_compaction_item(summary, retained)
    artifact_input: Final = _input((item.model_dump(),))
    compacted: Final = _input(replay_compaction_input(artifact_input))
    session.record(CompactionAccounting(usages=measured, item=item))
    proxy_request: Final = _mapping(request.get("proxy_server_request"))
    body: Final = _mapping(proxy_request.get("body"))
    logged_input: Final[Mapping[str, object]] = (
        MappingProxyType(
            {
                "proxy_server_request": {  # mutable-ok: proxy logging consumes a concrete JSON request
                    **proxy_request,
                    "body": {**body, "input": artifact_input},  # mutable-ok: request body is a JSON object
                }
            }
        )
        if proxy_request and body
        else MappingProxyType({})
    )
    final_prepared: Final = MappingProxyType({**prepared, "input": compacted, **logged_input})
    session.remember(request, final_prepared)
    return MappingProxyType({**request, **final_prepared})


async def prepare_compaction(
    request: Mapping[str, object], session: CompactionSession, execute: SummaryExecutor
) -> Mapping[str, object]:
    try:
        return await _prepare_compaction(request, session, execute)
    except ValueError as exc:
        message: Final = (
            "Invalid Responses compaction request configuration" if isinstance(exc, ValidationError) else str(exc)
        )
        model: Final = request.get("model")
        provider: Final = request.get("custom_llm_provider")
        raise litellm.BadRequestError(
            message=message,
            model=model if isinstance(model, str) else "",
            llm_provider=provider if isinstance(provider, str) else "",
        ) from None


def finish_compaction(result: ResponsesResult, session: CompactionSession) -> ResponsesResult:
    if not session.accounting.usages and session.accounting.item is None:
        return result
    if isinstance(result, ResponsesAPIResponse):
        return session.accounting.finalize(result)
    return CompactionResponsesStream(result, session.accounting)
