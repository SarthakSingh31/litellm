import inspect
from collections.abc import Awaitable, Callable, Coroutine, Mapping
from types import MappingProxyType
from typing import Final, TypeAlias, cast  # noqa: TID251  # native binding selects a sync result or an async awaitable

from litellm.responses import main
from litellm.responses.streaming_iterator import BaseResponsesAPIStreamingIterator
from litellm.rust_bridge.catalog import Context, Delivery, Route
from litellm.rust_bridge.dispatch import PublicDispatch, call_hook
from litellm.rust_bridge.public_call import bind, optional_bool, optional_mapping, optional_str, signature
from litellm.rust_bridge.responses.entrypoints import (
    NATIVE_ARESPONSES,
    NATIVE_RESPONSES,
    LiteLLMResponsesRequest,
)
from litellm.types.llms.openai import ResponsesAPIResponse

__all__ = ("aresponses", "responses")

ResponsesResult: TypeAlias = ResponsesAPIResponse | BaseResponsesAPIStreamingIterator
PythonResponses: TypeAlias = Callable[..., ResponsesResult | Coroutine[object, object, ResponsesResult]]
PythonAresponses: TypeAlias = Callable[..., Awaitable[ResponsesResult]]


def _python_responses() -> PythonResponses:
    return cast(  # cast-ok: forward the original call shape through the Python @client decorator
        PythonResponses,
        main.responses,  # noqa: TID251  # dispatch boundary owns this Python fallback
    )


def _python_aresponses() -> PythonAresponses:
    return cast(  # cast-ok: forward the original call shape through the Python @client decorator
        PythonAresponses,
        main.aresponses,  # noqa: TID251  # dispatch boundary owns this Python fallback
    )


_PYTHON_RESPONSES: Final = _python_responses()
_RESPONSES: Final = signature(_PYTHON_RESPONSES)
_PYTHON_ARESPONSES: Final = _python_aresponses()
_ARESPONSES: Final = signature(_PYTHON_ARESPONSES)


def _public_request(
    legacy: inspect.Signature, args: tuple[object, ...], kwargs: Mapping[str, object]
) -> LiteLLMResponsesRequest | None:
    fields: Final = bind(legacy, args, kwargs)
    if fields is None:
        return None
    model: Final = fields.get("model")
    extra: Final = optional_mapping(fields.get("kwargs")) or MappingProxyType({})
    if not isinstance(model, str):
        return None
    return LiteLLMResponsesRequest(
        model=model,
        input=fields.get("input"),
        stream=optional_bool(fields.get("stream")),
        api_key=optional_str(extra.get("api_key")),
        api_base=optional_str(extra.get("api_base")) or optional_str(extra.get("base_url")),
        custom_llm_provider=optional_str(fields.get("custom_llm_provider")),
        extra_headers=optional_mapping(fields.get("extra_headers")),
        kwargs=extra,
    )


def _context(request: LiteLLMResponsesRequest) -> Context:
    return Context(
        Route.RESPONSES,
        provider=request.custom_llm_provider,
        model=request.model,
        delivery=Delivery.STREAMING if request.stream else Delivery.COMPLETED,
    )


_DISPATCH: Final = PublicDispatch(
    route=Route.RESPONSES,
    request=lambda args, kwargs: _public_request(_RESPONSES, args, kwargs),
    context=_context,
    bypass=lambda request: request.kwargs.get("aresponses") is True,
)

_ADISPATCH: Final = PublicDispatch(
    route=Route.RESPONSES,
    request=lambda args, kwargs: _public_request(_ARESPONSES, args, kwargs),
    context=_context,
)


def _compaction_request(
    legacy: inspect.Signature, args: tuple[object, ...], kwargs: Mapping[str, object]
) -> Mapping[str, object] | None:
    fields: Final = bind(legacy, args, kwargs)
    if fields is None:
        return None
    explicit: Final = MappingProxyType({key: value for key, value in fields.items() if key != "kwargs"})
    extra: Final = optional_mapping(fields.get("kwargs")) or MappingProxyType({})
    return MappingProxyType({**explicit, **extra})


def responses(
    *args: object,
    **kwargs: object,  # kwargs-ok: preserve the public Responses call shape
) -> ResponsesResult | Coroutine[object, object, ResponsesResult]:
    from litellm.litellm_core_utils.asyncify import run_async_function
    from litellm.responses.compaction import (
        COMPACTION_CHILD_KEY,
        COMPACTION_SESSION_KEY,
        CompactionSession,
        finish_compaction,
        prepare_compaction,
    )

    session: Final = CompactionSession()
    managed: Final = (
        isinstance(kwargs.get(COMPACTION_SESSION_KEY), CompactionSession) or kwargs.get(COMPACTION_CHILD_KEY) is True
    )
    request: Final = _compaction_request(_RESPONSES, args, kwargs)
    prepared: Final = (
        run_async_function(prepare_compaction, request, session, _execute_summary)
        if request is not None and not managed and kwargs.get("aresponses") is not True
        else None
    )
    changed: Final = prepared is not None and prepared is not request
    forwarded: Final = MappingProxyType(
        {
            key: value
            for key, value in (prepared if changed and prepared is not None else kwargs).items()
            if key not in (COMPACTION_SESSION_KEY, COMPACTION_CHILD_KEY)
        }
    )
    python: Final = _PYTHON_RESPONSES
    result: Final = _DISPATCH.run(
        () if changed else args,
        forwarded,
        python=python,
        binding=NATIVE_RESPONSES,
        native=call_hook,
    )
    if isinstance(result, (ResponsesAPIResponse, BaseResponsesAPIStreamingIterator)) and not managed:
        return finish_compaction(result, session)
    return result


async def _execute_summary(request: Mapping[str, object]) -> ResponsesResult:
    from litellm.responses.compaction import COMPACTION_CHILD_KEY

    return await _PYTHON_ARESPONSES(
        **MappingProxyType({key: value for key, value in request.items() if key != COMPACTION_CHILD_KEY})
    )


async def aresponses(*args: object, **kwargs: object) -> ResponsesResult:  # kwargs-ok: preserve the public call shape
    from litellm.responses.compaction import (
        COMPACTION_CHILD_KEY,
        COMPACTION_SESSION_KEY,
        CompactionSession,
        finish_compaction,
        prepare_compaction,
    )

    session: Final = CompactionSession()
    managed: Final = (
        isinstance(kwargs.get(COMPACTION_SESSION_KEY), CompactionSession) or kwargs.get(COMPACTION_CHILD_KEY) is True
    )
    request: Final = _compaction_request(_ARESPONSES, args, kwargs)
    prepared: Final = (
        await prepare_compaction(request, session, _execute_summary) if request is not None and not managed else None
    )
    changed: Final = prepared is not None and prepared is not request
    forwarded: Final = MappingProxyType(
        {
            key: value
            for key, value in (prepared if changed and prepared is not None else kwargs).items()
            if key not in (COMPACTION_SESSION_KEY, COMPACTION_CHILD_KEY)
        }
    )
    python: Final = _PYTHON_ARESPONSES
    result: Final = await _ADISPATCH.arun(
        () if changed else args,
        forwarded,
        python=python,
        binding=NATIVE_ARESPONSES,
        native=call_hook,
    )
    return result if managed else finish_compaction(result, session)


responses.__doc__ = _PYTHON_RESPONSES.__doc__
responses.__wrapped__ = _PYTHON_RESPONSES  # pyright: ignore[reportFunctionMemberAccess]  # inspect.signature follows __wrapped__ to the legacy signature
aresponses.__doc__ = _PYTHON_ARESPONSES.__doc__
aresponses.__wrapped__ = _PYTHON_ARESPONSES  # pyright: ignore[reportFunctionMemberAccess]  # inspect.signature follows __wrapped__ to the legacy signature
