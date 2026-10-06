import base64
from collections.abc import Mapping, Sequence
from functools import reduce
from itertools import groupby
from typing import (
    Annotated,
    Final,
    cast,  # noqa: TID251  # SDK Iterable fields are materialized after schema validation
)
from uuid import uuid4

from openai.types.responses.response_input_param import ResponseInputItemParam, ResponseInputParam
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator

from litellm.responses.utils import ResponsesAPIRequestUtils
from litellm.types.responses.main import CompactionOutputItem

GATEWAY_COMPACTION_PREFIX: Final = "litellm_compaction:"
_PAYLOAD_VERSION: Final = 1
_SUMMARY_PREFIX: Final = "Previous conversation summary (serialized by LiteLLM):\n"
_USER_ROLE: Final = "user"
_ASSISTANT_ROLE: Final = "assistant"
_REASONING_TYPE: Final = "reasoning"
_TOOL_CALL_SUFFIX: Final = "_call"
_INSTRUCTION_ROLES: Final = frozenset(("system", "developer"))
_COMPACTION_TYPE: Final = "compaction"
_TOOL_PAIRS: Final = (
    ("function_call", "function_call_output"),
    ("custom_tool_call", "custom_tool_call_output"),
    ("computer_call", "computer_call_output"),
    ("local_shell_call", "local_shell_call_output"),
    ("shell_call", "shell_call_output"),
    ("apply_patch_call", "apply_patch_call_output"),
    ("tool_search_call", "tool_search_output"),
)
_TOOL_CALL_TYPES: Final = frozenset(call for call, _ in _TOOL_PAIRS)
_TOOL_OUTPUT_TYPES: Final = frozenset(output for _, output in _TOOL_PAIRS)
_RETAINED_INPUT_ADAPTER: Final = TypeAdapter(ResponseInputParam)


def _gateway_content(item: Mapping[str, object]) -> str | None:
    if item.get("type") != _COMPACTION_TYPE:
        return None
    content: Final = item.get("encrypted_content")
    if not isinstance(content, str):
        return None
    _, unwrapped = ResponsesAPIRequestUtils._unwrap_encrypted_content_with_model_id(  # pyright: ignore[reportPrivateUsage]  # existing affinity-envelope decoder
        content
    )
    return unwrapped if unwrapped.startswith(GATEWAY_COMPACTION_PREFIX) else None


def has_gateway_compaction(input: object) -> bool:
    """Recognize gateway artifacts without validating or normalizing native/vendor input."""
    if not isinstance(input, (list, tuple)):
        return False
    items: Final = cast(Sequence[object], input)  # cast-ok: list/tuple check establishes a sequence of opaque values
    return any(
        _gateway_content(cast(Mapping[str, object], item)) is not None  # cast-ok: checked mapping, read-only lookup
        for item in items
        if isinstance(item, Mapping)
    )


class _CompactionPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    version: Annotated[int, Field(strict=True, ge=_PAYLOAD_VERSION, le=_PAYLOAD_VERSION)]
    summary: Annotated[str, Field(min_length=1)]
    retained_input: ResponseInputParam

    @field_validator("retained_input")
    @classmethod
    def no_nested_gateway_artifacts(cls, value: ResponseInputParam) -> ResponseInputParam:
        if any(_gateway_content(item) is not None for item in value):
            raise ValueError("Retained input must replay gateway compaction artifacts before being stored")
        return cast(  # cast-ok: preserve validated SDK schema while materializing Iterable fields as JSON lists
            ResponseInputParam, _RETAINED_INPUT_ADAPTER.dump_python(value, mode="json")
        )


def create_compaction_item(summary: str, retained_input: ResponseInputParam) -> CompactionOutputItem:
    """Serialize a replayable artifact. The payload is not encrypted or authenticated."""
    payload: Final = _CompactionPayload(version=_PAYLOAD_VERSION, summary=summary, retained_input=retained_input)
    encoded: Final = base64.b64encode(payload.model_dump_json().encode("utf-8")).decode("ascii")
    return CompactionOutputItem(
        type=_COMPACTION_TYPE,
        id=f"cmp_{uuid4().hex}",
        encrypted_content=f"{GATEWAY_COMPACTION_PREFIX}{encoded}",
    )


def _decode(item: ResponseInputItemParam) -> _CompactionPayload | None:
    content: Final = _gateway_content(item)
    if content is None:
        return None
    serialized: Final = base64.b64decode(content[len(GATEWAY_COMPACTION_PREFIX) :], validate=True)
    return _CompactionPayload.model_validate_json(serialized)


def _must_retain(item: ResponseInputItemParam) -> bool:
    return item.get("role") in _INSTRUCTION_ROLES or (
        item.get("type") == _COMPACTION_TYPE and _gateway_content(item) is None
    )


def replay_compaction_input(input: str | ResponseInputParam) -> str | ResponseInputParam:
    """Replace summarized history using the latest gateway artifact, without decoding native artifacts."""
    if isinstance(input, str):
        return input
    artifacts: Final = tuple((index, payload) for index, item in enumerate(input) if (payload := _decode(item)))
    if not artifacts:
        return input
    index, latest = artifacts[-1]
    instructions: Final = tuple(
        item for item in input[:index] if _must_retain(item) and item not in latest.retained_input
    )
    summary: Final[ResponseInputItemParam] = {"role": _USER_ROLE, "content": f"{_SUMMARY_PREFIX}{latest.summary}"}
    return [  # mutable-ok: Responses input is a JSON list
        *instructions,
        summary,
        *latest.retained_input,
        *input[index + 1 :],
    ]


def _tool_key(item: ResponseInputItemParam) -> str | None:
    if item.get("type") not in _TOOL_CALL_TYPES | _TOOL_OUTPUT_TYPES:
        return None
    identifier: Final = item.get("call_id", item.get("id"))
    return identifier if isinstance(identifier, str) else None


def _is_assistant_action(item: ResponseInputItemParam) -> bool:
    kind: Final = item.get("type")
    return (
        item.get("role") == _ASSISTANT_ROLE
        or kind == _REASONING_TYPE
        or (isinstance(kind, str) and kind.endswith(_TOOL_CALL_SUFFIX))
    )


def _assistant_action_spans(input: ResponseInputParam) -> tuple[tuple[int, int], ...]:
    groups: Final = tuple(
        tuple(items)
        for is_action, items in groupby(
            ((index, item) for index, item in enumerate(input) if not _must_retain(item)),
            key=lambda indexed: _is_assistant_action(indexed[1]),
        )
        if is_action
    )
    return tuple((items[0][0], items[-1][0]) for items in groups)


def split_compaction_input(input: ResponseInputParam) -> tuple[ResponseInputParam, ResponseInputParam]:
    """Keep instructions, the current user task and latest complete action group; summarize all older exchanges."""
    effective: Final = replay_compaction_input(input)
    assert not isinstance(effective, str)
    latest_user: Final = next(
        (index for index in range(len(effective) - 1, -1, -1) if effective[index].get("role") == _USER_ROLE), None
    )
    if latest_user is None:
        return [], effective  # mutable-ok: Responses input is a JSON list
    action_spans: Final = _assistant_action_spans(effective)
    newest_boundary: Final = max(latest_user, action_spans[-1][0] if action_spans else latest_user)
    tool_entries: Final = sorted(
        (key, index, item.get("type") in _TOOL_OUTPUT_TYPES)
        for index, item in enumerate(effective)
        if (key := _tool_key(item)) is not None
    )
    groups: Final = tuple(tuple(entries) for _, entries in groupby(tool_entries, key=lambda entry: entry[0]))
    spans: Final = tuple((min(entry[1] for entry in group), max(entry[1] for entry in group)) for group in groups)
    unresolved: Final = tuple(min(entry[1] for entry in group) for group in groups if not any(e[2] for e in group))
    boundary: Final = reduce(
        lambda start, span: min(start, span[0]) if span[0] < start <= span[1] else start,
        sorted((*spans, *action_spans), key=lambda span: span[1], reverse=True),
        min((newest_boundary, *unresolved)),
    )
    if not any(
        index < boundary and index != latest_user and not _must_retain(item) for index, item in enumerate(effective)
    ):
        return [], effective  # mutable-ok: no history is discarded by summarizing the retained task alone
    return (
        [  # mutable-ok: Responses input is a JSON list
            item for index, item in enumerate(effective) if index < boundary and not _must_retain(item)
        ],
        [  # mutable-ok: Responses input is a JSON list
            item
            for index, item in enumerate(effective)
            if index >= boundary or index == latest_user or _must_retain(item)
        ],
    )
