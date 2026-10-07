"""
This hook is used to inject cache control directives into messages.

Users can define
- `cache_control_injection_points` in the completion params and litellm will inject the cache control directives into the messages at the specified injection points.

Supported for both `v1/chat/completions` (via the prompt-management hook) and
`v1/messages` (via `apply_to_anthropic_messages_request`).

"""

import copy
import itertools
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from functools import reduce
from typing import TYPE_CHECKING, Any, Final, cast
from urllib.parse import urlparse

from pydantic import TypeAdapter, ValidationError

from litellm._logging import verbose_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.custom_prompt_management import CustomPromptManagement
from litellm.integrations.prompt_management_base import PromptManagementClient
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    is_unsignable_thinking_block,
    with_prompt_cache_breakpoint,
)
from litellm.litellm_core_utils.prompt_templates.factory import (
    anthropic_surviving_tool_call_indices,
    find_anthropic_server_tool_result,
)
from litellm.llms.anthropic.common_utils import (
    is_claude_code_one_shot_subagent_request,
    supports_anthropic_cache_control,
    tool_call_is_rebuilt_as_server_tool_use,
)
from litellm.types.integrations.anthropic_cache_control_hook import (
    GATEWAY_INJECTED_CACHE_METADATA_KEY,
    GATEWAY_INJECTED_FOR_EVERY_DEPLOYMENT,
    CacheControlInjectionPoint,
    CacheControlMessageInjectionPoint,
)
from litellm.types.llms.anthropic import (
    ANTHROPIC_TOOL_SEARCH_TOOL_TYPES,
    AllAnthropicToolsValues,
    AnthropicSystemMessageContent,
)
from litellm.types.llms.openai import (
    AllMessageValues,
    ChatCompletionCachedContent,
    ChatCompletionTextObject,
    ChatCompletionToolParam,
    PromptCacheBreakpoint,
    PromptCacheOptions,
)
from litellm.types.prompts.init_prompts import PromptSpec
from litellm.types.utils import ChatCompletionMessageToolCall, Message, StandardCallbackDynamicParams

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


# Anthropic (and Bedrock Claude) reject requests with more than 4 cache_control
# breakpoints: "A maximum of 4 blocks with cache_control may be provided."
MAX_CACHE_CONTROL_BLOCKS: Final = 4

CACHE_BREAKPOINT_KEYS: Final = ("cache_control", "prompt_cache_breakpoint")
OPENAI_PROMPT_CACHE_BREAKPOINT_MIN_GPT_VERSION: Final = (5, 6)
_GPT_VERSION_PATTERN: Final = re.compile(r"^gpt-(\d+)(?:\.(\d+))?")
OPENAI_PROMPT_CACHE_BREAKPOINT_BLOCK_TYPES: Final = frozenset(
    {"text", "image", "image_url", "file", "input_audio", "input_text", "input_image", "input_file"}
)
OPENAI_API_HOST: Final = "api.openai.com"
OPENAI_API_BASE_ENV_VARS: Final = ("OPENAI_BASE_URL", "OPENAI_API_BASE")
_OBJECT_MAPPING_ADAPTER: Final = TypeAdapter(dict[object, object])
_OBJECT_LIST_ADAPTER: Final = TypeAdapter(list[object])
_OBJECT_SEQUENCE_ADAPTER: Final = TypeAdapter(tuple[object, ...])
_POINTS_MAPPING_ADAPTER: Final = TypeAdapter(tuple[Mapping[str, object], ...])

AllToolParamValues = ChatCompletionToolParam | AllAnthropicToolsValues


def _validated_object_mapping(value: object) -> dict[object, object] | None:
    try:
        return _OBJECT_MAPPING_ADAPTER.validate_python(value)
    except ValidationError:
        return None


def _validated_object_list(value: object) -> list[object] | None:
    try:
        return _OBJECT_LIST_ADAPTER.validate_python(value)
    except ValidationError:
        return None


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _provider_cache_params(params: Mapping[str, object]) -> Mapping[str, object]:
    return {
        key: value
        for key, value in params.items()
        if key not in (ANTHROPIC_RESPONSES_CACHE_SCOPE, RESPONSES_CACHE_PROVIDER)
        and (key != "cache_control_injection_points" or bool(value))
    }


def _annotate_cache_control(recipient: object, control: ChatCompletionCachedContent) -> None:
    if isinstance(recipient, (Message, ChatCompletionMessageToolCall)):
        setattr(recipient, "cache_control", control)
        return
    if not isinstance(recipient, dict):
        return
    writable: Final = cast(  # cast-ok: runtime dictionary; the helper's legacy API annotates in place
        dict[object, object], recipient
    )
    writable["cache_control"] = control


def supports_openai_prompt_cache_breakpoint(model: str) -> bool:
    model_map_flag: Final = _model_map_prompt_cache_breakpoint_flag(model)
    if model_map_flag is not None:
        return model_map_flag
    version_match: Final = _GPT_VERSION_PATTERN.match(model.rsplit("/", 1)[-1].lower())
    if version_match is None:
        return False
    version: Final = (int(version_match.group(1)), int(version_match.group(2) or 0))
    return version >= OPENAI_PROMPT_CACHE_BREAKPOINT_MIN_GPT_VERSION


def _model_map_prompt_cache_breakpoint_flag(model: str) -> bool | None:
    import litellm

    entries: Final = (litellm.model_cost.get(key) for key in (model, model.rsplit("/", 1)[-1]))
    flags: Final = (entry.get("supports_prompt_cache_breakpoint") for entry in entries if isinstance(entry, dict))
    return next((bool(flag) for flag in flags if flag is not None), None)


def targets_openai_api(api_base: object) -> bool:
    import litellm

    resolved: Final = next(
        (value for value in (api_base, litellm.api_base, *map(os.getenv, OPENAI_API_BASE_ENV_VARS)) if value),
        None,
    )
    if not isinstance(resolved, str):
        return True
    host: Final = urlparse(resolved).hostname
    return host is not None and (host == OPENAI_API_HOST or host.endswith(f".{OPENAI_API_HOST}"))


def _carries_cache_breakpoint(block: object) -> bool:
    return any(_attribute_or_key(block, key) is not None for key in CACHE_BREAKPOINT_KEYS)


def _attribute_or_key(value: object, key: str) -> object | None:
    if hasattr(value, key):
        return getattr(value, key)
    if isinstance(value, Mapping):
        return value.get(key)
    return None


def _as_object_list(value: object | None) -> list[object] | None:
    if not isinstance(value, list):
        return None
    return _validated_object_list(value)


def _tool_call_carries_cache_breakpoint(tool_call: object, message: object) -> bool:
    if _attribute_or_key(tool_call, "cache_control") is None:
        return False

    return not tool_call_is_rebuilt_as_server_tool_use(
        _attribute_or_key(tool_call, "id"), _attribute_or_key(message, "provider_specific_fields")
    )


def _tool_carries_cache_breakpoint(tool: object) -> bool:
    return _carries_cache_breakpoint(tool) or (
        isinstance(tool, dict) and _carries_cache_breakpoint(tool.get("function"))
    )


def _chat_transform_drops_tool_cache_control(tool: object) -> bool:
    return isinstance(tool, dict) and tool.get("type") in ANTHROPIC_TOOL_SEARCH_TOOL_TYPES


def _accepts_prompt_cache_breakpoint(block: object) -> bool:
    return isinstance(block, dict) and block.get("type") in OPENAI_PROMPT_CACHE_BREAKPOINT_BLOCK_TYPES


# Set by a caller whose message list is not the one that goes upstream -- today the
# Responses API layer, whose `instructions` only becomes a system message further down.
# Tells this hook to hand role-targeted points to the pass holding the final messages
# rather than spending them on a list that is still missing some of their targets.
CARRY_UNMATCHED_MESSAGE_POINTS: Final = "_litellm_carry_unmatched_cache_control_points"

EXTERNAL_BREAKPOINTS_STAMP: Final = "_litellm_external_breakpoints"
ANTHROPIC_RESPONSES_CACHE_SCOPE: Final = "_litellm_anthropic_responses_cache_scope"
RESPONSES_CACHE_PROVIDER: Final = "_litellm_responses_cache_provider"
_PREFIX_CONTROLS_STAMP: Final = "_litellm_cache_prefix_controls"
_AUTOMATIC_CONTROL_STAMP: Final = "_litellm_cache_automatic_control"
_COVERAGE_STAMP: Final = "_litellm_cache_coverage_skip"


def _anthropic_cacheable_block(block: object, role: object, native_messages: bool = False) -> bool:
    mapping: Final = _validated_object_mapping(block)
    if mapping is None:
        return False
    block_type: Final = mapping.get("type")
    if block_type == "text":
        text: Final = mapping.get("text")
        return isinstance(text, str) and bool(text.strip())
    if block_type in ("tool_use", "tool_result"):
        return native_messages
    if role in ("assistant", "system"):
        return False
    if block_type == "image":
        return native_messages
    if block_type in ("image_url", "document"):
        return True
    if block_type == "file":
        file: Final = _validated_object_mapping(mapping.get("file")) or {}
        file_id: Final = file.get("file_id")
        file_format: Final = file.get("format")
        return bool(file.get("file_data")) or (
            isinstance(file_id, str)
            and (
                file_id.startswith("http")
                or file_format
                in (
                    "application/pdf",
                    "text/plain",
                    "document",
                    "document_url",
                    "image/jpeg",
                    "image/png",
                    "image/gif",
                    "image/webp",
                )
            )
        )
    return False


def _message_cache_recipients(
    messages: Sequence[object], index: int, native_messages: bool = False
) -> tuple[tuple[str, int], ...]:
    message: Final = messages[index]
    content: Final = _attribute_or_key(message, "content")
    role: Final = _attribute_or_key(message, "role")
    if role in ("tool", "function"):
        return (("message", 0),) if isinstance(content, (str, list)) else ()
    ordinary: Final = (
        (("message", 0),)
        if isinstance(content, str) and content.strip()
        else tuple(
            ("content", i)
            for i, block in enumerate(_OBJECT_SEQUENCE_ADAPTER.validate_python(content))
            if _anthropic_cacheable_block(block, role, native_messages)
        )
        if isinstance(content, list)
        else ()
    )
    calls: Final = tuple(("call", i) for i in anthropic_surviving_tool_call_indices(messages, index))
    return (*ordinary, *calls)


def _recipient_control(message: object, recipient: tuple[str, int]) -> object:
    kind, index = recipient
    if kind == "message":
        return _attribute_or_key(message, "cache_control")
    values: Final = _as_object_list(_attribute_or_key(message, "tool_calls" if kind == "call" else "content")) or []
    return _attribute_or_key(values[index], "cache_control")


def _message_controls(messages: Sequence[object], index: int, native_messages: bool = False) -> tuple[object, ...]:
    return tuple(
        control
        for recipient in _message_cache_recipients(messages, index, native_messages)
        if (control := _recipient_control(messages[index], recipient)) is not None
    )


def _ordered_message_controls(messages: Sequence[object]) -> tuple[object, ...]:
    return tuple(itertools.chain.from_iterable(_message_controls(messages, index) for index in range(len(messages))))


def _contains_cache_control(value: object) -> bool:
    mapping: Final = _validated_object_mapping(value)
    if mapping is not None:
        return mapping.get("cache_control") is not None or any(
            _contains_cache_control(child) for child in mapping.values()
        )
    return isinstance(value, (list, tuple)) and any(
        _contains_cache_control(child) for child in _OBJECT_SEQUENCE_ADAPTER.validate_python(value)
    )


def _message_coverage_skip(message: object, automatic: bool = False) -> bool:
    mapping: Final = _validated_object_mapping(message) or {}
    role: Final = mapping.get("role")
    content: Final = mapping.get("content")
    fields: Final = _validated_object_mapping(mapping.get("provider_specific_fields")) or {}
    if role == "assistant" and (
        _contains_cache_control(fields.get("compaction_blocks"))
        or automatic
        and (bool(fields.get("compaction_blocks")) or mapping.get("function_call") is not None)
    ):
        return True
    calls: Final = _validated_object_list(mapping.get("tool_calls")) or []
    if role == "assistant" and any(_server_call_coverage_skip(call, fields, automatic) for call in calls):
        return True
    thinking: Final = _validated_object_list(mapping.get("thinking_blocks")) or []
    if any(_contains_cache_control(block) for block in thinking if not is_unsignable_thinking_block(block)):
        return True
    blocks: Final = _validated_object_list(content) or []
    if role in ("user", "assistant", "system") and (
        isinstance(content, str)
        and not content.strip()
        and mapping.get("cache_control") is not None
        or any(_empty_marked_text(block) for block in blocks)
    ):
        return True
    return any(_block_coverage_skip(block, role, automatic) for block in blocks)


def _empty_marked_text(block: object) -> bool:
    mapping: Final = _validated_object_mapping(block) or {}
    text: Final = mapping.get("text")
    return (
        mapping.get("type") == "text"
        and (not isinstance(text, str) or not text.strip())
        and mapping.get("cache_control") is not None
    )


def _server_call_coverage_skip(call: object, fields: Mapping[object, object], automatic: bool = False) -> bool:
    mapping: Final = _validated_object_mapping(call) or {}
    call_id: Final = mapping.get("id")
    if mapping.get("type") != "function" or not isinstance(call_id, str):
        return False
    result: Final = find_anthropic_server_tool_result(
        call_id,
        _validated_object_list(fields.get("web_search_results")),
        _validated_object_list(fields.get("tool_results")),
    )
    return result is not None and (automatic or _contains_cache_control(result))


def _block_coverage_skip(block: object, role: object, automatic: bool = False) -> bool:
    mapping: Final = _validated_object_mapping(block) or {}
    block_type: Final = mapping.get("type")
    if role == "assistant" and block_type == "thinking":
        return not is_unsignable_thinking_block(block) and _contains_cache_control(block)
    if isinstance(block_type, str) and (block_type == "server_tool_use" or block_type.endswith("_tool_result")):
        return automatic or _contains_cache_control(block)
    if block_type == "tool_result":
        return _contains_cache_control(mapping.get("content"))
    if role in ("tool", "function") and block_type in ("text", "image_url", "file"):
        return _contains_cache_control(block)
    return False


def _transcript_coverage_skip(messages: Sequence[object], automatic_control: object = None) -> bool:
    leading_end: Final = next(
        (index for index, message in enumerate(messages) if _attribute_or_key(message, "role") != "system"),
        len(messages),
    )
    return any(_attribute_or_key(message, "role") == "system" for message in messages[leading_end:]) or any(
        _message_coverage_skip(message, automatic_control is not None) for message in messages
    )


def _ttl_duration(control: object) -> int:
    return 3600 if _attribute_or_key(control, "ttl") == "1h" else 300


def _ttl_order_valid(controls: Sequence[object]) -> bool:
    return all(_ttl_duration(left) >= _ttl_duration(right) for left, right in zip(controls, controls[1:]))


def _scoped_envelope_facts(
    tools: Iterable[object] | None, params: object, request_kwargs: object
) -> tuple[tuple[object, ...], object, bool]:
    import litellm

    envelope: Final = {**(_validated_object_mapping(request_kwargs) or {}), **(_validated_object_mapping(params) or {})}
    extra: Final = _validated_object_mapping(envelope.get("extra_body")) or {}
    effective_tools: Final = _validated_object_list(extra.get("tools")) if "tools" in extra else tools
    automatic: Final = extra.get("cache_control", envelope.get("cache_control"))
    controls: Final = tuple(
        control
        for tool in effective_tools or ()
        if (
            control := _attribute_or_key(tool, "cache_control")
            or _attribute_or_key(_attribute_or_key(tool, "function"), "cache_control")
        )
        is not None
    )
    uncovered_extra: Final = any(
        _contains_cache_control(value) for key, value in extra.items() if key not in ("cache_control", "tools")
    )
    coverage_skip: Final = (
        litellm.modify_params is True
        or envelope.get("modify_params") is True
        or "messages" in extra
        or "system" in extra
        or envelope.get("system") is not None
        or uncovered_extra
        or ("tools" in extra and effective_tools is None)
    )
    return controls, automatic, coverage_skip


class AnthropicCacheControlHook(CustomPromptManagement):
    @staticmethod
    def _request_value(request_kwargs: object, key: str) -> object:
        request_mapping: Final = _validated_object_mapping(request_kwargs)
        if request_mapping is None:
            return None
        return request_mapping.get(key)

    @staticmethod
    def _request_user_agent(request_kwargs: object) -> str | None:
        proxy_server_request: Final = AnthropicCacheControlHook._request_value(request_kwargs, "proxy_server_request")
        proxy_server_request_mapping: Final = _validated_object_mapping(proxy_server_request)
        if proxy_server_request_mapping is None:
            return None
        headers: Final = proxy_server_request_mapping.get("headers")
        headers_mapping: Final = _validated_object_mapping(headers)
        if headers_mapping is None:
            return None
        user_agent: Final = next(
            (value for key, value in headers_mapping.items() if isinstance(key, str) and key.lower() == "user-agent"),
            None,
        )
        return user_agent if isinstance(user_agent, str) else None

    @staticmethod
    def _request_system(request_kwargs: object) -> str | list[object] | None:
        system: Final = AnthropicCacheControlHook._request_value(request_kwargs, "system")
        if isinstance(system, str):
            return system
        return _validated_object_list(system)

    def get_chat_completion_prompt(
        self,
        model: str,
        messages: list[AllMessageValues],
        non_default_params: dict[str, object],
        prompt_id: str | None,
        prompt_variables: dict | None,
        dynamic_callback_params: StandardCallbackDynamicParams,
        prompt_spec: PromptSpec | None = None,
        prompt_label: str | None = None,
        prompt_version: int | None = None,
        ignore_prompt_manager_model: bool | None = False,
        ignore_prompt_manager_optional_params: bool | None = False,
    ) -> tuple[str, list[AllMessageValues], dict[str, object]]:
        """
        Apply cache control directives based on specified injection points.

        Returns:
        - model: str - the model to use
        - messages: List[AllMessageValues] - messages with applied cache controls
        - non_default_params: dict - params with any global cache controls
        """
        # Extract cache control injection points
        scoped_bridge: Final = _attribute_or_key(non_default_params, ANTHROPIC_RESPONSES_CACHE_SCOPE) is True
        carry_unmatched: Final = bool(non_default_params.pop(CARRY_UNMATCHED_MESSAGE_POINTS, False))
        pending_points: Final = (
            non_default_params.get("cache_control_injection_points", [])
            if scoped_bridge
            else non_default_params.pop("cache_control_injection_points", [])
        )
        injection_points: Final = (
            cast(  # cast-ok: public point dictionaries allow omitted role, index, and control fields
                Sequence[CacheControlInjectionPoint], _POINTS_MAPPING_ADAPTER.validate_python(pending_points or ())
            )
        )
        if not injection_points:
            if scoped_bridge:
                non_default_params["cache_control_injection_points"] = pending_points  # rebind-ok: intent owner
            empty_params: Final = (
                dict(_provider_cache_params(non_default_params))
                if scoped_bridge and not carry_unmatched
                else non_default_params
            )
            return model, messages, empty_params

        if scoped_bridge and carry_unmatched:
            return model, messages, non_default_params

        # Create a deep copy of messages to avoid modifying the original list
        copied_messages: Final = copy.deepcopy(messages)

        message_points: Final = tuple(
            cast(CacheControlMessageInjectionPoint, point)
            for point in injection_points
            if point.get("location") == "message"
        )
        remaining_points: Final = tuple(point for point in injection_points if point.get("location") != "message")

        stamped_dialect: Final = injection_points[0].get("_litellm_openai_dialect")
        openai_dialect: Final = (
            stamped_dialect
            if isinstance(stamped_dialect, bool)
            else AnthropicCacheControlHook._targets_openai_prompt_cache_breakpoint(
                model,
                _optional_string(non_default_params.get("custom_llm_provider")),
                non_default_params.get("api_base") or non_default_params.get("base_url"),
                non_default_params.get("prompt_cache_options"),
            )
        )
        # A provisional message list defers every role-targeted point to the pass holding
        # the final one: a role with no message here may have one there, and settling all
        # of them in one pass is what lets config order decide the shared breakpoint
        # budget. An ordinal names a different message once a later layer builds its own
        # list, so it is placed here or not at all.
        carried_message_points: Final[Sequence[CacheControlMessageInjectionPoint]] = (
            tuple(point for point in message_points if point.get("index") is None) if carry_unmatched else ()
        )
        applied_message_points: Final[Sequence[CacheControlMessageInjectionPoint]] = (
            tuple(point for point in message_points if point.get("index") is not None)
            if carry_unmatched
            else tuple(message_points)
        )
        stamped_external: Final = injection_points[0].get(EXTERNAL_BREAKPOINTS_STAMP)
        external_breakpoints: Final = stamped_external if isinstance(stamped_external, int) else 0
        reserved_blocks: Final = AnthropicCacheControlHook._blocks_reserved_outside_messages(
            remaining_points, external_breakpoints, openai_dialect
        )
        breakpoints_before: Final = AnthropicCacheControlHook.count_request_cache_breakpoints(copied_messages)
        scoped_point: Final = _POINTS_MAPPING_ADAPTER.validate_python(pending_points)[0]
        coverage_skip: Final = bool(scoped_point.get(_COVERAGE_STAMP)) or _transcript_coverage_skip(
            copied_messages, scoped_point.get(_AUTOMATIC_CONTROL_STAMP)
        )
        processed_messages: Final = (
            list(
                self._apply_scoped_message_injections(
                    applied_message_points,
                    copied_messages,
                    MAX_CACHE_CONTROL_BLOCKS - reserved_blocks,
                    _OBJECT_SEQUENCE_ADAPTER.validate_python(scoped_point.get(_PREFIX_CONTROLS_STAMP) or ()),
                    scoped_point.get(_AUTOMATIC_CONTROL_STAMP),
                )
            )
            if scoped_bridge and not openai_dialect and not coverage_skip
            else copied_messages
            if scoped_bridge and not openai_dialect
            else self._apply_message_injections(
                points=applied_message_points,
                messages=copied_messages,
                max_blocks=MAX_CACHE_CONTROL_BLOCKS - reserved_blocks,
                openai_dialect=openai_dialect,
                anthropic_eligibility=supports_anthropic_cache_control(
                    model, _optional_string(non_default_params.get("custom_llm_provider"))
                ),
            )
        )
        if scoped_bridge and coverage_skip:
            verbose_logger.debug(
                "AnthropicCacheControlHook: Skipping configured additions outside supported cache coverage."
            )
        if (
            openai_dialect
            and AnthropicCacheControlHook.count_request_cache_breakpoints(processed_messages) > breakpoints_before
        ):
            non_default_params.setdefault("prompt_cache_options", PromptCacheOptions(mode="explicit"))

        # Points this pass did not place: non-message ones for the provider transform, and
        # the deferred role-targeted ones. Deferring is what reaches the Responses API's
        # `instructions`, which is only a system message once the bridge builds one. A later
        # pass re-applies them safely: a target that already carries a mark is skipped and
        # the census counts every mark on the wire, litellm's own included.
        carried_points: Final[Sequence[CacheControlInjectionPoint]] = (
            *AnthropicCacheControlHook._points_with_a_slot_left(
                remaining_points,
                AnthropicCacheControlHook.count_request_cache_breakpoints(processed_messages) + external_breakpoints,
                openai_dialect,
            ),
            *carried_message_points,
        )
        if scoped_bridge:
            non_default_params["cache_control_injection_points"] = list(remaining_points)  # rebind-ok: intent owner
            return (
                model,
                processed_messages,
                dict(_provider_cache_params(non_default_params)),
            )
        elif carried_points:
            non_default_params["cache_control_injection_points"] = list(carried_points)

        return model, processed_messages, non_default_params

    @staticmethod
    def _targets_openai_prompt_cache_breakpoint(
        model: str | None,
        custom_llm_provider: str | None,
        api_base: object = None,
        prompt_cache_options: object = None,
    ) -> bool:
        if model is None or not supports_openai_prompt_cache_breakpoint(model):
            return False
        if (custom_llm_provider or AnthropicCacheControlHook._resolve_provider(model)) != "openai":
            return False
        return prompt_cache_options is not None or targets_openai_api(api_base)

    @staticmethod
    def _resolve_provider(model: str) -> str | None:
        from litellm.exceptions import BadRequestError
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        try:
            _, provider, _, _ = get_llm_provider(model=model)
        except BadRequestError:
            return None
        return provider

    @staticmethod
    def count_request_cache_breakpoints(messages: Iterable[object], system: object = None) -> int:
        system_blocks: Final = (
            sum(1 for block in system if _carries_cache_breakpoint(block)) if isinstance(system, list) else 0
        )
        return system_blocks + sum(AnthropicCacheControlHook._count_cache_control_blocks(msg) for msg in messages)

    @staticmethod
    def count_external_cache_breakpoints(
        tools: Iterable[object] | None, cache_control: object = None, request_kwargs: object = None
    ) -> int:
        """Client breakpoints outside messages and system that the provider cap still counts.

        A tool carries its mark at the top level (Anthropic shape) or under ``function``
        (OpenAI shape). A top-level ``cache_control`` is Anthropic's automatic caching,
        which places one breakpoint of its own on top of the explicit ones. The
        ``extra_body`` envelope of ``request_kwargs`` is merged over the request on the
        wire, so a ``tools`` or ``cache_control`` it carries replaces the direct value
        and is counted in its place. Callers pass only the tools whose mark reaches the
        provider on their path.
        """
        extra_body: Final = (
            _validated_object_mapping(AnthropicCacheControlHook._request_value(request_kwargs, "extra_body")) or {}
        )
        wire_cache_control: Final = extra_body.get("cache_control", cache_control)
        wire_tools: Final = _validated_object_list(extra_body["tools"]) if "tools" in extra_body else tools
        tool_blocks: Final = sum(1 for tool in wire_tools or () if _tool_carries_cache_breakpoint(tool))
        envelope_blocks: Final = AnthropicCacheControlHook.count_request_cache_breakpoints(
            _validated_object_list(extra_body.get("messages")) or (), extra_body.get("system")
        )
        return int(wire_cache_control is not None) + tool_blocks + envelope_blocks

    @staticmethod
    def count_external_cache_breakpoints_on_messages_route(
        tools: Iterable[object] | None, cache_control: object, request_kwargs: object
    ) -> int:
        """The /v1/messages census before the route splits.

        The native messages transforms drop the ``extra_body`` envelope while the
        chat bridge merges it, so the cap reserves for whichever census is larger
        rather than letting an envelope that unmarks a direct tool free a slot the
        provider still counts.
        """
        return max(
            AnthropicCacheControlHook.count_external_cache_breakpoints(tools, cache_control),
            AnthropicCacheControlHook.count_external_cache_breakpoints(tools, cache_control, request_kwargs),
        )

    @staticmethod
    def _blocks_reserved_outside_messages(
        remaining_points: Sequence[CacheControlInjectionPoint], external_breakpoints: int, openai_dialect: bool
    ) -> int:
        """Slots of the provider cap that the message census cannot see.

        The client's breakpoints on tools and its automatic top-level ``cache_control``
        are already on the wire, and a ``tool_config`` point becomes one more cachePoint
        in the Bedrock converse transform. OpenAI's cap counts only its own block markers.
        """
        if openai_dialect:
            return 0
        tool_config_blocks: Final = 1 if any(p.get("location") == "tool_config" for p in remaining_points) else 0
        return external_breakpoints + tool_config_blocks

    @staticmethod
    def _points_with_a_slot_left(
        remaining_points: Sequence[CacheControlInjectionPoint], breakpoints_on_wire: int, openai_dialect: bool
    ) -> tuple[CacheControlInjectionPoint, ...]:
        """A ``tool_config`` point becomes a cachePoint the Bedrock converse transform never
        counts against the cap, so it is forwarded only while the wire still has a slot."""
        if openai_dialect or breakpoints_on_wire < MAX_CACHE_CONTROL_BLOCKS:
            return tuple(remaining_points)
        return tuple(point for point in remaining_points if point.get("location") != "tool_config")

    @staticmethod
    def _apply_message_injections(
        points: Sequence[CacheControlMessageInjectionPoint],
        messages: list[AllMessageValues],
        max_blocks: int,
        openai_dialect: bool = False,
        native_messages: bool = False,
        anthropic_eligibility: bool = True,
    ) -> list[AllMessageValues]:
        """Apply message-level cache control injection points in order.

        Anthropic allows at most ``MAX_CACHE_CONTROL_BLOCKS`` cache_control
        breakpoints per request. Client-supplied breakpoints count toward that
        limit, so we never inject onto a message that already carries
        cache_control (preserving the client's TTL) and we stop injecting once
        ``max_blocks`` is reached. Injection points are honored in config order,
        so earlier points win when slots are scarce.
        """
        used_blocks = AnthropicCacheControlHook.count_request_cache_breakpoints(messages)

        limit_reached = False
        for point in points:
            if used_blocks >= max_blocks:
                limit_reached = True
                break

            control: ChatCompletionCachedContent = point.get("control", None) or ChatCompletionCachedContent(
                type="ephemeral"
            )

            for target_index in AnthropicCacheControlHook._resolve_target_indices(point=point, messages=messages):
                if used_blocks >= max_blocks:
                    limit_reached = True
                    break

                supported_marks = (
                    AnthropicCacheControlHook._message_has_cache_control(messages[target_index])
                    if openai_dialect
                    or not anthropic_eligibility
                    or _attribute_or_key(messages[target_index], "role") in ("tool", "function")
                    else bool(_message_controls(messages, target_index, native_messages))
                )
                if supported_marks:
                    # Client already marked this message; don't overwrite it.
                    continue

                before_blocks = AnthropicCacheControlHook._count_cache_control_blocks(messages[target_index])
                messages[target_index] = AnthropicCacheControlHook._safe_insert_cache_control_in_message(
                    messages[target_index],
                    control,
                    openai_dialect,
                    _message_cache_recipients(messages, target_index, native_messages) if not openai_dialect else None,
                    anthropic_eligibility,
                )
                used_blocks += max(
                    0, AnthropicCacheControlHook._count_cache_control_blocks(messages[target_index]) - before_blocks
                )

            if limit_reached:
                break

        if limit_reached:
            verbose_logger.warning(
                "AnthropicCacheControlHook: Reached the provider limit of %s cache breakpoints. Skipping further injection.",
                MAX_CACHE_CONTROL_BLOCKS,
            )

        return messages

    @staticmethod
    def _apply_scoped_message_injections(
        points: Sequence[CacheControlMessageInjectionPoint],
        messages: Sequence[AllMessageValues],
        max_blocks: int,
        prefix_controls: Sequence[object],
        automatic_control: object,
    ) -> Sequence[AllMessageValues]:
        return reduce(
            lambda current, point: reduce(
                lambda transcript, index: AnthropicCacheControlHook._apply_scoped_target(
                    index,
                    transcript,
                    point.get("control") or ChatCompletionCachedContent(type="ephemeral"),
                    max_blocks,
                    prefix_controls,
                    automatic_control,
                ),
                AnthropicCacheControlHook._resolve_target_indices(point, list(current)),
                current,
            ),
            points,
            messages,
        )

    @staticmethod
    def _apply_scoped_target(
        index: int,
        messages: Sequence[AllMessageValues],
        control: ChatCompletionCachedContent,
        max_blocks: int,
        prefix_controls: Sequence[object],
        automatic_control: object,
    ) -> Sequence[AllMessageValues]:
        recipients: Final = _message_cache_recipients(messages, index)
        existing: Final = _ordered_message_controls(messages)
        if not recipients or _message_controls(messages, index):
            return messages
        if len(existing) >= max_blocks:
            verbose_logger.debug(
                "AnthropicCacheControlHook: Reached the provider limit of %s cache breakpoints. Skipping injection.",
                MAX_CACHE_CONTROL_BLOCKS,
            )
            return messages
        last_eligible: Final = next(
            (i for i in range(len(messages) - 1, -1, -1) if _message_cache_recipients(messages, i)), None
        )
        proposed_message: Final = AnthropicCacheControlHook._safe_insert_cache_control_in_message(
            copy.deepcopy(messages[index]), control, recipients=recipients
        )
        proposed: Final = [proposed_message if i == index else message for i, message in enumerate(messages)]
        automatic_matches: Final = (
            automatic_control is None
            or last_eligible != index
            or _ttl_duration(control) == _ttl_duration(automatic_control)
        )
        automatic_emits: Final = (
            automatic_control is not None
            and last_eligible is not None
            and _recipient_control(proposed[last_eligible], _message_cache_recipients(proposed, last_eligible)[-1])
            is None
        )
        automatic_controls: Final = (automatic_control,) if automatic_emits else ()
        valid_order: Final = _ttl_order_valid(
            (*prefix_controls, *_ordered_message_controls(proposed), *automatic_controls)
        )
        if not automatic_matches or not valid_order:
            verbose_logger.debug(
                "AnthropicCacheControlHook: Skipping configured addition incompatible with cache TTL order."
            )
        return proposed if automatic_matches and valid_order else messages

    @staticmethod
    def _resolve_target_indices(
        point: CacheControlMessageInjectionPoint, messages: list[AllMessageValues]
    ) -> list[int]:
        """Resolve which message indices an injection point targets."""
        _targetted_index: Final[int | str | None] = point.get("index", None)
        targetted_index: int | None = None
        if isinstance(_targetted_index, str):
            try:
                targetted_index = int(_targetted_index)
            except ValueError:
                pass
        else:
            targetted_index = _targetted_index

        # Case 1: Target by specific index
        if targetted_index is not None:
            original_index: Final = targetted_index
            if targetted_index < 0:
                targetted_index += len(messages)

            if 0 <= targetted_index < len(messages):
                return [targetted_index]

            verbose_logger.warning(
                "AnthropicCacheControlHook: Provided index %s is out of bounds for message list of length %s. Targeted index was %s. Skipping cache control injection for this point.",
                original_index,
                len(messages),
                targetted_index,
            )
            return []

        # Case 2: Target by role
        targetted_role: Final = point.get("role", None)
        if targetted_role is not None:
            return [idx for idx, msg in enumerate(messages) if msg.get("role") == targetted_role]

        return []

    @staticmethod
    def _count_cache_control_blocks(message: object) -> int:
        message_count: Final = 1 if _carries_cache_breakpoint(message) else 0
        content: Final = _as_object_list(_attribute_or_key(message, "content"))
        content_count: Final = sum(1 for block in content if _carries_cache_breakpoint(block)) if content else 0
        tool_calls: Final = _as_object_list(_attribute_or_key(message, "tool_calls"))
        tool_call_count: Final = (
            sum(1 for tool_call in tool_calls if _tool_call_carries_cache_breakpoint(tool_call, message))
            if tool_calls
            else 0
        )
        return message_count + content_count + tool_call_count

    @staticmethod
    def _message_has_cache_control(message: AllMessageValues) -> bool:
        """Return True if the message already carries any cache_control."""
        return AnthropicCacheControlHook._count_cache_control_blocks(message) > 0

    @staticmethod
    def _safe_insert_cache_control_in_message(
        message: AllMessageValues,
        control: ChatCompletionCachedContent,
        openai_dialect: bool = False,
        recipients: Sequence[tuple[str, int]] | None = None,
        anthropic_eligibility: bool = True,
    ) -> AllMessageValues:
        """
        Safe way to insert cache control in a message

        OpenAI Message content can be either:
            - string
            - list of objects

        This method handles inserting cache control in both cases.
        Per Anthropic's API specification, when using multiple content blocks,
        only the last content block can have cache_control.
        """
        if openai_dialect:
            return AnthropicCacheControlHook._insert_prompt_cache_breakpoint_in_message(message)

        if not anthropic_eligibility:
            content: Final = _attribute_or_key(message, "content")
            legacy_recipients: Final = (
                (("message", 0),)
                if isinstance(content, str)
                else (("content", len(_OBJECT_SEQUENCE_ADAPTER.validate_python(content)) - 1),)
                if isinstance(content, list) and content
                else ()
            )
            return AnthropicCacheControlHook._safe_insert_cache_control_in_message(
                message, control, recipients=legacy_recipients
            )
        eligible: Final = tuple(recipients) if recipients is not None else _message_cache_recipients((message,), 0)
        if not eligible:
            return message
        kind, index = eligible[-1]
        if kind == "message":
            _annotate_cache_control(message, control)
            return message
        key: Final = "tool_calls" if kind == "call" else "content"
        values: Final = _as_object_list(_attribute_or_key(message, key)) or []
        _annotate_cache_control(values[index], control)
        return message

    @staticmethod
    def _insert_prompt_cache_breakpoint_in_message(message: AllMessageValues) -> AllMessageValues:
        if message.get("role") == "assistant":
            return message
        message_content: Final = message.get("content", None)
        if isinstance(message_content, str):
            marked: Final = copy.copy(message)
            marked["content"] = [
                with_prompt_cache_breakpoint(
                    ChatCompletionTextObject(type="text", text=message_content), PromptCacheBreakpoint(mode="explicit")
                )
            ]
            return marked
        if isinstance(message_content, list):
            target_index: Final = next(
                (
                    index
                    for index in range(len(message_content) - 1, -1, -1)
                    if _accepts_prompt_cache_breakpoint(message_content[index])
                ),
                None,
            )
            if target_index is not None:
                message_content[target_index] = with_prompt_cache_breakpoint(
                    message_content[target_index], PromptCacheBreakpoint(mode="explicit")
                )
        return message

    @staticmethod
    def _system_block_with_breakpoint(
        block: Mapping[str, object], control: ChatCompletionCachedContent, openai_dialect: bool
    ) -> Mapping[str, object]:
        marker: Final = (
            ("prompt_cache_breakpoint", PromptCacheBreakpoint(mode="explicit"))
            if openai_dialect
            else ("cache_control", control)
        )
        return {**block, marker[0]: marker[1]}

    @staticmethod
    def apply_to_anthropic_messages_request(
        messages: list[dict],
        system: str | list | None,
        injection_points: Sequence[CacheControlInjectionPoint],
        openai_dialect: bool = False,
        external_breakpoints: int = 0,
    ) -> tuple[list[dict], str | list | None, list[CacheControlInjectionPoint]]:
        """Apply cache control injection for the Anthropic-native v1/messages endpoint.

        ``external_breakpoints`` is the client's breakpoint count outside ``messages`` and
        ``system`` (see ``count_external_cache_breakpoints``); it shrinks the budget so
        the request never exceeds the provider cap.

        Returns (messages, system, remaining_non_message_points).
        """
        if not injection_points:
            return messages, system, []

        processed_messages: list[dict] = copy.deepcopy(messages)
        processed_system = copy.deepcopy(system) if system is not None else None

        role_points: Final = tuple(
            cast(CacheControlMessageInjectionPoint, point)
            for point in injection_points
            if point.get("location") == "message"
        )
        system_points: Final = tuple(point for point in role_points if point.get("role") == "system")
        message_points: Final = tuple(point for point in role_points if point.get("role") != "system")
        remaining_points: Final = tuple(point for point in injection_points if point.get("location") != "message")

        reserved_blocks: Final = AnthropicCacheControlHook._blocks_reserved_outside_messages(
            remaining_points, external_breakpoints, openai_dialect
        )
        max_blocks: Final = MAX_CACHE_CONTROL_BLOCKS - reserved_blocks

        message_blocks: Final = AnthropicCacheControlHook.count_request_cache_breakpoints(processed_messages)
        system_blocks = AnthropicCacheControlHook.count_request_cache_breakpoints((), processed_system)

        if system_points and processed_system is not None and message_blocks + system_blocks < max_blocks:
            system_already_has_cc: Final = isinstance(processed_system, list) and any(
                _carries_cache_breakpoint(b) for b in processed_system
            )
            if not system_already_has_cc:
                control: Final = system_points[0].get("control") or ChatCompletionCachedContent(type="ephemeral")
                if isinstance(processed_system, str):
                    processed_system = [
                        AnthropicCacheControlHook._system_block_with_breakpoint(
                            AnthropicSystemMessageContent(type="text", text=processed_system), control, openai_dialect
                        )
                    ]
                    system_blocks += 1
                elif len(processed_system) > 0 and isinstance(processed_system[-1], dict):
                    processed_system[-1] = AnthropicCacheControlHook._system_block_with_breakpoint(
                        processed_system[-1], control, openai_dialect
                    )
                    system_blocks += 1

        for i, msg in enumerate(processed_messages):
            content = msg.get("content")
            if isinstance(content, str):
                processed_messages[i] = {**msg, "content": [{"type": "text", "text": content}]}

        processed_messages = AnthropicCacheControlHook._apply_message_injections(
            points=message_points,
            messages=cast(list[AllMessageValues], processed_messages),
            max_blocks=max_blocks - system_blocks,
            openai_dialect=openai_dialect,
            native_messages=True,
        )
        forwarded_points: Final = AnthropicCacheControlHook._points_with_a_slot_left(
            remaining_points,
            AnthropicCacheControlHook.count_request_cache_breakpoints(processed_messages, processed_system)
            + external_breakpoints,
            openai_dialect,
        )

        return processed_messages, processed_system, list(forwarded_points)

    @staticmethod
    def _default_control() -> ChatCompletionCachedContent:
        """Build the cache_control block for auto-injected breakpoints.

        Defaults to Anthropic's 5-minute ephemeral cache; honors the optional
        ``litellm.anthropic_prompt_caching_ttl`` override ("5m" or "1h").
        """
        import litellm

        ttl: Final = litellm.anthropic_prompt_caching_ttl
        if ttl == "5m" or ttl == "1h":
            return ChatCompletionCachedContent(type="ephemeral", ttl=ttl)
        return ChatCompletionCachedContent(type="ephemeral")

    @staticmethod
    def _stamped_for_prompt_hook(
        points: Sequence[Mapping[str, object]],
        external_breakpoints: int,
        model: str,
        custom_llm_provider: str | None,
        api_base: object,
        prompt_cache_options: object,
    ) -> Sequence[Mapping[str, object]]:
        """Carry onto the points what the prompt-management hook never receives.

        The hook sees neither the tools nor the request kwargs, so the target dialect
        and the client's breakpoint count outside the message list ride on the points.
        Builds copies because config-owned point dicts are shared across requests.
        """
        with_dialect: Final = AnthropicCacheControlHook._stamped_with_dialect(
            points, model, custom_llm_provider, api_base, prompt_cache_options
        )
        if external_breakpoints == 0:
            return with_dialect
        return AnthropicCacheControlHook._stamped(with_dialect, EXTERNAL_BREAKPOINTS_STAMP, external_breakpoints)

    @staticmethod
    def _stamped_with_dialect(
        points: Sequence[Mapping[str, object]],
        model: str,
        custom_llm_provider: str | None,
        api_base: object,
        prompt_cache_options: object,
    ) -> Sequence[Mapping[str, object]]:
        if not supports_openai_prompt_cache_breakpoint(model):
            return points
        return AnthropicCacheControlHook._stamped(
            points,
            "_litellm_openai_dialect",
            AnthropicCacheControlHook._targets_openai_prompt_cache_breakpoint(
                model, custom_llm_provider, api_base, prompt_cache_options
            ),
        )

    @staticmethod
    def _stamped(points: Sequence[Mapping[str, object]], key: str, value: object) -> Sequence[Mapping[str, object]]:
        return [{**point, key: value} for point in points]

    @staticmethod
    def _request_has_cache_control(
        messages: list[AllMessageValues],
        system: str | list | None,
        tools: list | None = None,
        cache_control: object = None,
        request_kwargs: object = None,
        on_messages_route: bool = False,
    ) -> bool:
        """Return True if the request already carries any client-supplied cache_control.

        Only the automatic defaults stand down on it: a client that marks its own
        breakpoints (Claude Code does) has a caching strategy the defaults would
        clash with, whether the marks sit in the request or in its ``extra_body``
        envelope. Configured injection points are an explicit instruction and are
        applied alongside the client's marks, bounded by the provider cap.
        """
        external_breakpoints: Final = (
            AnthropicCacheControlHook.count_external_cache_breakpoints_on_messages_route(
                tools, cache_control, request_kwargs
            )
            if on_messages_route
            else AnthropicCacheControlHook.count_external_cache_breakpoints(tools, cache_control, request_kwargs)
        )
        return AnthropicCacheControlHook.count_request_cache_breakpoints(messages, system) + external_breakpoints > 0

    @staticmethod
    def get_default_injection_points(
        messages: list[AllMessageValues],
        system: str | list | None,
        model: str,
        custom_llm_provider: str | None,
        tools: list | None = None,
        enable_prompt_caching: bool | None = None,
        cache_control: object = None,
        request_kwargs: object = None,
        on_messages_route: bool = False,
    ) -> list[CacheControlInjectionPoint]:
        """Default breakpoints when ``litellm.enable_anthropic_prompt_caching`` is on.

        ``enable_prompt_caching`` is the per-request override (stamped from key
        metadata by the proxy); True turns auto-injection on for this request
        even when the global flag is off. Caches the system prompt and the
        trailing turn, so the stable prefix (system + tools + history) is
        reused while the breakpoint advances with the conversation. Returns []
        (stand down) when neither flag is on, the model is not Claude on a
        supported explicit-cache transport, the model lacks prompt-caching
        support, or the request already carries client-supplied cache_control.
        """
        import litellm

        if litellm.enable_anthropic_prompt_caching is not True and enable_prompt_caching is not True:
            return []

        if not supports_anthropic_cache_control(model, custom_llm_provider):
            return []

        if AnthropicCacheControlHook._request_has_cache_control(
            messages, system, tools, cache_control, request_kwargs, on_messages_route
        ):
            return []

        if is_claude_code_one_shot_subagent_request(
            messages, system, tools, AnthropicCacheControlHook._request_user_agent(request_kwargs)
        ):
            return []

        control: Final = AnthropicCacheControlHook._default_control()
        points: Final[list[CacheControlInjectionPoint]] = [
            CacheControlMessageInjectionPoint(location="message", role="system", index=None, control=control),
            CacheControlMessageInjectionPoint(location="message", role=None, index=-1, control=control),
        ]
        return points

    @staticmethod
    def messages_with_default_injections(
        messages: list[AllMessageValues],
        models: Iterable[str],
        tools: list[AllToolParamValues] | None = None,
        enable_prompt_caching: bool | None = None,
        request_kwargs: object = None,
    ) -> list[AllMessageValues]:
        """Return the messages auto prompt caching will send, default breakpoints included.

        Router cache affinity depends on this. Deployment selection runs before the injection in
        `litellm.acompletion`, so it has to reproduce the markers to derive the same cache key the
        success event later writes from the sent messages. `models` is every candidate model of the
        group: the first that would auto-inject decides, since the default breakpoints (system
        prompt and trailing turn) do not depend on which deployment serves the call. Returns the
        input list itself when auto-injection would not apply
        """
        import litellm

        points: Final = next(
            (
                candidate
                for candidate in (
                    AnthropicCacheControlHook.get_default_injection_points(
                        messages=messages,
                        model=litellm.model_alias_map.get(model, model),
                        custom_llm_provider=None,
                        tools=tools,
                        enable_prompt_caching=enable_prompt_caching,
                        system=AnthropicCacheControlHook._request_system(request_kwargs),
                        cache_control=AnthropicCacheControlHook._request_value(request_kwargs, "cache_control"),
                        request_kwargs=request_kwargs,
                    )
                    for model in models
                )
                if candidate
            ),
            None,
        )
        if not points:
            return messages
        return AnthropicCacheControlHook._apply_message_injections(
            points=cast(  # cast-ok: the default points are all message-location points
                list[CacheControlMessageInjectionPoint], points
            ),
            messages=copy.deepcopy(messages),
            max_blocks=MAX_CACHE_CONTROL_BLOCKS,
        )

    @staticmethod
    def maybe_seed_default_injection_points(
        non_default_params: dict[str, Any],
        messages: list[AllMessageValues],
        model: str,
        custom_llm_provider: str | None,
        tools: list | None = None,
        enable_prompt_caching: bool | None = None,
        api_base: object = None,
        scoped_bridge: bool = False,
        request_kwargs: object = None,
    ) -> None:
        """For /chat/completions: resolve the injection points the request should carry.

        Configured injection points win over the automatic defaults and are applied
        even when the client marked its own cache_control elsewhere in the request;
        the provider's four-block cap bounds them, counting the client's marks on
        messages, tools and the top-level ``cache_control``. Only the defaults stand
        down on client marks. Seeding the param lets the existing prompt-management
        gate and the AnthropicCacheControlHook run unchanged.
        """
        import litellm

        if (
            scoped_bridge
            and "cache_control_injection_points" in non_default_params
            and not non_default_params["cache_control_injection_points"]
        ):
            return
        if scoped_bridge:
            non_default_params[ANTHROPIC_RESPONSES_CACHE_SCOPE] = True  # rebind-ok: legacy in-place seeder contract
        configured: Final = non_default_params.get("cache_control_injection_points")
        if configured:
            tools_keeping_marks: Final = tuple(
                tool
                for tool in _OBJECT_SEQUENCE_ADAPTER.validate_python(tools or ())
                if not _chat_transform_drops_tool_cache_control(tool)
            )
            configured_points: Final = (
                _POINTS_MAPPING_ADAPTER.validate_python(configured)
                if scoped_bridge
                else cast(  # cast-ok: preserve the existing unscoped configuration object
                    Sequence[Mapping[str, object]], configured
                )
            )
            fresh_points: Final = (
                [
                    {
                        key: value
                        for key, value in point.items()
                        if key
                        not in (
                            EXTERNAL_BREAKPOINTS_STAMP,
                            "_litellm_openai_dialect",
                            _PREFIX_CONTROLS_STAMP,
                            _AUTOMATIC_CONTROL_STAMP,
                            _COVERAGE_STAMP,
                        )
                    }
                    for point in configured_points
                ]
                if scoped_bridge
                else configured_points
            )
            current_envelope: Final = (
                {**(_validated_object_mapping(request_kwargs) or {}), **non_default_params}
                if scoped_bridge
                else non_default_params
            )
            stamped: Final = AnthropicCacheControlHook._stamped_for_prompt_hook(
                fresh_points,
                AnthropicCacheControlHook.count_external_cache_breakpoints(
                    tools_keeping_marks, current_envelope.get("cache_control"), current_envelope
                ),
                model,
                custom_llm_provider,
                api_base,
                non_default_params.get("prompt_cache_options"),
            )
            if scoped_bridge:
                controls, automatic, coverage_skip = _scoped_envelope_facts(
                    tools_keeping_marks, non_default_params, request_kwargs
                )
                non_default_params["cache_control_injection_points"] = [  # rebind-ok: legacy in-place seeder contract
                    {
                        **point,
                        _PREFIX_CONTROLS_STAMP: controls,
                        _AUTOMATIC_CONTROL_STAMP: automatic,
                        _COVERAGE_STAMP: coverage_skip,
                    }
                    for point in stamped
                ]
            else:
                non_default_params["cache_control_injection_points"] = stamped  # rebind-ok: seeder API
            return
        points: Final = AnthropicCacheControlHook.get_default_injection_points(
            messages=messages,
            system=None,
            model=litellm.model_alias_map.get(model, model),
            custom_llm_provider=custom_llm_provider,
            tools=tools,
            enable_prompt_caching=enable_prompt_caching,
            cache_control=non_default_params.get("cache_control"),
            request_kwargs=non_default_params,
        )
        if points:
            non_default_params["cache_control_injection_points"] = points

    @staticmethod
    def record_gateway_injection(
        request_kwargs: Mapping[str, object],
        added: int,
        injected_for_every_deployment: bool = False,
    ) -> None:
        """Name the deployment whose payload the gateway, not the client, put breakpoints on.

        Spend accounting only asks whether litellm acted, so what it needs is which
        deployment, not a count. Recording that is what makes the mark attempt-scoped: the
        metadata bucket is one dict shared by every retry, failover and fallback of a
        request, and ``litellm_call_id`` is shared with it, so anything request-scoped
        written by one attempt is read by all of them and each boundary would have to
        remember to strip it. The deployment is the part that actually changes when the
        request moves, so a leg that injected nothing is never credited for one that did.

        It also makes a zero delta (hook re-entry) and a negative one (a prompt manager
        replacing the messages) harmless, since neither rewrites an earlier mark.

        A pass that runs before a deployment is chosen, which is what the proxy does for
        prompt templates, injects into the payload every leg goes on to send, so it marks
        the request for all of them rather than for one. Such a pass says so with
        ``injected_for_every_deployment`` instead of relying on the shape of
        ``request_kwargs``: the router's prompt-management factory stamps a provisional
        deployment's ``model_info`` into kwargs before the prompt pass runs, and billing
        the request through any other deployment would silently drop the credit. An
        every-deployment mark, once written, also never narrows: a later per-leg stamp
        (the Bedrock converse tool_config one included) describes one leg of a payload
        every leg sends, so narrowing to it would uncredit whichever leg gets billed
        after a failover. Both losses are fail-closed under-crediting, which is why the
        guard only protects the sentinel and per-leg marks still overwrite each other.

        Only what this pass actually placed counts. A ``tool_config`` point is placed by
        the Bedrock converse transform, and only when the request carries tools, so the
        presence of one here says nothing about whether a breakpoint reaches the wire;
        claiming it marked three request shapes out of four that inject nothing. Missing
        that Bedrock credit is the fail-closed direction, and the alternative is a
        provider transform that carries spend-attribution state.

        Reads whichever bucket the request actually carries rather than asking the shared
        name resolver, which answers on key presence: ``litellm_params`` declares
        ``litellm_metadata`` as None on every request, so the resolver names a bucket that
        is not there and the mark is dropped.

        Never CREATES the bucket. The proxy seeds it on every request and is the marker's
        only reader, so a request without one is a bare SDK call nothing would consume it
        from. Creating it would also add a key to a dict call sites splat as ``**kwargs``,
        and on the Responses API ``metadata`` is both this bucket's default name and an
        explicit parameter, so the splat collides with the caller's own value.
        """
        if added <= 0:
            return
        bucket: Final = next(
            (
                candidate
                for candidate in (request_kwargs.get("litellm_metadata"), request_kwargs.get("metadata"))
                if isinstance(candidate, dict)
            ),
            None,
        )
        if bucket is None:
            return
        if bucket.get(GATEWAY_INJECTED_CACHE_METADATA_KEY) == GATEWAY_INJECTED_FOR_EVERY_DEPLOYMENT:
            return
        if injected_for_every_deployment:
            bucket[GATEWAY_INJECTED_CACHE_METADATA_KEY] = GATEWAY_INJECTED_FOR_EVERY_DEPLOYMENT
            return
        model_info: Final = request_kwargs.get("model_info")
        bucket[GATEWAY_INJECTED_CACHE_METADATA_KEY] = (
            model_info.get("id", GATEWAY_INJECTED_FOR_EVERY_DEPLOYMENT)
            if isinstance(model_info, dict)
            else GATEWAY_INJECTED_FOR_EVERY_DEPLOYMENT
        )

    @staticmethod
    def maybe_inject_cache_control(
        messages: list[dict],
        system: str | list | None,
        kwargs: dict[str, Any],
        model: str | None = None,
        custom_llm_provider: str | None = None,
        tools: list[dict] | None = None,
        api_base: str | None = None,
    ) -> tuple[list[dict], str | list | None]:
        """Extract cache_control_injection_points from kwargs and apply if present.

        Configured points are applied even when the client marked its own
        cache_control elsewhere in the request, bounded by the provider cap,
        which counts the client's marks on messages, system, tools and the
        top-level ``cache_control``. When none are configured but
        ``litellm.enable_anthropic_prompt_caching`` or the per-request
        ``enable_prompt_caching`` kwarg (stamped from key metadata) is on,
        synthesize default breakpoints for the native /v1/messages path; those
        defaults alone stand down on client marks. Pops both keys from kwargs;
        if remaining (non-message) points exist they are written back so
        downstream transforms can handle them.
        """
        typed_messages = cast(list[AllMessageValues], messages)  # cast-ok: Anthropic-shaped dicts from v1/messages
        enable_prompt_caching: Final = cast(  # cast-ok: kwargs is untyped; key stamped as bool by the proxy
            bool | None, kwargs.pop("enable_prompt_caching", None)
        )
        cache_control: Final = kwargs.get("cache_control")
        configured: Final = cast(  # cast-ok: kwargs is untyped; this key only holds the documented injection-point list
            list[CacheControlInjectionPoint] | None, kwargs.pop("cache_control_injection_points", None)
        )
        injection_points: Final[Sequence[CacheControlInjectionPoint]] = configured or (
            AnthropicCacheControlHook.get_default_injection_points(
                messages=typed_messages,
                system=system,
                tools=tools,
                model=model,
                custom_llm_provider=custom_llm_provider,
                enable_prompt_caching=enable_prompt_caching,
                cache_control=cache_control,
                request_kwargs=kwargs,
                on_messages_route=True,
            )
            if model is not None
            else ()
        )
        if not injection_points:
            return messages, system

        openai_dialect: Final = AnthropicCacheControlHook._targets_openai_prompt_cache_breakpoint(
            model, custom_llm_provider, api_base, kwargs.get("prompt_cache_options")
        )
        breakpoints_before: Final = AnthropicCacheControlHook.count_request_cache_breakpoints(messages, system)
        messages, system, remaining = AnthropicCacheControlHook.apply_to_anthropic_messages_request(
            messages=messages,
            system=system,
            injection_points=injection_points,
            openai_dialect=openai_dialect,
            external_breakpoints=AnthropicCacheControlHook.count_external_cache_breakpoints_on_messages_route(
                tools, cache_control, kwargs
            ),
        )
        breakpoints_added: Final = (
            AnthropicCacheControlHook.count_request_cache_breakpoints(messages, system) - breakpoints_before
        )
        AnthropicCacheControlHook.record_gateway_injection(kwargs, breakpoints_added)
        if openai_dialect and breakpoints_added > 0:
            kwargs.setdefault("prompt_cache_options", PromptCacheOptions(mode="explicit"))
        if remaining:
            kwargs["cache_control_injection_points"] = remaining
        return messages, system

    @property
    def integration_name(self) -> str:
        """Return the integration name for this hook."""
        return "anthropic_cache_control_hook"

    def should_run_prompt_management(
        self,
        prompt_id: str | None,
        prompt_spec: PromptSpec | None,
        dynamic_callback_params: StandardCallbackDynamicParams,
    ) -> bool:
        """Always return False since this is not a true prompt management system."""
        return False

    def _compile_prompt_helper(
        self,
        prompt_id: str | None,
        prompt_spec: PromptSpec | None,
        prompt_variables: dict | None,
        dynamic_callback_params: StandardCallbackDynamicParams,
        prompt_label: str | None = None,
        prompt_version: int | None = None,
    ) -> PromptManagementClient:
        """Not used - this hook only modifies messages, doesn't fetch prompts."""
        return PromptManagementClient(
            prompt_id=prompt_id,
            prompt_template=[],
            prompt_template_model=None,
            prompt_template_optional_params=None,
            completed_messages=None,
        )

    async def async_compile_prompt_helper(
        self,
        prompt_id: str | None,
        prompt_variables: dict | None,
        dynamic_callback_params: StandardCallbackDynamicParams,
        prompt_spec: PromptSpec | None = None,
        prompt_label: str | None = None,
        prompt_version: int | None = None,
    ) -> PromptManagementClient:
        """Not used - this hook only modifies messages, doesn't fetch prompts."""
        return self._compile_prompt_helper(
            prompt_id=prompt_id,
            prompt_spec=prompt_spec,
            prompt_variables=prompt_variables,
            dynamic_callback_params=dynamic_callback_params,
            prompt_label=prompt_label,
            prompt_version=prompt_version,
        )

    async def async_get_chat_completion_prompt(
        self,
        model: str,
        messages: list[AllMessageValues],
        non_default_params: dict,
        prompt_id: str | None,
        prompt_variables: dict | None,
        dynamic_callback_params: StandardCallbackDynamicParams,
        litellm_logging_obj: LiteLLMLoggingObj,
        prompt_spec: PromptSpec | None = None,
        tools: list[dict] | None = None,
        prompt_label: str | None = None,
        prompt_version: int | None = None,
        ignore_prompt_manager_model: bool | None = False,
        ignore_prompt_manager_optional_params: bool | None = False,
    ) -> tuple[str, list[AllMessageValues], dict]:
        """Async version - delegates to sync since no async operations needed."""
        return self.get_chat_completion_prompt(
            model=model,
            messages=messages,
            non_default_params=non_default_params,
            prompt_id=prompt_id,
            prompt_variables=prompt_variables,
            dynamic_callback_params=dynamic_callback_params,
            prompt_spec=prompt_spec,
            prompt_label=prompt_label,
            prompt_version=prompt_version,
            ignore_prompt_manager_model=ignore_prompt_manager_model,
            ignore_prompt_manager_optional_params=ignore_prompt_manager_optional_params,
        )

    @staticmethod
    def should_use_anthropic_cache_control_hook(non_default_params: dict) -> bool:
        if non_default_params.get("cache_control_injection_points", None):
            return True
        return False

    @staticmethod
    def get_custom_logger_for_anthropic_cache_control_hook(
        non_default_params: dict,
    ) -> CustomLogger | None:
        from litellm.litellm_core_utils.litellm_logging import (
            _init_custom_logger_compatible_class,
        )

        if AnthropicCacheControlHook.should_use_anthropic_cache_control_hook(non_default_params):
            return _init_custom_logger_compatible_class(
                logging_integration="anthropic_cache_control_hook",
                internal_usage_cache=None,
                llm_router=None,
            )
        return None
