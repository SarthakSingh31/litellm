import asyncio
import json
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, TypeAlias, cast

from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter, field_validator

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.constants import (
    LITELLM_TRUNCATED_PAYLOAD_FIELD,
    REDACTED_BY_LITELLM,
    REDACTED_TOOL_CALL_ARGUMENTS_PLACEHOLDER,
)
from litellm.proxy._types import SpendLogsMetadata, SpendLogsPayload
from litellm.proxy.spend_tracking.cold_storage_handler import ColdStorageHandler
from litellm.responses.compaction_history import replay_compaction_input
from litellm.responses.utils import ResponsesAPIRequestUtils
from litellm.types.llms.openai import (
    AllMessageValues,
    ChatCompletionResponseMessage,
    GenericChatCompletionMessage,
    ResponseInputParam,
)
from litellm.types.utils import ChatCompletionMessageToolCall, Message, ModelResponse

if TYPE_CHECKING:
    from litellm.responses.litellm_completion_transformation.transformation import (
        ChatCompletionSession,
    )
else:
    ChatCompletionSession = Any

########################################################
# Cold Storage Handler
########################################################
COLD_STORAGE_HANDLER: Final = ColdStorageHandler()
########################################################

_MAX_COMPACTION_ANCESTRY: Final = 256
_INPUT_ADAPTER: Final[TypeAdapter[str | ResponseInputParam]] = TypeAdapter(str | ResponseInputParam)
_JSON_ADAPTER: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)
_JSON_OBJECT_ADAPTER: Final = TypeAdapter(Mapping[str, JsonValue])
_RESPONSE_OBJECT: Final = "response"
_EMPTY_OBJECT: Final[Mapping[str, JsonValue]] = MappingProxyType({})
_ChatHistoryItem: TypeAlias = (
    AllMessageValues
    | GenericChatCompletionMessage
    | ChatCompletionMessageToolCall
    | ChatCompletionResponseMessage
    | Message
)


class _StoredResponseBody(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True)

    input: str | ResponseInputParam
    previous_response_id: str | None = None
    instructions: str | None = None

    @field_validator("input")
    @classmethod
    def materialize_input(cls, value: str | ResponseInputParam) -> str | ResponseInputParam:
        return _materialized_input(value)


class _StoredReplayRow(BaseModel):
    request_id: str
    session_id: str | None = None
    proxy_server_request: str | Mapping[str, JsonValue] | None = None
    response: JsonValue = None
    metadata: str | Mapping[str, JsonValue] | None = None


_REPLAY_ROWS_ADAPTER: Final = TypeAdapter(tuple[_StoredReplayRow, ...])


def _materialized_input(value: object) -> str | ResponseInputParam:
    validated: Final = _INPUT_ADAPTER.validate_python(value)
    return cast(  # cast-ok: SDK validation precedes materialization of Iterable fields to JSON lists
        str | ResponseInputParam, _INPUT_ADAPTER.dump_python(validated, mode="json")
    )


def _has_unavailable_content(value: JsonValue) -> bool:
    if isinstance(value, str):
        return REDACTED_BY_LITELLM in value or LITELLM_TRUNCATED_PAYLOAD_FIELD in value
    if isinstance(value, list):
        return any(_has_unavailable_content(item) for item in value)
    if isinstance(value, dict):
        return LITELLM_TRUNCATED_PAYLOAD_FIELD in value or any(
            _has_unavailable_content(item) for item in value.values()
        )
    return False


def _stored_output(spend_log: Mapping[str, object]) -> Mapping[str, JsonValue]:
    raw: Final = spend_log.get("response")
    parsed: Final = _JSON_ADAPTER.validate_json(raw) if isinstance(raw, str) else _JSON_ADAPTER.validate_python(raw)
    return _JSON_OBJECT_ADAPTER.validate_python(parsed) if parsed is not None else _EMPTY_OBJECT


def _normalize_redacted_tool_call_arguments(message: Message) -> None:
    """Redaction stores the bare sentinel (invalid JSON) in tool-call arguments;
    normalize replayed history to "{}" so provider converters can parse it."""
    for tool_call in message.tool_calls or []:
        if (function := getattr(tool_call, "function", None)) is not None and function.arguments == REDACTED_BY_LITELLM:
            function.arguments = REDACTED_TOOL_CALL_ARGUMENTS_PLACEHOLDER
    function_call: Final = message.function_call
    if function_call is not None and function_call.arguments == REDACTED_BY_LITELLM:
        function_call.arguments = REDACTED_TOOL_CALL_ARGUMENTS_PLACEHOLDER


class ResponsesSessionHandler:
    @staticmethod
    async def get_chat_completion_message_history_for_previous_response_id(
        previous_response_id: str,
        *,
        compaction_ancestry_only: bool = False,
    ) -> ChatCompletionSession:
        """
        Return the chat completion message history for a previous response id
        """
        from litellm.responses.litellm_completion_transformation.transformation import (
            ChatCompletionSession,
        )

        verbose_proxy_logger.debug("inside get_chat_completion_message_history_for_previous_response_id")
        all_spend_logs: Final = (
            await ResponsesSessionHandler.get_compaction_ancestry(previous_response_id)
            if compaction_ancestry_only
            else await ResponsesSessionHandler.get_all_spend_logs_for_previous_response_id(previous_response_id)
        )
        verbose_proxy_logger.debug("found %s spend logs for this response id", len(all_spend_logs))

        litellm_session_id: str | None = None
        if len(all_spend_logs) > 0:
            litellm_session_id = all_spend_logs[0].get("session_id")

        chat_completion_message_history: list[
            AllMessageValues
            | GenericChatCompletionMessage
            | ChatCompletionMessageToolCall
            | ChatCompletionResponseMessage
            | Message
        ] = []
        for spend_log in all_spend_logs:
            chat_completion_message_history = (
                await ResponsesSessionHandler.extend_chat_completion_message_with_spend_log_payload(
                    spend_log=spend_log,
                    chat_completion_message_history=chat_completion_message_history,
                )
            )

        verbose_proxy_logger.debug(
            "chat_completion_message_history %s",
            json.dumps(chat_completion_message_history, indent=4, default=str),
        )
        return ChatCompletionSession(
            messages=chat_completion_message_history,
            litellm_session_id=litellm_session_id,
        )

    @staticmethod
    async def get_compaction_ancestry(
        previous_response_id: str,
        *,
        _visited: frozenset[str] = frozenset(),
    ) -> tuple[SpendLogsPayload, ...]:
        from litellm.proxy.proxy_server import prisma_client

        if prisma_client is None:
            raise ValueError("Gateway compaction needs stored response history or full client input replay")
        decoded: Final = ResponsesAPIRequestUtils._decode_responses_api_response_id(  # pyright: ignore[reportPrivateUsage]  # reuse the existing response affinity decoder
            previous_response_id
        )
        response_id: Final = decoded.get("response_id", previous_response_id)
        if response_id in _visited:
            raise ValueError("Invalid cycle in stored previous_response_id history")
        if len(_visited) >= _MAX_COMPACTION_ANCESTRY:
            raise ValueError("Stored response ancestry exceeds the replay limit; replay the complete input")
        rows: Final = _REPLAY_ROWS_ADAPTER.validate_python(
            await prisma_client.db.query_raw(  # pyright: ignore[reportAny]  # generated Prisma raw-query result is validated here
                'SELECT * FROM "LiteLLM_SpendLogs" WHERE request_id = $1 ORDER BY "endTime" DESC LIMIT 1',
                response_id,
            )
        )
        if not rows:
            raise ValueError("Previous response history is not stored yet; retry or replay the complete input")
        row: Final = cast(  # cast-ok: validated replay fields consumed by legacy session helpers
            SpendLogsPayload, rows[0].model_dump()
        )
        body: Final = _StoredResponseBody.model_validate(
            await ResponsesSessionHandler.get_proxy_server_request_from_spend_log(row)  # pyright: ignore[reportUnknownMemberType]  # legacy cold-storage result is validated here
        )
        if not body.input:
            raise ValueError("Previous response input was not retained; replay the complete input for compaction")
        output: Final = _stored_output(row)
        if not output or not ("choices" in output or "output" in output):
            raise ValueError("Previous response output was not retained; replay the complete input for compaction")
        if _has_unavailable_content(_JSON_ADAPTER.validate_python(body.model_dump())) or _has_unavailable_content(
            _JSON_ADAPTER.validate_python(output)
        ):
            raise ValueError(
                "Stored response history is redacted or truncated; replay the complete input for compaction"
            )
        previous_input: Final = body.input
        if replay_compaction_input(previous_input) is not previous_input:
            return (row,)
        parent: Final = body.previous_response_id
        if not parent:
            return (row,)
        ancestors: Final = await ResponsesSessionHandler.get_compaction_ancestry(
            parent, _visited=_visited | frozenset((response_id,))
        )
        return (*ancestors, row)

    @staticmethod
    async def extend_chat_completion_message_with_spend_log_payload(
        spend_log: SpendLogsPayload,
        chat_completion_message_history: Sequence[_ChatHistoryItem],
    ) -> list[_ChatHistoryItem]:  # mutable-ok: existing session API returns JSON messages
        """
        Extend the chat completion message history with the spend log payload
        """
        from litellm.responses.litellm_completion_transformation.transformation import (
            LiteLLMCompletionResponsesConfig,
        )

        request: Final = _JSON_OBJECT_ADAPTER.validate_python(
            await ResponsesSessionHandler.get_proxy_server_request_from_spend_log(  # pyright: ignore[reportUnknownMemberType]  # legacy cold-storage result is validated here
                spend_log=spend_log,
            )
            or _EMPTY_OBJECT
        )
        request_for_conversion: Final = dict(request)  # mutable-ok: existing converter accepts JSON request dicts
        raw_input: Final = request.get("input") or request.get("messages")
        response_input: Final = (
            _materialized_input(
                [raw_input]  # mutable-ok: lone Responses item uses a JSON list
                if isinstance(raw_input, dict)
                else raw_input
            )
            if raw_input
            else None
        )
        replayed: Final = replay_compaction_input(response_input) if response_input is not None else None
        prior: Final = () if replayed is not response_input else tuple(chat_completion_message_history)
        input_messages: Final = (
            LiteLLMCompletionResponsesConfig.transform_responses_api_input_to_messages(  # pyright: ignore[reportUnknownMemberType]  # legacy converter carries partially typed request annotations
                input=replayed, responses_api_request=request_for_conversion, replay_reasoning=True
            )
            if replayed is not None
            else ()
        )
        output: Final = _stored_output(spend_log)
        if output.get("object") == _RESPONSE_OBJECT or "output" in output:
            output_messages: Final = LiteLLMCompletionResponsesConfig.transform_responses_api_input_to_messages(  # pyright: ignore[reportUnknownMemberType]  # legacy converter carries partially typed request annotations
                input=_materialized_input(output.get("output")),
                responses_api_request={},  # mutable-ok: existing converter accepts JSON request dicts
                replay_reasoning=True,
            )
            return [*prior, *input_messages, *output_messages]  # mutable-ok: existing session API returns JSON messages
        model_response: Final = ModelResponse.model_validate(output) if output else None
        choices: Final = tuple(model_response.choices) if model_response is not None else ()
        for choice in choices:
            if hasattr(choice, "message"):
                _normalize_redacted_tool_call_arguments(choice.message)
        return [  # mutable-ok: existing session API returns JSON messages
            *prior,
            *input_messages,
            *(choice.message for choice in choices if hasattr(choice, "message")),
        ]

    @staticmethod
    async def get_proxy_server_request_from_spend_log(
        spend_log: SpendLogsPayload,
    ) -> dict | None:
        """
        Get the parsed proxy server request from the spend log
        """
        proxy_server_request: Final[str | dict] = spend_log.get("proxy_server_request") or "{}"
        proxy_server_request_dict: dict | None = None
        if isinstance(proxy_server_request, dict):
            proxy_server_request_dict = proxy_server_request
        else:
            proxy_server_request_dict = json.loads(proxy_server_request)

        ############################################################
        # Check if user has setup cold storage for session handling
        ############################################################
        if ResponsesSessionHandler._should_check_cold_storage_for_full_payload(proxy_server_request_dict):
            # Try to get cold storage object key from spend log metadata
            _proxy_server_request_dict: dict | None = None
            cold_storage_object_key = ResponsesSessionHandler._get_cold_storage_object_key_from_spend_log(spend_log)
            if cold_storage_object_key:
                # Use the object key directly from metadata
                _proxy_server_request_dict = (
                    await ResponsesSessionHandler.get_proxy_server_request_from_cold_storage_with_object_key(
                        object_key=cold_storage_object_key,
                    )
                )
            if _proxy_server_request_dict:
                proxy_server_request_dict = _proxy_server_request_dict

        return proxy_server_request_dict

    @staticmethod
    def _get_cold_storage_object_key_from_spend_log(
        spend_log: SpendLogsPayload,
    ) -> str | None:
        """
        Extract the cold storage object key from spend log metadata.

        Args:
            spend_log: The spend log payload containing metadata

        Returns:
            Optional[str]: The cold storage object key if found, None otherwise
        """
        try:
            metadata_str: Final = spend_log.get("metadata", "{}")
            if isinstance(metadata_str, str):
                metadata_dict: Final[SpendLogsMetadata] = json.loads(metadata_str)
                return metadata_dict.get("cold_storage_object_key")
            elif isinstance(metadata_str, dict):
                return metadata_str.get("cold_storage_object_key")
            return None
        except (json.JSONDecodeError, TypeError, AttributeError):
            verbose_proxy_logger.debug("Failed to parse metadata from spend log to extract cold storage object key")
            return None

    @staticmethod
    async def get_proxy_server_request_from_cold_storage_with_object_key(
        object_key: str,
    ) -> dict | None:
        """
        Get the proxy server request from cold storage using the object key directly.

        Args:
            object_key: The S3/GCS object key to retrieve

        Returns:
            Optional[dict]: The proxy server request dict or None if not found
        """
        verbose_proxy_logger.debug("inside get_proxy_server_request_from_cold_storage_with_object_key...")

        proxy_server_request_dict: Final = (
            await COLD_STORAGE_HANDLER.get_proxy_server_request_from_cold_storage_with_object_key(
                object_key=object_key,
            )
        )

        return proxy_server_request_dict

    @staticmethod
    def _should_check_cold_storage_for_full_payload(
        proxy_server_request_dict: dict | None,
    ) -> bool:
        """
        Only check cold storage when both are true
        1. `LITELLM_TRUNCATED_PAYLOAD_FIELD` is in the proxy server request dict
        2. `litellm.cold_storage_custom_logger` is not None
        """
        from litellm.constants import LITELLM_TRUNCATED_PAYLOAD_FIELD

        configured_cold_storage_custom_logger: Final = litellm.cold_storage_custom_logger
        if configured_cold_storage_custom_logger is None:
            return False
        if proxy_server_request_dict is None:
            return True
        if len(proxy_server_request_dict) == 0:
            return True
        if LITELLM_TRUNCATED_PAYLOAD_FIELD in str(proxy_server_request_dict):
            return True
        return False

    @staticmethod
    async def get_all_spend_logs_for_previous_response_id(
        previous_response_id: str,
    ) -> list[SpendLogsPayload]:
        """
        Get all spend logs for a previous response id


        SQL query

        SELECT session_id FROM spend_logs WHERE response_id = previous_response_id, SELECT * FROM spend_logs WHERE session_id = session_id

        A just-finished turn gets a short second chance: the worker that served it may
        still be writing its spend log when the follow-up arrives, and an empty result
        drops the whole conversation instead of erroring. Deployments that write no spend
        logs at all have nothing to wait for, so they keep the single original query.
        """
        from litellm.constants import (
            RESPONSES_SESSION_LOOKUP_MAX_ATTEMPTS,
            RESPONSES_SESSION_LOOKUP_RETRY_INTERVAL,
        )
        from litellm.proxy.proxy_server import disable_spend_logs, prisma_client

        verbose_proxy_logger.debug("decoding response id=%s", previous_response_id)

        decoded_response_id: Final = ResponsesAPIRequestUtils._decode_responses_api_response_id(previous_response_id)
        response_id: Final = decoded_response_id.get("response_id", previous_response_id)
        if prisma_client is None:
            return []

        query: Final = """
            WITH matching_session AS (
                SELECT session_id
                FROM "LiteLLM_SpendLogs"
                WHERE request_id = $1
            )
            SELECT *
            FROM "LiteLLM_SpendLogs"
            WHERE session_id IN (SELECT session_id FROM matching_session)
            ORDER BY "endTime" ASC;
        """

        max_attempts: Final = 1 if disable_spend_logs else RESPONSES_SESSION_LOOKUP_MAX_ATTEMPTS
        for attempt in range(max_attempts):
            if attempt:
                await asyncio.sleep(RESPONSES_SESSION_LOOKUP_RETRY_INTERVAL)
            if spend_logs := await prisma_client.db.query_raw(query, response_id):
                verbose_proxy_logger.debug(
                    "Found the following spend logs for previous response id %s: %s",
                    response_id,
                    json.dumps(spend_logs, indent=4, default=str),
                )
                return spend_logs

        verbose_proxy_logger.debug("Found no spend logs for previous response id %s", response_id)
        return []  # mutable-ok: an empty result the caller only reads
