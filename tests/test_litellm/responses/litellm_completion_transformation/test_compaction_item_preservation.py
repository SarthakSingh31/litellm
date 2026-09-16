"""
Round-trip of Anthropic ``compaction`` content blocks through the
Responses API -> chat completions bridge: ``compaction_blocks`` on the
chat message become a ``compaction`` output item, and a replayed
``compaction`` input item becomes an assistant message carrying
``provider_specific_fields["compaction_blocks"]`` again.
"""

import json
import logging
from collections.abc import Mapping, Sequence

import pytest

from litellm.litellm_core_utils.prompt_templates.factory import anthropic_messages_pt
from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
)
from litellm.types.llms.openai import ResponsesAPIResponse
from litellm.types.utils import Choices, Message, ModelResponse

COMPACTION_BLOCK = {"type": "compaction", "content": "Summary: the user is building a web scraper."}
SIGNED_THINKING = {"type": "thinking", "thinking": "hidden", "signature": "sig-1"}


def _chat_response(message: Message) -> ModelResponse:
    return ModelResponse(
        id="chatcmpl-1",
        created=1,
        model="claude-sonnet-5",
        object="chat.completion",
        choices=[Choices(index=0, finish_reason="stop", message=message)],
    )


def _to_responses(message: Message) -> ResponsesAPIResponse:
    return LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
        request_input="hi", responses_api_request={}, chat_completion_response=_chat_response(message)
    )


def _transform_item(item: Mapping[str, object]) -> list:
    return LiteLLMCompletionResponsesConfig._transform_responses_api_input_item_to_chat_completion_message(
        input_item=item, replay_reasoning=True
    )


def _transform_input(items: Sequence[Mapping[str, object]]) -> list:
    return LiteLLMCompletionResponsesConfig._transform_response_input_param_to_chat_completion_message(
        input=items, replay_reasoning=True
    )


def _compaction_item(encrypted_content: str = json.dumps([COMPACTION_BLOCK])) -> dict:
    return {"type": "compaction", "id": "cmp_1", "encrypted_content": encrypted_content}


class TestCompactionOutputItem:
    def test_compaction_block_becomes_first_output_item(self):
        response = _to_responses(
            Message(
                role="assistant",
                content="Continuing after compaction.",
                thinking_blocks=[SIGNED_THINKING],
                provider_specific_fields={"compaction_blocks": [COMPACTION_BLOCK]},
            )
        )
        assert [item.type for item in response.output] == ["compaction", "reasoning", "message"]
        compaction = response.output[0]
        assert compaction.id.startswith("cmp_")
        assert json.loads(compaction.encrypted_content) == [COMPACTION_BLOCK]

    def test_message_without_compaction_blocks_emits_no_compaction_item(self):
        response = _to_responses(
            Message(role="assistant", content="plain", provider_specific_fields={"citations": None})
        )
        assert [item.type for item in response.output] == ["message"]

    def test_block_without_summary_emits_no_compaction_item(self):
        response = _to_responses(
            Message(
                role="assistant",
                content="plain",
                provider_specific_fields={"compaction_blocks": [{"type": "compaction", "content": ""}]},
            )
        )
        assert [item.type for item in response.output] == ["message"]


class TestCompactionInputItem:
    def test_compaction_item_becomes_assistant_carrier(self):
        messages = _transform_item(_compaction_item())
        assert len(messages) == 1
        assert messages[0]["role"] == "assistant"
        assert messages[0]["content"] is None
        assert messages[0]["provider_specific_fields"] == {"compaction_blocks": [COMPACTION_BLOCK]}

    def test_compaction_merges_into_following_assistant_message(self):
        messages = _transform_input(
            [
                {"role": "user", "content": "long history"},
                _compaction_item(),
                {"type": "reasoning", "id": "rs_1", "encrypted_content": json.dumps([SIGNED_THINKING])},
                {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "answer"}]},
                {"role": "user", "content": "next"},
            ]
        )
        assistants = [m for m in messages if m["role"] == "assistant"]
        assert len(assistants) == 1
        assert assistants[0]["provider_specific_fields"]["compaction_blocks"] == [COMPACTION_BLOCK]
        assert assistants[0]["thinking_blocks"] == [SIGNED_THINKING]
        assert [m["role"] for m in messages] == ["user", "assistant", "user"]

    def test_compaction_before_function_call_keeps_one_assistant_turn(self):
        messages = _transform_input(
            [
                _compaction_item(),
                {"type": "function_call", "call_id": "call_1", "name": "get_weather", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "call_1", "output": "sunny"},
            ]
        )
        assert [m["role"] for m in messages] == ["assistant", "tool"]
        assert messages[0]["provider_specific_fields"]["compaction_blocks"] == [COMPACTION_BLOCK]
        assert messages[0]["tool_calls"][0]["id"] == "call_1"

    def test_compaction_not_followed_by_assistant_stays_standalone(self):
        messages = _transform_input([_compaction_item(), {"role": "user", "content": "next"}])
        assert [m["role"] for m in messages] == ["assistant", "user"]
        assert messages[0]["provider_specific_fields"]["compaction_blocks"] == [COMPACTION_BLOCK]

    def test_opaque_compaction_item_dropped_with_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger="LiteLLM"):
            messages = _transform_item(_compaction_item(encrypted_content="not-written-by-litellm"))
        assert messages == []
        assert any("compaction" in record.getMessage() for record in caplog.records if record.name == "LiteLLM")

    def test_pending_compaction_precedes_blocks_already_on_the_target(self):
        older_block = {"type": "compaction", "content": "Older summary."}
        messages = LiteLLMCompletionResponsesConfig._merge_reasoning_only_assistant_messages(
            [
                _transform_item(_compaction_item())[0],
                {
                    "role": "assistant",
                    "content": "answer",
                    "provider_specific_fields": {"compaction_blocks": [older_block], "citations": ["c1"]},
                },
            ]
        )
        assert len(messages) == 1
        assert messages[0]["provider_specific_fields"] == {
            "compaction_blocks": [COMPACTION_BLOCK, older_block],
            "citations": ["c1"],
        }

    def test_inspection_mode_drops_compaction_item(self):
        messages = LiteLLMCompletionResponsesConfig.transform_responses_api_input_to_messages(
            input=[_compaction_item(), {"role": "user", "content": "next"}], responses_api_request={}
        )
        assert [m["role"] for m in messages] == ["user"]

    def test_unknown_item_without_content_is_still_dropped(self):
        assert _transform_item({"type": "mystery_item", "id": "x"}) == []


class TestCompactionRoundTrip:
    def test_output_item_decodes_back_to_the_same_blocks(self):
        response = _to_responses(
            Message(role="assistant", content="x", provider_specific_fields={"compaction_blocks": [COMPACTION_BLOCK]})
        )
        replayed = _transform_item(
            {
                "type": "compaction",
                "id": response.output[0].id,
                "encrypted_content": response.output[0].encrypted_content,
            }
        )
        assert replayed[0]["provider_specific_fields"]["compaction_blocks"] == [COMPACTION_BLOCK]

    def test_vendor_fields_on_the_block_survive_the_round_trip(self):
        block = {**COMPACTION_BLOCK, "id": "compact_abc", "future_field": {"nested": 1}}
        response = _to_responses(
            Message(role="assistant", content="x", provider_specific_fields={"compaction_blocks": [block]})
        )
        replayed = _transform_item({"type": "compaction", "encrypted_content": response.output[0].encrypted_content})
        assert replayed[0]["provider_specific_fields"]["compaction_blocks"] == [block]

    def test_replayed_block_is_first_in_anthropic_assistant_content(self):
        messages = _transform_input(
            [
                {"role": "user", "content": "long history"},
                _compaction_item(),
                {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "answer"}]},
                {"role": "user", "content": "next"},
            ]
        )
        anthropic_messages = anthropic_messages_pt(messages=messages, model="claude-sonnet-5", llm_provider="anthropic")
        assistant = anthropic_messages[1]
        assert assistant["role"] == "assistant"
        assert assistant["content"][0] == COMPACTION_BLOCK
        assert [block["type"] for block in assistant["content"]] == ["compaction", "text"]


class TestCompactionProviderGating:
    @staticmethod
    def _request_messages(custom_llm_provider: str | None):
        request = LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request(
            model="any-model",
            input=[
                _compaction_item(),
                {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "answer"}]},
                {"role": "user", "content": "next"},
            ],
            responses_api_request={},
            custom_llm_provider=custom_llm_provider,
        )
        return request["messages"]

    @pytest.mark.parametrize("provider", ["anthropic", "bedrock", "vertex_ai", None])
    def test_consumers_keep_compaction_blocks(self, provider):
        messages = self._request_messages(provider)
        assert messages[0]["provider_specific_fields"]["compaction_blocks"] == [COMPACTION_BLOCK]

    def test_non_consumer_drops_compaction_blocks_with_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger="LiteLLM"):
            messages = self._request_messages("openai")
        assert [m["role"] for m in messages] == ["assistant", "user"]
        assert "compaction_blocks" not in (messages[0].get("provider_specific_fields") or {})
        assert any("compaction" in record.getMessage() for record in caplog.records if record.name == "LiteLLM")

    def test_non_consumer_keeps_sibling_provider_fields(self):
        messages = LiteLLMCompletionResponsesConfig._without_compaction_blocks_for_provider(
            [
                {
                    "role": "assistant",
                    "content": "answer",
                    "provider_specific_fields": {"compaction_blocks": [COMPACTION_BLOCK], "citations": ["c1"]},
                },
                {"role": "user", "content": "next"},
            ],
            custom_llm_provider="openai",
        )
        assert messages[0]["provider_specific_fields"] == {"citations": ["c1"]}
        assert messages[0]["content"] == "answer"

    def test_non_consumer_drops_standalone_carrier_entirely(self):
        request = LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request(
            model="any-model",
            input=[_compaction_item(), {"role": "user", "content": "next"}],
            responses_api_request={},
            custom_llm_provider="openai",
        )
        assert [m["role"] for m in request["messages"]] == ["user"]
