import base64
import binascii
import json
from typing import Final

import pytest
from openai.types.responses.response_input_param import ResponseInputParam
from pydantic import TypeAdapter, ValidationError

from litellm.responses.compaction_history import (
    GATEWAY_COMPACTION_PREFIX,
    create_compaction_item,
    has_gateway_compaction,
    replay_compaction_input,
    split_compaction_input,
)
from litellm.responses.utils import ResponsesAPIRequestUtils


def _input(value: object) -> ResponseInputParam:
    return TypeAdapter(ResponseInputParam).validate_python(value)


def _artifact(summary: str, retained: ResponseInputParam) -> ResponseInputParam:
    item: Final = create_compaction_item(summary, retained)
    return _input(json.loads(f"[{item.model_dump_json()}]"))


def test_serialized_roundtrip_preserves_instructions_active_task_and_tool_pairs() -> None:
    history: Final = _input(
        [
            {"role": "system", "content": "Keep the safety requirements"},
            {"role": "user", "content": "Old task"},
            {"role": "assistant", "content": "Old answer"},
            {"role": "developer", "content": "Keep the output format"},
            {"role": "user", "content": "Current task"},
            {"type": "function_call", "call_id": "call_1", "name": "lookup", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call_1", "output": "found"},
        ]
    )
    prefix, tail = split_compaction_input(history)
    assert prefix == [*history[1:3], history[4]]
    assert tail == [history[0], *history[3:]]
    newer: Final = _input([{"role": "user", "content": "Follow up"}])
    replayed: Final = replay_compaction_input([*history, *_artifact("Old work summary", tail), *newer])
    assert isinstance(replayed, list)
    assert replayed[0].get("role") == "user"
    assert "Old work summary" in str(replayed[0].get("content"))
    assert replayed[1:] == [*tail, *newer]
    assert history[1] not in replayed


def test_latest_artifact_only_and_previous_summary_is_summarizable() -> None:
    active: Final = _input([{"role": "user", "content": "Task"}])
    old: Final = _artifact("First summary", active)
    latest: Final = _artifact("Combined second summary", active)
    replayed: Final = replay_compaction_input([*old, *active, *latest])
    assert isinstance(replayed, list)
    assert len(replayed) == 2
    assert "First summary" not in str(replayed)
    assert "Combined second summary" in str(replayed[0])
    restored: Final = _input(json.loads(json.dumps(replayed)))
    prefix, tail = split_compaction_input(restored)
    assert prefix == replayed[:1]
    assert tail == active


def test_model_affinity_wrapper_is_unwrapped() -> None:
    item: Final = create_compaction_item("Wrapped summary", _input([{"role": "user", "content": "Task"}]))
    wrapped: Final = ResponsesAPIRequestUtils._wrap_encrypted_content_with_model_id(item.encrypted_content, "model-id")
    replayed: Final = replay_compaction_input(
        _input([{"type": "compaction", "id": item.id, "encrypted_content": wrapped}])
    )
    assert isinstance(replayed, list)
    assert "Wrapped summary" in str(replayed[0])
    assert replayed[1].get("content") == "Task"


def test_native_and_plain_input_pass_through_unchanged() -> None:
    native: Final = _input([{"type": "compaction", "id": "native", "encrypted_content": "opaque-provider-value"}])
    assert replay_compaction_input("hello") == "hello"
    assert replay_compaction_input(native) is native
    replayed: Final = replay_compaction_input([*native, *_artifact("Gateway summary", [])])
    assert isinstance(replayed, list)
    assert replayed[0] == native[0]


@pytest.mark.parametrize(
    "suffix",
    [
        "not-base64!",
        base64.b64encode(b"not json").decode(),
        base64.b64encode(b'{"summary":"x","retained_input":[]}').decode(),
        base64.b64encode(b'{"version":2,"summary":"x","retained_input":[]}').decode(),
        base64.b64encode(b'{"version":true,"summary":"x","retained_input":[]}').decode(),
        base64.b64encode(b'{"version":1,"summary":"","retained_input":[]}').decode(),
        base64.b64encode(b'{"version":1,"summary":"x","retained_input":[{"type":"made-up"}]}').decode(),
    ],
)
def test_malformed_recognized_artifacts_error_even_when_superseded(suffix: str) -> None:
    malformed: Final = _input(
        [{"type": "compaction", "id": "bad", "encrypted_content": f"{GATEWAY_COMPACTION_PREFIX}{suffix}"}]
    )
    with pytest.raises((binascii.Error, ValidationError)):
        replay_compaction_input([*malformed, *_artifact("Valid later summary", [])])


def test_nested_gateway_artifacts_are_rejected() -> None:
    with pytest.raises(ValidationError, match="must replay"):
        create_compaction_item("summary", _artifact("nested", []))


def test_boundary_moves_back_through_interleaved_tool_pairs() -> None:
    history: Final = _input(
        [
            {"role": "user", "content": "Earlier task"},
            {"type": "function_call", "call_id": "a", "name": "first", "arguments": "{}"},
            {"type": "custom_tool_call", "call_id": "b", "name": "second", "input": "query"},
            {"type": "function_call_output", "call_id": "a", "output": "first result"},
            {"role": "user", "content": "Urgent current task"},
            {"type": "custom_tool_call_output", "call_id": "b", "output": "second result"},
        ]
    )
    prefix, tail = split_compaction_input(history)
    assert prefix == history[:1]
    assert tail == history[1:]


def test_unanswered_calls_and_absent_user_are_not_lost() -> None:
    pending: Final = _input(
        [
            {"role": "user", "content": "Earlier task"},
            {"type": "function_call", "call_id": "pending", "name": "work", "arguments": "{}"},
            {"role": "user", "content": "New task"},
        ]
    )
    prefix, tail = split_compaction_input(pending)
    assert prefix == pending[:1]
    assert tail == pending[1:]
    assert split_compaction_input(pending[1:2]) == ([], pending[1:2])


def test_client_tool_search_pair_stays_together() -> None:
    history: Final = _input(
        [
            {"role": "user", "content": "Old task"},
            {"type": "tool_search_call", "call_id": "search", "arguments": {}, "execution": "client"},
            {"role": "user", "content": "Current task"},
            {"type": "tool_search_output", "call_id": "search", "tools": [], "execution": "client"},
        ]
    )
    assert split_compaction_input(history) == (history[:1], history[1:])


def test_new_instructions_before_artifact_survive_and_input_is_not_mutated() -> None:
    retained: Final = _input([{"role": "user", "content": "Task"}])
    instructions: Final = _input([{"role": "developer", "content": "New policy"}])
    history: Final = [*instructions, *_artifact("summary", retained)]
    before: Final = json.dumps(history)
    replayed: Final = replay_compaction_input(history)
    assert isinstance(replayed, list)
    assert replayed[0] == instructions[0]
    assert json.dumps(history) == before


def test_summary_like_instruction_is_not_mistaken_for_gateway_summary() -> None:
    replayed: Final = replay_compaction_input(_artifact("summary", []))
    assert isinstance(replayed, list)
    instruction: Final = _input([{"role": "system", "content": replayed[0].get("content")}])
    active: Final = _input([{"role": "user", "content": "Task"}])
    prefix, tail = split_compaction_input([*instruction, *replayed, *active])
    assert prefix == replayed
    assert tail == [*instruction, *active]


def test_serialized_expanded_history_recompacts_summary_without_promoting_instructions() -> None:
    retained: Final = _input(
        [
            {"role": "system", "content": "System constraints"},
            {"role": "developer", "content": "Developer constraints"},
            {"role": "user", "content": "Active task"},
        ]
    )
    first: Final = replay_compaction_input(_artifact("First summary", retained))
    restored: Final = _input(json.loads(json.dumps(first)))
    assert restored[0].get("role") == "user"
    prefix, tail = split_compaction_input(restored)
    assert prefix == restored[:1]
    assert tail == retained
    second: Final = replay_compaction_input(_artifact("Combined summary", tail))
    assert isinstance(second, list)
    assert "First summary" not in str(second)
    assert "Combined summary" in str(second[0])
    assert second[1:] == retained


def test_summary_without_a_retained_task_waits_for_next_user_before_compacting() -> None:
    replayed: Final = replay_compaction_input(_artifact("Summary only", []))
    restored: Final = _input(json.loads(json.dumps(replayed)))
    assert split_compaction_input(restored) == ([], restored)
    active: Final = _input([{"role": "user", "content": "New current task"}])
    assert split_compaction_input([*restored, *active]) == (restored, active)


def test_artifact_roundtrip_materializes_sdk_iterable_content() -> None:
    retained: Final = _input(
        [
            {"role": "user", "content": [{"type": "input_text", "text": "Current task"}]},
            {
                "type": "message",
                "role": "assistant",
                "id": "message-1",
                "status": "completed",
                "content": [{"type": "output_text", "text": "Partial progress", "annotations": []}],
            },
        ]
    )
    replayed: Final = replay_compaction_input(_artifact("summary", retained))
    assert isinstance(replayed, list)
    assert isinstance(replayed[1].get("content"), list)
    assert isinstance(replayed[2].get("content"), list)
    assert "Current task" in json.dumps(replayed)
    assert "Partial progress" in json.dumps(replayed)


def test_gateway_detection_does_not_validate_unknown_provider_shapes() -> None:
    assert not has_gateway_compaction("plain input")
    assert not has_gateway_compaction(None)
    assert not has_gateway_compaction([None, 7, {"type": "vendor-extension", "opaque": object()}])
    assert not has_gateway_compaction([{"type": "compaction", "encrypted_content": "provider-opaque"}])
    assert not has_gateway_compaction([{"role": "user", "content": GATEWAY_COMPACTION_PREFIX}])


@pytest.mark.parametrize("wrap_affinity", [False, True])
def test_gateway_detection_recognizes_malformed_gateway_payload_before_validation(wrap_affinity: bool) -> None:
    malformed: Final = f"{GATEWAY_COMPACTION_PREFIX}not-base64!"
    content: Final = (
        ResponsesAPIRequestUtils._wrap_encrypted_content_with_model_id(malformed, "fixture-model")
        if wrap_affinity
        else malformed
    )
    assert has_gateway_compaction([{"type": "compaction", "encrypted_content": content}])
    with pytest.raises(binascii.Error):
        replay_compaction_input(_input([{"type": "compaction", "id": "bad", "encrypted_content": content}]))


def _completed_round(call_id: str) -> ResponseInputParam:
    return _input(
        [
            {"type": "reasoning", "id": f"reason-{call_id}", "summary": [], "encrypted_content": f"signed-{call_id}"},
            {"type": "function_call", "call_id": call_id, "name": "lookup", "arguments": "{}"},
            {"type": "function_call_output", "call_id": call_id, "output": f"result-{call_id}"},
        ]
    )


@pytest.mark.parametrize("completed_rounds", [2, 6])
def test_one_user_long_tool_run_compacts_completed_rounds_and_preserves_current_task(completed_rounds: int) -> None:
    instructions: Final = _input([{"role": "developer", "content": "Do not change the requirements"}])
    task: Final = _input([{"role": "user", "content": "Investigate every failing subsystem and repair the root cause"}])
    rounds: Final = tuple(_completed_round(f"call-{index}") for index in range(completed_rounds))
    history: Final = [*instructions, *task, *(item for group in rounds for item in group)]
    prefix, tail = split_compaction_input(history)
    assert prefix == [*task, *(item for group in rounds[:-1] for item in group)]
    assert tail == [*instructions, *task, *rounds[-1]]
    assert tail[-3].get("encrypted_content") == f"signed-call-{completed_rounds - 1}"
    assert task[0] in prefix and task[0] in tail


def test_parallel_calls_keep_latest_entire_action_group_and_original_output_order() -> None:
    task: Final = _input([{"role": "user", "content": "Run both checks, then explain their differences"}])
    earlier: Final = _completed_round("earlier")
    latest: Final = _input(
        [
            {"type": "reasoning", "id": "reason-latest", "summary": [], "encrypted_content": "signed-latest"},
            {"role": "assistant", "content": "Checking both sources"},
            {"role": "developer", "content": "Preserve this instruction inside the action group"},
            {"type": "function_call", "call_id": "a", "name": "first", "arguments": "{}"},
            {"type": "custom_tool_call", "call_id": "b", "name": "second", "input": "query"},
            {"type": "custom_tool_call_output", "call_id": "b", "output": "second result"},
            {"type": "function_call_output", "call_id": "a", "output": "first result"},
        ]
    )
    prefix, tail = split_compaction_input([*task, *earlier, *latest])
    assert prefix == [*task, *earlier]
    assert tail == [*task, *latest]


@pytest.mark.parametrize("finish_pending", [False, True])
def test_backward_closure_keeps_earlier_reasoning_and_parallel_group(finish_pending: bool) -> None:
    task: Final = _input([{"role": "user", "content": "Finish the whole investigation"}])
    earlier: Final = _completed_round("earlier")
    parallel: Final = _input(
        [
            {"type": "reasoning", "id": "reason-parallel", "summary": [], "encrypted_content": "signed-parallel"},
            {"role": "assistant", "content": "Both tool calls belong to this action"},
            {"type": "function_call", "call_id": "a", "name": "first", "arguments": "{}"},
            {"type": "function_call", "call_id": "b", "name": "second", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "a", "output": "first result"},
        ]
    )
    latest: Final = _completed_round("latest")
    late_output: Final = (
        _input([{"type": "function_call_output", "call_id": "b", "output": "late second result"}])
        if finish_pending
        else []
    )
    prefix, tail = split_compaction_input([*task, *earlier, *parallel, *latest, *late_output])
    assert prefix == [*task, *earlier]
    assert tail == [*task, *parallel, *latest, *late_output]
    assert tail[1].get("encrypted_content") == "signed-parallel"


def test_new_user_is_retained_while_earlier_finished_tool_rounds_are_summarized() -> None:
    original: Final = _input([{"role": "user", "content": "Original task"}])
    earlier: Final = _completed_round("earlier")
    current: Final = _input([{"role": "user", "content": "Change only the reported failing condition"}])
    latest: Final = _completed_round("latest")
    prefix, tail = split_compaction_input([*original, *earlier, *current, *latest])
    assert prefix == [*original, *earlier, *current]
    assert tail == [*current, *latest]


def test_single_current_action_group_has_no_discardable_history() -> None:
    instructions: Final = _input([{"role": "system", "content": "Required policy"}])
    task: Final = _input([{"role": "user", "content": "Current task"}])
    latest: Final = _completed_round("only")
    history: Final = [*instructions, *task, *latest]
    assert split_compaction_input(history) == ([], history)
    assert split_compaction_input([*instructions, *task]) == ([], [*instructions, *task])


def test_native_call_and_signed_reasoning_stay_in_the_latest_action_group() -> None:
    task: Final = _input([{"role": "user", "content": "Compare the sources"}])
    earlier: Final = _completed_round("earlier")
    latest: Final = _input(
        [
            {"type": "reasoning", "id": "reason-native", "summary": [], "encrypted_content": "signed-native"},
            {
                "type": "web_search_call",
                "id": "search-native",
                "status": "completed",
                "action": {"type": "search", "query": "sources"},
            },
            {"role": "assistant", "content": "The sources agree"},
        ]
    )
    prefix, tail = split_compaction_input([*task, *earlier, *latest])
    assert prefix == [*task, *earlier]
    assert tail == [*task, *latest]


def test_latest_artifact_replays_before_compacting_a_continued_single_task_tool_run() -> None:
    task: Final = _input([{"role": "user", "content": "Continue until every test passes"}])
    retained: Final = [*task, *_completed_round("retained")]
    old: Final = _artifact("Obsolete summary", task)
    latest: Final = _artifact("Latest authoritative summary", retained)
    new_round: Final = _completed_round("new")
    prefix, tail = split_compaction_input([*old, *latest, *new_round])
    assert "Obsolete summary" not in str(prefix)
    assert "Latest authoritative summary" in str(prefix[0])
    assert prefix[1].get("content") == task[0].get("content")
    assert any(item.get("call_id") == "retained" for item in prefix)
    assert tail == [*task, *new_round]
    assert all(item.get("type") != "compaction" for item in [*prefix, *tail])
