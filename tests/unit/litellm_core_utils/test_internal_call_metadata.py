"""Unit tests for internal-call metadata forwarding: budget-reservation stripping and origin stamping."""

from copy import deepcopy
from typing import Final

import pytest
from pydantic import TypeAdapter

from litellm.constants import INTERNAL_CALL_ORIGIN_METADATA_KEY
from litellm.litellm_core_utils.internal_call_metadata import (
    forwarded_internal_call_metadata,
    is_unbilled_non_inference_call,
    is_unbilled_non_inference_call_from_params,
    responses_compaction_history_metadata,
    sanitized_forwardable_call_metadata,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.hooks.proxy_track_cost_callback import _get_budget_reservation_from_metadata
from litellm.types.utils import (
    BACKGROUND_RESPONSE_COST_POLL_CALL_ORIGIN,
    RESPONSES_COMPACTION_CALL_ORIGIN,
    SHADOW_EVAL_ROUTER_CALL_ORIGIN,
)

PARENT = {
    "user_api_key": "sk-hash",
    "user_api_key_hash": "sk-hash",
    "user_api_key_team_id": "team-1",
    "user_api_key_budget_reservation": {"amount": 1.0},
    "user_api_key_auth": {"api_key": "sk-hash", "budget_reservation": {"amount": 1.0}},
    "routing_decision": {"router_model_name": "my-router"},
    "headers": {"x-request-id": "abc"},
}


def test_forwarded_metadata_strips_reservation_everywhere_and_stamps_origin():
    result = forwarded_internal_call_metadata(PARENT, "autorouter_classifier")

    assert result[INTERNAL_CALL_ORIGIN_METADATA_KEY] == "autorouter_classifier"
    assert "user_api_key_budget_reservation" not in result
    assert result["user_api_key_auth"] == {"api_key": "sk-hash"}
    assert result["routing_decision"] == {"router_model_name": "my-router"}
    assert PARENT["user_api_key_auth"]["budget_reservation"] is not None


def test_forwarded_metadata_empty_parent_stays_unstamped():
    assert forwarded_internal_call_metadata(None, "autorouter_classifier") == {}
    assert forwarded_internal_call_metadata({}, "autorouter_classifier") == {}


def test_sanitized_forwardable_metadata_keeps_only_identity_and_always_stamps():
    result = sanitized_forwardable_call_metadata(PARENT, SHADOW_EVAL_ROUTER_CALL_ORIGIN)

    assert result[INTERNAL_CALL_ORIGIN_METADATA_KEY] == SHADOW_EVAL_ROUTER_CALL_ORIGIN
    assert result["user_api_key"] == "sk-hash"
    assert result["user_api_key_team_id"] == "team-1"
    assert result["user_api_key_auth"] == {"api_key": "sk-hash"}
    assert "routing_decision" not in result
    assert "headers" not in result
    assert "user_api_key_budget_reservation" not in result

    assert sanitized_forwardable_call_metadata({}, SHADOW_EVAL_ROUTER_CALL_ORIGIN) == {
        INTERNAL_CALL_ORIGIN_METADATA_KEY: SHADOW_EVAL_ROUTER_CALL_ORIGIN
    }


class TestSubCallMetadataSanitization:
    """The proxy cost callback must not be able to recover the parent budget reservation
    from sub-call metadata, in either of the shapes it knows how to read."""

    def test_cost_callback_cannot_recover_reservation_from_sanitized_metadata(self):
        from litellm.proxy._types import UserAPIKeyAuth
        from litellm.proxy.hooks.proxy_track_cost_callback import (
            _get_budget_reservation_from_metadata,
        )

        reservation = {"reserved_cost": 1.0}
        auth_shapes = (
            {"models": ["gpt-4o"], "budget_reservation": dict(reservation)},
            UserAPIKeyAuth(api_key="sk-abc", budget_reservation=dict(reservation)),
        )
        for auth in auth_shapes:
            metadata = {
                "user_api_key_hash": "hash-abc",
                "user_api_key_budget_reservation": dict(reservation),
                "user_api_key_auth": auth,
            }
            assert _get_budget_reservation_from_metadata(metadata) == reservation

            sanitized = forwarded_internal_call_metadata(metadata, "autorouter_classifier")
            assert sanitized is not None
            assert sanitized["user_api_key_auth"] is not None
            assert _get_budget_reservation_from_metadata(sanitized) is None

    def test_classifier_buckets_keep_non_spend_fields_on_a_chat_completions_parent(self):
        """Drives the real resolver over the buckets the embedding classifier builds.

        An absent bucket must stay empty rather than carry a lone origin stamp:
        get_litellm_metadata_from_kwargs prefers litellm_metadata whenever truthy, so an
        origin-only dict would make an empty litellm_metadata win and silently drop
        requester_ip_address, tags and spend_logs_metadata from the classifier's row."""
        from litellm.litellm_core_utils.core_helpers import get_litellm_metadata_from_kwargs

        parent = {
            "user_api_key": "sk-abc",
            "requester_ip_address": "10.0.0.1",
            "spend_logs_metadata": {"team_note": "keep me"},
            "tags": ["prod"],
        }
        resolved = get_litellm_metadata_from_kwargs(
            {
                "litellm_params": {
                    "metadata": forwarded_internal_call_metadata(parent, "autorouter_classifier"),
                    "litellm_metadata": forwarded_internal_call_metadata(None, "autorouter_classifier"),
                }
            }
        )
        assert resolved["internal_call_origin"] == "autorouter_classifier"
        assert resolved["requester_ip_address"] == "10.0.0.1"
        assert resolved["spend_logs_metadata"] == {"team_note": "keep me"}
        assert resolved["tags"] == ["prod"]

    def test_sanitized_auth_keeps_access_group_fields_and_leaves_original_untouched(self):
        from litellm.proxy._types import UserAPIKeyAuth

        auth = UserAPIKeyAuth(
            api_key="sk-abc",
            team_id="team-1",
            budget_reservation={"reserved_cost": 1.0},
        )
        sanitized = forwarded_internal_call_metadata({"user_api_key_auth": auth}, "autorouter_classifier")
        sanitized_auth = sanitized["user_api_key_auth"]
        assert sanitized_auth.budget_reservation is None
        assert sanitized_auth.team_id == "team-1"
        assert sanitized_auth.api_key == auth.api_key
        assert auth.budget_reservation == {"reserved_cost": 1.0}


@pytest.mark.parametrize("auth_shape", ["mapping", "model"])
def test_compaction_history_metadata_strips_parent_reservation_without_mutation(auth_shape: str) -> None:
    reservation: Final = {"reserved_cost": 2.0, "finalized": False}
    auth: Final = (
        {"api_key": "test-key", "budget_reservation": reservation}
        if auth_shape == "mapping"
        else UserAPIKeyAuth(api_key="test-key", budget_reservation=reservation)
    )
    parent: Final = {
        "user_api_key": "test-key",
        "user_api_key_auth": auth,
        "user_api_key_budget_reservation": reservation,
        "routing_decision": "selected-deployment",
    }
    before: Final = deepcopy(parent)
    history: Final = responses_compaction_history_metadata(parent)
    assert _get_budget_reservation_from_metadata(history) is None
    assert _get_budget_reservation_from_metadata(parent) is reservation
    assert history[INTERNAL_CALL_ORIGIN_METADATA_KEY] == RESPONSES_COMPACTION_CALL_ORIGIN
    assert history["user_api_key"] == "test-key"
    assert history["routing_decision"] == "selected-deployment"
    assert parent == before


@pytest.mark.parametrize("call_type", ["get_responses", "aget_responses", "list_input_items", "alist_input_items"])
def test_trusted_history_read_suppresses_old_background_usage_but_not_inference(call_type: str) -> None:
    history: Final = responses_compaction_history_metadata(None)
    copied: Final = deepcopy(history)
    response: Final = {"background": True, "usage": {"input_tokens": 1234, "output_tokens": 56}}
    assert is_unbilled_non_inference_call(call_type, copied, response)
    assert is_unbilled_non_inference_call_from_params(call_type, {"litellm_metadata": copied}, response)
    assert not is_unbilled_non_inference_call("aresponses", copied, response)
    assert not is_unbilled_non_inference_call("acompletion", copied, response)
    assert not is_unbilled_non_inference_call(call_type, None, response)


@pytest.mark.parametrize("forged_marker", [None, {}, True, "trusted", []])
def test_public_json_cannot_forge_compaction_history_cost_suppression(forged_marker: object) -> None:
    forged: Final = {
        **dict.fromkeys(responses_compaction_history_metadata(None), forged_marker),
        INTERNAL_CALL_ORIGIN_METADATA_KEY: RESPONSES_COMPACTION_CALL_ORIGIN,
    }
    assert not is_unbilled_non_inference_call("aget_responses", forged, {"background": True})


def test_history_marker_loses_authority_on_json_roundtrip_and_public_polling_still_bills() -> None:
    adapter: Final = TypeAdapter(dict[str, object])
    trusted: Final = responses_compaction_history_metadata({"user_api_key": "test-key"})
    public: Final = adapter.validate_json(adapter.dump_json(trusted))
    assert is_unbilled_non_inference_call("aget_responses", trusted, {"background": True})
    assert not is_unbilled_non_inference_call("aget_responses", public, {"background": True})
    assert not is_unbilled_non_inference_call(
        "aget_responses", {INTERNAL_CALL_ORIGIN_METADATA_KEY: RESPONSES_COMPACTION_CALL_ORIGIN}, {"background": True}
    )
    assert not is_unbilled_non_inference_call(
        "aget_responses", {INTERNAL_CALL_ORIGIN_METADATA_KEY: BACKGROUND_RESPONSE_COST_POLL_CALL_ORIGIN}, {}
    )
    assert is_unbilled_non_inference_call("aget_responses", None, {"background": False})
