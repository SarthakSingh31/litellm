# Gateway Responses compaction

Enable same-deployment summaries for Responses requests with `context_management`

```yaml
general_settings:
  context_management_summary_model: same_as_request
  context_management_summary_max_tokens: 4096

model_list:
  - model_name: selected-model
    litellm_params:
      model: openai/your-deployed-model
      api_key: os.environ/OPENAI_API_KEY
    model_info:
      supports_native_compaction: false
      max_input_tokens: 128000
      max_output_tokens: 16000
```

The model and limits above are illustrative. Use the limits of the actual deployment, or leave the limits to LiteLLM's model catalog. Mark `model_info.supports_native_compaction: true` only when that particular deployment supports native compaction. Such requests retain their native route and usage. Native Responses support alone does not establish native compaction support. With the same-model selector enabled, an undeclared capability is treated as unsupported

Existing explicit summary-model aliases retain their Messages-polyfill behavior. The `same_as_request` selector is implemented for Responses, not the Messages polyfill. Missing configuration leaves ordinary compaction forwarding unchanged. Gateway artifacts remain replayable even after disabling new gateway summaries

```json
{
  "model": "selected-model",
  "input": [
    {"role": "user", "content": "Earlier task"},
    {"role": "assistant", "content": "Earlier progress"},
    {"role": "user", "content": "Continue the task"}
  ],
  "context_management": [{"type": "compaction", "compact_threshold": 100000}],
  "stream": true
}
```

The threshold is measured against effective input, instructions and tool definitions. A positive explicit threshold is honored, including small-window models; omission defaults to 90% of the input limit. Compaction begins at the threshold. Summary output is capped by the configured summary budget, the model output limit and the remaining context space. A prompt that cannot fit is rejected without dropping arbitrary history

Summaries run against the selected model, provider, endpoint and credentials. Router calls retain deployment eligibility, budgets, region restrictions and separate per-call concurrency slots. User tools are provided only as summary context, never executable summary tools. The summary has its own logging identity and a bounded timeout. Once summary execution starts, retries stay pinned to that deployment and reuse prepared context; cross-model fallbacks are refused. Requests without gateway summary work retain normal fallback behavior

The response includes a standard `compaction` item with stable added/done stream events and matching terminal output indices. The parent response ID belongs to main generation. Replay the item together with newer output and input items to replace summarized history. The artifact retains instructions, the active task and recent tool call/result pairs. Its versioned `encrypted_content` payload is serialized, **not encrypted or authenticated**; do not treat it as a confidentiality boundary. Unrecognized provider-native payloads are not decoded as gateway summaries

A long tool loop can compact without waiting for another user message. The original user task and latest complete assistant action group stay verbatim; earlier completed exchanges enter the summary. Pending calls, parallel results and associated reasoning are kept together

Native `previous_response_id` continuation restores input items and output through the provider. Bridged continuation uses the existing stored request/response history, including prompt-storage and redaction policies. Missing or unavailable stored history is an error, not an empty conversation. Full client input replay is the alternative when storage is disabled. The original client transcript need not be deleted

## Usage and billing

The outward terminal usage adds reported summary and main input, output, cached input, cache-write input and reasoning output componentwise. Input is cache-inclusive and reasoning follows existing provider normalization. Total tokens are input plus output. Summary cache hits and replayed artifacts do not incur an additional summary contribution

Provider operation logs retain each call's own usage. Outward aggregation uses a separate response object, is request-local and idempotent, and clears main-only `usage.cost`. A consumer can keep its existing typed Responses client and record one usage row for the assistant turn. Pricing tiers selected from combined input intentionally apply to the whole turn, even when individual calls would have fallen below that tier. This does not force unrelated requests onto the model's highest tier

Summary and history reads do not inherit the main call's budget reservation. History reads use an internal process-local marker to avoid billing previously generated tokens again, including background response tokens. Caller-supplied JSON cannot forge this marker; public background retrieval and its cost poller keep their existing billing behavior

Combined usage describes billable work, not live context occupancy. Count effective input for future compaction thresholds; do not feed aggregate usage back as a context-size estimate

## Failure and validation limits

Summary extraction failure retains original effective context and measured consumption. A valid failed or incomplete main terminal includes reported summary consumption. If main usage is unavailable, the returned known consumption is a lower bound and a diagnostic is emitted. Missing summary usage fails explicitly rather than assuming a free execution

An exception or disconnected stream before a terminal reaches the caller can leave incurred usage unsettled by a terminal-only billing consumer. Normal operation logs remain available, but this feature adds no reconciliation service or guaranteed settlement for that case. Cancellation never fabricates a successful terminal

Stored prompt references must be expanded before gateway compaction so prompt management cannot change the selected model after summary work. Native passthrough and provider-specific history/authentication paths need deployment-level validation before rollout

Deterministic regressions exercise actual native and bridged HTTP transformations, serialized Responses streaming and replay, five-counter usage arithmetic, separate operation logs, retry pinning, deployment eligibility and concurrency. These tests do not establish live-provider behavior, downstream database settlement or a built deployment image
