# Jetson AGX Thor validation notes

These results describe one Jetson AGX Thor with 128 GB unified memory. They are
acceptance measurements for the exact software and model combination below,
not a general performance or accuracy claim for all Thor configurations.

## Runtime shape

- Per-request context: 262,144 tokens
- Shared KV pool: 557,056 tokens (544 Ki-token)
- Running request slots: 4
- Mamba state slots: 16
- Target and draft KV: FP8 E4M3
- Chunked prefill: 2,048 tokens
- Decode CUDA graph maximum batch size: 4
- Static memory fraction: 0.95
- Speculative decoding: NEXTN 3/1/4 with speculative ReplaySSM

Four request slots share the pool. Four full 262K requests do not fit; two
nearly full contexts do, while four simultaneous contexts have an average
budget of 139,264 tokens each.

## Dual-client long-context soak

The test ran for 2,400.94 seconds. Two client threads each completed three
cycles using the Project Gutenberg text of *The Count of Monte Cristo*:

1. Build a chat prompt close to 250K tokens.
2. Request a 1,024-token continuation.
3. Ask the model for a dense continuity record.
4. Keep the latter half of the source plus the record and continuation.
5. Replay source text to bring the next cycle back near 250K.

The replay in step 5 makes this a repeatable KV high-water and compaction test;
it is not equivalent to spending hours generating every intervening token.

- Actual continuation prompt: 249,933 tokens
- Actual compaction prompt: 249,949 tokens
- Requests completed: 12/12
- Compacted context: 126,994–128,084 tokens (50.81%–51.24%)
- Peak concurrent decode requests: 2
- Peak KV use: 503,232 / 557,056 tokens (90.34%)
- Lowest `MemAvailable`: 15,788,300 KiB (15.06 GiB)
- Swap growth: 21 MiB
- Service restarts: 0
- OOM, NaN, CUDA error, request retraction, or kernel OOM kill: none observed

KV use returned to zero at completion. After the client process exited,
`MemAvailable` stabilized around 16.75 GiB during a 40-second post-test sample.
This 40-minute run did not show continued growth, but it cannot exclude a much
slower leak that would require an 8–12 hour soak to observe.

## Mamba/NEXTN uptime guard

The pinned aarch64 runtime predates the accepted-state and empty-checkpoint
guards from SGLang PR #35821. `patches/sglang-mamba-radix-uptime.patch` ports
the three execution-path changes used by Flash-Next:

- never insert a zero-length Mamba radix checkpoint;
- bound eager/KDA tracking to the accepted speculative path;
- apply the same bound in the fused Triton commit kernel.

Without these guards, a zero-length ghost radix node or a state selected past
the accepted draft path can contaminate later Mamba state reuse. The installer
checks the exact before/after SHA-256 hashes and refuses unknown source states.

The public API name is the checkpoint that is actually served:
`hn7305/Qwen3.8-Flash-Next-NVFP4-Spark`.

## Reasoning and tool-call regression

Before the restart, `scripts/reasoning-tool-regression.py` exercised thinking
on/off with tools present/absent, both values of `preserve_thinking`, and eight
synthetic agent turns. All tool requests finished as structured `tool_calls`;
no output token was ID 0 and the longest identical-token run was one. A single
post-patch/post-restart reproduction of SGLang issue #36537 also returned a
valid `shell_probe` tool call (`finish_reason=tool_calls`) with no token ID 0.

This means issue #36537 was not reproduced on this exact Thor checkpoint and
runtime; it does not prove that every long-lived Kilo session is unaffected.
The Mamba patch addresses the separate, source-confirmed uptime defect tracked
by issue #37326 and must not be presented as a proven fix for the observed text
fragmentation.

Both target and draft startup logs state that the checkpoint has no FP8 KV
scaling factors, so this runtime uses scale 1.0. The weights themselves load as
NVFP4 without missing-layer or quantization-fallback warnings. Scale 1.0 is an
explicit SGLang fallback with a possible accuracy cost; changing to BF16 KV
would invalidate the current 557,056-token capacity, so the deployed profile
keeps FP8 KV and records this limitation instead of silently changing it.

## Automatic truncation boundary

The regression client constructed a chat prompt estimated at 262.9K tokens and
requested 1,024 tokens with EOS ignored. The server logged both input-only and
input-plus-completion truncation paths:

- Original server-counted input: 262,871 tokens
- Effective input after truncation: 261,110 tokens
- Completion: 1,024 tokens
- Total: 262,134 tokens
- Finish reason: `length`
- End-to-end time: 208.261 seconds

The request returned normally and KV dropped to zero immediately. No service or
kernel error was recorded, and the model service restart count remained zero.
