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
