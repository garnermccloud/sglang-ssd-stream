# RTX PRO 6000: Qualified Adaptive MTP Bundle

Source and benchmark publication, qualified September 9, 2026 and published
September 10. This is the exact 17-file overlay deployed on one RTX PRO 6000
Blackwell Workstation Edition (96 GB), not a new checkpoint or a general SGLang
release. The source bytes were checked against both the qualification archive
and the running production image before publication.

**The normal SSD Stream installer does not enable this bundle.** Its pinned
SGLang revision and plugin interface differ from this historical runtime.
Do not copy these files into an arbitrary SGLang checkout or install the legacy
`sglang_ple_ssd/plugin.py` over today's `sglang_ssd_stream` package. Porting and
clean-install acceptance remain separate work. No model weights changed.

## Matched Results

Both arms used corrected sparse-index packing. The control used fixed
four-token MTP windows; the candidate used the adaptive bundle below.

| Workload | Fixed-four tok/s | Adaptive bundle tok/s | Median change |
| --- | ---: | ---: | ---: |
| List | 175.57 | 186.13 | +6.01% |
| Prose | 187.64 | 194.61 | +3.71% |
| Code | 238.76 | 282.12 | +18.16% |
| Reasoning | 163.35 | 163.97 | +0.38% |

These are wall-clock completion-throughput medians, including time to first
token. Each arm/workload has six measured 1,024-token completions. The order was
control-before, candidate-before-functional, candidate-after-functional,
control-after, with three measurements per workload in each pass. Sixteen
warmups were excluded, leaving 48 measured responses; all finished at the token
limit without retractions. Temperature was zero and the sampling seed was 1234.

This supports a workload-specific code-generation gain, **not a universal 18%
speedup or a coding-quality improvement**. All six outputs in every arm/workload
were distinct. The smaller differences are sensitive to output and run order.
One candidate code request was 241.15 tok/s; the six-request range was
241.15-294.80 tok/s. Candidate code medians before/after functional checks were
283.91/280.33 tok/s, versus control medians of 236.22/241.29 tok/s.

Code accepted tokens per round increased from 3.40 to 5.00 while round cost rose
from 14.28 to 17.70 ms. Adaptive proposal denominators reported by that server
are not reliable per-position acceptance denominators and are not used to make
such a claim. The table also does not isolate each component's contribution.

Do not compare these figures directly with the original 164.7 tok/s SSD-versus-RAM
demonstration on the main model card: that was a different experiment.

## Included Changes

- Stable-prefix sparse-index packing preserves valid index order and duplicates,
  rejects out-of-range indices, and handles wider layouts through 2,059 columns.
- Adaptive MTP switches between three/seven speculative steps, giving four/eight
  total draft-token windows. The pending-state ring accommodates the wide window.
- Corrected ReplaySSM commits PLE state after verification.
- The two widths share attention scratch space and the CUDA-graph capture stream.
- The draft head is initialized before memory profiling and graph capture.

Original weights, strict acceptance thresholds of 1, FP32 SSM state, FP8 KV,
CUDA graphs, and a 262,144-token context/pool were retained. Expert-tactic
experiments and trained projection weights were not promoted. The settings and
adaptive policy are included in `evidence/settings.json` and
`payload/opt/sglang-performance/adaptive.json`.

## Evidence and Verification

- `manifest.json`: exact deployed and baseline file hashes plus local image IDs.
- `qualified-changes.patch`: readable diff against the retained pre-update image.
- `payload/`: complete, byte-identical deployed source overlay.
- `evidence/`: all 48 timing/verifier records, exact input token IDs, synthetic
  prompts, recomputable medians, and the 12 functional-check outcomes.
- `runtime-versions.json`: package versions observed in the deployed runtime.
- `tests/test_qsa_pack_gpu.py`: the unchanged GPU qualification test source.

The historical GPU run passed five tests without skips, including wide packing
and CUDA graph replay. The historical functional run passed 12 checks, including
images, tool-call/result round trips, CRT reasoning, and retrieval at 120,081 and
255,681 input tokens. The candidate logs recorded 21 width switches in each
direction and actual CUDA-graph execution. There was no container restart or
OOM. These are scoped functional checks, not comprehensive agent-quality trials.

Raw generated outputs, serving logs, credentials, deployment configuration,
model tensors, and activation fixtures are not distributed. Output hashes are
retained, but this compact publication alone cannot reconstruct generated text
or independently rescore its quality. One GPU test needs the original saved
activation fixture; it skips without `QSA_PASS3_FIXTURE`. The five-pass historical
result must not be confused with a fresh run that skips that fixture.

Verify the source bytes and recompute the table without a GPU or dependencies:

```bash
python3 verify_bundle.py
python3 -m unittest discover -s tests -p test_publication.py -v
```

## Applying or Porting

The qualified base has local image ID
`sha256:056a88812978e09cb4d82c0d75f285bb5f303bb9a8814175816364c5c2bffa6a`.
It includes additional Qwen/SSD integration changes beyond Git HEAD
`d91c3682b0b429e4c70df63cd57f819588ce29b0`. Neither this commit alone nor the
current public installer reproduces that base. The local image IDs are
provenance identifiers, **not pullable registry references**. No container image
or complete legacy baseline is published with this source bundle.

For maintainers who already have that exact baseline, the included Dockerfile
checks every overwritten file before applying the overlay, then verifies the
result. The base-image identity must be checked separately with `docker image
inspect`; the in-image file check is not an assertion that all other files match.

```bash
# Only after confirming QUALIFIED_BASE has the exact image ID above:
docker build --build-arg QUALIFIED_BASE="$QUALIFIED_BASE" -t rtx-pro-qualified .
```

Keep the baseline's model/SSD arguments and add:

```text
--speculative-adaptive
--speculative-adaptive-config /opt/sglang-performance/adaptive.json
--enable-linear-replayssm-spec
```

The image sets `SGLANG_RAGGED_VERIFY_MODE=static`. Start with MTP steps/top-k/draft
tokens 3/1/4, both acceptance thresholds 1, FP32 SSM, FP8 KV, CUDA graphs enabled,
and context/pool 262144, as in the evidence. Keep a rollback environment. Do not
infer readiness for other GPUs, concurrent workloads, or other SGLang versions.

For a newer upstream checkout, use the patch and source as a porting reference,
not an unconditional replacement. Re-run the GPU tests and matched serving,
long-context, multimodal, and tool checks before changing a default profile.

## Attribution

SGLang and the RadixArk Flash-Next integration provide the underlying runtime.
Their existing notices remain in the source. This independent overlay is
distributed under this repository's Apache-2.0 license. It is not an official
SGLang release or a model-weight update.
