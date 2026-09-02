#!/usr/bin/env bash
set -euo pipefail

export SGLANG_PLUGINS=ssd_stream
# Safety net only: the 261,888-token instrumented run produced no NaN/Inf.
export SGLANG_SANITIZE_NAN_LOGITS=1
# Cap each full-attention layer's dominant FP32 QSA prefill score matrix.
# Qwen3.8 Flash-Next has 12 such layers; 32 MiB bounds their aggregate cached
# allocator footprint to roughly 384 MiB instead of roughly 1.5 GiB.
export SGLANG_QSA_PREFILL_LOGITS_BUDGET_MB=${SGLANG_QSA_PREFILL_LOGITS_BUDGET_MB:-32}
# Release only unused CUDA allocator blocks after the scheduler has remained
# fully idle for this many seconds. This does not flush the radix/KV cache.
export SGLANG_EMPTY_CACHE_INTERVAL=${SGLANG_EMPTY_CACHE_INTERVAL:-60}
RUNTIME_ROOT=${SGLANG_SSD_STREAM_RUNTIME_ROOT:-${HOME}/.local/share/sglang-ssd-stream}
export CUDA_HOME=${CUDA_HOME:-${RUNTIME_ROOT}/cuda}
export PATH=${RUNTIME_ROOT}/venv/bin:${PATH}
export MAX_JOBS=1
export FLASHINFER_NVCC_THREADS=1
export CMAKE_BUILD_PARALLEL_LEVEL=1

MODEL=${MODEL:-${HOME}/models/Qwen3.8-Flash-Next-NVFP4-Spark}
PYTHON=${PYTHON:-${RUNTIME_ROOT}/venv/bin/python}
DRAFT=${DRAFT:-${HOME}/models/sglang-ssd-stream-runtime/huggingface/hub/models--garnermccloud--Qwen3.8-Flash-Next-NVFP4-SSD-Stream/snapshots/83325b75b7cb498ef5d7a5477171cadf92ad21f5/mtp}

test -f "${MODEL}/config.json"
test -f "${DRAFT}/config.json"
test -x "${PYTHON}"

# The 544-Ki-token KV pool is shared by up to four running requests. With the
# 262-K per-request context limit it can hold two nearly-full coding contexts.
# NEXTN needs four Mamba state slots per request (one target plus three draft
# states), hence 16 slots for four-way concurrency.
exec "${PYTHON}" -m sglang.launch_server \
  --trust-remote-code \
  --model-path "${MODEL}" \
  --served-model-name Qwen/Qwen3.6-27B-NVFP4 \
  --host 0.0.0.0 \
  --port 8000 \
  --quantization modelopt_fp4 \
  --fp4-gemm-backend flashinfer_cutlass \
  --attention-backend triton \
  --moe-runner-backend flashinfer_cutlass \
  --kv-cache-dtype fp8_e4m3 \
  --speculative-draft-kv-cache-dtype fp8_e4m3 \
  --page-size 64 \
  --mamba-radix-cache-strategy extra_buffer_lazy \
  --mamba-track-interval 64 \
  --mamba-ssm-dtype float32 \
  --max-mamba-cache-size 16 \
  --chunked-prefill-size 2048 \
  --max-running-requests 4 \
  --sleep-on-idle \
  --cuda-graph-max-bs-decode 4 \
  --disable-prefill-cuda-graph \
  --context-length 262144 \
  --max-total-tokens 557056 \
  --mem-fraction-static 0.95 \
  --speculative-algorithm NEXTN \
  --speculative-draft-model-path "${DRAFT}" \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --enable-linear-replayssm-spec \
  --allow-auto-truncate \
  --enable-multimodal \
  --reasoning-parser auto \
  --tool-call-parser auto
