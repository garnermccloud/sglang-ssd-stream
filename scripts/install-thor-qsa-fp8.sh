#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
RUNTIME_ROOT=${SGLANG_SSD_STREAM_RUNTIME_ROOT:-${HOME}/.local/share/sglang-ssd-stream}
PYTHON=${PYTHON:-${RUNTIME_ROOT}/venv/bin/python}
SITE=$(
  "${PYTHON}" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])'
)
ATTN_DIR="${SITE}/sglang/srt/layers/attention"
PLUGIN_DST="${SITE}/sglang_ssd_stream/plugin.py"
PATCH_FILE="${REPO_DIR}/patches/sglang-qsa-fp8-thor.patch"
BUDGET_PATCH_FILE="${REPO_DIR}/patches/sglang-qsa-prefill-budget.patch"
TRUNCATE_PATCH_FILE="${REPO_DIR}/patches/sglang-auto-truncate-page-safe.patch"

test -d "${ATTN_DIR}/qsa"
test -f "${PLUGIN_DST}"
test -f "${PATCH_FILE}"
test -f "${BUDGET_PATCH_FILE}"
test -f "${TRUNCATE_PATCH_FILE}"
test -f "${REPO_DIR}/src/sglang_ssd_stream/plugin.py"

BASE_KERNEL=7e369f09293fb9b0872c21f0010247ec1e3a696b5ad4809f04d9a730b1031095
BASE_SPARSE=f3801cc37453278e884873a821350def23c58453eb91c56f2c96d8f62a3709f5
BASE_BACKEND=7cb54a4440a3f6f9619227138398c173993a9f8deec8e6a6be50ce9067d50153
PATCHED_KERNEL=2462e638aa5e26a1e715283fbc68f92c7db9149a5f30e0947eb78eb51970aceb
PATCHED_SPARSE=73a953b6d1ef843e9cebf620ef2f9e2aa513de6775dddca08dbf7f4bbc247d97
PATCHED_BACKEND=3ba25953f98a75f3104e1bf53e7742b0e227563d1ddc6dfefc5e6e5561198221
BASE_INDEXER=a916967133fdb6b06e6094bfe3a9aa1ba275bf7d858d2f5c1bb39f8d0149461f
PATCHED_INDEXER=05c74339829dd6a919ae5b06a783a14c6b655d6a0046a852b98a9abda3dda9c4
BASE_TOKENIZER_MANAGER=92b42e5b0445111c60dc61751de4e9843d9dc8f2631c2509899be0ce680ee3ef
PATCHED_TOKENIZER_MANAGER=bfdca1daef6d1407ebb7d62532cd482ceac4440593412d9c7b3de1d985995e28
BASE_SCHEDULER=e7751263dba7a1a008a0a2467ed1b68db58e2d3d6c728d53fbba6b63b565a9cd
PATCHED_SCHEDULER=b522757ed15dce4a33617f17a7c6a10a653acbe2334b81d874130fb73b93c851

kernel="${ATTN_DIR}/qsa/kernel.py"
sparse="${ATTN_DIR}/qsa/sparse_attn.py"
backend="${ATTN_DIR}/qwen_sparse_attn_backend.py"
indexer="${ATTN_DIR}/qsa/qsa_indexer.py"
tokenizer_manager="${SITE}/sglang/srt/managers/tokenizer_manager.py"
scheduler="${SITE}/sglang/srt/managers/scheduler.py"
read_hash() { sha256sum "$1" | cut -d' ' -f1; }

kh=$(read_hash "${kernel}")
sh=$(read_hash "${sparse}")
bh=$(read_hash "${backend}")

if [[ "${kh}:${sh}:${bh}" == "${PATCHED_KERNEL}:${PATCHED_SPARSE}:${PATCHED_BACKEND}" ]]; then
  echo "QSA FP8 patch is already installed."
elif [[ "${kh}:${sh}:${bh}" == "${BASE_KERNEL}:${BASE_SPARSE}:${BASE_BACKEND}" ]]; then
  backup="${REPO_DIR}/backups/sglang-qsa-$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p "${backup}/qsa"
  cp -- "${kernel}" "${backup}/qsa/kernel.py"
  cp -- "${sparse}" "${backup}/qsa/sparse_attn.py"
  cp -- "${backend}" "${backup}/qwen_sparse_attn_backend.py"
  patch --dry-run -d "${SITE}" -p2 < "${PATCH_FILE}"
  patch -d "${SITE}" -p2 < "${PATCH_FILE}"
  echo "Installed QSA FP8 patch; backup: ${backup}"
else
  echo "Refusing to patch an unknown SGLang source state:" >&2
  echo "  kernel=${kh}" >&2
  echo "  sparse=${sh}" >&2
  echo "  backend=${bh}" >&2
  exit 1
fi

ih=$(read_hash "${indexer}")
if [[ "${ih}" == "${PATCHED_INDEXER}" ]]; then
  echo "QSA prefill workspace budget patch is already installed."
elif [[ "${ih}" == "${BASE_INDEXER}" ]]; then
  backup="${REPO_DIR}/backups/sglang-qsa-budget-$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p "${backup}/qsa"
  cp -- "${indexer}" "${backup}/qsa/qsa_indexer.py"
  patch --dry-run -d "${SITE}" -p2 < "${BUDGET_PATCH_FILE}"
  patch -d "${SITE}" -p2 < "${BUDGET_PATCH_FILE}"
  echo "Installed configurable QSA prefill workspace budget; backup: ${backup}"
else
  echo "Refusing to patch an unknown QSA indexer source state: ${ih}" >&2
  exit 1
fi

tmh=$(read_hash "${tokenizer_manager}")
schedh=$(read_hash "${scheduler}")
if [[ "${tmh}:${schedh}" == "${PATCHED_TOKENIZER_MANAGER}:${PATCHED_SCHEDULER}" ]]; then
  echo "Page-safe auto-truncate patch is already installed."
elif [[ "${tmh}:${schedh}" == "${BASE_TOKENIZER_MANAGER}:${BASE_SCHEDULER}" ]]; then
  backup="${REPO_DIR}/backups/sglang-auto-truncate-$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p "${backup}"
  cp -- "${tokenizer_manager}" "${backup}/tokenizer_manager.py"
  cp -- "${scheduler}" "${backup}/scheduler.py"
  patch --dry-run -d "${SITE}" -p1 < "${TRUNCATE_PATCH_FILE}"
  patch -d "${SITE}" -p1 < "${TRUNCATE_PATCH_FILE}"
  echo "Installed page-safe auto-truncate patch; backup: ${backup}"
else
  echo "Refusing to patch an unknown SGLang request-manager source state:" >&2
  echo "  tokenizer_manager=${tmh}" >&2
  echo "  scheduler=${schedh}" >&2
  exit 1
fi

install -m 0644 "${REPO_DIR}/src/sglang_ssd_stream/plugin.py" "${PLUGIN_DST}"
"${PYTHON}" -m py_compile "${kernel}" "${sparse}" "${backend}" "${indexer}" \
  "${tokenizer_manager}" "${scheduler}" "${PLUGIN_DST}"

test "$(read_hash "${kernel}")" = "${PATCHED_KERNEL}"
test "$(read_hash "${sparse}")" = "${PATCHED_SPARSE}"
test "$(read_hash "${backend}")" = "${PATCHED_BACKEND}"
test "$(read_hash "${indexer}")" = "${PATCHED_INDEXER}"
test "$(read_hash "${tokenizer_manager}")" = "${PATCHED_TOKENIZER_MANAGER}"
test "$(read_hash "${scheduler}")" = "${PATCHED_SCHEDULER}"
echo "QSA FP8, page-safe truncation, and SSD-stream integrity guards are consistent."
