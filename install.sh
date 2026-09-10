#!/bin/sh
set -eu

REPOSITORY="garnermccloud/sglang-ssd-stream"
VERSION="0.3.0"
RTX_SGLANG_COMMIT="3df8e1e7dbc5807696622afe2929b6c33c185ca3"
SPARK_SGLANG_COMMIT="0a79825b7baa3e2aafd54e89097a5aba83d00b4e"
FLASHINFER_VERSION="0.6.17"
CUDA_NVCC_VERSION="13.0.88"
CUDA_RUNTIME_VERSION="13.0.96"
CUDA_CCCL_VERSION="13.0.85"

if [ "$(uname -s)" != "Linux" ]; then
    echo "sglang-ssd-stream requires Linux" >&2
    exit 1
fi

ARCH="$(uname -m)"
case "$ARCH" in
    x86_64) SGLANG_COMMIT="$RTX_SGLANG_COMMIT" ;;
    aarch64) SGLANG_COMMIT="$SPARK_SGLANG_COMMIT" ;;
    *)
        echo "sglang-ssd-stream does not have a wheel for $ARCH" >&2
        exit 1
        ;;
esac

TAG="v$VERSION"
WHEEL="sglang_ssd_stream-${VERSION}-cp312-cp312-manylinux_2_28_${ARCH}.whl"
URL="${SGLANG_SSD_STREAM_WHEEL_URL:-https://github.com/$REPOSITORY/releases/download/$TAG/$WHEEL}"
RUNTIME_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/sglang-ssd-stream"
mkdir -p "$RUNTIME_DIR"
# Never move this directory: generated console-script shebangs use its final path.
# Keep previous and failed environments for recovery; only the launcher is promoted.
VENV="$(mktemp -d "$RUNTIME_DIR/venv-$VERSION.XXXXXXXX")"
PYTHON="$VENV/bin/python"

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh -o "$VENV/install-uv.sh"
    sh "$VENV/install-uv.sh"
fi
# This is the unique directory just allocated above, never the active venv.
uv venv --allow-existing --managed-python --python 3.12 "$VENV"

SGLANG_BUILD_RUST_EXTS=none uv pip install \
    --python "$PYTHON" \
    "sglang @ https://github.com/sgl-project/sglang/archive/$SGLANG_COMMIT.tar.gz#subdirectory=python" \
    "flashinfer-python==$FLASHINFER_VERSION" \
    "nvidia-modelopt==0.45.0" \
    "nvidia-cuda-cccl==$CUDA_CCCL_VERSION" \
    "nvidia-cuda-crt==$CUDA_NVCC_VERSION" \
    "nvidia-cuda-nvcc==$CUDA_NVCC_VERSION" \
    "nvidia-cuda-nvrtc==$CUDA_NVCC_VERSION" \
    "nvidia-cuda-runtime==$CUDA_RUNTIME_VERSION" \
    "nvidia-nvjitlink==$CUDA_NVCC_VERSION" \
    "nvidia-nvvm==$CUDA_NVCC_VERSION"

uv pip install \
    --python "$PYTHON" \
    --no-deps \
    --index https://flashinfer.ai/whl \
    "flashinfer-cubin==$FLASHINFER_VERSION"

uv pip install \
    --python "$PYTHON" \
    --no-deps \
    --index https://flashinfer.ai/whl/cu130 \
    "flashinfer-jit-cache==$FLASHINFER_VERSION+cu130"

uv pip install \
    --python "$PYTHON" \
    --no-deps \
    --reinstall \
    "$URL"

BIN_DIR="$HOME/.local/bin"
"$PYTHON" -c 'import sys; from importlib.metadata import version; from sglang_ssd_stream import _io; assert version("sglang-ssd-stream") == sys.argv[1]' "$VERSION"
"$VENV/bin/sglang-ssd-stream" --help >/dev/null
mkdir -p "$BIN_DIR"
LINK_DIR="$(mktemp -d "$BIN_DIR/.sglang-ssd-stream.XXXXXXXX")"
trap 'rm -rf "$LINK_DIR"' EXIT
trap 'exit 1' HUP INT TERM
ln -s "$VENV/bin/sglang-ssd-stream" "$LINK_DIR/launcher"
# Linux rename in the same filesystem is atomic; -T refuses a directory target.
mv -Tf "$LINK_DIR/launcher" "$BIN_DIR/sglang-ssd-stream"

echo
echo "Installed sglang-ssd-stream $VERSION"
echo "Run: $BIN_DIR/sglang-ssd-stream serve"
