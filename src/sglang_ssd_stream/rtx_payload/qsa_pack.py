"""Stable sparse-index packing without host readbacks or sort kernels."""

import torch
import triton
import triton.language as tl


@triton.jit
def _pack_indices(indices, lengths, output, width: tl.constexpr,
                  row_stride: tl.constexpr, col_stride: tl.constexpr,
                  length_stride: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK)
    values = tl.load(indices + row * row_stride + columns * col_stride,
                     mask=columns < width, other=-1)
    length = tl.load(lengths + row * length_stride)
    valid = (columns < width) & (values >= 0) & (values < length)
    prefix = tl.cumsum(valid.to(tl.int32), axis=0)
    total = tl.sum(valid.to(tl.int32), axis=0)
    # Valid and invalid lanes form disjoint, complete destination permutations.
    # Every real column writes once, so no fill kernel or scatter race is needed.
    destination = tl.where(valid, prefix - 1, total + columns - prefix)
    tl.store(output + row * width + destination, tl.where(valid, values, -1),
             mask=columns < width)


def compact_sparse_indices(indices, lengths):
    if (indices.ndim != 2 or lengths.ndim != 1 or indices.shape[0] != lengths.numel()
            or indices.dtype not in (torch.int32, torch.int64)
            or lengths.dtype not in (torch.int32, torch.int64)
            or indices.device != lengths.device or not indices.is_cuda):
        raise ValueError('Expected CUDA integer [rows, width] indices and [rows] lengths')
    rows, width = indices.shape
    output = torch.empty((rows, width), dtype=indices.dtype, device=indices.device)
    if rows and width:
        block = triton.next_power_of_2(width)
        _pack_indices[(rows,)](indices, lengths, output, width, *indices.stride(),
                              lengths.stride(0), block, num_warps=4 if block <= 2048 else 8)
    return output
