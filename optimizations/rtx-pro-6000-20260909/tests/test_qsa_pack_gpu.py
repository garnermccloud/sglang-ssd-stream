"""GPU-only qualification; no model construction or serving monkey patches.

Run: python -m pytest -q -rs test_qsa_pack_gpu.py
QSA_PASS3_FIXTURE may name the saved attention-nonfinite.safetensors file.
Missing CUDA/optional attention dependencies skip; a missing qsa_pack on CUDA fails.
"""

import gc
import importlib
import math
import os
from pathlib import Path
import unittest

try:
    import torch
except ImportError:
    torch = None


def reference_pack(indices, lengths):
    result = torch.full_like(indices, -1)
    counts = []
    for row, length in enumerate(lengths.tolist()):
        # Python scalar comparison can narrow a large length to the index dtype.
        comparison = indices[row].to(torch.int64)
        valid = indices[row][(comparison >= 0) & (comparison < length)]
        result[row, :valid.numel()] = valid
        counts.append(valid.numel())
    return result, counts


@unittest.skipUnless(torch is not None, "PyTorch required")
class ReferencePackTests(unittest.TestCase):
    def test_int32_indices_with_large_int64_length(self):
        indices = torch.tensor([[2, -1, 0, 2, 2147483647]], dtype=torch.int32)
        packed, counts = reference_pack(indices, torch.tensor([2**40 + 1], dtype=torch.int64))
        self.assertEqual(packed.dtype, indices.dtype)
        self.assertEqual(packed.tolist(), [[2, 0, 2, 2147483647, -1]])
        self.assertEqual(counts, [4])


def capture(call):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            call()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = call()
    return graph, result


@unittest.skipUnless(torch is not None and torch.cuda.is_available(), "CUDA required")
class QsaPackGpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Import the installed original, never the candidate sparse_attn module.
        sparse = importlib.import_module("sglang.srt.layers.attention.qsa.sparse_attn")
        cls.gather = staticmethod(sparse.qwen_sparse_kv_extraction_compact_triton)
        cls.compact = staticmethod(importlib.import_module("qsa_pack").compact_sparse_indices)

    def setUp(self):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self.baseline = torch.cuda.memory_allocated()

    def tearDown(self):
        torch.cuda.synchronize()
        self.assertLess(torch.cuda.max_memory_allocated() - self.baseline, 1024**3)
        gc.collect()
        torch.cuda.empty_cache()

    def assert_bytes(self, actual, expected):
        torch.testing.assert_close(actual.contiguous().view(torch.uint8).cpu(),
                                   expected.contiguous().view(torch.uint8).cpu(),
                                   atol=0, rtol=0)

    def test_helper_corners(self):
        for index_dtype in (torch.int32, torch.int64):
            for length_dtype in (torch.int32, torch.int64):
                for width in (0, 1, 7, 31, 32, 33, 2047, 2048, 2049, 2055, 2059):
                    with self.subTest(width=width, indices=index_dtype, lengths=length_dtype):
                        x = (torch.arange(4 * width).reshape(4, width) % 23 - 3).to(index_dtype)
                        lengths = torch.tensor([0, 1, 11, 20], dtype=length_dtype)
                        expected, _ = reference_pack(x, lengths)
                        device_x = x.cuda()
                        result = self.compact(device_x, lengths.cuda())
                        self.assertEqual(result.dtype, index_dtype)
                        self.assertEqual(result.shape, x.shape)
                        torch.testing.assert_close(result.cpu(), expected, atol=0, rtol=0)
                        torch.testing.assert_close(device_x.cpu(), x, atol=0, rtol=0)

    def test_strided_helper_graph(self):
        for index_dtype in (torch.int32, torch.int64):
            for length_dtype in (torch.int32, torch.int64):
                for width in (1, 33, 2048, 2055, 2059):
                    with self.subTest(width=width, indices=index_dtype, lengths=length_dtype):
                        backing = torch.full((8, width * 3 + 2), -99,
                                             dtype=index_dtype, device="cuda")
                        indices = backing[::2, 1:1 + width * 3:3]
                        length_backing = torch.full((8,), -99, dtype=length_dtype, device="cuda")
                        lengths = length_backing[::2]
                        lengths.fill_(7)
                        graph, output = capture(lambda: self.compact(indices, lengths))
                        for replay in range(4):
                            x = (torch.arange(4 * width).reshape(4, width) % 13 - 3).to(index_dtype)
                            lens = torch.tensor([0, 1, 7, 10], dtype=length_dtype)
                            if replay == 1:
                                x.fill_(-1)
                            elif replay == 2:
                                x.fill_(0)
                                lens.fill_(1)
                            elif replay == 3:
                                if index_dtype == torch.int64:
                                    x[:, ::2] = 2**40
                                if length_dtype == torch.int64:
                                    lens[1:] = 2**40 + 1
                            expected, _ = reference_pack(x, lens)
                            indices.copy_(x)
                            lengths.copy_(lens)
                            before = backing.cpu()
                            before_lengths = length_backing.cpu()
                            output.fill_(-77)
                            graph.replay()
                            torch.cuda.synchronize()
                            torch.testing.assert_close(output.cpu(), expected, atol=0, rtol=0)
                            torch.testing.assert_close(backing.cpu(), before, atol=0, rtol=0)
                            torch.testing.assert_close(length_backing.cpu(), before_lengths, atol=0, rtol=0)

    def test_gather_graph_replays(self):
        modes = ((torch.bfloat16, torch.bfloat16),
                 (torch.float8_e4m3fn, torch.float8_e4m3fn),
                 (torch.float8_e4m3fn, torch.bfloat16))
        for batch in (1, 4, 8):
            for width in (1, 17, 2047, 2048, 2049, 2055, 2059):
                for layout in ("page", "tight"):
                    for mode, (src_dtype, dst_dtype) in enumerate(modes):
                        with self.subTest(batch=batch, width=width, layout=layout, mode=mode):
                            self.check_gather(batch, width, layout, src_dtype, dst_dtype, mode)

    def check_gather(self, batch, width, layout, src_dtype, dst_dtype, mode):
        index_dtype = (torch.int32, torch.int64)[(batch + mode) % 2]
        length_dtype = (torch.int32, torch.int64)[(width + mode) % 2]
        capacity, heads, dim = max(width + 8, 128), 2, 64
        slots = (batch + 1) * capacity
        stride = math.ceil(width / 64) * 64
        indices = torch.empty((batch, width), dtype=index_dtype, device="cuda")
        lengths = torch.empty(batch, dtype=length_dtype, device="cuda")
        reqs = torch.empty(batch, dtype=torch.int32, device="cuda")
        mapping = torch.empty((batch + 1, capacity), dtype=torch.int32, device="cuda")
        cu = torch.empty(batch + 1, dtype=torch.int32, device="cuda")
        k = torch.empty((slots, heads, dim), dtype=src_dtype, device="cuda")
        v = torch.empty_like(k)
        # Guard rows after the last allocation and all gaps must remain untouched.
        out_k = torch.empty((batch * stride + 3, heads, dim), dtype=dst_dtype, device="cuda")
        out_v = torch.empty_like(out_k)
        dequant = src_dtype != dst_dtype
        scales = (0.375, 1.75) if dequant else (1., 1.)

        def launch():
            packed = self.compact(indices, lengths)
            self.gather(k, v, mapping, reqs, packed, lengths, cu, out_k, out_v,
                        batch, width, k_scale=scales[0], v_scale=scales[1])
            return packed

        def update(replay):
            generator = torch.Generator().manual_seed(9700 + replay)
            lens = torch.tensor([capacity - 3 - r * 3 for r in range(batch)], dtype=length_dtype)
            x = torch.arange(width).expand(batch, -1).clone().to(index_dtype)
            if replay == 1:
                x[:, 1::3] = -1
                x[:, 2::7] = capacity + 99
                x[:, ::11] = 0  # Repeated logical and physical selections are intentional.
            elif replay == 2:
                lens[:] = 9
                lens[0] = 0  # Zero-valid gather rows must not write any destination.
                x[:] = -7
                x[:, -min(width, 5):] = torch.arange(min(width, 5))
            elif replay == 3:
                x[:] = -1
                x[:, :min(width, 126)] = torch.arange(min(width, 126))
                if width > 128:
                    x[:, -4:-2] = torch.tensor([126, 127])
                lens[:] = 128
            # Full -> holes -> zero/short -> distant tail -> full, same graph/shapes.
            packed, counts = reference_pack(x, lens)
            offsets = ([r * stride for r in range(batch + 1)] if layout == "page"
                       else [0] + list(torch.tensor(counts).cumsum(0).tolist()))
            requests = (torch.arange(batch) + replay) % (batch + 1)
            table = torch.randperm(slots, generator=generator).reshape(batch + 1, capacity).int()
            cpu_k = torch.randn((slots, heads, dim), generator=generator).to(src_dtype)
            cpu_v = torch.randn((slots, heads, dim), generator=generator).to(src_dtype)
            for target, source in ((indices, x), (lengths, lens), (reqs, requests),
                                   (mapping, table), (cu, torch.tensor(offsets)),
                                   (k, cpu_k), (v, cpu_v)):
                target.copy_(source)
            poison = float("nan") if replay % 2 == 0 else -37.
            out_k.copy_(torch.full(out_k.shape, poison, dtype=torch.float32).to(dst_dtype))
            out_v.copy_(torch.full(out_v.shape, poison, dtype=torch.float32).to(dst_dtype))
            expected_k, expected_v = out_k.cpu(), out_v.cpu()
            written = torch.zeros(out_k.shape[0], dtype=torch.bool)
            for r, count in enumerate(counts):
                selected = table[requests[r], packed[r, :count].long()].long()
                start = offsets[r]
                written[start:start + count] = True
                for source, expected, scale in ((cpu_k, expected_k, scales[0]),
                                                 (cpu_v, expected_v, scales[1])):
                    # CPU FP8 advanced indexing is unsupported on some torch versions.
                    values = source.float()[selected]
                    expected[start:start + count] = (values * scale).to(dst_dtype)
            return x, packed, expected_k, expected_v, written

        update(0)
        graph, packed = capture(launch)
        for replay in (0, 1, 2, 3, 4):
            original, expected, ek, ev, written = update(replay)
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(packed.cpu(), expected, atol=0, rtol=0)
            torch.testing.assert_close(indices.cpu(), original, atol=0, rtol=0)
            for output, reference in ((out_k, ek), (out_v, ev)):
                if not dequant:
                    self.assert_bytes(output, reference)
                else:
                    actual = output.cpu()
                    torch.testing.assert_close(actual[written].float(), reference[written].float(),
                                               atol=0.015625, rtol=0.008)
                    self.assert_bytes(actual[~written], reference[~written])
                self.assertTrue(torch.isfinite(output.float().cpu()[written]).all().item())

    def test_saved_pass3_attention_graph(self):
        path = Path(os.environ.get("QSA_PASS3_FIXTURE",
                                   "/experiment/native-decode-pairs-late-pass3/attention-nonfinite.safetensors"))
        if not path.is_file():
            self.skipTest(f"Saved pass3 fixture absent: {path}")
        try:
            from safetensors.torch import load_file
            from flashinfer.decode import trtllm_batch_decode_with_kv_cache
        except ImportError as error:
            self.skipTest(f"Attention dependency absent: {error}")
        saved = load_file(str(path), device="cpu")
        q_cpu, original = saved["q"], saved["topk"].int()
        self.assertEqual(tuple(saved["k"].shape), (128, 2, 256))
        self.assertEqual(saved["k"].shape, saved["v"].shape)
        self.assertEqual(tuple(original.shape), (1, 2055))
        base_k = saved["k"].to(torch.float8_e4m3fn).float()
        base_v = saved["v"].to(torch.float8_e4m3fn).float()
        packed_cpu, counts = reference_pack(original, torch.tensor([128]))
        self.assertEqual(counts, [128])
        torch.testing.assert_close(packed_cpu[0, :128], torch.arange(128).int())
        q = q_cpu.cuda()
        indices = original.cuda()
        lengths = torch.tensor([128], dtype=torch.int32, device="cuda")
        counts_gpu = torch.tensor([128], dtype=torch.int32, device="cuda")
        mapping = torch.arange(128, dtype=torch.int32, device="cuda")[None, :]
        reqs = torch.zeros(1, dtype=torch.int32, device="cuda")
        k, v = base_k.cuda().to(torch.float8_e4m3fn), base_v.cuda().to(torch.float8_e4m3fn)
        page, stride = 64, math.ceil(original.shape[1] / 64) * 64
        cu = torch.tensor([0, stride], dtype=torch.int32, device="cuda")
        blocks = torch.arange(stride // page, dtype=torch.int32, device="cuda")[None, :]
        pk = torch.empty((stride, 2, 256), dtype=k.dtype, device="cuda")
        pv = torch.empty_like(pk)
        workspace = torch.zeros(128 * 1024**2, dtype=torch.uint8, device="cuda")

        def launch():
            packed = self.compact(indices, lengths)
            self.gather(k, v, mapping, reqs, packed, lengths, cu, pk, pv, 1, original.shape[1])
            return trtllm_batch_decode_with_kv_cache(
                query=q, kv_cache=(pk.view(-1, page, 2, 256).permute(0, 2, 1, 3),
                                   pv.view(-1, page, 2, 256).permute(0, 2, 1, 3)),
                workspace_buffer=workspace, block_tables=blocks, seq_lens=counts_gpu,
                max_seq_len=stride, bmm1_scale=1 / math.sqrt(256), bmm2_scale=1.)

        graph, output = capture(launch)
        for replay in range(4):
            x = original.clone()
            length = 128 if replay % 2 == 0 else 63
            if replay == 2:
                x[:] = -1
                x[0, -128:] = torch.arange(128).flip(0)
            packed, counts = reference_pack(x, torch.tensor([length]))
            self.assertGreater(counts[0], 0)  # Never send zero lengths to attention.
            table = torch.arange(128).roll(replay * 17)
            ck = (base_k * (1 - replay * .125)).to(k.dtype)
            cv = (base_v + replay * .125).to(v.dtype)
            indices.copy_(x)
            lengths.fill_(length)
            counts_gpu.fill_(counts[0])
            mapping.copy_(table[None, :])
            k.copy_(ck)
            v.copy_(cv)
            q.copy_(q_cpu * (1 - replay * .0625))
            for dest in (pk, pv):
                dest.copy_(torch.full(dest.shape, float("nan") if replay % 2 == 0 else -37.).to(dest.dtype))
            graph.replay()
            torch.cuda.synchronize()
            selected = table[packed[0, :counts[0]].long()]
            rk, rv = ck.float()[selected], cv.float()[selected]
            self.assert_bytes(pk[:counts[0]], rk.to(k.dtype))
            self.assert_bytes(pv[:counts[0]], rv.to(v.dtype))
            # CPU float32 SDPA is independent of the captured GPU attention backend.
            repeats = q.shape[1] // 2
            reference = torch.nn.functional.scaled_dot_product_attention(
                q.float().cpu().transpose(0, 1).unsqueeze(0),
                rk.repeat_interleave(repeats, dim=1).transpose(0, 1).unsqueeze(0),
                rv.repeat_interleave(repeats, dim=1).transpose(0, 1).unsqueeze(0),
            ).squeeze(0).transpose(0, 1)
            self.assertTrue(torch.isfinite(output).all().item())
            torch.testing.assert_close(output.float().cpu(), reference, atol=.1, rtol=.03)


if __name__ == "__main__":
    unittest.main()
