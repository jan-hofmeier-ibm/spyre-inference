# Copyright 2026 The Spyre-Inference Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
exp_spyre.py - Exponential(lambda) and log-Exponential variates, self-contained:
an int16 Threefry-4x16 kernel plus the exponential transform, all on the Spyre
device. This file does not import the other Threefry implementations.

Variants
--------
    variant="24bit" (default): 2.5 variates per Threefry call, each from exactly
    24 random bits (the fp32 significand width, so the conversion is exact).
    Per 32-bit lane (lo and hi int16 halfword) of the four cipher words L_0..L_3:
        variates 0..3:  s = L_w mod 2^24                         i16_low24_to_fp32
        variate 4:      s = top8(L_0)*2^16 + top8(L_1)*2^8 + top8(L_2)   i16_top3_to_fp32
        u = (s + 1) * 2^-24                                      (u in (0, 1])
        E = max(-ln(u), 2^-25) / lam
    top8(L_3) is unused. Output order is [x0 | x1 | x2 | x3 | top-bytes block], each
    block calls//2 long. u == 1.0 (s == 2^24 - 1, once per 2^24 variates) would give
    E = 0 and, in the log variant, -inf; the floor 2^-25 (midpoint of the cell [0, 2^-24)
    that u == 1 stands for) replaces it, so E > 0 and every other grid point keeps its own
    value. E ranges over [2^-25, 24 ln 2] / lam = [2.98e-8, 16.64] / lam.
    variant="32bit": the previous pipeline, 2 variates per call (described below).
    The two variants give different sample streams for the same key and counter.
    Plan and measurements: docs/24bit-variates-plan.md.

32bit pipeline (per Threefry call, 4 variates)
-----------------------------------------------
    x0 is a (calls,) int16 vector.  i16tofp32 converts one int16 stick (64
    elements, 32 lanes × 2 halfwords) to two fp32 sticks:
        fp[0..31]  = unsigned(x0[0::2])  — lo halfwords of the 32 lanes
        fp[32..63] = unsigned(x0[1::2])  — hi halfwords of the 32 lanes
    Reshaped to (-1, 2, 32):
        fp_3d[k, 0, :] = lo halfwords of int16 stick k
        fp_3d[k, 1, :] = hi halfwords of int16 stick k
    Each (lo, hi) pair then gives one uniform:
        s = hi * 65536 + lo + 0.5          (fp32, s in [0.5, 2^32))
        u = s * 2^-32                      (u in (0, 1), never 0)
        E = -ln(u) / lam                   (inverse CDF)
    log variant -> ln(E)

    All four cipher words x0, x1, x2, x3 are used.  Each word gives
    calls//2 variates, so one Threefry call produces 2*calls variates.
    calls = ceil(n/32)*32, so 2*calls variates cover n samples.
    Output order is [all from x0 | all from x1 | all from x2 | all from x3]
    (torch.cat).

Why this form
-------------
* "+ 0.5" keeps u strictly inside (0, 1): the maximum raw int is 2^32-1, so
  u_max = (2^32-1 + 0.5) * 2^-32 = 1 - 2^-33, which is strictly below 1.0 in
  fp32 (next representable value down is 1 - 2^-24). No clamp is needed: log(u)
  is always finite for both the exp and ln-exp paths.
* -ln(u)/lam is one LOG and one multiply by the constant -1/lam. The negation is
  folded into that constant.
* The log variant is just log() over E. A dedicated integrated form was measured
  and is not cheaper (scripts/bench/bench_log_integrated_spyre.py, docs/SPYRE_RNG_STATUS_AND_COMMANDS.md 7.3).
"""

from __future__ import annotations

import functools

import torch
from torch import Tensor

try:
    import torch_spyre  # noqa: F401  (ensures spyre:: ops are registered)
except ImportError:
    pass

_TWO_M32 = 2.0 ** -32


# ---------------------------------------------------------------------------
# Threefry-4x16 in native int16 (bit-exact vs the int32 CPU reference)
#
# Spyre constraints that shape this code:
# * int16 right shift is arithmetic, so a logical shift is (x >> k) & (0xFFFF >> k)
#   with the mask as a Python int (no device fill, which is unsupported for int16).
# * int16 add wraps mod 2^16 on device (ADDI16TOI16), so no masking after add.
# * Scalar int16 tensors are built on the CPU and moved to the device.
# * The compiled kernel takes only 1-D tensors; row-selects of a 2-D counter in
#   one graph fail with "Multi-arg pointwise: no supported output layout".
# ---------------------------------------------------------------------------

SKEIN_KS_PARITY_16_I16: int = 0x1BDA

_ROT16 = (
    (5, 13), (6, 10), (6, 14), (12, 2),
    (3, 10), (8, 6), (12, 5), (9, 10),
)


def _rotl16_i16(x: Tensor, n: int) -> Tensor:
    """Logical rotate-left by n inside a 16-bit window; x is int16."""
    if x.device.type == "spyre":
        return torch.ops.spyre.rotl16i16(x, n)   # one fused kernel
    rn = 16 - n
    lo = x.bitwise_left_shift(n)
    hi = x.bitwise_right_shift(rn) & int(0xFFFF >> rn)
    return lo | hi


def _rotl16_xor_i16(x: Tensor, n: int, y: Tensor) -> Tensor:
    """rotl(x, n) ^ y for int16 x, y of the same shape (one MIX step of Threefry)."""
    if x.device.type == "spyre":
        return torch.ops.spyre.rotlxori16(x, y, n)   # one fused two-input kernel
    return _rotl16_i16(x, n) ^ y


def _threefry4x16_i16(X0: Tensor, X1: Tensor, X2: Tensor, X3: Tensor,
                      ks: tuple, rounds: int):
    """
    Batched Threefry-4x16 on four (N,) int16 counter rows.

    ks[0..4]        : extended key schedule (ks4 = parity XOR)
    ks[5..5+max_s-1]: pre-folded X3 injection values ks[(s+3)%5] + s for
                      s = 1..rounds//4; eliminates one device add per injection.
    """
    X0 = torch.add(X0, ks[0])
    X1 = torch.add(X1, ks[1])
    X2 = torch.add(X2, ks[2])
    X3 = torch.add(X3, ks[3])

    for d in range(rounds):  # unrolled by torch.compile
        r0, r1 = _ROT16[d % 8]
        if d % 2 == 0:
            X0 = torch.add(X0, X1)
            X2 = torch.add(X2, X3)
            X1 = _rotl16_xor_i16(X1, r0, X0)
            X3 = _rotl16_xor_i16(X3, r1, X2)
        else:
            X0 = torch.add(X0, X3)
            X2 = torch.add(X2, X1)
            X3 = _rotl16_xor_i16(X3, r0, X0)
            X1 = _rotl16_xor_i16(X1, r1, X2)

        if (d + 1) % 4 == 0:  # key injection after rounds 3, 7, 11, ...
            s = (d + 1) // 4
            X0 = torch.add(X0, ks[s % 5])
            X2 = torch.add(X2, ks[(s + 2) % 5])
            X1 = torch.add(X1, ks[(s + 1) % 5])
            # X3 += ks[(s+3)%5] + s — +s is pre-folded into ks[5+s-1]
            X3 = torch.add(X3, ks[5 + s - 1])

    return X0, X1, X2, X3


_compiled_inner = torch.compile(_threefry4x16_i16, dynamic=False)


def _prepare_rows(counter: Tensor):
    """Row-preparation half of _prepare_inputs: (4, N) counter -> (X0, X1, X2, X3).

    clone() gives each row a fresh allocation with zero offset into HBM. On a device,
    row i starts at element i*N, so Spyre only accepts this split when N is a multiple
    of 64; callers with any other N build the four 1-D rows on the host instead.
    """
    if counter.dtype != torch.int16:
        counter = counter.to(torch.int16)
    return tuple(counter[i].clone() for i in range(4))


def _key_schedule(key: Tensor, rounds: int, dev: torch.device) -> tuple:
    """Key-schedule half of _prepare_inputs: (4,) key -> ks tuple on *dev*.

    ks[0..4]        : extended Threefry key schedule (ks4 = parity XOR)
    ks[5..5+max_s-1]: pre-folded X3 injection values ks[(s+3)%5] + s for
                      s = 1..rounds//4; eliminates one device add per injection.
    """
    if key.dtype != torch.int16:
        key = key.to(torch.int16)

    # Copy the whole key to the host first and index there: .cpu() of a sliced device
    # view (key[i]) ignores the storage offset on the rebased torch_spyre and returns
    # key[0] for every i (invisible with an all-zero key).
    key_cpu = key.cpu()
    k = [key_cpu[i] for i in range(4)]
    par = torch.tensor(SKEIN_KS_PARITY_16_I16, dtype=torch.int16)
    k4 = k[0] ^ k[1] ^ k[2] ^ k[3] ^ par
    ks_base = (*k, k4)

    # Pre-fold the +s scalar into the X3 key-injection constant.
    max_s = rounds // 4
    ks3_adj = tuple(
        ks_base[(s + 3) % 5] + torch.tensor(s, dtype=torch.int16)
        for s in range(1, max_s + 1)
    )
    return tuple(t.to(dev) for t in ks_base + ks3_adj)


def _prepare_inputs(counter: Tensor, key: Tensor, rounds: int):
    """Eager-mode setup for _threefry4x16_i16: returns (X0, X1, X2, X3, ks).

    The row split of a (4, N) counter (_prepare_rows) plus the key schedule
    (_key_schedule).
    """
    X0, X1, X2, X3 = _prepare_rows(counter)
    return X0, X1, X2, X3, _key_schedule(key, rounds, counter.device)


@functools.lru_cache(maxsize=4)
def _zero_row(calls: int, dev: torch.device) -> Tensor:
    """All-zero int16 row of *calls* elements on *dev*, built once per (size, device).

    The cipher never writes to its inputs, so one row can be shared by every call. It is
    used for word 3 only. Word 2 stays a separate tensor even when it is zero: the
    generator replaces it after its first advance, and sharing one tensor between two
    cipher inputs would change the aliasing the compiled graph sees.
    """
    return torch.zeros(calls, dtype=torch.int16).to(dev)


def _counter_rows_on_device(calls: int, offset: int, inv: int, dev: torch.device):
    """The four 1-D int16 counter rows of `make_counter(calls, offset)` with word 2 set to
    *inv*, on *dev*, without a host (4, N) tensor.

    Values are identical to `_prepare_rows(make_counter(...).to(dev))`. Only rows 0 and 1
    (lane index, low and high 16 bits) and row 2 (a constant) are built on the host; row 3
    is zero and comes from `_zero_row`. This avoids the four device clone kernels of
    `_prepare_rows` and the four host `counter[i].clone()` copies that the first row-wise
    version needed (about 1.5 ms at 419,840 columns).
    """
    idx = torch.arange(offset, offset + calls, dtype=torch.int32)
    x2 = torch.full((calls,), _wrap_i16(inv), dtype=torch.int16)
    return (idx.to(torch.int16).to(dev), (idx >> 16).to(torch.int16).to(dev),
            x2.to(dev), _zero_row(calls, dev))


# ---------------------------------------------------------------------------
# Device-side math (all 1-D tensors, no selects, scalars are Python constants)
# ---------------------------------------------------------------------------

def _u16(w: Tensor) -> Tensor:
    """int16 bit pattern -> fp32 holding the unsigned 16-bit value."""
    f = w.to(torch.float32)
    return f + 65536.0 * (f < 0).to(torch.float32)


def _uniform(lo: Tensor, hi: Tensor) -> Tensor:
    """Two 16-bit words -> fp32 uniform in (0, 1)."""
    s = _u16(hi) * 65536.0 + _u16(lo) + 0.5
    return s * _TWO_M32


def _finish(lo0, hi0, lo2, hi2, neg_inv_lam: float, log_variant: bool):
    """Arithmetic-only finish: (lo, hi) pairs -> exponential variates."""
    u0 = (hi0 * 65536.0 + lo0 + 0.5) * _TWO_M32
    u1 = (hi2 * 65536.0 + lo2 + 0.5) * _TWO_M32
    e0 = torch.log(u0) * neg_inv_lam
    e1 = torch.log(u1) * neg_inv_lam
    if log_variant:
        return torch.log(e0), torch.log(e1)
    return e0, e1


def _demux(fp: Tensor):
    """Split an i16tofp32 result into its lo and hi fp32 vectors.

    i16tofp32 maps each int16 stick of 64 elements to two fp32 sticks of 32:
        fp[k*64 : k*64+32]    = unsigned values of the even-indexed elements (lo)
        fp[k*64+32 : k*64+64] = unsigned values of the odd-indexed elements (hi)
    so view(-1, 2, 32) gives [stick, lo/hi, lane].
    """
    f3 = fp.view(-1, 2, 32)
    return f3[:, 0, :].reshape(-1), f3[:, 1, :].reshape(-1)


def _pipeline(X0, X1, X2, X3, ks, rounds, neg_inv_lam, log_variant):
    """The whole pipeline as ONE graph: cipher, i16_pair_to_fp32, transform, cat.

    Returns the (4 * calls // 2,) fp32 variates:
    [all from x0 | all from x1 | all from x2 | all from x3].
    """
    r = _threefry4x16_i16(X0, X1, X2, X3, ks, rounds)
    u0 = torch.ops.spyre.i16_pair_to_fp32(r[0])
    u1 = torch.ops.spyre.i16_pair_to_fp32(r[1])
    u2 = torch.ops.spyre.i16_pair_to_fp32(r[2])
    u3 = torch.ops.spyre.i16_pair_to_fp32(r[3])
    e0 = torch.log(u0) * neg_inv_lam
    e1 = torch.log(u1) * neg_inv_lam
    e2 = torch.log(u2) * neg_inv_lam
    e3 = torch.log(u3) * neg_inv_lam
    if log_variant:
        return torch.cat([torch.log(e0), torch.log(e1), torch.log(e2), torch.log(e3)])
    return torch.cat([e0, e1, e2, e3])


_compiled_pipeline = torch.compile(_pipeline, dynamic=False)

_E_FLOOR = 2.0 ** -25  # smallest E (for lam = 1) of the 24bit variant, see _pipeline_24


def _finish_24(u, neg_inv_lam, log_variant):
    """u (a sequence of fp32 tensors in (0, 1]) -> E = max(-ln(u), _E_FLOOR) / lam, cat.

    u == 1.0 (s == 2^24 - 1, once per 2^24 variates) would give E = 0 and, in the log
    variant, -inf. ln(u) is therefore limited to at most -_E_FLOOR before the scaling, which
    makes E >= _E_FLOOR / lam. The floor is the midpoint of the cell [0, 2^-24) that u == 1
    stands for, so every other grid point keeps its own value and no value is produced twice.
    torch.clamp is unsupported on fp32 by the backend, hence torch.minimum.
    """
    ln_u = [torch.log(x) for x in u]
    e = [torch.minimum(x, torch.full_like(x, -_E_FLOOR)) * neg_inv_lam for x in ln_u]
    if log_variant:
        return torch.cat([torch.log(x) for x in e])
    return torch.cat(e)


def _pipeline_24(X0, X1, X2, X3, ks, rounds, neg_inv_lam, log_variant):
    """The 24-bit pipeline as ONE graph: cipher, fused 24-bit conversions, transform, cat.

    Per 32-bit lane of the four cipher words L_0..L_3 (see the module docstring):
        variates 0..3:  s_w = L_w mod 2^24                     (i16_low24_to_fp32)
        variate 4:      s_4 = top8(L_0)*2^16 + top8(L_1)*2^8 + top8(L_2)   (i16_top3_to_fp32)
        u = (s + 1) * 2^-24 in (0, 1]; top8(L_3) is unused.
    E is floored at _E_FLOOR / lam, see `_finish_24`.

    Returns the (5 * calls // 2,) fp32 variates:
    [x0 low24 | x1 low24 | x2 low24 | x3 low24 | top bytes].
    """
    r = _threefry4x16_i16(X0, X1, X2, X3, ks, rounds)
    u = (
        torch.ops.spyre.i16_low24_to_fp32(r[0]),
        torch.ops.spyre.i16_low24_to_fp32(r[1]),
        torch.ops.spyre.i16_low24_to_fp32(r[2]),
        torch.ops.spyre.i16_low24_to_fp32(r[3]),
        torch.ops.spyre.i16_top3_to_fp32(r[0], r[1], r[2])
        )
    return _finish_24(u, neg_inv_lam, log_variant)


_compiled_pipeline_24 = torch.compile(_pipeline_24, dynamic=False)


# ---------------------------------------------------------------------------
# CPU-native pipelines (no torch.ops.spyre.* calls)
# Used when device is "cpu"; bit-compatible with the Spyre pipelines above.
# ---------------------------------------------------------------------------

def _i16_pair_to_fp32_cpu(w: Tensor) -> Tensor:
    """Pure-PyTorch equivalent of torch.ops.spyre.i16_pair_to_fp32.

    Maps each int16 stick of 64 elements to two fp32 sticks of 32 elements:
        fp[k*64 : k*64+32]    = unsigned(w[k*64 : k*64+64 : 2])  — lo halfwords
        fp[k*64+32 : k*64+64] = unsigned(w[k*64+1 : k*64+65 : 2]) — hi halfwords

    Interleaved layout: even indices = lo, odd indices = hi.
    Result: (len(w),) fp32 with unsigned int16 values, lo/hi interleaved to [lo|hi].
    """
    lo = _u16(w[0::2])   # even-indexed = lo halfwords
    hi = _u16(w[1::2])   # odd-indexed  = hi halfwords
    n = lo.shape[0]
    # Reshape to (-1, 64//2) = (-1, 32) sticks then interleave lo and hi blocks
    # to match the Spyre layout: [lo_stick0 | hi_stick0 | lo_stick1 | hi_stick1 | ...]
    lo3 = lo.view(-1, 32)
    hi3 = hi.view(-1, 32)
    return torch.cat([lo3, hi3], dim=1).reshape(2 * n)


def _i16_low24_to_fp32_cpu(w: Tensor) -> Tensor:
    """Pure-PyTorch equivalent of torch.ops.spyre.i16_low24_to_fp32.

    Per 32-bit lane (lo = w[2k], hi = w[2k+1]):
        s = (L mod 2^24)  where L = unsigned(lo) | (unsigned(hi) << 16)
        u = (s + 1) * 2^-24  in (0, 1]

    Output: (len(w)//2,) fp32, one value per 32-bit lane.
    """
    lo = _u16(w[0::2]).to(torch.float32)   # unsigned lo halfwords
    hi = _u16(w[1::2]).to(torch.float32)   # unsigned hi halfwords
    L = lo + hi * 65536.0                  # 32-bit lane value
    s = L.remainder(16777216.0)            # L mod 2^24 (fp32-safe for L < 2^32)
    return (s + 1.0) * (2.0 ** -24)


def _i16_top3_to_fp32_cpu(w0: Tensor, w1: Tensor, w2: Tensor) -> Tensor:
    """Pure-PyTorch equivalent of torch.ops.spyre.i16_top3_to_fp32.

    Per 32-bit lane of each word:
        top8(L_i) = unsigned(hi_i) >> 8  (top 8 bits of the unsigned hi halfword)
        s = top8(L_0)*2^16 + top8(L_1)*2^8 + top8(L_2)
        u = (s + 1) * 2^-24  in (0, 1]

    Output: (len(w0)//2,) fp32.
    """
    hi0 = (_u16(w0[1::2]) / 256.0).floor()   # top 8 bits of hi halfword of w0
    hi1 = (_u16(w1[1::2]) / 256.0).floor()
    hi2 = (_u16(w2[1::2]) / 256.0).floor()
    s = hi0 * 65536.0 + hi1 * 256.0 + hi2
    return (s + 1.0) * (2.0 ** -24)


def _pipeline_cpu(X0, X1, X2, X3, ks, rounds, neg_inv_lam, log_variant):
    """CPU-native 32-bit pipeline: same semantics as _pipeline but without spyre ops."""
    r = _threefry4x16_i16(X0, X1, X2, X3, ks, rounds)
    # _i16_pair_to_fp32_cpu returns (calls,) fp32 in [lo_block | hi_block] per stick.
    # The log computation mirrors _pipeline: each word gives calls//2 uniforms as
    # u = (hi*65536 + lo + 0.5) * 2^-32, which matches _finish via (lo|hi) pairs.
    def _word_to_exp(w):
        lo = _u16(w[0::2])
        hi = _u16(w[1::2])
        u = (hi * 65536.0 + lo + 0.5) * _TWO_M32
        e = torch.log(u) * neg_inv_lam
        return torch.log(e) if log_variant else e

    return torch.cat([_word_to_exp(r[i]) for i in range(4)])


def _pipeline_24_cpu(X0, X1, X2, X3, ks, rounds, neg_inv_lam, log_variant):
    """CPU-native 24-bit pipeline: same semantics as _pipeline_24 but without spyre ops."""
    r = _threefry4x16_i16(X0, X1, X2, X3, ks, rounds)
    u = (
        _i16_low24_to_fp32_cpu(r[0]),
        _i16_low24_to_fp32_cpu(r[1]),
        _i16_low24_to_fp32_cpu(r[2]),
        _i16_low24_to_fp32_cpu(r[3]),
        _i16_top3_to_fp32_cpu(r[0], r[1], r[2]),
    )
    return _finish_24(u, neg_inv_lam, log_variant)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def padded_calls(n: int) -> int:
    """Threefry calls used for n samples, padded to a multiple of 64 (one int16 stick).

    All four cipher words x0, x1, x2, x3 are used.  i16_pair_to_fp32 pairs the lo/hi
    halfwords within each 32-bit lane of an int16 stick: each word yields
    ``calls//2`` variates, so one Threefry call produces ``4 * (calls//2) =
    2*calls`` variates.  We therefore need ``2*calls >= n``, i.e.
    ``calls >= ceil(n/2)``, padded to a multiple of 64 to keep the tensors
    stick-aligned (64 int16 elements per stick).
    """
    return -((-((n + 1) // 2)) // 64) * 64


def padded_calls_24(n: int) -> int:
    """Threefry calls for n samples at 2.5 variates per call (the "24bit" variant).

    Five blocks of ``calls // 2`` variates, so ``5 * (calls // 2) >= n``, i.e.
    ``calls >= ceil(2n / 5)``. Padded to a multiple of 2048 = 64 elements per stick
    * 32 cores: with only a multiple of 64 the stick count can have no divisor near
    32 (e.g. 1,639 = 11 * 149 sticks at n = 262,144) and the work division then runs
    every SDSC on 11 cores instead of 32.
    """
    need = -(-(2 * n) // 5)
    return -(-need // 2048) * 2048


DEFAULT_VARIANT = "24bit"


def _resolve_variant(variant: str):
    """Variant name -> (compiled pipeline, padded_calls function)."""
    if variant == "24bit":
        return _compiled_pipeline_24, padded_calls_24
    if variant == "32bit":
        return _compiled_pipeline, padded_calls
    raise ValueError(f'variant must be "24bit" or "32bit", got {variant!r}')


def _resolve_variant_cpu(variant: str):
    """Variant name -> (cpu pipeline function, padded_calls function)."""
    if variant == "24bit":
        return _pipeline_24_cpu, padded_calls_24
    if variant == "32bit":
        return _pipeline_cpu, padded_calls
    raise ValueError(f'variant must be "24bit" or "32bit", got {variant!r}')


def make_counter(calls: int, offset: int = 0) -> Tensor:
    """
    (4, calls) int16 counter on CPU. Call index i = offset + j is a 32-bit value
    split over rows 0 (low 16 bits) and 1 (high 16 bits); rows 2-3 are zero.
    A bare int16 index would wrap after 65536 calls and repeat the stream.
    int32 arithmetic is used for speed, so offset + calls must stay below 2**31.
    """
    idx = torch.arange(offset, offset + calls, dtype=torch.int32)
    out = torch.zeros(4, calls, dtype=torch.int16)
    out[0] = idx.to(torch.int16)  # int32 -> int16 keeps the low 16 bits
    out[1] = (idx >> 16).to(torch.int16)
    return out


def _wrap_i16(v: int) -> int:
    """Python int -> the signed int16 value with the same low 16 bits."""
    return ((int(v) + 0x8000) & 0xFFFF) - 0x8000


@torch.compiler.disable
def _generate(n, lam, key, rounds, device, offset, log_variant, inv=0,
              variant=DEFAULT_VARIANT):
    if lam <= 0:
        raise ValueError("lam must be > 0")
    dev = torch.device(device)
    # Host key: _key_schedule works on the host, so uploading it only to copy it back
    # (key.cpu()) would be a wasted roundtrip.
    key_t = torch.tensor(list(key), dtype=torch.int32).to(torch.int16)

    neg_inv_lam = -1.0 / lam
    # Rows are built on the host and copied one by one: no device clone kernels.
    # Word 2 is the invocation counter.

    if dev.type == "cpu":
        # CPU path: use pure-PyTorch pipelines (no torch.ops.spyre.* calls).
        pipe, padded = _resolve_variant_cpu(variant)
        calls = padded(n)
        X0, X1, X2, X3 = _counter_rows_on_device(calls, offset, inv, dev)
        ks = _key_schedule(key_t, rounds, dev)
        out = pipe(X0, X1, X2, X3, ks, rounds, neg_inv_lam, log_variant)
    else:
        pipe, padded = _resolve_variant(variant)
        # Row i of the (4, calls) counter starts at element i*calls, and Spyre rejects
        # slices that are not stick-aligned, hence the padding.
        calls = padded(n)
        X0, X1, X2, X3 = _counter_rows_on_device(calls, offset, inv, dev)
        ks = _key_schedule(key_t, rounds, dev)
        out = pipe(X0, X1, X2, X3, ks, rounds, neg_inv_lam, log_variant)
    return out[:n]


def _advance(x: Tensor, one: Tensor) -> Tensor:
    return x + one  # int16 add wraps mod 2^16 on device


_compiled_advance = torch.compile(_advance, dynamic=False)


class ExpSpyreGenerator:
    """Stateful generator: the counter lives on the device and is only advanced.

    ``variant`` selects the pipeline: "24bit" (default, 2.5 variates per Threefry call,
    24 random bits each) or "32bit" (2 variates per call, the previous pipeline). The
    sample stream differs between the two variants.

    The counter rows, key schedule and the scalar used for the advance are built once
    in ``__init__``. Every ``generate()`` call then costs one compiled int16 add
    (``X2 += 1``, about 0.1 ms) plus the compiled pipeline; the CPU counter build, the
    host-to-device transfer and ``_prepare_inputs`` are not repeated.

    Stream layout: counter words 0/1 hold the lane index (low/high 16 bits), word 3 is
    zero and word 2 is the invocation counter. Invocation ``k`` uses ``X2 = start_invocation
    + k`` (mod 2**16), so the stream repeats after 65,536 invocations. This differs from
    the former ``offset`` semantics, where ``offset`` shifted the lane index.

    ``n`` (hence the output shape) is fixed at construction. The first ``generate()``
    call compiles ``_compiled_pipeline``, and the second one ``_compiled_advance``
    (unless already cached); no explicit warm-up is needed. On CPU the device-side
    advance is skipped and ``generate()`` delegates to ``_generate`` with the same
    invocation index.
    """

    def __init__(self, n: int, lam: float = 1.0,
                 key: tuple[int, int, int, int] = (0, 0, 0, 0),
                 rounds: int = 8, device: str | torch.device = "spyre",
                 start_invocation: int = 0, variant: str = DEFAULT_VARIANT):
        if lam <= 0:
            raise ValueError("lam must be > 0")
        self._pipe, padded = _resolve_variant(variant)
        self._variant = variant
        self._n = n
        self._lam = lam
        self._key = tuple(key)
        self._rounds = rounds
        self._device = torch.device(device)
        self._start = start_invocation
        self._neg_inv_lam = -1.0 / lam
        self._call_count = 0
        if self._device.type == "cpu":
            # CPU path: stateless, delegates to _generate on each call.
            self._cpu_pipe = _resolve_variant_cpu(variant)[0]
            return

        dev = self._device
        calls = padded(n)
        key_t = torch.tensor(list(key), dtype=torch.int32).to(torch.int16)
        self._X0, self._X1, self._X2, self._X3 = _counter_rows_on_device(
            calls, 0, start_invocation, dev)
        self._ks = _key_schedule(key_t, rounds, dev)
        self._one = torch.tensor(1, dtype=torch.int16).to(dev)

    @property
    def offset(self) -> int:
        """Number of ``generate()`` calls since construction."""
        return self._call_count

    def generate(self, log_variant: bool = False) -> Tensor:
        """(n,) float32 Exponential(lam), or ln(E) if ``log_variant``.

        Each call uses the next invocation index. The first call uses
        ``start_invocation``; later calls advance word 2 on the device first.
        """
        if self._device.type == "cpu":
            out = _generate(self._n, self._lam, self._key, self._rounds,
                            self._device, 0, log_variant,
                            self._start + self._call_count, self._variant)
        else:
            if self._call_count > 0:
                self._X2 = _compiled_advance(self._X2, self._one)
            out = self._pipe(self._X0, self._X1, self._X2, self._X3,
                             self._ks, self._rounds, self._neg_inv_lam,
                             log_variant)[:self._n]
        self._call_count += 1
        return out


def generate_exp_spyre(n: int, lam: float = 1.0,
                       key: tuple[int, int, int, int] = (0, 0, 0, 0),
                       rounds: int = 8, device: str | torch.device = "spyre",
                       offset: int = 0, variant: str = DEFAULT_VARIANT) -> Tensor:
    """(n,) float32 Exponential(lam). `offset` is the starting invocation index
    (counter word 2), not a lane offset. `variant` is "24bit" or "32bit" (see the
    module docstring).

    Stateless convenience wrapper: it builds an `ExpSpyreGenerator` per call, so the
    setup cost is paid every time. For repeated calls hold an `ExpSpyreGenerator`.
    Works on both "cpu" and "spyre" devices.
    """
    return ExpSpyreGenerator(n, lam, key, rounds, device, offset,
                             variant).generate(False)


def generate_log_exp_spyre(n: int, lam: float = 1.0,
                           key: tuple[int, int, int, int] = (0, 0, 0, 0),
                           rounds: int = 8, device: str | torch.device = "spyre",
                           offset: int = 0, variant: str = DEFAULT_VARIANT) -> Tensor:
    """(n,) float32 ln(E), E ~ Exponential(lam). Equals ln(E_lam1) - ln(lam).

    `offset` is the starting invocation index and `variant` the pipeline
    (see `generate_exp_spyre`). Works on both "cpu" and "spyre" devices.
    """
    return ExpSpyreGenerator(n, lam, key, rounds, device, offset,
                             variant).generate(True)
