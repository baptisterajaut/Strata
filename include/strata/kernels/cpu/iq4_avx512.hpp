// include/strata/kernels/cpu/iq4_avx512.hpp - IQ4_XS gate/up and IQ4_NL down expert rows in 512-bit lanes for the CPUs
// with AVX-512 F/BW/VL/DQ + VNNI but no VBMI (Cascade Lake, Skylake-SP's successor: Xeon Gold 6226 / 6230R), where
// cpu_avx512_ok() is false and every other AVX-512 kernel stays off.  Called only where iq4_512_on() (native_expert.cpp).
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels::cpu {

/// ff[t][r] = silu(gate_r . a[t]) * (up_r . a[t]) for IQ4_XS (type 23) rows against Q8_K activations, rows [r0, r1).
void iq4xs512_gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, int nt,
                      float* const* ff, int r0, int r1);
/// out[t][r] = w_r . h[t] for IQ4_NL (type 20) rows against Q8_0 activations, rows [r0, r1); n a multiple of 64.
void iq4nl512_down_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* hq, int nt, float* const* out,
                        int r0, int r1);

}  // namespace strata::kernels::cpu
