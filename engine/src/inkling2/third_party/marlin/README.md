# Marlin MoE kernels (vendored)

The mixed-precision MoE GEMM kernels of vLLM, used by `strata-ink2 --marlin 1` for the NVFP4 experts of
Inkling-2-Mini (glue: `../../ink2_marlin.cuh`).

- Source: https://github.com/vllm-project/vllm, commit `0e3c8dbdae9d55149b0fba4e6c1014b4b08c9c73` (main, 2026-11-22).
- License: Apache-2.0, see `LICENSE` (vLLM's). The kernels derive from Marlin (https://github.com/IST-DASLab/marlin,
  Elias Frantar; modified by Neural Magic and the vLLM contributors); the file headers are kept.
- Strata itself is MIT; this directory stays under Apache-2.0.

## Files

| Here | vLLM | State |
| --- | --- | --- |
| `quantization/marlin/marlin.cuh`, `marlin_dtypes.cuh`, `marlin_mma.h`, `dequant.h` | `csrc/quantization/marlin/` | verbatim |
| `core/scalar_type.hpp` | `csrc/core/scalar_type.hpp` | modified |
| `moe/kernel.h`, `moe/marlin_template.h` | `csrc/moe/marlin_moe_wna16/` | modified |
| `moe/ops.cu` | `csrc/moe/marlin_moe_wna16/ops.cu` | modified (included by `ink_marlin.cu`, not compiled alone) |
| `moe/sm80_kernel_float16_fe2m1f_float16.cu`, `moe/kernel_selector.h` | output of `csrc/quantization/marlin/generate_kernels.py 8.6` | generated, selector filtered |
| `ink_marlin.h`, `ink_marlin.cu` | new (launcher wrapper and weight repack) | ours, Apache-2.0 as a derived work |

## Modifications

- `core/scalar_type.hpp`: `STD_TORCH_CHECK` is a local macro (prints and aborts); the torch include is gone. Nothing
  here needs torch.
- `moe/kernel.h`, `moe/marlin_template.h`: one more kernel parameter, `expert_tab`. When it is set, the weights and
  scales of an expert are found through a pointer table (`expert_tab[expert_id]` points at `[weights | scales]` of
  that expert), not at `expert_id * stride` in one stacked tensor: strata-ink2 keeps experts in pool slots. The B
  read indices are 64-bit for that reason. With `expert_tab == nullptr` the kernel addresses experts as upstream.
- `moe/ops.cu`: the torch entry point (`moe_wna16_marlin_gemm` and its registration) and the torch includes are
  removed; `marlin_mm` takes `expert_tab` and passes it to the kernel.
- `moe/kernel_selector.h`: only the entries of the fp16-activation NVFP4 kernels (`kFloat16`, `kFE2M1f`, scales
  `kFE4M3fn`) are kept, the only family instantiated here.
- `ink_marlin.cu` `repack_part`: the weight permutation of `csrc/quantization/marlin/gptq_marlin_repack.cu` and the
  scale permutation and S0E5M3 encoding of `marlin_permute_scales` / `nvfp4_marlin_process_scales`
  (`vllm/model_executor/layers/quantization/utils/marlin_utils.py`, `marlin_utils_fp4.py`), rewritten as two gather
  kernels that read the checkpoint layout (row-major `[N][K/2]` nibbles, `[N][K/16]` scales) directly.

## Build

Two translation units (`ink_marlin.cu`, `moe/sm80_kernel_float16_fe2m1f_float16.cu`) with
`-DMARLIN_NAMESPACE_NAME=marlin_moe_wna16 --expt-relaxed-constexpr -static-global-template-stub=false` and this
directory and `moe/` on the include path (`engine/cmake/strata_gpu_engine.cmake`, target `strata_ink2_marlin`).

To regenerate the kernel list: run `generate_kernels.py 8.6 OUTDIR` of the commit above (it needs jinja2), copy
`sm80_kernel_float16_fe2m1f_float16.cu`, and keep the `kernel_selector.h` entries of that family, the first as `if`.
