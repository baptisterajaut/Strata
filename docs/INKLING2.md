# Inkling-2-Mini (`strata-ink2`)

`strata-ink2` is Strata's engine program for [Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini)
(Thinking Machines Lab, Apache 2.0): a 128-expert, top-8 mixture of experts whose layers 6-17 run three times, so a
token goes through 48 layer applications. The engine reads the published checkpoint as it is: NVFP4 experts in layers
0-19, MXFP8 experts in layers 20-23, BF16 for everything else, and the ViT that reads pictures. Nothing is requantized.

To install and start it, see [MODELS.md](MODELS.md#inkling-2-mini): `./setup.sh --family inkling2` (the official
checkpoint, one of the two sources of setup's Inkling family; the `.strata` SQ builds run `strata-inkling` instead)
picks the flags below for your card. This page is for reading what setup chose, or for running the engine by hand.

NVIDIA (RTX 20 series or newer) or AMD RDNA (wave32, gfx10 to gfx12) through ROCm, one GPU, Linux. The AMD port is
jingx8885's, measured on a gfx1151 (Radeon 8060S); the speeds below are NVIDIA ones.

## What setup chooses

Measured on an RTX 3090 with its VRAM limited to the card size (the expert pools only see that much), 2x Xeon Gold
6252, 257-token chat prompt, 512 greedy tokens; prompt speed on a 16K-token prompt.

| Card | Engine flags (besides `--model`, `--chunk 1024 --tc 1`) | Decode | Prompt |
| --- | --- | --- | --- |
| 24 GB | `--max-context 65536 --kv f16 --spec 1` (every expert on the card) | 88-90 tok/s (108 on short prompts with `--spec`) | 2,380 tok/s |
| 16 GB | `--max-context 131072 --kv f16`, offload as below | not measured | not measured |
| 24 GB, longer contexts | `--max-context 131072 --kv f16 --kv-grow 1 --vram-reserve-mib R --cpu-threads N --spec 1` | 84-86 tok/s (103 on short prompts) | not measured |
| 12 GB, with a desktop | `--max-context 65536 --kv f16 --kv-grow 1 --vram-reserve-mib R --cpu-threads N` | 73-75 tok/s | 717 tok/s |
| 8 GB, nothing else on it | `--max-context 32768 --kv f16 --kv-grow 1 --vram-reserve-mib R --cpu-threads N` | 66 tok/s | 568 tok/s |
| 8 GB, with a desktop | the same with `--kv i8r` | 60-61 tok/s | 560 tok/s |

Every config also gets `--conversation-cache-mib 4096` (2048, or off, when RAM is short). In offload the engine pins
about 17 GiB of RAM for the experts that are not on the card; setup asks for 32 GB of RAM.

With the offload, setup adds `--spec 1` only where the card still holds 70% of the experts beside the drafter at the
full context. The drafter takes about 2 GiB from the expert pools (its weights, its K/V, and the prompt buffers, which
stay mapped with it), and a check computes eight tokens' experts to keep three to five, so the experts that are not on
the card cost more per token than in plain decoding. Decode in tok/s, plain and with `--spec 1` (same box, 512 greedy
tokens, mean of two runs; 16 GB: one run):

| Card and context | Experts on the card with the drafter | Chat prompt (257 tokens) | Short prompt (24 tokens) |
| --- | --- | --- | --- |
| 8 GB, 32K | 19% | 66.9, with `--spec 1` 51.8 | 64.6, with `--spec 1` 59.9 |
| 12 GB, 64K | 45% | 80.6, with `--spec 1` 72.3 | 81.5, with `--spec 1` 87.9 |
| 16 GB, 128K | 70% | 84.1, with `--spec 1` 84.4 | 85.7, with `--spec 1` 100.2 |
| 24 GB, 128K | 98% | 85.9, with `--spec 1` 83.9 | 86.7, with `--spec 1` 103.1 |
| 24 GB, 256K, `--kv i8r`, no `--kv-grow` | 92% | 82.9, with `--spec 1` 84.7 | 83.3, with `--spec 1` 100.0 |

These are speeds at the start of a conversation: with `--kv-grow` the K/V of a long conversation takes expert slots,
which is why the 16 GB card at 128K stays without the drafter.

## Flags

**Model and context**

- `--model DIR` the checkpoint folder (or a GGUF file, below).
- `--max-context N` (alias `--ctx`) the longest conversation, prompt plus answer.
- `--chunk N` prompt tokens read per step (`--serve` needs one; setup uses 1024). With `--kv-grow` a chunk shrinks by
  itself when the card is short of room.
- `--tc 0|1` tensor cores for the prompt (1, the default in setup). `--tc 0` is the exact fp32 path, slower, used to
  check the other one.
- `--serve` the line protocol the Strata server speaks.

**Memory**

- `--vram-reserve-mib N` leaves N MiB of VRAM free and turns on the expert offload: the experts that do not fit stay
  in pinned RAM, and the card keeps the most used ones (decayed use counts, promotions in the background).
- `--cpu-threads N` computes the experts that are not on the card on the CPU instead of copying them over PCIe: one
  pool of N threads per NUMA node (at most two), each expert's memory bound to its node. Needs `--vram-reserve-mib`,
  AVX2 + FMA + F16C, and Linux. `STRATA_NUMA_POOL=0` turns the NUMA placement off.
- `--kv-grow 1` the K/V cache starts small and grows into the expert pools as the conversation grows, so a long
  context costs expert slots only when it is used. Each pool keeps at least 64 slots: a prompt that would need more
  ends with a "context is full" error, and the engine keeps serving. `--kv-grow-step N` sets the growth step.
- `--kv FORMAT` the K/V format. Mean KL against fp32 K/V (expert routing fixed, 23K-token mixed text):

  | `--kv` | Bytes per token and layer pass | KL mean | Top-1 |
  | --- | --- | --- | --- |
  | `f16` (default) | 4,096 | 0 | 100% |
  | `i8r` (int8, rotated) | 2,112 | 3.5e-5 | 99.80% |
  | `k8v8` | 2,112 | 5.3e-5 | 99.73% |
  | `i8` | 2,112 | 1.3e-4 | 99.61% |
  | `k8v4` | 1,632 | 1.5e-3 (1.6K-token text) | 98.75% |

  `f32` exists for checks.

**Speed**

- `--spec 1` speculative decoding with the model's own drafter (`mtp.safetensors`): a draft token is kept only when it
  is the model's own choice (greedy) or passes the usual acceptance test (sampled). It falls back to plain decoding
  while that is faster. The check reads the drafts through other kernels than plain decoding, so a near-tie can go the
  other way: a 512-token greedy answer matched plain decoding for 273 tokens, then parted where the top two logits were
  0.2 apart. With `--cpu-threads` the CPU also computes a check's experts that are not on the card (same answer as
  with every expert on the card over those 512 tokens); it pays only where most experts are on the card (the table
  under [What setup chooses](#what-setup-chooses)).
- `--verify-tc 1` (off by default) runs the dense products of a `--spec` check and of the drafter (up to 8 vectors,
  matrices of more than 256 rows) on tensor cores: fp16 weights and inputs, fp32 accumulation, the numbers of the
  prompt path's products. Without it, those products follow plain decoding's order. CUDA cards of compute capability
  8.0 or later. RTX 3090, every expert on the card, `--kv i8r`, a check at every step, three runs each: 36.0 ms per
  check without, 32.6 with. With the expert routing fixed, the 8-token prompt lane is at mean KL 3.0e-7 against the
  fp32 token path, against 8.7e-8 without the flag, top-1 100%.

- `--marlin 1` (off by default) keeps the NVFP4 experts (layers 0-19) in the Marlin layout and multiplies them with
  the Marlin MoE kernels vendored from vLLM (`engine/src/inkling2/third_party/marlin`, Apache-2.0): fp16 activations
  and products, fp32 accumulation, for prompts, `--spec` checks and plain decoding alike. It needs the checkpoint,
  every expert on the card (no `--vram-reserve-mib`) and an NVIDIA card of the RTX 30 series or newer; elsewhere the
  engine says so and runs as without it. The MXFP8 experts (layers 20-23) keep their kernels. RTX 3090, every expert
  on the card, two runs each:

  | | Without | `--marlin 1` |
  | --- | --- | --- |
  | 16K-token prompt, `--chunk` 512 / 1024 / 2048 | 1,900 / 2,450 / 2,830 tok/s | 2,910 / 3,360 / 3,670 tok/s |
  | 1K-token prompt | 2,510 tok/s | 3,450 tok/s |
  | Decode, chat prompt (257 tokens) | 91.0, with `--spec 1` 88.6 | 89.8, with `--spec 1` 94-95 |
  | Decode, short prompt (24 tokens) | 92.5, with `--spec 1` 108.8 | 91.2, with `--spec 1` 127.3 |

  A prompt chunk's expert products run twice as fast at `--chunk 2048` (3.4 to 1.7 ms per launch) and a `--spec`
  check's about twice (estimated from the kernel totals); one token at a time the two kernels are level, both read
  the eight experts at about two thirds of the card's memory bandwidth, and decoding is 1% slower. Loading takes
  about 550 MiB more VRAM while a layer is repacked, and 21 MiB after. Quality is in [Quality](#quality).

**Conversations and pictures**

- `--conversation-cache-mib N` (0, off, by default) keeps conversations in pinned RAM so that a request never reads
  again a prompt the engine already read: a checkpoint at the end of each prompt lets the next turn continue from
  there even when the client sends the history back without the model's reasoning, conversations set aside (an
  agent and its subagents taking turns) are taken up again, and conversations sharing a long beginning (a system
  prompt and tools) read it once. `--conversation-cache-min-tokens N` (512): shorter conversations are not kept.
- Pictures need no flag. The ViT is read into RAM at the first picture and streamed to the card for each one, so
  text-only use costs no VRAM; with offload it borrows a few hundred MiB of expert slots while it runs. A picture
  over `--image-max-tokens N` (4096; one token is a 40x40 pixel tile) is scaled down.

**Checks and debugging**: `--tokens IDS` / `--tokens-file F` with `--n N` run one prompt without the server,
`--dump-logits F`, `--record-routing F` / `--force-routing F`, `--image F`, `--dump-image-emb F`, `--vis-pixels F`.
`terminal/tools/inkling2/README.md` describes the comparison against the transformers implementation.

## GGUF files

`--model FILE.gguf` reads llama.cpp's merged `inkling2_loop` layout with IQ3_S, IQ4_NL and Q8_0 experts (work by
merbanan). GGUF models run without `--cpu-threads` and `--spec` for now, and setup does not install them.
Layers 20-23 may use a different expert type than layers 0-19, as in ggml-org's `Inkling-2-Mini-IQ4_NL.gguf` (IQ4_NL,
Q8_0 in layers 20-23): on the 1,600-token text its mean KL against the fp32 reference is 5.9e-3 (llama.cpp on the same
file: 5.4e-3). Any other mix, or an expert type outside these three (NVFP4, for example), stops the load with an error
that names the layers and types.

## Quality

Against the transformers implementation in fp32 on a 1,600-token text: the token-by-token path with fp32 K/V is at
mean KL 1.4e-12 (top-1 100%), and the tensor-core prompt path at 2.1e-7 with the expert routing fixed. Pictures: mean
KL 1e-7 or less in fp32, with the same greedy answers.

With `--marlin 1` (fp16 expert products) and the expert routing fixed, the prompt path is at mean KL 3.4e-7 against
the fp32 token-by-token path (1.3e-7 without the flag, same run) and a `--spec` check block at 1.0e-7 (8.7e-8), top-1
100% in each case. With free routing and fp16 K/V, against the transformers fp32 reference: token-by-token 1.41e-3 and
top-1 98.37% (1.32e-3, 98.44% without the flag), prompt path 1.15e-3 and 98.37% (1.03e-3, 98.31%); those figures move
with single routing flips, not with the kernels. As without the flag, a `--spec 1` answer can part from the plain one
at a near-tie (512 greedy tokens: the same for 181 and 316 tokens on two prompts; 273 without the flag).

## Compared with vLLM

vLLM 0.34.0rc1.dev1145+g0e3c8dbda (the nightly wheel of 2026-11-22), same RTX 3090, same checkpoint. That build asks
for compute capability 9.0; it ran on the 3090 after four local changes: the capability gate lowered, FlashAttention 2
for the vision tower, programmatic dependent launch turned off in Triton, and the Triton fallback instead of the
native conv/mix kernel. It keeps bf16 K/V and its expert products come from the Humming kernels with bf16
activations (not from Marlin). With 18.2 GiB of weights it could not hold a 16K-token context in 24 GB (it asked for
4.4 GiB of K/V and had 1.6 to 2.2), so it was measured at `--max-model-len 1536` (680 with DSpark), batch 1, greedy:

| | vLLM | strata-ink2 | with `--marlin 1` |
| --- | --- | --- | --- |
| Decode, chat prompt (257 tokens) | 93.6 tok/s | 91.0 | 89.8 |
| the same with the drafter (DSpark, 7 drafts; ours `--spec 1`) | 126.1 (400 tokens) | 88.6 | 94-95 |
| Decode, short prompt (24 tokens) | 94.1 | 92.5 | 91.2 |
| the same with the drafter | 221.6 | 108.8 | 127.3 |
| 1K-token prompt | 4,700 to 5,240 tok/s | 2,510 | 3,450 |

A public card (localmaxxing.com: vLLM v0.1.dev25855+g7cb5dae22, RTX 3090, 65,536-token context, 22.9 GB) reports
5,245 tok/s on a 16,384-token prompt and 242 tok/s with DSpark. That commit is not in the vLLM repository and the
build above could not run that shape, so it is not reproduced here. On it (16,384 tokens in, 1,174 out,
`--max-context 65536 --chunk 2048`) strata-ink2 reads the prompt at 2,830 tok/s and writes 84.8 tok/s (100.8 with
`--spec 1`); with `--marlin 1`, 3,680 tok/s and 83.9 tok/s (109 to 114 with `--spec 1`).

Where the difference is. One token at a time the engines are level: both are bound by reading eight experts per layer
pass. With the drafter, vLLM spends 17 to 21 ms on a step (seven drafts and the check of eight positions) and keeps
2.7 to 3.9 tokens of it; strata-ink2 keeps 3 to 5 but spends about 36 ms (30 with `--marlin 1`), of which 12 ms are
the attention and output projections read once per position group, so it often goes back to plain decoding. On
prompts, after `--marlin 1` the expert products are a quarter of the GPU time at 16K and the attention kernel another
quarter.

## Limits

- One GPU. `--cpu-threads` on Linux only, at most two NUMA nodes.
- `--spec` with the expert offload and no `--cpu-threads` copies a check's missing experts over PCIe: 16 tok/s on an
  8 GB card. Setup does not write that pair.
- `--marlin 1` only with every expert on the card: with the expert offload or a GGUF file the flag is ignored, and
  setup does not write it.
- Not offered in `strata launch` or the desktop app yet.
