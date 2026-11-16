# GLM-5.3 monokernels on gfx950

This document explains `sgl_amd_mono/glm53/`: what its launches run, which weights they read or copy, how they plug
into SGLang and how they are tested. It covers both checkpoints the spec serves: GLM-5.3's FP8 block-128 checkpoint
(`zai-org/GLM-5.3`) and GLM-5.2's Quark MXFP4 checkpoint (`amd/GLM-5.2-MXFP4`), which share the architecture
(`GlmMoeDsaForCausalLM`, NextN draft `GlmMoeDsaForCausalLMNextN`). The SGLang interface is the one of RFC
sgl-project/sglang#43949. The kernel comes from ROCm/aiter#6173 at 56d6390def; each file names its origin.

## 1. Scope

| | target (`spec.py`) | NextN draft (`draft.py`) |
| --- | --- | --- |
| SGLang steps | `DECODE`, and EAGLE `TARGET_VERIFY` with topk 1 and draft tokens = steps + 1 | `DRAFT_EXTEND_V2` and the chain's draft `DECODE` steps |
| rows | 1-8 (widths 2, 4, 8; a step runs at the next width, the rest pad rows) | extend: the verify rows; decode: 1 a request |
| launches per step | 78 layers (0 from the tokens, 1-2 dense, 3-77 MoE) and the tail | the NextN layer and its tail |
| other steps | SGLang's forward | SGLang's forward |

GPU: gfx950 (MI355X), TP 4 or 8, `--page-size 64` (SGLang's DSA cache on ROCm). A one-row launch never completes on
gfx950, so one row runs at width 2. With MTP 3 (4 verify rows a request) the launches take steps of 1 and 2
requests; larger batches run SGLang's forward.

| file | contents |
| --- | --- |
| `spec.py` | the target's `MonoSpec`, its refusal reasons, `NUMERICS` and its forward |
| `draft.py` | the NextN draft's `MonoSpec`, its forward and the wrapper that routes the draft's own steps |
| `check.py` | check mode (`SGLANG_AMD_MONO_CHECK=1`) |
| `tables.py` | a step's row tables and the draft's top-k seed, each one Triton launch |
| `gfx950/kernel.py` | the FlyDSL decoder-layer launch (`build_glm5_monokernel`) |
| `gfx950/config.py`, `layout.py`, `tiles.py` | the geometry, the scratch and peer-buffer layouts, the CTA schedule |
| `gfx950/weights.py` | the layer views of SGLang's params (`Glm53Weights`, `Glm53HeadWeights`) and the copies |
| `gfx950/packing.py` | the host packing of the copied matrices into the MFMA tile order |
| `gfx950/runner.py` | the build key (`Glm53Build`), one runtime per (width, kind) and one op per layer |

## 2. One layer, one launch

Each covered layer is one launch of 256 CTAs x 512 threads (one CTA per CU, all resident) for this rank's TP shard:

```
input RMSNorm -> q_a | kv_a -> q_a RMSNorm -> q_b (+RoPE) -> KV RMSNorm / k_pe RoPE -> KV cache write
  -> indexer: k / q / weights projections, k norm, RoPE, Hadamard, index-K cache write
  -> index scores -> exact radix top-2048
  -> absorbed q (W_UK) -> sparse MLA split softmax -> merge -> W_UV -> o_proj
  -> attention TP reduce + residual
  -> post-attention RMSNorm -> router (sigmoid, top-8) -> 1 shared + 8 routed experts' gate | up, SiLU
  -> expert down, route-weighted -> MoE TP reduce + residual
```

The kinds (`runner.py`) are the MoE layers that run the indexer (`index`) and those that reuse the previous layer's
top-k (`shared`), the dense layers 1-2 (`dense`: the same launch with the FP8 SwiGLU MLP in place of the MoE, as
expert-shaped slices), layer 0 from the step's token ids (`embed`: SGLang's embedding is not run) and the tail
(`head`: the final norm, the LM head and the logits' all-gather, whose logits SGLang's logits processor returns).

Stages hand values on as tagged pairs (a 32-bit value and the launch's epoch tag in one store) in a global scratch,
and the TP reduces push each rank's partial rows into every peer's symmetric buffer; every rank sums the partials in
rank order, so all ranks hold bit-identical hidden states and routing. There is no grid-wide barrier: dependencies
point to earlier stages only. Every wait has a deadline (half the watchdog timeout for a step's first launch, where
the ranks' hosts meet); a failed wait stops the launch on every rank with a GPU fault, so no token of the step is
returned.

A pad row (SGLang's padding slot 0, or a width's row past the step) writes no cache and attends no key.

## 3. Weights

`gfx950/weights.py` reads SGLang's loaded tensors in place where their layout fits, and copies the rest once at
startup; the copies are allocated before SGLang sizes the KV cache and logged with their sizes. A tensor in another
dtype, shape or layout refuses the server and names it.

### 3.1 GLM-5.3 (FP8 block-128)

- In place: the block-FP8 projections (`fused_qkv_a_proj_with_mqa`, `q_b_proj`, `o_proj`, the indexer's `wk` and
  `wq_b`), row-major or in AITER's (16, 16) shuffle, with their 128 x 128 scales; the routed and shared experts
  (`w13` / `w2`, the shared expert fused at slot 256) with their block scales; the embedding and the LM head.
- Copied: W_UK and W_UV packed from SGLang's per-tensor FP8 `w_kc` / `w_vc`, the router and the indexer's
  `weights_proj` packed for the MFMA tiles, the indexer's k LayerNorm in f32, one bf16 rotary table, and the dense
  layers' MLPs as expert-shaped slices.

### 3.2 GLM-5.2 (Quark MXFP4)

The spec takes a Quark config whose global weight and input quantization is MXFP4 (FP4, a group of 32, E8M0 scales)
with no per-layer config. In it the routed and shared experts of layers 3-77 are MXFP4; every other linear, and the
whole NextN layer (layer 78: its experts, `eh_proj`, W_UK / W_UV), is bf16.

- In place: the experts, as SGLang's Quark MoE keeps them for AITER's A4W4 path (`w13_weight` [257, 2I, H / 2]
  packed FP4, `w13_weight_scale` [257, 2I, H / 32] E8M0, and `w2` alike). They run on the FP4 MFMA with the rows
  quantized to MXFP8 (`Glm53Build.native_fp4`, set from the layer's own experts for the target and the NextN layer
  alike); the bf16-row dequant path stays in the kernel (`native_fp4=False`) for the kernel tests.
- Copied as block FP8, in the layout the FP8 checkpoint has in place: every bf16 projection (128 x 128 blocks, each
  block's amax / 448, AITER-shuffled), the dense layers' MLPs, W_UK / W_UV of the NextN layer (128 x 64 and
  128 x 128 blocks) and the NextN layer's bf16 experts (unshuffled from AITER's bf16 layout first when SGLang
  shuffled them). At TP8 that is 3.2 GiB a rank for the target (2.7 GiB of it the projections) and 1.2 GiB for the
  NextN layer (its experts).
- `eh_proj` is an unquantized `ReplicatedLinear` there (an `nn.Linear` in the FP8 checkpoint); the draft reads its
  bf16 rows in place either way.

So the attention, the indexer and the dense layers compute in block FP8 on both checkpoints, the MXFP4 experts in
MXFP8 x MXFP4. SGLang's own forward on the MXFP4 checkpoint runs those linears in bf16; check mode's bounds
(section 6) are written against that.

## 4. The NextN draft

EAGLE's NextN draft runs its own steps, which monokernel API 1 does not route, so `draft.py` puts a wrapper in front of
the draft model's forward at build. It routes by forward mode, batch size, rows and the presence of the reused top-k
alone, so a captured graph replays the decision made at capture; it logs each draft key at its first launch and
refuses, during capture, a draft-shaped step it cannot take. A routed step is two launches: the NextN layer from
the tokens and its entry `eh_proj(enorm(e) | hnorm(h))`, and the tail with the NextN norm and the shared LM head.
A draft extend runs the indexer kind and hands its top-k to SGLang as the seed; a draft decode step runs the shared
kind on the seed's valid keys. A sampled draft decode (rejection sampling on a batch that is not all greedy) runs
SGLang's path.

## 5. SGLang integration

- **Spec**: declares the steps, the device limits and `NUMERICS`, checks the config, the server arguments, the
  environment and the loaded model, and refuses unsupported setups at startup with every reason (e.g. another
  quantization, `--page-size` other than 64, MTP topk other than 1, a layer whose weights do not fit).
- **Build keys** (`runner.Glm53Build`): width, TP, kind, the KV dtype, the shuffle, the dense slices, `mxfp4` and
  `native_fp4`. One runtime per (width, kind) serves every layer of that kind; `Glm53LayerOp` refuses weights whose
  `mxfp4` or shuffle is not its runtime's. FlyDSL keeps the compiled launches in its disk cache.
- **CUDA graphs**: every captured decode or verify batch the widths take runs the launches; other batches and eager
  steps run SGLang's forward.
- **No SGLang changes**: the modules read only the SGLang names they declare in `SGLANG_READS`
  (`test/unit/test_contract.py` checks them against the installed SGLang).

## 6. Correctness checks

- **Unit tests** (CPU): `test/unit/test_glm53_spec.py` (registration, the config and checkpoint refusals, the Quark
  MXFP4 config, MXFP4 experts read only in AITER's A4W4 layout, the bf16 unshuffle and the block-FP8 copy, a bf16
  dense MLP copied) and `test/unit/test_glm53_draft.py` (the draft's refusals, its `eh_proj` and experts, the
  routing wrapper).
- **Kernel tests** (`test/kernels/test_glm53_kernels.py`, 8 gfx950 GPUs, no checkpoint: random weights in SGLang's
  layouts at TP4 and TP8, `glm53/check_layer.py`, `check_topk.py`, `check_nextn.py`): every launch kind writes
  exactly its live rows' cache rows on the bf16 and E4M3 caches; a graph of the model's launches replays the eager
  step bitwise; selected tokens carry their indexer launch's tag; the dense MLP and the tail's logits are within
  their bounds against f32 references; the indexer's top-k is exact at 8K to 300K keys; each NextN launch equals the
  plain launch on its own entry. The MXFP4 cases (`--mxfp4 --native-fp4`) rerun the writes, the replays and the
  tags with MXFP4 experts.
- **Check mode** (`SGLANG_AMD_MONO_CHECK=1`, eager): each covered layer also runs SGLang's layer on its input and
  caches, and the launch's stream, cache rows and top-k are compared against `NUMERICS`; the dense MLPs and the tail
  against f32 references; the draft's entry, seed and NextN layer against SGLang's. On an MXFP4 checkpoint the dense
  MLP's excess over an FP8 MLP (`dense.excess.rel`) is taken against SGLang's FP8 MLP on the launch's own copy,
  since SGLang runs that MLP in bf16; and SGLang's `eh_proj` there runs on AITER's bf16 GEMM, farther from f32 than
  the launch, so the draft entry is held to f32 (`draft.eh_vs_fp32.rel`) and only loosely to SGLang's.
- **E2E** (`test/e2e/test_glm53_server.py`, `GLM_53_MODEL_PATH` set): GSM8K and decode-against-prefill KL, MTP
  accept length, check mode (decode and verify steps, the FP8 KV cache, rows past the parallel top-k), returned
  hidden states, and the guarded-wait fault path.
- **Timing**: `torchrun --nproc-per-node 8 test/kernels/glm53/time_layers.py --kind model --rows 2,4,8`
  (`--mxfp4 --native-fp4` for GLM-5.2's experts; `--kind full` for the whole served step).

## 7. Limitations

- gfx950 only; TP 4 or 8; up to 8 rows a step (2 requests with MTP 3)
- the FP8 block-128 checkpoint or the Quark MXFP4 checkpoint described above
- MTP: one draft chain (topk 1); the draft follows the target's widths
- on the MXFP4 checkpoint the attention, indexer and dense projections run on block-FP8 copies of the bf16 weights,
  not in bf16 as SGLang's forward does, and the copies take 3.2 GiB (target) and 1.2 GiB (NextN) a rank at TP8
