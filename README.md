---
license: apache-2.0
pretty_name: Decision Index runs of Jev-Style models
tags: [decision-index, jev-style]
---

# Decision Index runs of Jev-Style models

Self-scored, complete runs of the [Decision Index](https://github.com/apolinario/decision-index) public suites,
scored with the kit's own scorer. Every text run's `scores.json` and every vision run's `vision-scores.json` says
`"complete": true`. `results.jsonl.gz` is the kit's compact format (no payload, image or raw output), so the files
carry no benchmark text (GPQA and HLE ask that their questions not be redistributed).

## Jev-Style 2B v4: Decision Index 0.3 public suite (text) and the public Vision suite

| Run | Public index | Raw index | Knowledge | Language | Retrieval | Tools | Arts | Requests | Scores |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| [Jev-Style-2B-Decision-v4](https://huggingface.co/chaoliangUNSW/Jev-Style-2B-Decision-v4) @ `3f2c063f`, text | **44.37** | 57.68 | 23.6 | 42.5 | 65.4 | 61.8 | 29.0 | 140,178 scoreable, all answered (0 unsupported) | [scores.json](runs/jev-style-2b-decision-v4/scores.json) |

| Run | Public vision index | Rows | Scores |
|---|---:|---|---|
| [Jev-Style-2B-Decision-v4](https://huggingface.co/chaoliangUNSW/Jev-Style-2B-Decision-v4) @ `3f2c063f`, vision | **57.55** | 12,210, all answered | [vision-scores.json](runs/jev-style-2b-decision-v4-vision/vision-scores.json) |

- Engines: `macjev.eval.di_v4:V4TextEngine` and `macjev.eval.di_v4:V4VisionEngine`, shipped in the model repo under
  `src/` (the scorer the model was trained, calibrated and evaluated with). Kit `25c6ec7` (edition 0.3), its runners
  and scorers unmodified. bf16 autocast, fp32 readout, one calibrated temperature per question type
  (`readout_config.json`), no text generated, nothing truncated. Text: fast path on (each request's state computed
  once and cached, its questions batched). Vision: every image at native resolution up to 1.6 MP.
- Hardware: one NVIDIA A100-SXM4-80GB (Google Colab). The text run is 5 parallel shards (`sha256(group_id) mod 5`);
  the vision run is the `vision pipeline` run plus the rows not yet answered split by `sha256(row id) mod N`; the
  processes shared the GPU through NVIDIA MPS, so the per-request times in the results include sharing. Every
  `run_id` appears exactly once; `environment.json` lists every part.
- Latency without sharing (one process, one request at a time, 750 text requests sampled uniformly from the suite
  files, the model repo's code only, bf16): with CUDA graphs (`--option cuda_graphs=true`) median 16.4 ms, mean
  58.8 ms, p80 60.9 ms, p95 211 ms; with the fast path alone (the setting of the full run) 71.5 / 104.0 / 80.5 /
  214 ms. 663 of the 750 requests are in the run: with the fast path their answers matched the full run on all 1,585
  of their questions, with CUDA graphs 1,579 of the 1,585 have the same top option.
- Suites rebuilt from the pinned sources and verified. Text: `suite rebuild` on `25c6ec7` no longer reproduces the
  pinned `release-v1` hash (PR #165 changed the SGD rows without updating the pins), so the SGD builder line from
  before #165 was patched in-process (kit files unmodified); all four pins then matched. Vision: rebuilt with
  Pillow 12.3.0 (the kit's documented version) and verified.
- Model identity: the engine loaded the released weights. The run label `v4-main-colab-seed1` in `response.model`
  and `environment.json` is the label written in the model repository's `macjev_readout.json`; `weights_digest`
  `d03c41b67e4205b3` is the sha256 over the `SHA256SUMS.json` entries of the three weight shards, the same for
  revision `3f2c063f`.
- These runs used the main repository's PyTorch runtime. The text-only
  [GGUF](https://huggingface.co/chaoliangUNSW/Jev-Style-2B-Decision-v4-GGUF) and
  [MLX](https://huggingface.co/chaoliangUNSW/Jev-Style-2B-Decision-v4-MLX) builds were not run on the index; their
  agreement with this runtime is on their cards.
- Several benchmarks' official train splits are in v4's training data (listed on the model card, including CORD of
  the Vision KIE benchmark), so those results are not zero-shot.

## Jev-Style v3 (0.8B and 2B): Decision Index 0.2.1

Self-scored runs of the Decision Index 0.2.1 suite (kit `87d4650`, the
rebuilt suite is byte-identical to the lab's: rows `b2b56d6f…`, added rows `7429f3c9…`), made at the request of the board
maintainers ([Space discussions #23 and #32](https://huggingface.co/spaces/multimodalart/jev-decision-index/discussions)).

| Model | Decision Index 0.2.1 | Raw index | Knowledge | Language | Retrieval | Tools | Arts | Scored requests | Median latency | Scores |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---|
| [Jev-Style-0.8B-Decision-v3](https://huggingface.co/chaoliangUNSW/Jev-Style-0.8B-Decision-v3) @ `633729d8` | **16.64** | 35.98 | 5.6 | 10.6 | 26.2 | 35.4 | 7.3 | 150,317 (84 unsupported) | 61 ms | [scores.json](runs/jev-style-0.8b-decision-v3/scores.json) |
| [Jev-Style-2B-Decision-v3](https://huggingface.co/chaoliangUNSW/Jev-Style-2B-Decision-v3) @ `fbb9d676` | **34.83** | 50.14 | 22.3 | 29.0 | 51.6 | 52.8 | 15.9 | 150,317 (0 unsupported) | 136 ms | [scores.json](runs/jev-style-2b-decision-v3/scores.json) |

Area columns are the chance-corrected area skills × 100. Latencies are in-process request wall times on the hardware below
(not the maintainers' RTX PRO 6000 measurement).

### How the v3 runs were made

- Engine: [`engine/jev_engine_torch.py`](engine/jev_engine_torch.py), a kit `Engine` over each model repo's own PyTorch runtime
  (`jev_style_decision.py`) at the pinned revision above, manifest-verified. float32 on CUDA; transformers 5.17.0,
  torch 2.11.0+cu128, flash-linear-attention 0.5.2 for the Gated DeltaNet layers (checked against the pure PyTorch
  reference on a 60-request sample: same top answer on 805/805 (2B) and 698/699 (0.8B) questions, max |Δp| ≤ 0.0025).
- One fixed rendering and one global temperature for every benchmark; no prompt tuning, no option filtering, nothing
  truncated. Requests over a declared limit raise the kit's `Unsupported` and count as unanswered:
  - 0.8B: 25,600 tokens in total, and question + options + readout within 2,048 tokens; a choice question whose options
    do not fit is scored in option chunks with one softmax over all options, so only questions whose text alone exceeds
    the budget are unsupported.
  - 2B: 25,600 tokens in total; no separate question/options budget (numbered-option catalogue layout).
- Run with `python -m decision_index run --engine jev_engine_torch:JevStyleTorchEngine --option model=<0.8b|2b>
  --option revision=<rev> --edition 0.2.1 --compact`. Colab reclaimed the VMs several times, so each run was resumed
  from its own `results.jsonl`, and the last part was split into contiguous row shards run in parallel and merged by
  `run_id` (every scored request appears exactly once; no `error` rows). `environment.json` lists every part.
- `jev-style-0.8b-decision-v3`: 92,613 requests on an NVIDIA A100 (Colab), 58,146 on NVIDIA L4 (Colab).
- `jev-style-2b-decision-v3`: 86,890 requests on an NVIDIA A100 (Colab), 63,869 on NVIDIA L4 (Colab).
- The 2B's `environment.json` shows `temperature: global (category=None) = None`: the engine's provenance line reads a
  0.8B-style attribute. The 2B runtime applies its one calibrated global temperature from `readout_config.json`
  (T = 0.8278650621) to every question; the engine file is published exactly as it was run.

Models are Apache-2.0. Jev-Style 2B v4 was trained in part on soft labels distilled from TypeSafe Jev 1.14 via its
public API (see its model card); the v3 models used no Jev outputs. Not affiliated with, endorsed by or connected to
TypeSafe AI or Jev.
