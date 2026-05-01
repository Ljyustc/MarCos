# Deep Thinking by Markov Chain of Continuous Thoughts (MarCos)

This repository contains the official implementation of *Deep Thinking by
Markov Chain of Continuous Thoughts*. The method augments a pretrained LLM with
an iterative "thinking" module that operates on a learnable bank of continuous
neuron embeddings, producing a chain of latent thoughts before decoding the
final answer.

Two complementary variants are provided. Each lives in its own folder, and
**each variant supports both the Qwen2.5 and Llama-3.2 backbones via a
`--backbone` flag** — there are no separate per-backbone subfolders.

| Variant | Folder | Training scheme |
| --- | --- | --- |
| **With randomness factor** | [`with_randomness/`](with_randomness/) | Two-phase: Phase 1 trains the encoder/thinker/decoder/projection MLPs; Phase 2 freezes them and trains a stochastic latent predictor (`pred_mlp1`). |
| **Without randomness** | [`without_randomness/`](without_randomness/) | Single-stage: encoder/thinker/decoder are initialised directly from the pretrained backbone and fine-tuned end-to-end on the prediction loss. |

All hyper-parameters are CLI flags. Concrete configurations live in the
`run_*.sh` / `sample.sh` shell scripts in each folder — you should not need to
edit any `.py` file to run a standard experiment.

---

## Repository layout

```
MarCos/
├── README.md
├── requirements.txt
├── .gitignore
├── with_randomness/                # variant 1: phase-1 + phase-2 training with random factor
│   ├── train.py                    # all hyper-params via argparse
│   ├── sample.py
│   ├── model.py                    # ModelM(..., backbone={'qwen','llama'})
│   ├── dataloader.py
│   ├── custom_qwen2_lambda.py      # Qwen2 with patched 4D causal mask
│   ├── custom_llama_lambda.py      # Llama with patched 4D causal mask
│   ├── run_qwen_phase1.sh          # phase-1 driver, Qwen backbone
│   ├── run_qwen_phase2.sh          # phase-2 driver, Qwen backbone
│   ├── run_llama_phase1.sh         # phase-1 driver, Llama backbone
│   ├── run_llama_phase2.sh         # phase-2 driver, Llama backbone
│   └── sample.sh                   # evaluation driver
└── without_randomness/             # variant 2: direct fine-tune from backbone
    ├── train.py
    ├── sample.py
    ├── model.py                    # ModelM(..., backbone={'qwen','llama'})
    ├── dataloader.py
    ├── custom_qwen2_lambda.py
    ├── custom_llama_lambda.py
    ├── run_qwen.sh                 # Qwen training driver
    ├── run_llama.sh                # Llama training driver
    └── sample.sh
```

---

## Environment

* GPU: tested on 8 × H200 (141 GB)
* Python ≥ 3.10
* CUDA-capable PyTorch 2.1+

```bash
pip install -r requirements.txt
```

The first run downloads the relevant backbone weights from the Hugging Face Hub
(`Qwen/Qwen2.5-0.5B`, `meta-llama/Llama-3.2-1B-Instruct`). For the Llama
backbone you need to accept the licence on its model page and run
`huggingface-cli login` first.

---

## Data format

All scripts expect a JSON-Lines file where each line is

```json
{"problem": "...", "answer": "step1\nstep2\nfinal answer"}
```

The dataloader splits `answer` into `--num_iterations` chunks (one per thinking
step) and constructs per-step loss masks. Pass your file paths via the
`TRAIN_DATA` and `VAL_DATA` environment variables when invoking the shell
scripts (or the `--train_data` / `--val_data` flags directly).

---

## Variant 1 — Two-phase training **with randomness factor**

The model contains an additional `pred_mlp1` head that predicts a per-step
distribution over a continuous latent. Training is split into two phases:

* **Phase 1** — train the encoder / thinker / decoder / projection MLPs
  jointly on prediction loss + sparsity regularisation. The random predictor
  is frozen.
* **Phase 2** — freeze everything from Phase 1 and only train the random
  predictor against the latents collected in Phase 1.

```bash
cd with_randomness

# Phase 1, Qwen backbone, 4 GPUs:
NPROC=4 TRAIN_DATA=path/to/train.json VAL_DATA=path/to/valid.json \
    bash run_qwen_phase1.sh

# Phase 2 (consumes the best checkpoint from Phase 1):
NPROC=4 TRAIN_DATA=path/to/train.json VAL_DATA=path/to/valid.json \
    bash run_qwen_phase2.sh out_qwen/<phase1_checkpoint>.pt

# Same flow with Llama:
bash run_llama_phase1.sh
bash run_llama_phase2.sh out_llama/<phase1_checkpoint>.pt

# Sampling / evaluation:
BACKBONE=qwen bash sample.sh out_qwen/<phase2_checkpoint>.pt path/to/test.jsonl
```

Each shell script is a thin wrapper around `torchrun ... train.py`: any
additional flags (e.g. `--max_iters 500`, `--wandb_log`) you pass after the
required positional arguments are forwarded verbatim.

---

## Variant 2 — Direct fine-tune **without randomness**

This variant skips the random predictor and the Phase-2 stage. Encoder /
thinker / decoder are all initialised from the same pretrained backbone and
trained end-to-end on the prediction loss.

```bash
cd without_randomness

# Qwen backbone:
NPROC=4 TRAIN_DATA=path/to/train.json VAL_DATA=path/to/valid.json \
    bash run_qwen.sh

# Llama backbone:
bash run_llama.sh

# Sampling:
BACKBONE=llama bash sample.sh out_llama_no_random/<ckpt>.pt path/to/test.jsonl
```

---

## Available CLI flags (both variants)

The full list lives in `train.py`'s `parse_args()`. The most useful are:

| Flag | Default | Notes |
| --- | --- | --- |
| `--backbone` | `qwen` (with-randomness) / `llama` (without-randomness) | Picks `MyQwen2Model` or `MyLlamaModel` for the thinker, plus the matching default `--model_path`. |
| `--model_path` | derived from `--backbone` | HF id or local path. Override to point at a local snapshot or a different size. |
| `--phase` | `1` | `1` = main modules, `2` = random predictor (with-randomness only). |
| `--init` | `pretrained` | `pretrained` / `config` / `resume`. |
| `--resume_ckpt` | — | Required when `--init=resume`. |
| `--step` | `3` | Number of thinking iterations supervised. |
| `--neuron_dim_{t,s,r}` | see scripts | Sizes of the *thinker / short-term memory / random* slots in the learnable neuron matrix. |
| `--max_iters` | `200` | Total training iterations. |
| `--batch_size` × `--gradient_accumulation_steps` × `world_size` | — | Effective batch size. |
| `--learning_rate`, `--min_lr`, `--warmup_iters`, `--lr_decay_iters` | — | Cosine schedule with linear warm-up. |
| `--L1_weight`, `--L1_TARGET` | `1e-4`, `10.0` / `1.0` | Initial sparsity weight and target sparsity loss for the DWA controller. |
| `--train_data`, `--val_data` | required | JSONL paths. |
| `--out_dir` | `out` | Where checkpoints / logs go. |
| `--wandb_log` | off | Enables W&B logging; set `WANDB_API_KEY` and optionally `--wandb_project`. |

Run `python train.py --help` (in either folder) for the complete list.

---

## Outputs

* Checkpoints — written under `--out_dir` as
  `ckpt_<timestamp>_ddp_<task>_<backbone>_phase<n>_step<k>...pt`. Two files
  are saved each eval interval: the best-so-far and a `_final.pt` snapshot.
* Sampling outputs — under `<output_dir>/samples_<ckpt-stem>.json`.
* Training logs — `*.log` next to the checkpoint; W&B (set `WANDB_API_KEY`).

---

## Troubleshooting

* **Llama gated repo / permission error** — accept the licence on
  `huggingface.co/meta-llama/Llama-3.2-1B-Instruct` and run
  `huggingface-cli login`. If you have an offline copy, point `--model_path` at
  it directly.
* **Cross-variant checkpoints** — the with-randomness variant has additional
  parameters (`pred_mlp1`, `projection_mlp{1,2}`) that don't exist in the
  no-randomness variant. Loading is done with `strict=False` for the resume
  path, but you should not expect a no-randomness checkpoint to fully populate
  a with-randomness model (or vice versa).
* **OOM** — drop `--batch_size` and bump `--gradient_accumulation_steps`
  proportionally to keep the effective batch size constant.

---

## Citation

```bibtex
@inproceedings{marcos2025,
  title  = {Deep Thinking by Markov Chain of Continuous Thoughts},
  author = {Anonymous},
  year   = {2025}
}
```
