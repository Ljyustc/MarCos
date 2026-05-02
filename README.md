# Deep Thinking by Markov Chain of Continuous Thoughts (MarCos)

This repository contains the official implementation of [*Deep Thinking by
Markov Chain of Continuous Thoughts*](./paper.pdf). The method designs an iterative "thinking" layer that operates on a learnable bank of continuous neuron embeddings, producing a chain of latent thoughts before decoding the final answer.

Two complementary variants are provided. Each lives in its own folder, and
**each variant supports both the Qwen2.5 and Llama-3.2 backbones via a
`--backbone` flag**.

| Variant | Folder | Training scheme |
| --- | --- | --- |
| **With randomness factor** | [`with_randomness/`](with_randomness/) | Two-phase: Phase 1 trains the encoder/thinker/decoder/projection MLPs from scratch; Phase 2 freezes them and trains a stochastic latent predictor (`pred_mlp1`). |
| **Without randomness** | [`without_randomness/`](without_randomness/) | Single-stage: encoder/thinker/decoder are initialised directly from the pretrained backbone and fine-tuned end-to-end on the prediction loss. |


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

---

## Data format

All scripts expect a JSON-Lines file where each line is

```json
{"question": "...", "answer": "step1\nstep2\nfinal answer"}
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

# Edit the variables at the top of run_qwen_phase1.sh
# (NPROC, TRAIN_DATA, VAL_DATA, OUT_DIR), then:
bash run_qwen_phase1.sh

# After Phase 1 finishes, edit RESUME_CKPT in run_qwen_phase2.sh to point at
# the checkpoint you want to resume from, then:
bash run_qwen_phase2.sh

# Same flow for the Llama backbone:
bash run_llama_phase1.sh
bash run_llama_phase2.sh

# Sampling / evaluation: edit CKPT, INPUT, BACKBONE in sample.sh, then:
bash sample.sh
```

---

## Variant 2 — Direct fine-tune **without randomness**

This variant skips the random predictor and the Phase-2 stage. Encoder /
thinker / decoder are all initialised from the same pretrained backbone and
trained end-to-end on the prediction loss.

```bash
cd without_randomness

# Edit run_qwen.sh / run_llama.sh (NPROC, TRAIN_DATA, VAL_DATA, OUT_DIR), then:
bash run_qwen.sh
# or
bash run_llama.sh

# Sampling: edit CKPT, INPUT, BACKBONE in sample.sh, then:
bash sample.sh
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
| `--neuron_dim_{t,s,r}` | see scripts | Numbers of the *deep neurons / shallow neurons / random variables*. |
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

## Citation

```bibtex
@article{liu2025marcos,
  title={Deep thinking by markov chain of continuous thoughts},
  author={Liu, Jiayu and Huang, Zhenya and Yang, Xuan and Ji, Tianyun and Sims, Anya and Xu, Hao and Chen, Enhong and Teh, Yee Whye and Miao, Ning},
  journal={arXiv preprint arXiv:2509.25020},
  year={2025}
}
```
