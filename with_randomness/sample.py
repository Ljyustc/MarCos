"""
Sample / evaluate from a trained MarCos checkpoint (with-randomness variant).

All hyper-parameters are CLI flags; see sample.sh for a concrete invocation.
"""
import argparse
import json
import os
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from dataloader import CollateFn, ProblemAnswerDataset
from model import ModelM


DEFAULT_MODEL_PATH = {
    'qwen': 'Qwen/Qwen2.5-0.5B',
    'llama': 'meta-llama/Llama-3.2-1B-Instruct',
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--backbone', choices=['qwen', 'llama'], default='qwen')
    p.add_argument('--model_path', default=None)
    p.add_argument('--checkpoint', required=True, help='Path to a .pt checkpoint produced by train.py')
    p.add_argument('--input_file', required=True, help='JSONL file of {"problem": ..., "answer": ...} examples')
    p.add_argument('--output_dir', default='test_results')
    p.add_argument('--output_file', default=None, help='Override the auto-generated output file name')
    p.add_argument('--num_iterations', type=int, default=3)
    p.add_argument('--neuron_dim_t', type=int, default=1)
    p.add_argument('--neuron_dim_s', type=int, default=100)
    p.add_argument('--neuron_dim_r', type=int, default=16)
    p.add_argument('--random_dim', type=int, default=-1)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--max_new_tokens', type=int, default=100)
    p.add_argument('--temperature', type=float, default=0.0)
    p.add_argument('--top_k', type=int, default=200)
    p.add_argument('--seed', type=int, default=1337)
    p.add_argument('--backend', default='nccl')
    p.add_argument('--target_append', action='store_true',
                   help='Append ground-truth previous-step answers when decoding (teacher forcing)')
    args = p.parse_args()
    if args.model_path is None:
        args.model_path = DEFAULT_MODEL_PATH[args.backbone]
    return args


def main():
    args = parse_args()

    rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    device = f'cuda:{rank}' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    if args.random_dim < 0:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(args.model_path)
        args.random_dim = args.neuron_dim_r * cfg.hidden_size

    print(f"loading checkpoint {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    enc = checkpoint['encoder_name'].replace('resume', 'config')
    thk = checkpoint['think_name'].replace('resume', 'config')
    dec = checkpoint['decoder_name'].replace('resume', 'config')
    model = ModelM(tokenizer, model_path=args.model_path, init_from=(enc, thk, dec),
                   backbone=args.backbone, neuron_dim_t=args.neuron_dim_t,
                   neuron_dim_s=args.neuron_dim_s, neuron_dim_r=args.neuron_dim_r,
                   num_iterations=args.num_iterations, random_dim=args.random_dim)
    model.load_state_dict(checkpoint['model'], strict=False)
    if 'best_val_loss' in checkpoint:
        print(f"checkpoint best_val_loss: {checkpoint['best_val_loss']}")
    model.to(device)
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)
    if args.output_file is None:
        ckpt_stem = os.path.splitext(os.path.basename(args.checkpoint))[0]
        args.output_file = os.path.join(args.output_dir, f'samples_{ckpt_stem}.json')

    val_dataset = ProblemAnswerDataset(args.input_file, tokenizer, num_splits=args.num_iterations)
    chunk_size = int(np.ceil(len(val_dataset) / world_size))
    start = rank * chunk_size
    end = (rank + 1) * chunk_size if rank < world_size - 1 else len(val_dataset)
    sub_dataset = torch.utils.data.Subset(val_dataset, list(range(start, end)))
    val_loader = DataLoader(sub_dataset, batch_size=args.batch_size,
                            collate_fn=CollateFn(tokenizer.eos_token_id, target_append=args.target_append),
                            num_workers=4, pin_memory=True)

    eos_str = tokenizer.decode([tokenizer.eos_token_id])
    outputs = []
    t0 = time.time()
    with torch.no_grad(), ctx:
        for batch in tqdm(val_loader, desc=f"Rank {rank}", unit="batch", ncols=80):
            x = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = [t.to(device) for t in batch['targets']]
            outputs_batch, _ = model.generate_with_answer(
                input_ids=x, attention_mask=attention_mask, targets=targets,
                max_length=args.max_new_tokens, temperature=args.temperature,
                top_k=args.top_k, target_append=args.target_append,
            )
            bsz = x.size(0)
            output_i = [[] for _ in range(bsz)]
            for i, step_tensor in enumerate(outputs_batch):
                for idx, output in enumerate(step_tensor):
                    text = tokenizer.decode(output).split(eos_str)[0]
                    output_i[idx].append(text)
            outputs += output_i
    print(f"rank {rank}: {time.time() - t0:.2f}s")

    if world_size > 1:
        if not dist.is_initialized():
            dist.init_process_group(backend=args.backend)
        gathered = [None] * world_size
        dist.all_gather_object(gathered, outputs)
        all_outputs = [o for chunk in gathered for o in chunk if o] if rank == 0 else []
    else:
        all_outputs = outputs

    if rank == 0:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            for d in all_outputs:
                f.write(json.dumps(d, ensure_ascii=False) + '\n')
        print(f"saved samples to {args.output_file}")

    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
