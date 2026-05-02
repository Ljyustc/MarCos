"""
Single-stage training (without randomness factor) for the MarCos model.

Encoder / Thinker / Decoder are all initialised directly from the same
pretrained backbone checkpoint and trained end-to-end on the prediction loss.
The auxiliary random predictor used in the with-randomness variant is absent.

All hyper-parameters are exposed as CLI flags; see run_*.sh for concrete
invocations against the Qwen and Llama backbones.
"""
import argparse
import logging
import math
import os
import time
from contextlib import nullcontext

import numpy as np
import torch
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer

from dataloader import CollateFn, ProblemAnswerDataset
from model import ModelM


DEFAULT_MODEL_PATH = {
    'qwen': 'Qwen/Qwen2.5-0.5B',
    'llama': 'meta-llama/Llama-3.2-1B-Instruct',
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Backbone
    p.add_argument('--backbone', choices=['qwen', 'llama'], default='llama')
    p.add_argument('--model_path', default=None)
    # Init / resume
    p.add_argument('--init', choices=['pretrained', 'config', 'resume'], default='pretrained')
    p.add_argument('--resume_ckpt', default=None)
    # Architecture
    p.add_argument('--phase', choices=['1', '2'], default='1',
                   help='Phase tag for checkpoint naming and loss switching; the no-randomness '
                        'variant typically only uses phase 1.')
    p.add_argument('--step', type=int, default=3)
    p.add_argument('--num_iterations', type=int, default=3)
    p.add_argument('--neuron_dim_t', type=int, default=0)
    p.add_argument('--neuron_dim_s', type=int, default=4)
    p.add_argument('--neuron_dim_r', type=int, default=0)
    p.add_argument('--random_dim', type=int, default=0,
                   help='Latent dim of the random predictor; 0 disables it (default for no-randomness).')
    # Optimization
    p.add_argument('--max_iters', type=int, default=200)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--gradient_accumulation_steps', type=int, default=64)
    p.add_argument('--learning_rate', type=float, default=1e-6)
    p.add_argument('--min_lr', type=float, default=1e-7)
    p.add_argument('--warmup_iters', type=int, default=2)
    p.add_argument('--lr_decay_iters', type=int, default=-1)
    p.add_argument('--decay_lr', action='store_true', default=True)
    p.add_argument('--no_decay_lr', dest='decay_lr', action='store_false')
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--beta1', type=float, default=0.9)
    p.add_argument('--beta2', type=float, default=0.95)
    p.add_argument('--grad_clip', type=float, default=1.0)
    # Sparsity / DWA
    p.add_argument('--L1_weight', type=float, default=1e-4)
    p.add_argument('--L1_TARGET', type=float, default=1.0)
    p.add_argument('--dwa_update_freq', type=int, default=500)
    # Data
    p.add_argument('--train_data', required=True)
    p.add_argument('--val_data', required=True)
    p.add_argument('--target_append', action='store_true',
                   help='If set, concatenate previous-step answers in front of the current '
                        'step when building the target sequence (teacher forcing across '
                        'steps). Default off — each step is a stand-alone target, matching '
                        'the with-randomness recipe.')
    p.add_argument('--use_chat_template', action='store_true',
                   help='Wrap the problem with the tokenizer chat template (needed for '
                        'Instruct models such as Llama-3.2-1B-Instruct).')
    # I/O
    p.add_argument('--out_dir', default='out')
    p.add_argument('--task', default='gsm8k-stage')
    p.add_argument('--eval_interval', type=int, default=1)
    p.add_argument('--log_interval', type=int, default=1)
    p.add_argument('--always_save_checkpoint', action='store_true')
    p.add_argument('--eval_only', action='store_true')
    # W&B
    p.add_argument('--wandb_log', action='store_true')
    p.add_argument('--wandb_project', default='marcos-no-randomness')
    p.add_argument('--wandb_run_name', default=None)
    # System
    p.add_argument('--seed', type=int, default=1337)
    p.add_argument('--backend', default='nccl')
    p.add_argument('--compile', action='store_true')
    args = p.parse_args()
    if args.model_path is None:
        args.model_path = DEFAULT_MODEL_PATH[args.backbone]
    if args.lr_decay_iters < 0:
        args.lr_decay_iters = args.max_iters
    if args.init == 'resume' and not args.resume_ckpt:
        p.error('--init=resume requires --resume_ckpt')
    return args


def compute_loss(model, batch, step, device, weights=None):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    targets = [t.to(device) for t in batch['targets']]
    loss_mask = [t.to(device) for t in batch['loss_masks']]

    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    _, logits_all = model(input_ids, attention_mask=attention_mask, targets=targets)
    if weights is None:
        weights = np.ones(max(step, 1), dtype=np.float32)

    loss_predict = 0
    no_mask_counts, no_mask_step_counts = 0, 0
    for i in np.arange(0, step):
        logits, nar_logits = logits_all[i][0][0], logits_all[i][0][1]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_nar_logits = nar_logits[:, :-1, :].contiguous()
        shift_labels = targets[i][:, 1:]
        shift_mask = loss_mask[i][:, :-1].contiguous()
        loss = loss_fn(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
        loss_nar = loss_fn(shift_nar_logits.reshape(-1, shift_nar_logits.size(-1)), shift_labels.reshape(-1))
        if torch.isnan(loss).any():
            print("NaN detected in loss")

        skip_mask = shift_mask.sum(dim=1, keepdim=True) != 1
        loss = loss.view(shift_labels.size()) * shift_mask * skip_mask
        loss_nar = loss_nar.view(shift_labels.size()) * shift_mask * skip_mask
        no_mask_counts += (shift_mask * skip_mask).sum()
        no_mask_step_counts += skip_mask.sum()
        loss_predict += loss.sum() * weights[i]

    loss_nll = loss_predict / no_mask_counts.clamp(min=1)
    return loss_nll, no_mask_counts, 0


class DWAController:
    def __init__(self, target=5.0, update_freq=50, init_weight=1e-3, min_w=1e-6, max_w=1e-3):
        self.target = target
        self.weight = init_weight
        self.update_freq = update_freq
        self.loss_accumulator = 0.0
        self.steps = 0
        self.min_w = min_w
        self.max_w = max_w

    def step(self, batch_sparse_loss):
        self.loss_accumulator += batch_sparse_loss
        self.steps += 1
        if self.steps < self.update_freq:
            return self.weight
        avg_loss = self.loss_accumulator / self.steps
        self.weight *= 1.01 if avg_loss > self.target else 0.99
        self.weight = max(self.min_w, min(self.max_w, self.weight))
        self.loss_accumulator = 0.0
        self.steps = 0
        return self.weight

    def get_weight(self):
        return self.weight


def get_lr(it, args):
    if it < args.warmup_iters:
        return args.learning_rate * (it + 1) / (args.warmup_iters + 1)
    if it > args.lr_decay_iters:
        return args.min_lr
    decay_ratio = (it - args.warmup_iters) / (args.lr_decay_iters - args.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return args.min_lr + coeff * (args.learning_rate - args.min_lr)


def main():
    args = parse_args()

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False

    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        init_process_group(backend=args.backend)
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
        seed_offset = ddp_rank
        assert args.gradient_accumulation_steps % ddp_world_size == 0
        args.gradient_accumulation_steps //= ddp_world_size
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        master_process = True
        seed_offset = 0
        ddp_world_size = 1

    if master_process:
        os.makedirs(args.out_dir, exist_ok=True)

    ckpt_prefix = (f"ckpt_{time.strftime('%Y%m%d_%H%M%S')}_ddp_{args.task}_{args.backbone}"
                   f"_phase{args.phase}_step{args.step}_dwaT{args.L1_TARGET}")
    ckpt_path = os.path.join(args.out_dir, ckpt_prefix + '.pt')
    ckpt_path_final = os.path.join(args.out_dir, ckpt_prefix + '_final.pt')

    if master_process:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler(os.path.join(args.out_dir, ckpt_prefix + '.log')),
                      logging.StreamHandler()],
        )
        logging.info("config: %s", vars(args))

    torch.manual_seed(args.seed + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device_type = 'cuda' if 'cuda' in device else 'cpu'
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    if args.init == 'resume':
        print(f"Resuming from {args.resume_ckpt}, phase={args.phase}")
        checkpoint = torch.load(args.resume_ckpt, map_location=device)
        enc, thk, dec = (
            checkpoint['encoder_name'].replace('resume', 'config'),
            checkpoint['think_name'].replace('resume', 'config'),
            checkpoint['decoder_name'].replace('resume', 'config'),
        )
        model = ModelM(tokenizer, model_path=args.model_path, init_from=(enc, thk, dec),
                       backbone=args.backbone, neuron_dim_t=args.neuron_dim_t,
                       neuron_dim_s=args.neuron_dim_s, neuron_dim_r=args.neuron_dim_r,
                       num_iterations=args.num_iterations, random_dim=args.random_dim,
                       phase=args.phase)
        model.load_state_dict(checkpoint['model'], strict=False)
    elif args.init == 'config':
        model = ModelM(tokenizer, model_path=args.model_path,
                       init_from=('config', 'config', 'config'),
                       backbone=args.backbone, neuron_dim_t=args.neuron_dim_t,
                       neuron_dim_s=args.neuron_dim_s, neuron_dim_r=args.neuron_dim_r,
                       num_iterations=args.num_iterations, random_dim=args.random_dim,
                       phase=args.phase)
        checkpoint = None
    else:
        model = ModelM(tokenizer, model_path=args.model_path,
                       init_from=(args.model_path, args.model_path, args.model_path),
                       backbone=args.backbone, neuron_dim_t=args.neuron_dim_t,
                       neuron_dim_s=args.neuron_dim_s, neuron_dim_r=args.neuron_dim_r,
                       num_iterations=args.num_iterations, random_dim=args.random_dim,
                       phase=args.phase)
        checkpoint = None

    model.to(device)

    scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16'))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    checkpoint = None

    if args.compile:
        model = torch.compile(model)
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=True)

    dwa_controller = DWAController(
        target=args.L1_TARGET, update_freq=args.dwa_update_freq,
        init_weight=args.L1_weight, min_w=1e-6, max_w=1e-3,
    )

    if args.wandb_log and master_process:
        import wandb
        run_name = args.wandb_run_name or f"{args.backbone}-{args.task}-phase{args.phase}-{int(time.time())}"
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

    train_dataset = ProblemAnswerDataset(args.train_data, tokenizer,
                                         num_splits=args.num_iterations,
                                         use_chat_template=args.use_chat_template)
    val_dataset = ProblemAnswerDataset(args.val_data, tokenizer,
                                       num_splits=args.num_iterations,
                                       use_chat_template=args.use_chat_template)

    collate = CollateFn(tokenizer.eos_token_id, target_append=args.target_append)
    if ddp:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler,
                                  collate_fn=collate, num_workers=4, pin_memory=True)
    else:
        train_sampler = None
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  collate_fn=collate, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            collate_fn=collate, num_workers=4, pin_memory=True)

    best_val_loss = 1e9
    t0 = time.time()
    global_step = 0
    for iter_num in range(args.max_iters):
        if ddp:
            train_sampler.set_epoch(iter_num)
        lr = get_lr(iter_num, args) if args.decay_lr else args.learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        if iter_num % args.eval_interval == 0 and master_process:
            model.eval()
            l1_weight = dwa_controller.get_weight()
            with torch.no_grad(), ctx:
                total_nll, total_tokens, total_sparsity = 0, 0, 0
                for batch in val_loader:
                    nll, tokens, spa = compute_loss(model, batch, args.step, device)
                    total_nll += nll
                    total_tokens += tokens
                    total_sparsity += spa
            val_loss = total_nll / max(len(val_loader), 1)
            spa_loss = total_sparsity / max(len(val_loader), 1)
            print(f"step {iter_num}: val loss {val_loss:.4f}, spar loss {spa_loss}")
            if args.wandb_log:
                import wandb
                wandb.log({"iter": iter_num, "val/loss": float(val_loss)})
            if val_loss < best_val_loss or args.always_save_checkpoint:
                best_val_loss = val_loss
                ckpt = {
                    'encoder_name': args.model_path if args.init == 'pretrained' else 'config',
                    'think_name': args.model_path if args.init == 'pretrained' else 'config',
                    'decoder_name': args.model_path if args.init == 'pretrained' else 'config',
                    'optimizer': optimizer.state_dict(),
                    'iter_num': iter_num,
                    'best_val_loss': float(best_val_loss),
                    'config': vars(args),
                    'backbone': args.backbone,
                    'model': (model.module if ddp else model).state_dict(),
                }
                torch.save(ckpt, ckpt_path)
                torch.save(ckpt, ckpt_path_final)
                print(f"saved checkpoint to {ckpt_path}")

        if args.eval_only:
            break

        model.train()
        micro_step = 0
        train_loss, train_count, spar_train_loss = 0, 0, 0
        for batch in train_loader:
            l1_weight = dwa_controller.get_weight()
            if micro_step == 0:
                optimizer.zero_grad(set_to_none=True)
            if ddp:
                model.require_backward_grad_sync = (micro_step == args.gradient_accumulation_steps - 1)
            with ctx:
                loss, tokens, spa_loss = compute_loss(model, batch, args.step, device)
            scaler.scale(loss).backward()
            train_loss += loss.item()
            train_count += tokens
            batch_spa = spa_loss if isinstance(spa_loss, int) else spa_loss.item()
            spar_train_loss += batch_spa
            dwa_controller.step(batch_spa)
            micro_step += 1
            global_step += 1
            if master_process and args.wandb_log:
                import wandb
                wandb.log({"batch/loss": loss.item(), "batch/l1_weight": l1_weight}, step=global_step)
            if micro_step == args.gradient_accumulation_steps:
                if args.grad_clip != 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                micro_step = 0

        t1 = time.time()
        dt, t0 = t1 - t0, t1
        if iter_num % args.log_interval == 0 and master_process:
            denom = max(len(train_loader), 1)
            print(f"iter {iter_num}: loss {train_loss/denom:.4f}, time {dt:.3f}s")

    if ddp:
        destroy_process_group()


if __name__ == '__main__':
    main()
