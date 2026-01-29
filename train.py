"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import numpy as np
from contextlib import nullcontext

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from model import ModelM
from dataloader import ProblemAnswerDataset, CollateFn 

# ----------------------------------------------------------------------------- 
# default config values designed to train a qwen2
# I/O
phase = '1' # '1' for training main modules, '2' for training random predictor
step = 3 # train which step
if phase == '1':
    if step == 1: # only train the first step
        init_from = ('Qwen/Qwen2.5-0.5B', 'Qwen/Qwen2.5-0.5B', 'Qwen/Qwen2.5-0.5B') # 'scratch' or 'resume' or 'qwen2' for encoder/think_model/decoder
    else: # train all steps
        init_from = ('config', 'config', 'config') 
    max_iters = 10 # total number of training iterations
else:
    init_from = ('resume', 'resume', 'resume') # 'resume' for encoder/think_model/decoder
    max_iters = 10 # total number of training iterations

task = 'gsm-stage-'+phase 
if 'stage-1' in task:
    learning_rate = 1e-4 # max learning rate
    min_lr = 1e-4 # minimum learning rate
    batch_size = 16 # if gradient_accumulation_steps > 1, this is the micro-batch size
    warmup_iters = 0 # how many steps to warm up for
    lr_decay_iters = -1
    gradient_accumulation_steps = 16 # used to simulate larger batch sizes
else:
    learning_rate = 1e-5
    min_lr = 1e-6
    batch_size = 64
    warmup_iters = 2
    lr_decay_iters = max_iters 
    gradient_accumulation_steps = 4

L1_weight = 1e-4
L1_TARGET = 10.0
out_dir = 'out'
eval_interval = 1 
log_interval = 1
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = False # if True, always save a checkpoint after each eval
num_iterations = 3
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'Qwen' + str(time.time()) # 'run' + str(time.time())
# data
neuron_dim_t = 10
neuron_dim_s = 100
neuron_dim_r = 16
random_dim = neuron_dim_r * 896
# adamw optimizer
weight_decay = 0.01
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = False # use PyTorch 2.0 to compile the model to be faster
# ----------------------------------------------------------------------------- 
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
config = {k: globals()[k] for k in config_keys} # will be useful for logging
checkpoint_name_pre = f"ckpt_{time.strftime('%Y%m%d_%H%M%S')}_ddp_" + task + "_qwen2_phase" + phase + "_step" + str(step)
checkpoint_name = checkpoint_name_pre + ".pt"
checkpoint_name_final = checkpoint_name_pre + "_final.pt"
# ----------------------------------------------------------------------------- 
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

def compute_loss(model, batch, phase="1", L1_weight=L1_weight):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    targets = [t.to(device) for t in batch['targets']]
    loss_mask = [t.to(device) for t in batch['loss_masks']]

    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    _, logits_all, _ = model(input_ids, attention_mask=attention_mask, targets=targets)
    loss_predict, loss_sparsity = 0, 0
    no_mask_counts = 0
    if phase == '1':
        for i in np.arange(0, step):
            logits, nar_logits = logits_all[i][1][0], logits_all[i][1][1]

            # prediction loss & l2 loss
            shift_logits = logits[:, :-1, :].contiguous()
            shift_nar_logits = nar_logits[:, :-1, :].contiguous()
            shift_labels = targets[i][:, 1:]
            shift_mask = loss_mask[i][:, :-1].contiguous()
            loss = loss_fn(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
            loss_nar = loss_fn(shift_nar_logits.reshape(-1, shift_nar_logits.size(-1)), shift_labels.reshape(-1))
            if torch.isnan(loss).any():
                print(f"NaN detected in loss")

            # If shift_mask only contains one token, it's the eos_token and we don't need to train
            skip_mask = shift_mask.sum(dim=1, keepdim=True) != 1  # (batch,1)

            loss = loss.view(shift_labels.size()) * shift_mask * skip_mask
            loss_nar = loss_nar.view(shift_labels.size()) * shift_mask * skip_mask

            # loss_predict += (loss.sum() + loss_nar.sum())/2
            loss_predict += loss.sum() 
            no_mask_counts += (shift_mask * skip_mask).sum()

            sparsity= logits_all[i][0]
            loss_predict += L1_weight * (sparsity * skip_mask).sum()
            loss_sparsity += (sparsity * skip_mask).sum()
        return loss_predict, no_mask_counts, loss_sparsity
    elif phase == '2':
        for i in range(len(logits_all)):
            random_prob = logits_all[i][0]
            loss_predict += random_prob
        return loss_predict, len(targets), 0


tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B')

if init_from[0] == 'resume':
    print(f"Resuming training from {out_dir}, phase = {phase}")
    ckpt_path = os.path.join(out_dir, 'phase1_saved.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    # print(checkpoint['best_val_loss'])
    encoder_name, think_name, decoder_name = checkpoint['encoder_name'], checkpoint['think_name'], checkpoint['decoder_name']
    model = ModelM(tokenizer, init_from=(encoder_name, think_name, decoder_name), neuron_dim_t=neuron_dim_t, neuron_dim_s=neuron_dim_s, neuron_dim_r=neuron_dim_r, num_iterations=num_iterations, random_dim=random_dim, phase=phase)
    model.load_state_dict(checkpoint['model'], strict=True)
else:
    print(f"Initializing from Weights: encoder-{init_from[0]}, thinker-{init_from[1]}, decoder-{init_from[2]}, phase = {phase}")
    model = ModelM(tokenizer, init_from=init_from, neuron_dim_t=neuron_dim_t, neuron_dim_s=neuron_dim_s, neuron_dim_r=neuron_dim_r, num_iterations=num_iterations, random_dim=random_dim, phase=phase)

model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16'))

# optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(beta1, beta2), weight_decay=weight_decay)
if init_from[0] == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=True)

# L1 weight scheduler
class DWAController:
    def __init__(self, target=5.0, update_freq=50, init_weight=1e-3, 
                 min_w=1e-6, max_w=0.1):
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
        
        # 更新逻辑
        if avg_loss > self.target:
            self.weight *= 1.01
        else:
            self.weight *= 0.99
            
        # --- 关键：添加截断保护 ---
        self.weight = max(self.min_w, min(self.max_w, self.weight))
        
        self.loss_accumulator = 0.0
        self.steps = 0
        return self.weight

    def get_weight(self):
        return self.weight

dwa_controller = DWAController(target=L1_TARGET, update_freq=500, init_weight=L1_weight, min_w=1e-6, max_w=0.1)

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)


train_dataset = ProblemAnswerDataset('train_data/train.json', tokenizer, num_splits=num_iterations)
val_dataset = ProblemAnswerDataset('train_data/valid.json', tokenizer, num_splits=num_iterations)

if ddp:
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, collate_fn=CollateFn(tokenizer.eos_token_id, target_append=False), num_workers=4, pin_memory=True)
else:
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=CollateFn(tokenizer.eos_token_id, target_append=False), num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=CollateFn(tokenizer.eos_token_id, target_append=False), num_workers=4, pin_memory=True)

best_val_loss = 1e9

# training loop
t0 = time.time()
local_iter_num = 0
running_mfu = -1.0
for iter_num in range(max_iters):
    if ddp:
        train_sampler.set_epoch(iter_num)
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        model.eval()
        l1_weight = dwa_controller.get_weight()
        with torch.no_grad():
            with ctx:
                total_nll, total_tokens, total_sparsity = 0, 0, 0
                for batch in val_loader:
                    _nll, _tokens, _spa = compute_loss(model, batch, phase=phase, L1_weight=l1_weight)
                    # print(_nll, _tokens)
                    total_nll += _nll
                    total_tokens += _tokens
                    total_sparsity += _spa
        val_loss = total_nll / total_tokens
        spa_loss = total_sparsity / total_tokens
            # train_loss = sum(compute_loss(model, batch).item() for batch in train_loader) / len(train_loader)
                # val_loss = sum(compute_loss(model, batch)[0].item() for batch in val_loader) / len(val_loader)
        print(f"step {iter_num}: val loss {val_loss:.4f}, spar loss {spa_loss:.4f}")
        if wandb_log:
            # wandb.log({"iter": iter_num, "train/loss": train_loss, "val/loss": val_loss, "lr": lr, "mfu": running_mfu * 100})
            wandb.log({"iter": iter_num, "val/loss": val_loss, "lr": lr})
        if val_loss < best_val_loss or always_save_checkpoint:
            best_val_loss = val_loss
            checkpoint = {
                'encoder_name': init_from[0], 
                'think_name': init_from[1], 
                'decoder_name': init_from[2],
                'optimizer': optimizer.state_dict(),
                'iter_num': iter_num,
                'best_val_loss': best_val_loss,
                'config': config,
            }
            if ddp:
                checkpoint['model'] = model.module.state_dict()
                # for n, p in model.module.named_parameters():
                #     print(f"[save] {n}: {p.dtype}")
            else:
                checkpoint['model'] = model.state_dict()

            print(f"saving checkpoint to {out_dir}")
            torch.save(checkpoint, os.path.join(out_dir, checkpoint_name))
        torch.save(checkpoint, os.path.join(out_dir, checkpoint_name_final))
        

    if eval_only:
        break
    model.train()
    
    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    micro_step = 0  # track accumulation steps
    train_loss, train_count, spar_train_loss = 0, 0, 0
    for batch in train_loader:
        l1_weight = dwa_controller.get_weight()
        if micro_step == 0:
            optimizer.zero_grad(set_to_none=True)
        
        ### new for ddp training
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            loss, train_tokens, spa_loss = compute_loss(model, batch, phase=phase, L1_weight=l1_weight)

        scaler.scale(loss).backward()
        
        train_loss += loss.item()
        train_count += train_tokens
        spar_train_loss += spa_loss.item()
        dwa_controller.step(spa_loss.item())
        micro_step += 1

        if micro_step == gradient_accumulation_steps:
            if grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()

            micro_step = 0

    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = train_loss/train_count
        spar_lossf = spar_train_loss/train_count
        # if local_iter_num >= 5:
        #     mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
        #     running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        # print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        print(f"iter {iter_num}: loss {lossf:.4f}, spar loss {spar_lossf:.4f}, time {dt:.3f}s")
    local_iter_num += 1


if ddp:
    destroy_process_group()
