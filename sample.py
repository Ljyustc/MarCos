"""
Sample from a trained model
"""
import os
import torch
import json
import time
import numpy as np
from contextlib import nullcontext
from tqdm import tqdm
from transformers import AutoTokenizer
from model_Qwen_1 import ModelM
from dataloader1 import ProblemAnswerDataset, CollateFn 
import torch.distributed as dist
from torch.utils.data import DataLoader

# -----------------------------------------------------------------------------
init_from = ('resume', 'resume', 'resume')  
checkpoint_name = 'phase2_saved.pt'
out_dir = 'out'  
start = "\n"  
num_iterations = 3
neuron_dim_t = 1
neuron_dim_s = 100
neuron_dim_r = 16
random_dim = 16*896
max_new_tokens = 100  
temperature = 0  # fixed
top_k = 200  # doesn't work
seed = 1337
backend = 'nccl' # 'nccl', 'gloo', etc.
device = 'cuda'  
compile = False  

input_file = 'test.jsonl'
output_file = 'test_results/test_output_' + checkpoint_name[:-3] + '_test.json'
loss_output_file = 'test_loss/test_output_' + checkpoint_name[:-3] + '_test.json'
batch_size = 64
# -----------------------------------------------------------------------------

rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])

device = f"cuda:{rank}"
torch.cuda.set_device(device)
device_type = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B')
# model
if init_from[0] == 'resume':
    ckpt_path = os.path.join(out_dir, checkpoint_name)
    checkpoint = torch.load(ckpt_path, map_location=device)
    encoder_name, think_name, decoder_name = checkpoint['encoder_name'], checkpoint['think_name'], checkpoint['decoder_name']
    model = ModelM(tokenizer, init_from=(encoder_name, think_name, decoder_name), neuron_dim_t=neuron_dim_t, neuron_dim_s=neuron_dim_s, neuron_dim_r=neuron_dim_r, num_iterations=num_iterations, random_dim=random_dim)
    # for name, param in model.encoder.layers[0].named_parameters():
    #     print(f"unitial: {name}: {param.dtype}")
    model.load_state_dict(checkpoint['model'], strict=True)
    print(checkpoint['best_val_loss'])
    checkpoint_keys = checkpoint['model'].keys()
elif init_from[0].startswith('Qwen'):
    model = ModelM(tokenizer, init_from=init_from, neuron_dim_t=neuron_dim_t, neuron_dim_s=neuron_dim_s, neuron_dim_r=neuron_dim_r, num_iterations=num_iterations, random_dim=random_dim)

model.to(device)
model.eval()

if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

def load_jsonl(file):
    data = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def write_jsonl(res, outfile):
    with open(outfile, 'w', encoding='utf-8') as f:
        for d in res:
            f.writelines(json.dumps(d, ensure_ascii=False))
            f.writelines('\n')

def compute_loss(model, batch):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    targets = [t.to(device) for t in batch['targets']]
    loss_mask = [t.to(device) for t in batch['loss_masks']]
    
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    _, logits_all, random_last_hidden = model(input_ids, attention_mask=attention_mask, targets=targets, test_mode=False, nar=False)
    loss_all = [0 for _ in range(len(logits_all))]
    no_mask_counts = [0 for _ in range(len(logits_all))]
    for i in range(len(logits_all)):
        logits = logits_all[i][1] 
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = targets[i][:, 1:]
        shift_mask = loss_mask[i][:, :-1].contiguous()
        loss = loss_fn(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
        if torch.isnan(loss).any():
            print(f"NaN detected in loss")

        # If shift_mask only contains one token, it's the eos_token and we don't need to train
        skip_mask = shift_mask.sum(dim=1, keepdim=True) != 1  # (batch,1)
        loss = loss.view(shift_labels.size()) * shift_mask * skip_mask

        loss_all[i] += loss.sum() 
        no_mask_counts[i] += (shift_mask * skip_mask).sum()
    return [(loss_all[i].item(), no_mask_counts[i]) for i in range(len(logits_all))], random_last_hidden

def process_batches(input_file, output_file, loss_output_file, model, tokenizer, device, batch_size=4, max_new_tokens=100, temperature=1.0, top_k=40):
    eos_str = tokenizer.decode([tokenizer.eos_token_id])  # EOS token
    outputs = []
    perplexities_all = []
    
    val_dataset = ProblemAnswerDataset(input_file, tokenizer, num_splits=num_iterations)
    chunk_size = int(np.ceil(len(val_dataset) / world_size))
    start = rank * chunk_size
    end = (rank + 1) * chunk_size if rank < world_size - 1 else len(val_dataset)
    sub_dataset = torch.utils.data.Subset(val_dataset, list(range(start, end)))

    val_loader = DataLoader(sub_dataset, batch_size=batch_size,
                        collate_fn=CollateFn(tokenizer.eos_token_id, target_append=False),
                        num_workers=4, pin_memory=True)
    val_loss = [[] for _ in range(num_iterations)]
    val_loss_tokens = [[] for _ in range(num_iterations)]
    val_random_hiddens = []
    start_time = time.time()
    with torch.no_grad():
        with ctx:
            for batch in tqdm(val_loader, desc=f"Rank {rank} Evaluating", unit="batch", ncols=80):
                batch_loss, random_last_hidden = compute_loss(model, batch)
                # val_random_hiddens.append(random_last_hidden)
                # print(batch_loss)
                x = batch["input_ids"].to(device)
                targets = [t.to(device) for t in batch['targets']]
                attention_mask = batch["attention_mask"].to(device)
        
                outputs_batch, perplexities = model.generate_with_answer(
                    input_ids=x,
                    attention_mask=attention_mask, 
                    targets=targets,
                    max_length=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    target_append=False,
                )
            
                output_i = [[] for _ in range(batch_size)]
                for i in range(len(outputs_batch)):
                    for idx, output in enumerate(outputs_batch[i]):
                        output_text = tokenizer.decode(output)
                        output_text = output_text.split(eos_str)[0]  # split by EOS
                        output_i[idx].append(output_text)
                
                outputs += output_i
                perplexities_all.append(perplexities)
                for i in range(num_iterations):
                    val_loss[i].append(batch_loss[i][0])
                    val_loss_tokens[i].append(batch_loss[i][1])
    
    end_time = time.time()
    total_time = end_time - start_time
    
    if world_size > 1:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        gathered_outputs = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_outputs, outputs)
    
        if rank == 0:
            final_outputs = []
            for o in gathered_outputs:
                for d in o:
                    if d: 
                        final_outputs.append(d)
        else:
            final_outputs = outputs
    
    if rank == 0:
        print(f"total time: {total_time:.4f} seconds")
        write_jsonl(final_outputs, output_file)
        print(f"Processed outputs saved to {output_file}")

    # val_random_hiddens = torch.cat(val_random_hiddens, dim=0)
    # torch.save(val_random_hiddens, 'val_random_embedding.pt')

    # print("perplexity: ", torch.concatenate(perplexities_all).mean())
        

process_batches(input_file, output_file, loss_output_file, model, tokenizer, device, batch_size, max_new_tokens, temperature, top_k)

if world_size > 1:
    dist.destroy_process_group()