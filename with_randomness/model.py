import numpy as np
from scipy.stats import beta
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
from transformers import BertModel
import torch.nn.functional as F
from torch.nn.functional import pad


def _resolve_thinker_class(backbone):
    if backbone == 'qwen':
        from custom_qwen2_lambda import MyQwen2Model
        return MyQwen2Model
    if backbone == 'llama':
        from custom_llama_lambda import MyLlamaModel
        return MyLlamaModel
    raise ValueError(f"unknown backbone: {backbone!r} (expected 'qwen' or 'llama')")


# Define our model
class ModelM(nn.Module):
    def __init__(self, tokenizer, model_path, init_from, backbone='qwen',
                 neuron_dim_t=100, neuron_dim_s=100, neuron_dim_r=1,
                 num_iterations=5, random_dim=1, phase='1'):
        super().__init__()
        self.tokenizer = tokenizer
        self.backbone = backbone

        ThinkerClass = _resolve_thinker_class(backbone)
        config = AutoConfig.from_pretrained(model_path)

        # Initialize modules. init_from[i] == 'config' means random init from the
        # backbone architecture; anything else is treated as a path / HF id.
        if init_from[0] == 'config':
            self.encoder = AutoModel.from_config(config)
        else:
            self.encoder = AutoModel.from_pretrained(init_from[0])
        self.random_encoder = AutoModel.from_config(config)

        if init_from[1] == 'config':
            self.think_model = ThinkerClass(config)
        else:
            self.think_model = ThinkerClass.from_pretrained(init_from[1])

        if init_from[2] == 'config':
            self.decoder = AutoModelForCausalLM.from_config(config)
        else:
            self.decoder = AutoModelForCausalLM.from_pretrained(init_from[2])
        
        # Extra Learnable Matrix
        self.neuron_dim_t, self.neuron_dim_s, self.neuron_dim_r = neuron_dim_t, neuron_dim_s, neuron_dim_r
        self.neuron_dim = neuron_dim_t + neuron_dim_s + neuron_dim_r
        self.neuron_matrix = nn.Parameter(torch.randn(self.neuron_dim, self.decoder.config.hidden_size))
        
        self.num_iterations = num_iterations
        hidden_dim = self.think_model.config.hidden_size
        if random_dim == -1:
            self.random_dim = hidden_dim
        else:
            self.random_dim = random_dim

        self.projection_mlp1 = nn.Sequential(
            nn.Linear(hidden_dim, self.random_dim),
            nn.ReLU()
        )
        self.projection_mlp2 = nn.Sequential(
            nn.Linear(self.random_dim, self.random_dim),
            nn.ReLU(),
            nn.Linear(self.random_dim, self.random_dim)            
        )
        self.pred_mlp1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2*self.random_dim)  # mean, log_var
        )
        self.phase = phase
        if phase == '1':
            self.pred_mlp1.requires_grad_(False)
        elif phase == '2':
            self.encoder.requires_grad_(False)
            self.random_encoder.requires_grad_(False)
            self.think_model.requires_grad_(False)
            self.decoder.requires_grad_(False)
            self.neuron_matrix.requires_grad_(False)
            self.projection_mlp1.requires_grad_(False)
            self.projection_mlp2.requires_grad_(False)

    def forward(self, input_ids, attention_mask, targets=None, test_mode=False, nar=True):
        """
        input_ids: (batch, seq_len)
        attention_mask: (batch, seq_len), where 1 = real token, 0 = padding
        decode_text_ids: list of (batch, dec_len)
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device

        ###-------Encoder--------
        encoder_outputs = self.encoder(input_ids, attention_mask=attention_mask).last_hidden_state
        # Mean of encoded representations
        encoder_mask = attention_mask.unsqueeze(-1).float()  # [batch_size, seq_len, 1]
        masked_outputs = encoder_outputs * encoder_mask  # [batch_size, seq_len, hidden_size]
        valid_token_counts = encoder_mask.sum(dim=1)  # [batch_size, 1]
        valid_token_counts = valid_token_counts.clamp(min=1e-9)
        encode_mean_outputs = masked_outputs.sum(dim=1) / valid_token_counts  # [batch_size, hidden_size]

        ###--------Thinking_module--------
        # Expand Neuron Matrix to Match Batch Size
        neuron_matrix = self.neuron_matrix.unsqueeze(0).repeat(batch_size, 1, 1)

        # Update Attention Mask
        extra_attention_mask = torch.ones((batch_size, self.neuron_dim), dtype=torch.long, device=device)  
        updated_attention_mask, _ = self.concat_neuron_text(attention_mask, attention_mask, extra_attention_mask)  
        extra_decode_mask = torch.ones((batch_size, self.neuron_dim_s), dtype=torch.long, device=device)  # (batch, 100)

        neuron_matrixes = []
        return_logits = [[] for _ in range(self.num_iterations)]
        
        ###--------Batch Processing Random_Encoder--------
        target_lengths = [t.shape[1] for t in targets]
        max_len = max(target_lengths)
        padded_targets = [pad(t, (0, max_len - t.shape[1]), value=self.tokenizer.eos_token_id) for t in targets]  # --> (B, max_len)

        all_targets = torch.cat(padded_targets, dim=0)  # (B * num_iterations, max_len)
        all_target_masks = (all_targets != self.tokenizer.eos_token_id).long()

        random_outputs = self.random_encoder(input_ids=all_targets, attention_mask=all_target_masks)
        random_hidden = random_outputs.last_hidden_state  # (B * num_iterations, L, dim)

        random_hidden = random_hidden.view(self.num_iterations, batch_size, max_len, -1)  # (N, B, L, D)
        target_lengths = all_target_masks.sum(dim=1) - 1  # (B * N,)
        target_lengths = target_lengths.view(self.num_iterations, batch_size) # (N, B)

        batch_indices = torch.arange(batch_size, device=device).unsqueeze(0).expand(self.num_iterations, -1)  # (N, B)

        random_last_hiddens = random_hidden[torch.arange(self.num_iterations, device=device).unsqueeze(1), batch_indices, target_lengths, :]  # (N, B, D)

        ###-------Decoder--------
        for i in range(self.num_iterations):
            random_last_hidden = random_last_hiddens[i]
            neuron_avg = neuron_matrix[:, :-1, :].mean(dim=1)+encode_mean_outputs  # (batch, hidden)
            pred_stats = self.pred_mlp1(neuron_avg)  # (batch, 2xself.think_model.config.hidden_size)
            mean = pred_stats[:, :-self.random_dim]
            log_var = pred_stats[:, -self.random_dim:]
            var = torch.exp(log_var)
            std = torch.exp(0.5 * log_var)

            random_last_hidden = self.projection_mlp1(random_last_hidden)

            if self.phase == '2':
                # randomness prediction loss
                # log-likelihood of the last value in random_last_hidden
                x = random_last_hidden[:, -self.random_dim:]
                log_likelihood = -((x - mean) / (2 * var)) ** 2 - 0.5 * torch.log(2 * torch.pi * var)
                return_logits[i].append(-log_likelihood.mean())
            else:
                # sparsity loss
                return_logits[i].append(torch.norm(random_last_hidden, p=1, dim=1, keepdim=True))  # (batch, 1)         
            
            if test_mode:
                std = torch.exp(0.5 * log_var) 
                epsilon = torch.randn_like(mean, device=device)
                sample = mean + epsilon * std # (batch_size,)
                random_last_hidden[:, -self.random_dim:] = sample

            projected_hidden = self.projection_mlp2(random_last_hidden)

            neuron_matrix[:, -self.neuron_dim_r:, :] = projected_hidden.reshape(batch_size, self.neuron_dim_r, -1)
            thinking_input, pad_positions = self.concat_neuron_text(encoder_outputs, attention_mask, neuron_matrix)
            
            # Pass to Thinking Model
            output = self.think_model(inputs_embeds=thinking_input, attention_mask=updated_attention_mask).last_hidden_state  # (batch, seq_len + 201, dim)

            batch_indices = torch.arange(attention_mask.size(0), device=device).unsqueeze(1)
            seq_indices = pad_positions.unsqueeze(1) + torch.arange(neuron_matrix.size(1), device=device).unsqueeze(0)  # (batch, decode_len)
            neuron_matrix = output[batch_indices, seq_indices] 
            
            # Restore the fixed columns
            # neuron_matrix[:, :, :50] = fixed_columns  # Ensure the first 50 columns remain fixed
            part1 = neuron_matrix[:, self.neuron_dim_t:self.neuron_dim_t + self.neuron_dim_s, :]  # 取出 s 个
            neuron_matrixes.append(part1)

            if targets != None:
                # 1. Token prediction loss
                logits = self.decode(part1, extra_decode_mask, targets[i])
                if nar:
                    nar_logits = self.decode_with_NAR(part1, extra_decode_mask, targets[i], mask_ratio=1)
                    return_logits[i].append([logits, nar_logits])
                else:
                    return_logits[i].append(logits)
        return neuron_matrixes, return_logits, random_last_hidden

    def concat_neuron_text(self, neuron_matrix, attention_mask, text_embeds):
        """
        Concatenate neuron_matrix and text_embeds along dim=1 while ensuring text_embeds start from the first padding position.
        
        Args:
            neuron_matrix (torch.Tensor): Tensor of shape (batch, seq, dim)
            text_embeds (torch.Tensor): Tensor of shape (batch, seq1, dim)
            attention_mask (torch.Tensor): Tensor of shape (batch, seq)
    
        Returns:
            torch.Tensor: Concatenated tensor of shape (batch, seq + valid_seq1, dim)
        """
        batch_size, seq = neuron_matrix.size()[:2]
        pad_positions = (attention_mask == 0).int().argmax(dim=1)
        pad_positions[attention_mask.sum(dim=1) == seq] = seq  

        concatenated = []
        for i in range(batch_size):
            split_idx = pad_positions[i].item()
            combined = torch.cat([neuron_matrix[i, :split_idx], text_embeds[i]], dim=0)
            concatenated.append(combined)

        max_len = max(x.shape[0] for x in concatenated)
        if len(neuron_matrix.size()) == 3:
            padded_result = torch.stack([torch.cat([x, torch.zeros(max_len - x.shape[0], neuron_matrix.size(-1), device=x.device)]) for x in concatenated])
        else:
            padded_result = torch.stack([torch.cat([x, torch.zeros(max_len - x.shape[0], device=x.device)]) for x in concatenated])
        return padded_result, pad_positions

    def decode(self, neuron_matrix, encoder_attention_mask, decode_text_ids, attention_mask=None):
        device = neuron_matrix.device  
        
        decode_text_embeds = self.decoder.model.embed_tokens(decode_text_ids)  # (batch, decode_len, dim)
        decoder_input_embeds, pad_positions = self.concat_neuron_text(neuron_matrix, encoder_attention_mask, decode_text_embeds)  # (batch, 100 + decode_len, dim)
        decoder_outputs = self.decoder(inputs_embeds=decoder_input_embeds, attention_mask=attention_mask)

        # extract logits from positions
        batch_indices = torch.arange(encoder_attention_mask.size(0), device=device).unsqueeze(1)
        seq_indices = pad_positions.unsqueeze(1) + torch.arange(decode_text_ids.size(1), device=device).unsqueeze(0)  # (batch, decode_len)
        logits = decoder_outputs.logits[batch_indices, seq_indices] 
        return logits
    
    def decode_with_NAR(self, neuron_matrix, encoder_attention_mask, decode_text_ids, mask_ratio=1, attention_mask=None):
        device = neuron_matrix.device  
        batch_size, seq_len = decode_text_ids.shape
        masked_input_ids = decode_text_ids.clone()

        # NAR loss
        if attention_mask == None:
            attention_mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=device)
        for i in range(batch_size):
            # mask_ratio = self.sample_ratio()
            num_to_mask = int(seq_len * mask_ratio)
            mask_indices = torch.randperm(seq_len, device=device)[:num_to_mask]
            attention_mask[i, mask_indices] = False  
            masked_input_ids[i, mask_indices] = self.tokenizer.eos_token_id
        nar_attention_mask, _ = self.concat_neuron_text(encoder_attention_mask, encoder_attention_mask, attention_mask)  # (batch, 100 + decode_len, dim)
        
        # print(masked_input_ids)
        # print(nar_attention_mask)
        nar_logits = self.decode(neuron_matrix, encoder_attention_mask, masked_input_ids, attention_mask=nar_attention_mask)
        return nar_logits
    
    def resort_decode(self, decode_text_ids, batch_size, device):
        previous_inputs = [[] for _ in range(batch_size)]
        if len(decode_text_ids) != 0:
            for tensor in decode_text_ids:
                for i in range(batch_size):
                    sequence = tensor[i]
                    eos_idx = (sequence == self.tokenizer.eos_token_id).nonzero(as_tuple=True)[0]  # find EOS 
                    if len(eos_idx) > 0:
                        eos_idx = eos_idx[0]  
                        valid_tokens = sequence[:eos_idx]  
                    else:
                        valid_tokens = sequence  
                    previous_inputs[i] += valid_tokens
        return [torch.tensor(x, device=device) for x in previous_inputs]
    
    def pad_token(self, decode_text_ids_i, previous_inputs=None):
        device = decode_text_ids_i.device
        batch_size = decode_text_ids_i.size(0)

        if previous_inputs != None:
            concatenated_sentences = [torch.cat([previous_inputs[i], decode_text_ids_i[i]], dim=-1) for i in range(batch_size)]
        else:
            concatenated_sentences = decode_text_ids_i

        max_new_seq_len = max(s.shape[-1] for s in concatenated_sentences)
        take_logit_pos = torch.tensor([s.shape[-1]-1 for s in concatenated_sentences], device=device)

        padded_output = torch.full((batch_size, max_new_seq_len), self.tokenizer.eos_token_id, dtype=decode_text_ids_i.dtype, device=device)
        for i in range(batch_size):
            seq_len = concatenated_sentences[i].shape[-1]
            padded_output[i, :seq_len] = concatenated_sentences[i]
        return padded_output, take_logit_pos

    def generate_with_answer(self, input_ids, attention_mask, targets, max_length=256, temperature=1.0, top_k=40, target_append=True):
        """
        input_ids: token IDs of the initial input text
        attention_mask: attention mask
        targets: ground-truth answers
        max_length: maximum number of new tokens to generate
        temperature: sampling temperature
        top_k: restrict sampling to the top-k tokens with the highest probability
        target_append: whether to append the ground-truth answer to the input sequence
        """
        device = input_ids.device
        batch_size = input_ids.shape[0]

        # start with '\n'
        newline_token_id = self.tokenizer.encode("\n")[0]
        decode_text_ids = [torch.full((batch_size, 1), newline_token_id, dtype=torch.long, device=device) for _ in range(self.num_iterations)]

        total_log_probs = [torch.zeros(batch_size, device=device) for _ in range(self.num_iterations)]  # store cumulative log probabilities
        token_counts = [torch.zeros(batch_size, device=device) for _ in range(self.num_iterations)]  # record the number of valid tokens
        
        # track which samples have already generated eos_token
        eos_reached = [torch.zeros(batch_size, dtype=torch.bool, device=device) for _ in range(self.num_iterations)]

        # obtain the thinking results
        neuron_matrixes, _, _ = self.forward(input_ids, attention_mask, targets, test_mode=True) # when test_mode=True, targets do not matter

        # Update Attention Mask
        extra_attention_mask = torch.ones((batch_size, self.neuron_dim_s), dtype=torch.long, device=device)  # (batch, 100)

        # decoding loop
        skip_i = []
        for i in range(self.num_iterations):
            if target_append:
                previous_inputs = self.resort_decode(decode_text_ids[:i], batch_size, device)
            else:
                previous_inputs = None
            for _ in range(max_length):
                current_inputs, take_logit_pos = self.pad_token(decode_text_ids[i], previous_inputs)
                if i in skip_i:
                    break
                logits = self.decode(neuron_matrixes[i], extra_attention_mask, current_inputs)
                logits = logits[torch.arange(batch_size), take_logit_pos].squeeze(1) 
                
                # sampling: Top-K + Temperature (default = 0)
                if temperature == 0:  # greedy decoding
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                elif top_k > 0:
                    probs = torch.softmax(logits / temperature, dim=-1)
                    values, indices = torch.topk(logits, top_k)
                    top_probs = torch.softmax(values / temperature, dim=-1)
                    next_token = torch.gather(indices, 1, torch.multinomial(top_probs, num_samples=1))
                else:
                    probs = torch.softmax(logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)

                take_logit_pos += 1
                
                # only update samples that have not yet generated EOS
                active_mask = ~eos_reached[i]  
                token_log_probs = torch.log(torch.gather(probs, 1, next_token).squeeze(1) + 1e-9)  
                total_log_probs[i][active_mask] += token_log_probs[active_mask]
                token_counts[i][active_mask] += 1  

                decode_text_ids[i] = torch.cat([decode_text_ids[i], next_token], dim=1)  

                # if all samples in the batch generate `eos_token_id`, stop early
                if torch.all(next_token == self.tokenizer.eos_token_id):
                    skip_i.append(i)

                eos_reached[i] |= (next_token == self.tokenizer.eos_token_id).squeeze(1)
        
        # compute Perplexity
        avg_log_prob = sum(total_log_probs) / sum(token_counts)
        perplexities = torch.exp(-avg_log_prob)

        return decode_text_ids, perplexities

    def generate_with_answer_nar(self, input_ids, attention_mask, targets, max_length=256, temperature=1.0, top_k=40, num_refinement_steps=1, target_append=False):
        """
        Generate text using Non-Autoregressive (NAR) decoding with iterative refinement.

        Args:
            input_ids (torch.Tensor): Token IDs of the initial input text.
            attention_mask (torch.Tensor): Attention mask.
            targets (list): Ground-truth answers.
            max_length (int): Maximum sequence length to generate.
            temperature (float): Sampling temperature (not used in greedy NAR decoding).
            top_k (int): Top-k sampling parameter (not used in greedy NAR decoding).
            num_refinement_steps (int): Number of refinement steps in NAR generation.
            target_append (bool): Whether to append ground-truth answers to the input (unused in this NAR version).

        Returns:
            list: A list of generated token sequence tensors.
            None: Perplexity is not applicable for NAR models, returns None.
        """
        device = input_ids.device
        batch_size = input_ids.shape[0]

        # 1. Obtain intermediate "thinking" thoughts (neuron_matrix), same as in AR decoding
        neuron_matrixes, _, _ = self.forward(input_ids, attention_mask, targets, test_mode=True) # when test_mode=True, targets do not matter
        extra_attention_mask = torch.ones((batch_size, self.neuron_dim_s), dtype=torch.long, device=device) 

        # 2. Get special token IDs from the tokenizer
        eos_token_id = self.tokenizer.eos_token_id

        final_generated_sequences = []
        # Outer loop remains the same: perform N different generation tasks (num_iterations)
        for i in range(self.num_iterations):
            fixed_generate_length = max_length
            # ===== Core logic of Non-Autoregressive generation begins =====

            # 3. Initialization: create a sequence filled with [MASK]/EOS as the starting point
            generated_ids = torch.full((batch_size, fixed_generate_length), eos_token_id, dtype=torch.long, device=device)

            # 4. Iterative refinement loop (Mask-Predict)
            for step in range(num_refinement_steps):
                # Call the decoder to obtain logits for all positions at once
                # logits shape: (batch_size, max_length, vocab_size)
                is_mask = (generated_ids != eos_token_id)
                logits = self.decode_with_NAR(neuron_matrixes[i], extra_attention_mask, generated_ids, mask_ratio=0, attention_mask=is_mask)

                # Greedy strategy
                predicted_ids = torch.argmax(logits, dim=-1)

                if step == num_refinement_steps - 1:
                    generated_ids = predicted_ids
                    break
                
                # a. Compute confidence scores
                probs = torch.softmax(logits, dim=-1)
                confidence_scores = torch.gather(probs, 2, predicted_ids.unsqueeze(-1)).squeeze(-1)

                # b. Use cosine scheduling to determine the number of tokens to mask at this step
                mask_ratio = np.cos((step / num_refinement_steps) * (np.pi / 2))
                num_to_mask = int(mask_ratio * fixed_generate_length)

                # c. Identify indices of tokens with the lowest confidence
                low_confidence_indices = torch.topk(confidence_scores, k=num_to_mask, dim=1, largest=False).indices

                # d. Reset low-confidence positions to [MASK]/EOS for the next refinement step
                generated_ids = predicted_ids.clone()
                generated_ids.scatter_(1, low_confidence_indices, eos_token_id)

            final_generated_sequences.append(generated_ids)

        # Perplexity calculation is not applicable to this model
        return final_generated_sequences, None
    
    def sample_ratio(self, min_range=0.6, max_range=1):
        alpha_param = 10  
        beta_param = 2   
        random_values_0_1 = beta.rvs(a=alpha_param, b=beta_param, size=1)[0]
        final_value = min_range + random_values_0_1 * (max_range - min_range)
        return final_value