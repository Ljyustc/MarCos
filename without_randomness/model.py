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


class MiniTransformer(nn.Module):
    def __init__(self, hidden_dim, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=nhead,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
                activation="relu"  # 或者 "gelu"
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # x: [B, L, H]
        for layer in self.layers:
            x = layer(x)  # 每层都会做 attention + ffn
        return self.norm(x)  # [B, L, H]

# Define our model
class ModelM(nn.Module):
    def __init__(self, tokenizer, model_path, init_from, backbone='llama',
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
        
        # Extra Learnable Matrix (201 x embedding_dim)
        self.neuron_dim_t, self.neuron_dim_s, self.neuron_dim_r = neuron_dim_t, neuron_dim_s, neuron_dim_r
        self.neuron_dim = neuron_dim_t + neuron_dim_s + neuron_dim_r
        self.neuron_matrix = nn.Parameter(torch.randn(self.neuron_dim, self.decoder.config.hidden_size))
        
        self.num_iterations = num_iterations
        # self.random_encoder = self.encoder
        
        # self.decoder.model
        # self.encoder = self.decoder.model
        hidden_dim = self.think_model.config.hidden_size
        # if random_dim == -1:
        #     self.random_dim = hidden_dim
        # else:
        #     self.random_dim = random_dim
        # # self.batchnorm = nn.BatchNorm1d(hidden_dim)
        # self.projection_mlp1 = nn.Sequential(
        #     nn.Linear(hidden_dim, self.random_dim),
        #     nn.ReLU()
        # )
        # self.projection_mlp2 = nn.Sequential(
        #     nn.Linear(self.random_dim, self.random_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.random_dim, self.random_dim)            
        # )
        # self.pred_mlp1 = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, 2*self.random_dim)  # mean, log_var
        # )
        # self.mean_mlp = nn.Sequential(
        #     nn.Linear(hidden_dim, self.random_dim*4),
        #     nn.LeakyReLU(),
        #     nn.Linear(self.random_dim*4, self.random_dim*2),
        #     nn.LeakyReLU(),
        #     nn.Linear(self.random_dim*2, self.random_dim),  # mean
        # )
        self.phase = phase

        # self.random_encode_process = MiniTransformer(hidden_dim, num_layers=3)

        # if phase == '1':
        #     self.pred_mlp1.requires_grad_(False)
        # elif phase == '2':
        #     self.encoder.requires_grad_(False)
        #     self.random_encoder.requires_grad_(False)
        #     self.think_model.requires_grad_(False)
        #     self.decoder.requires_grad_(False)
        #     self.neuron_matrix.requires_grad_(False)
        #     self.projection_mlp1.requires_grad_(False)
        #     self.projection_mlp2.requires_grad_(False)

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
        # print("inputs", input_ids)
        # print("encoder.training:", self.encoder.training)
        # print("encoder: ", encoder_outputs)
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
        extra_attention_mask = torch.ones((batch_size, self.neuron_dim), dtype=torch.long, device=device)  # (batch, 201)
        updated_attention_mask, _ = self.concat_neuron_text(attention_mask, attention_mask, extra_attention_mask)  # (batch, seq_len + 201)
        # updated_attention_mask, _ = self.concat_neuron_text(extra_attention_mask, extra_attention_mask, attention_mask)  # (batch, seq_len + 201)
        # updated_decode_mask, _ = self.concat_neuron_text(attention_mask, attention_mask, extra_decode_mask)  # (batch, seq_len + 100)

        neuron_matrixes = []
        return_logits = [[] for _ in range(self.num_iterations)]
        
        ###--------Batch Processing Random_Encoder--------
        # target_lengths = [t.shape[1] for t in targets]
        # max_len = max(target_lengths)
        # padded_targets = [pad(t, (0, max_len - t.shape[1]), value=self.tokenizer.eos_token_id) for t in targets]  # --> (B, max_len)

        # all_targets = torch.cat(padded_targets, dim=0)  # (B * num_iterations, max_len)
        # all_target_masks = (all_targets != self.tokenizer.eos_token_id).long()
        # # print(all_targets[0], all_target_masks[0])
        # random_outputs = self.random_encoder(input_ids=all_targets, attention_mask=all_target_masks)
        # random_hidden = random_outputs.last_hidden_state  # (B * num_iterations, L, dim)

        # random_hidden = random_hidden.view(self.num_iterations, batch_size, max_len, -1)  # (N, B, L, D)
        # target_lengths = all_target_masks.sum(dim=1) - 1  # (B * N,)
        # target_lengths = target_lengths.view(self.num_iterations, batch_size) # (N, B)
        # # print(target_lengths)

        # batch_indices = torch.arange(batch_size, device=device).unsqueeze(0).expand(self.num_iterations, -1)  # (N, B)

        # random_last_hiddens = random_hidden[torch.arange(self.num_iterations, device=device).unsqueeze(1), batch_indices, target_lengths, :]  # (N, B, D)
        # random_last_hiddens = random_hidden[:,:,41,:]
        # print("firts random_last: ", random_last_hiddens)
        # torch.save(random_last_hiddens, '1663k_sparse0.5_nar_phase1_step2_embedding.pt')

        ####  平均池化randomness
        # all_target_masks = all_target_masks.view(self.num_iterations, batch_size, max_len)  # (N, B, L)

        # # 使用 mask 做平均池化（仅平均非 eos_token 的 hidden states）
        # masked_hidden = random_hidden * all_target_masks.unsqueeze(-1)  # (N, B, L, D)
        # sum_hidden = masked_hidden.sum(dim=2)  # (N, B, D)
        # lengths = all_target_masks.sum(dim=2).clamp(min=1)  # 避免除以 0

        # random_last_hiddens = sum_hidden / lengths.unsqueeze(-1)  # (N, B, D)
        ####  
        # encode_mean_outputs = encode_mean_outputs.unsqueeze(1)  # [B, 1, H]

        ###-------Decoder--------
        for i in range(self.num_iterations):
            extra_decode_mask = torch.ones((batch_size, self.neuron_dim_s * (1)), dtype=torch.long, device=device)  # (batch, 100)
            # random_last_hidden = random_last_hiddens[i]
            # # print(self.layernorm(random_last_hidden).mean(dim=0)[400:410])
            # # neuron_avg = neuron_matrix[:, :-self.neuron_dim_r, :].mean(dim=1)+encode_mean_outputs  # (batch, hidden)

            # sub_neurons = neuron_matrix[:, :-self.neuron_dim_r, :]   # [B, R, H]
            # # 把 encode_mean_outputs 拼接到最后
            # seq = torch.cat([sub_neurons, encode_mean_outputs.unsqueeze(1)], dim=1)  # [B, R+1, H]
            # # 送入 TinyTransformer
            # transformed = self.random_encode_process(seq)  # [B, R+1, H]
            # # 拿最后一个位置（即 encode_mean_outputs 对应的位置）
            # neuron_avg = transformed[:, -1, :]   # [B, H]

            # #     # print("neuron_avg", neuron_avg)
            # pred_stats = self.pred_mlp1(neuron_avg)  # (batch, 2xself.think_model.config.hidden_size)
            # # mean = pred_stats[:, :-self.random_dim]
            # # mean = torch.clamp(mean, min=0)
            # mean = self.mean_mlp(neuron_avg)
            # # print("begin: ",mean.size(), self.random_dim)
            # log_var = pred_stats[:, -self.random_dim:]
            # var = torch.exp(log_var)
            # std = torch.exp(0.5 * log_var)
            # torch.save(mean, f'eq_mean_{i}.pt')
            # torch.save(std, f'eq_std_{i}.pt')

            # if i==0:
            #     sparse_mask = torch.abs(random_last_hidden) <= 1
            #     random_last_hidden[sparse_mask] = 0
            # random_last_hidden[:, 570] = -10
            # random_last_hidden[:,:-self.random_dim] = 0
            # random_last_hidden[:,:] = 0
            # print("r: ", random_last_hidden)
            # if i == 0:
            #     torch.save(random_last_hidden, 'nl-step1_embedding.pt')
            # else:
            #     torch.save(random_last_hidden, 'sparse0.1_step12_2_embedding.pt')
            # print("random_last input: ", torch.where(random_last_hidden!=0))
            # random_last_hidden = self.projection_mlp1(random_last_hidden)
            # # print(torch.norm(random_last_hidden, dim=1, keepdim=True))
            # # print("mlp: ", random_last_hidden.dtype)
            # random_last_hidden = F.normalize(random_last_hidden, p=2, dim=1)
            # print("normalize: ", random_last_hidden.dtype)
            # random_last_hidden[:, 1641] = 0.1
            # random_last_hidden[:, 0] = 0.9
            # print("random_last: ", torch.where(random_last_hidden!=0))

            # if self.phase == '2':
            #     # randomness prediction loss
            #     # log-likelihood of the last value in random_last_hidden
            #     x = random_last_hidden[:, -self.random_dim:]
            #     # print("x", x)
            #     # log_likelihood = -((x - mean)) ** 2 / (2 * var) - 0.5 * torch.log(2 * torch.pi * var)
            #     log_likelihood = -torch.sqrt(((x - mean) ** 2).mean(dim=1, keepdim=True))
            #     return_logits[i].append(-log_likelihood.mean(dim=1, keepdim=True))  # (batch, 1)
            # else:
            #     # sparsity loss
            #     entropy = -(random_last_hidden * torch.log(random_last_hidden + 1e-12)).sum(dim=1, keepdim=True)
            #     return_logits[i].append(entropy)  # (batch, 1)
            
            # top1_idx = random_last_hidden.argmax(dim=-1, keepdim=True)
            # top1_mask = torch.zeros_like(random_last_hidden)
            # top1_mask.scatter_(-1, top1_idx, 1.0)
            # random_last_hidden = random_last_hidden * top1_mask
            
            # torch.save(random_last_hidden, f'nl_step{i}_embedding.pt')
            # mask = (random_last_hidden != 0).to(random_last_hidden.dtype)
            # variance = 0.01  # * random_last_hidden.detach().abs()
            # noise = torch.randn_like(random_last_hidden) * variance * mask
            # noise = torch.randn_like(random_last_hidden) * 0.01 # * mask
            # random_last_hidden = random_last_hidden + noise
            # if i == 0:
            #     # random_last_hidden[0, :] = random_last_hidden[1, :]
            #     random_last_hidden[:, :] = 0
            #     random_last_hidden[:, 31] = 0.1402
            #     random_last_hidden[:, 48] = 0.9901
            #     random_last_hidden[:, 636] = 0.0122
            # elif i == 2:
            #     random_last_hidden[:, :] = 0
            #     random_last_hidden[:, [923, 1558, 2530]] = 1
        #         random_last_hidden[:, [  29,   62,  191,  263,  297,  300,  422,  439,  492,  515,  530,  589,
        #   813,  854,  979,  984, 1012, 1021, 1087, 1170, 1272, 1322, 1406, 1495,
        #  1513, 1551, 1558, 1647, 1661, 1721, 1744, 1745, 1861, 1889, 2041, 2057,
        #  2060, 2154, 2297, 2340, 2447, 2530, 2563, 2780, 2912, 3020, 3103, 3111,
        #  3209, 3229, 3397, 3481, 3534, 3560, 3569]] = torch.tensor([2.4657e-02, 8.5059e-03, 2.5828e-03, 4.5112e-03, 1.8079e-04, 5.0622e-03,
        # 8.7125e-03, 1.4532e-02, 5.2688e-03, 2.4967e-03, 9.0913e-03, 1.6805e-02,
        # 4.1539e-04, 3.9258e-03, 1.1571e-02, 1.5841e-02, 3.7536e-03, 4.0635e-03,
        # 3.0718e-02, 3.1269e-02, 3.0821e-03, 6.6807e-03, 5.9231e-03, 6.8529e-03,
        # 4.0222e-02, 8.2648e-03, 5.1655e-03, 6.7000e-01, 1.4550e-03, 2.3245e-03,
        # 2.4574e-01, 1.1364e-02, 1.3844e-02, 2.0576e-03, 2.3004e-02, 1.5497e-03,
        # 3.9947e-02, 2.0318e-03, 2.2900e-03, 2.2728e-02, 5.8543e-03, 6.9204e-01,
        # 8.6781e-03, 7.3006e-03, 2.5070e-02, 9.5734e-03, 2.5483e-02, 1.5910e-02,
        # 3.0718e-02, 2.6344e-03, 9.8575e-04, 1.4326e-02, 1.0744e-02, 7.7483e-04,
        # 1.1089e-02]).to(random_last_hidden.device)
        #     else:
        #         random_last_hidden[:, :] = 0
        #         # random_last_hidden[:, 2475] = 1
        #         random_last_hidden[:, [  48,   92,  366,  900,  984, 1064, 1078, 2165, 2290, 2384, 2475, 2522,
        #  2537, 2632, 3132, 3182]] = torch.tensor([3.2687e-02, 1.0125e-02, 1.2357e-02, 3.9544e-02, 8.0722e-04, 1.8337e-02,
        # 3.1093e-03, 5.1343e-02, 2.9817e-02, 1.5467e-02, 9.9497e-01, 5.3257e-02,
        # 1.1162e-02, 4.9081e-04, 6.9760e-03, 4.5643e-03]).to(random_last_hidden.device)
        #     random_last_hidden[:, :] = 0
        #     random_last_hidden[:, [  48,   54,   55,   98,  100,  129,  173,  179,  196,  215,  394,  504,
        #   582,  589,  803,  829,  871, 1020, 1105, 1206, 1220, 1243, 1284, 1309,
        #  1315, 1319, 1335, 1380, 1412, 1442, 1652, 1735, 1739, 1866, 1882, 1935,
        #  2049, 2160, 2179, 2182, 2431, 2440, 2459, 2493, 2510, 2530, 2590, 2611,
        #  2752, 2851, 2872, 3004, 3065, 3115, 3119, 3166, 3264, 3316, 3568]] = torch.tensor([3.4135e-04, 2.5992e-03, 7.1506e-03, 3.0708e-03, 1.9741e-03, 7.3480e-04,
        # 1.0879e-02, 7.2383e-03, 3.9482e-03, 5.2642e-04, 4.3046e-04, 8.3789e-03,
        # 1.2810e-02, 6.2328e-01, 2.2071e-04, 1.0660e-02, 1.0693e-04, 1.7547e-02,
        # 1.1581e-02, 7.0190e-01, 1.1077e-03, 1.6560e-03, 1.2020e-02, 6.2732e-03,
        # 2.4742e-02, 1.0046e-02, 3.1585e-03, 6.6680e-03, 1.1406e-02, 1.0528e-03,
        # 1.4652e-02, 3.9701e-03, 1.4389e-02, 6.4487e-03, 1.7657e-03, 3.2024e-03,
        # 1.9083e-03, 1.0419e-03, 2.0618e-03, 1.0616e-02, 1.3161e-02, 1.5354e-02,
        # 1.6560e-03, 2.0289e-04, 1.3599e-02, 2.8918e-01, 1.2886e-03, 1.4828e-02,
        # 4.8694e-03, 2.7856e-03, 5.3958e-03, 9.1246e-03, 5.9661e-03, 1.0923e-02,
        # 7.5454e-03, 4.2552e-03, 7.7318e-04, 2.5663e-03, 1.7688e-01]).to(random_last_hidden.device)
            
            # if i==1:
            # #     pos_ = torch.where(random_last_hidden[1]>=0.02)
            #     print(pos_)
                # print(random_last_hidden[1,pos_])
            #     # print(random_last_hidden[0,67])
            #     sparse_mask = torch.abs(random_last_hidden) >= 0.1
            #     print(torch.where(sparse_mask))
                # for k in pos_:
                #     random_last_hidden[0,k]=random_last_hidden[1,k]
                # random_last_hidden[0,:1000] = random_last_hidden[1,:1000]
                # random_last_hidden[0,3000:4000] = random_last_hidden[1,3000:4000]
                # random_last_hidden[0,4000:5000] = random_last_hidden[1,4000:5000]
                # random_last_hidden[0,5000:15000] = random_last_hidden[1,5000:15000]
                # random_last_hidden[0,10000:15000] = random_last_hidden[1,10000:15000]
                # a = random_last_hidden[:, 6801]
                # random_last_hidden[:, 6801] = random_last_hidden[:, 10768]
                # random_last_hidden[:, 6801] = 0

                # a = random_last_hidden[1]
                # random_last_hidden[1] = random_last_hidden[0]
                # random_last_hidden[0] = 0
                # random_last_hidden = random_last_hidden * 0
                # random_last_hidden[:, 847] = 0.06

                # a = encoder_outputs[2]
                # encoder_outputs[2] = encoder_outputs[1]
                # encoder_outputs[1] = 0
                
            #     a = random_last_hidden[3, 4591]
            #     random_last_hidden[3, 4591] = random_last_hidden[3, 13141]
            #     random_last_hidden[3, 13141] = a
            #     random_last_hidden[sparse_mask] = 0
            # if i == 2:
            #     # print("now:", random_last_hidden.dtype)
            #     torch.save(random_last_hidden, 'nl_step1_embedding.pt')
            # random_last_hidden = self.projection_mlp1(random_last_hidden)
            
            
            # if test_mode:
            #     x = random_last_hidden[:, -self.random_dim:]
            #     # print("x", x)
            #     # print(x.size(), mean.size())
            #     mean_loss = (x - mean) ** 2 
            #     std = torch.exp(0.5 * log_var)  # 注意取平方根，因为 log(var) = 2*log(std)
            #     # print("mean", mean)
            #     # print("std", std)
            #     # print("mean loss", torch.sqrt(mean_loss.mean()))
            #     # exit()
            #     # 采用重参数化技巧从 N(mean, std^2) 中采样
            #     epsilon = torch.randn_like(mean, device=device)
            #     sample = mean + epsilon * 0.01 # (batch_size,)
            #     random_last_hidden[:, -self.random_dim:] = sample
            # # torch.save(random_last_hidden, f'nl_step{i}_embedding.pt')
            
            # # print("out", self.projection_mlp[0](random_last_hidden)[:, 408])
            # projected_hidden = self.projection_mlp2(random_last_hidden)
            # # print("third random_last: ", projected_hidden)

            # neuron_matrix[:, -self.neuron_dim_r:, :] = projected_hidden.reshape(batch_size, self.neuron_dim_r, -1)

            # neuron_matrix[:, -self.neuron_dim_r:, :] = random_last_hidden.reshape(batch_size, self.neuron_dim_r, -1)
            
            # if i == 1:
            #     neuron_matrix[:, self.neuron_dim_t: self.neuron_dim_t+self.neuron_dim_s] = 0
            # if i == 0 or i ==1:
            #     print(neuron_matrix[:, self.neuron_dim_t: self.neuron_dim_t+self.neuron_dim_s])
            if i == 0: # >-1 before 12.2
                thinking_input, pad_positions = self.concat_neuron_text(encoder_outputs, attention_mask, neuron_matrix)
                # thinking_input, _ = self.concat_neuron_text(neuron_matrix, extra_attention_mask, encoder_outputs)
                # pad_positions = torch.zeros(batch_size, dtype=torch.long, device=device)
            
                # # Pass to Thinking Model
                output = self.think_model(inputs_embeds=thinking_input, attention_mask=updated_attention_mask).last_hidden_state  # (batch, seq_len + 201, dim)

                #thinking_input, pad_positions = neuron_matrix, torch.zeros(batch_size, dtype=torch.long).to(device)

                # Pass to Thinking Model
                #output = self.think_model(inputs_embeds=thinking_input, attention_mask=extra_attention_mask).last_hidden_state  # (batch, seq_len + 201, dim)
            
                # Extract the Extra Matrix Part for Next Iteration
                batch_indices = torch.arange(attention_mask.size(0), device=device).unsqueeze(1)
                seq_indices = pad_positions.unsqueeze(1) + torch.arange(neuron_matrix.size(1), device=device).unsqueeze(0)  # (batch, decode_len)
                neuron_matrix = output[batch_indices, seq_indices] 
            else:
                neuron_matrix = self.think_model(inputs_embeds=neuron_matrix, attention_mask=extra_attention_mask).last_hidden_state  # (batch, seq_len + 201, dim)
            
            # Restore the fixed columns
            # neuron_matrix[:, :, :50] = fixed_columns  # Ensure the first 50 columns remain fixed
            part1 = neuron_matrix[:, self.neuron_dim_t:self.neuron_dim_t + self.neuron_dim_s, :]  # 取出 s 个
            # output_filtered, _ = self.concat_neuron_text(encoder_outputs, attention_mask, part1)
            neuron_matrixes.append(part1)
            # decode_input = torch.cat(neuron_matrixes, dim=1)
            decode_input = part1
            # print(i, part1)

            if targets != None:
                # 1. Token prediction loss
                # logits = self.decode(output_filtered, updated_decode_mask, targets[i])
                logits = self.decode(decode_input, extra_decode_mask, targets[i])
                if nar:
                    nar_logits = self.decode_with_NAR(decode_input, extra_decode_mask, targets[i], mask_ratio=1)
                    return_logits[i].append([logits, nar_logits])
                else:
                    return_logits[i].append(logits)
                # print("i: ", i, "predict targets: ", targets[i])
                # print("i: ", i, "forward logits: ", logits)        
        # neuron_matrixes[1][:] = 0
        return neuron_matrixes, return_logits

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
        # print(neuron_matrix.size(), attention_mask.size())
        
        # 找到第一个 padding (0值) 的位置
        pad_positions = (attention_mask == 0).int().argmax(dim=1)
    
        # 处理可能没有padding的情况
        pad_positions[attention_mask.sum(dim=1) == seq] = seq  # 如果全是1，说明没有0，应该拼在最后
    
        # 逐个batch进行拼接
        concatenated = []
        for i in range(batch_size):
            split_idx = pad_positions[i].item()
            combined = torch.cat([neuron_matrix[i, :split_idx], text_embeds[i]], dim=0)
            concatenated.append(combined)
    
        # 进行padding，确保batch内对齐
        max_len = max(x.shape[0] for x in concatenated)
        if len(neuron_matrix.size()) == 3:
            padded_result = torch.stack([torch.cat([x, torch.zeros(max_len - x.shape[0], neuron_matrix.size(-1), device=x.device)]) for x in concatenated])
        else:
            padded_result = torch.stack([torch.cat([x, torch.zeros(max_len - x.shape[0], device=x.device)]) for x in concatenated])
        return padded_result, pad_positions

    def decode(self, neuron_matrix, encoder_attention_mask, decode_text_ids, attention_mask=None):
        device = neuron_matrix.device  
        
        # 用 decoder 计算 decode_text_ids 的表征
        decode_text_embeds = self.decoder.model.embed_tokens(decode_text_ids)  # (batch, decode_len, dim)
        # if 'qwen' in self.decoder.config.model_type:
        #     decode_text_embeds = self.decoder.model.embed_tokens(decode_text_ids)  # (batch, decode_len, dim)
        # else:
        #     decode_text_embeds = self.decoder.transformer.wte(decode_text_ids)  # (batch, decode_len, dim)

        # 拼接 neuron_matrix 和 decode_text_embeds
        decoder_input_embeds, pad_positions = self.concat_neuron_text(neuron_matrix, encoder_attention_mask, decode_text_embeds)  # (batch, 100 + decode_len, dim)
        
        decoder_outputs = self.decoder(inputs_embeds=decoder_input_embeds, attention_mask=attention_mask)
        # if attention_mask != None:
        #     decoder_outputs = self.decoder(inputs_embeds=decoder_input_embeds, attention_mask=attention_mask)
        # else:
        #     decoder_outputs = self.decoder(inputs_embeds=decoder_input_embeds)
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
            attention_mask[i, mask_indices] = False  # 0 表示 masked，不让模型看到
            masked_input_ids[i, mask_indices] = self.tokenizer.eos_token_id
        nar_attention_mask, _ = self.concat_neuron_text(encoder_attention_mask, encoder_attention_mask, attention_mask)  # (batch, 100 + decode_len, dim)
        
        # print(masked_input_ids)
        # print(nar_attention_mask)
        nar_logits = self.decode(neuron_matrix, encoder_attention_mask, masked_input_ids, attention_mask=nar_attention_mask)
        return nar_logits
    
    def resort_decode(self, decode_text_ids, batch_size, device):
        previous_inputs = [[] for _ in range(batch_size)]
        if len(decode_text_ids) != 0:
            # 1. 逐个 batch 处理，去除 pad_token_id 并拼接
            for tensor in decode_text_ids:
                for i in range(batch_size):
                    sequence = tensor[i]
                    eos_idx = (sequence == self.tokenizer.eos_token_id).nonzero(as_tuple=True)[0]  # 找到 EOS 位置
                    if len(eos_idx) > 0:
                        eos_idx = eos_idx[0]  # 取第一个 EOS 出现的位置
                        valid_tokens = sequence[:eos_idx]  # 取 EOS 之前的部分
                    else:
                        valid_tokens = sequence  # 如果没有 EOS，就保留整个序列
                    previous_inputs[i] += valid_tokens
        return [torch.tensor(x, device=device) for x in previous_inputs]
    
    def resort_decode_problem(self, decode_text_ids, batch_size, device, input_ids):
        previous_inputs = []
        for b in range(batch_size):
            # 提取当前样本的有效问题部分 (去除 padding)
            p_seq = input_ids[b]
            p_valid = p_seq[p_seq != self.tokenizer.eos_token_id]
            previous_inputs.append(list(p_valid))

        if len(decode_text_ids) != 0:
            for step_idx, tensor in enumerate(decode_text_ids):
                for i in range(batch_size):
                    sequence = tensor[i]
                    # 找到第一个 EOS 位置进行截断
                    eos_indices = (sequence == self.tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
                    if len(eos_indices) > 0:
                        valid_tokens = sequence[:eos_indices[0]]
                    else:
                        valid_tokens = sequence
                    # Step 0 starts with BOS — skip it when appending to avoid duplicate BOS
                    tokens_list = valid_tokens.tolist()
                    if step_idx == 0 and len(tokens_list) > 0 and tokens_list[0] == self.tokenizer.bos_token_id:
                        tokens_list = tokens_list[1:]
                    previous_inputs[i] += tokens_list

        return [torch.tensor(x, device=device) for x in previous_inputs]

    def pad_token(self, decode_text_ids_i, previous_inputs=None):
        device = decode_text_ids_i.device
        batch_size = decode_text_ids_i.size(0)
        # 1. 在 batch 维度内拼接
        if previous_inputs != None:
            concatenated_sentences = [torch.cat([previous_inputs[i], decode_text_ids_i[i]], dim=-1) for i in range(batch_size)]
        else:
            concatenated_sentences = decode_text_ids_i

        # 2. 计算最大拼接长度
        max_new_seq_len = max(s.shape[-1] for s in concatenated_sentences)
        take_logit_pos = torch.tensor([s.shape[-1]-1 for s in concatenated_sentences], device=device)

        # 3. 填充到相同长度
        padded_output = torch.full((batch_size, max_new_seq_len), self.tokenizer.eos_token_id, dtype=decode_text_ids_i.dtype, device=device)
        for i in range(batch_size):
            seq_len = concatenated_sentences[i].shape[-1]
            padded_output[i, :seq_len] = concatenated_sentences[i]
        return padded_output, take_logit_pos

    def generate(self, input_ids, attention_mask, max_length=256, temperature=1.0, top_k=40, target_append=True):
        """
        input_ids: 初始输入文本的 token ids
        attention_mask: 注意力掩码
        max_length: 最多生成多少个新 token
        temperature: 采样温度
        top_k: 限制最高概率的前 k 个 token 进行采样
        """
        device = input_ids.device
        batch_size = input_ids.shape[0]

        # 以 '\n' 作为起始 token
        newline_token_id = self.tokenizer.encode("\n")[0]
        decode_text_ids = [torch.full((batch_size, 1), newline_token_id, dtype=torch.long, device=device) for _ in range(self.num_iterations)]
        total_log_probs = [torch.zeros(batch_size, device=device) for _ in range(self.num_iterations)]  # 存储累计 log 概率
        token_counts = [torch.zeros(batch_size, device=device) for _ in range(self.num_iterations)]  # 记录有效 token 数量
        
        # 追踪哪些样本已生成 eos_token
        eos_reached = [torch.zeros(batch_size, dtype=torch.bool, device=device) for _ in range(self.num_iterations)]

        # 获取thinking结果
        neuron_matrixes, _ = self.forward(input_ids, attention_mask)
        # Update Attention Mask
        extra_attention_mask = torch.ones((batch_size, self.neuron_dim_s), dtype=torch.long, device=device)  # (batch, 100)
        updated_attention_mask, _ = self.concat_neuron_text(attention_mask, attention_mask, extra_attention_mask)  # (batch, seq_len + 100)
        
        # 生成循环
        skip_i = []
        for i in range(self.num_iterations):
            if target_append:
                previous_inputs = self.resort_decode(decode_text_ids[:i], batch_size, device)
            else:
                previous_inputs = None
            # print(i, previous_inputs)
            for _ in range(max_length):
                current_inputs, take_logit_pos = self.pad_token(decode_text_ids[i], previous_inputs)
                if i in skip_i or current_inputs.size(-1)+neuron_matrixes[i].size(1)>=924:
                    break
                logits = self.decode(neuron_matrixes[i], updated_attention_mask, current_inputs)
                logits = logits[torch.arange(batch_size), take_logit_pos].squeeze(1) # 取take_logit_pos位置token 的 logits
                
                # 采样：Top-K + Temperature
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
                
                active_mask = ~eos_reached[i]  # 只对尚未生成 EOS 的样本进行更新
                token_log_probs = torch.log(torch.gather(probs, 1, next_token).squeeze(1) + 1e-9)  # 避免 log(0)
                total_log_probs[i][active_mask] += token_log_probs[active_mask]
                token_counts[i][active_mask] += 1  # 仅增加未生成 EOS 的样本的 token 计数

                decode_text_ids[i] = torch.cat([decode_text_ids[i], next_token], dim=1)  # 追加新 token

                # 如果 batch 内的所有样本都生成了 `eos_token_id`，提前终止
                if torch.all(next_token == self.tokenizer.eos_token_id):
                    skip_i.append(i)
                
                # 确保 eos_reached 一旦为 True，就不再变回 False
                eos_reached[i] |= (next_token == self.tokenizer.eos_token_id).squeeze(1)
        
        # 计算 Perplexity
        avg_log_prob = sum(total_log_probs) / sum(token_counts)
        perplexities = torch.exp(-avg_log_prob)

        return decode_text_ids, perplexities  # 返回完整生成的 token 序列, num_iterations长度的list, 每一个元素batch x dec_len

    def generate_with_answer(self, input_ids, attention_mask, targets, max_length=256, temperature=1.0, top_k=40, target_append=True):
        """
        input_ids: 初始输入文本的 token ids
        attention_mask: 注意力掩码
        targets: ground_truth答案
        max_length: 最多生成多少个新 token
        temperature: 采样温度
        top_k: 限制最高概率的前 k 个 token 进行采样
        """
        device = input_ids.device
        batch_size = input_ids.shape[0]

        # Step 0 starts with BOS, subsequent steps start with \n, consistent with training
        bos_token_id = self.tokenizer.bos_token_id
        newline_token_id = self.tokenizer.encode("\n", add_special_tokens=False)[0]
        decode_text_ids = [
            torch.full((batch_size, 1), bos_token_id if i == 0 else newline_token_id, dtype=torch.long, device=device)
            for i in range(self.num_iterations)
        ]
        total_log_probs = [torch.zeros(batch_size, device=device) for _ in range(self.num_iterations)]
        token_counts = [torch.zeros(batch_size, device=device) for _ in range(self.num_iterations)]

        # 追踪哪些样本已生成 eos_token
        eos_reached = [torch.zeros(batch_size, dtype=torch.bool, device=device) for _ in range(self.num_iterations)]

        # 获取thinking结果
        neuron_matrixes, _ = self.forward(input_ids, attention_mask, targets, test_mode=True)
        # print("neuron: ", neuron_matrixes[0])
        # Update Attention Mask
        # extra_attention_mask = torch.ones((batch_size, self.neuron_dim_s), dtype=torch.long, device=device)  # (batch, 100)
        # updated_attention_mask, _ = self.concat_neuron_text(attention_mask, attention_mask, extra_attention_mask)  # (batch, seq_len + 100)
        
        # 生成循环
        skip_i = []
        for i in range(self.num_iterations):
            if target_append:
                previous_inputs = self.resort_decode_problem(decode_text_ids[:i], batch_size, device, input_ids)
            else:
                previous_inputs = None
            extra_attention_mask = torch.ones((batch_size, self.neuron_dim_s *(1)), dtype=torch.long, device=device)  # (batch, 100)
            # decode_inputs = torch.cat(neuron_matrixes[:i+1], dim=1)
            decode_inputs = neuron_matrixes[i]
            for _ in range(max_length):
                current_inputs, take_logit_pos = self.pad_token(decode_text_ids[i], previous_inputs)
                if i in skip_i or current_inputs.size(-1)+neuron_matrixes[i].size(1)>=924:
                    break
                logits = self.decode(decode_inputs, extra_attention_mask, current_inputs)
                # print("i: ", "test targets", current_inputs)
                logits = logits[torch.arange(batch_size), take_logit_pos].squeeze(1) # 取take_logit_pos位置token 的 logits
                # print("i: ", i, "test logits: ", logits)
                
                # 采样：Top-K + Temperature
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
                
                active_mask = ~eos_reached[i]  # 只对尚未生成 EOS 的样本进行更新
                token_log_probs = torch.log(torch.gather(probs, 1, next_token).squeeze(1) + 1e-9)  # 避免 log(0)
                total_log_probs[i][active_mask] += token_log_probs[active_mask]
                token_counts[i][active_mask] += 1  # 仅增加未生成 EOS 的样本的 token 计数

                decode_text_ids[i] = torch.cat([decode_text_ids[i], next_token], dim=1)  # 追加新 token

                # 如果 batch 内的所有样本都生成了 `eos_token_id`，提前终止
                if torch.all(next_token == self.tokenizer.eos_token_id):
                    skip_i.append(i)
                
                # 确保 eos_reached 一旦为 True，就不再变回 False
                eos_reached[i] |= (next_token == self.tokenizer.eos_token_id).squeeze(1)
        
        # 计算 Perplexity
        avg_log_prob = sum(total_log_probs) / sum(token_counts)
        perplexities = torch.exp(-avg_log_prob)

        return decode_text_ids, perplexities  # 返回完整生成的 token 序列, num_iterations长度的list, 每一个元素batch x dec_len
    
    def generate_with_answer_parallel(self, input_ids, attention_mask, targets, max_length=256, temperature=1.0, top_k=40, target_append=True):
        """
        input_ids: 初始输入文本的 token ids
        attention_mask: 注意力掩码
        targets: ground_truth答案
        max_length: 最多生成多少个新 token
        temperature: 采样温度
        top_k: 限制最高概率的前 k 个 token 进行采样
        """
        device = input_ids.device
        batch_size = input_ids.shape[0]

        # 以 '\n' 作为起始 token
        newline_token_id = self.tokenizer.encode("\n")[0]
        decode_text_ids = torch.full((self.num_iterations*batch_size, 1), newline_token_id, dtype=torch.long, device=device)
        # decode_text_ids = [t[:, :1].to(dtype=torch.long, device=device).clone() for t in targets]
        total_log_probs = torch.zeros(self.num_iterations*batch_size, device=device)   # 存储累计 log 概率
        token_counts = torch.zeros(self.num_iterations*batch_size, device=device)  # 记录有效 token 数量
        
        # 追踪哪些样本已生成 eos_token
        eos_reached = torch.zeros(self.num_iterations*batch_size, dtype=torch.bool, device=device)

        # 获取thinking结果
        neuron_matrixes, _, _ = self.forward(input_ids, attention_mask, targets, test_mode=False)
        # print("neuron: ", neuron_matrixes[0])
        # Update Attention Mask
        extra_attention_mask = torch.ones((batch_size, self.neuron_dim_s), dtype=torch.long, device=device)  # (batch, 100)

        neuron_matrixes = torch.stack(neuron_matrixes, dim=0)      # (num_iter, batch, seq, dim)
        extra_attention_mask = extra_attention_mask.unsqueeze(0).expand(self.num_iterations, -1, -1)  # (num_iter, batch, seq)
        
        # 拼到一起
        neuron_matrixes = neuron_matrixes.reshape(self.num_iterations*batch_size, -1, neuron_matrixes.size(-1))
        extra_attention_mask = extra_attention_mask.reshape(self.num_iterations*batch_size, -1)
        
        # 生成循环
        skip_i = []
        for _ in range(max_length):
            current_inputs, take_logit_pos = self.pad_token(decode_text_ids, None)
            if current_inputs.size(-1)+neuron_matrixes.size(1)>=356:
                break
            logits = self.decode(neuron_matrixes, extra_attention_mask, current_inputs)
            # print("i: ", "test targets", current_inputs)
            logits = logits[torch.arange(self.num_iterations*batch_size), take_logit_pos].squeeze(1) # 取take_logit_pos位置token 的 logits
            # print("i: ", i, "test logits: ", logits)
                
            # 采样：Top-K + Temperature
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
                
            active_mask = ~eos_reached  # 只对尚未生成 EOS 的样本进行更新
            token_log_probs = torch.log(torch.gather(probs, 1, next_token).squeeze(1) + 1e-9)  # 避免 log(0)
            total_log_probs[active_mask] += token_log_probs[active_mask]
            token_counts[active_mask] += 1  # 仅增加未生成 EOS 的样本的 token 计数

            decode_text_ids = torch.cat([decode_text_ids, next_token], dim=1)  # 追加新 token

            # 如果 batch 内的所有样本都生成了 `eos_token_id`，提前终止
            if torch.all(next_token == self.tokenizer.eos_token_id):
                break
                
            # 确保 eos_reached 一旦为 True，就不再变回 False
            eos_reached |= (next_token == self.tokenizer.eos_token_id).squeeze(1)
        
        # 计算 Perplexity
        avg_log_prob = sum(total_log_probs) / sum(token_counts)
        perplexities = torch.exp(-avg_log_prob)

        decode_text_ids = decode_text_ids.view(self.num_iterations, batch_size, -1)
        decode_text_ids = [decode_text_ids[i] for i in range(self.num_iterations)]

        return decode_text_ids, perplexities  # 返回完整生成的 token 序列, num_iterations长度的list, 每一个元素batch x dec_len

    def generate_with_answer_nar(self, input_ids, attention_mask, targets, max_length=256, temperature=1.0, top_k=40, num_refinement_steps=1, target_append=False):
        """
        使用非自回归（Non-Autoregressive）解码和迭代优化来生成文本。
    
        Args:
            input_ids (torch.Tensor): 初始输入文本的 token ids。
            attention_mask (torch.Tensor): 注意力掩码。
            targets (list): ground_truth 答案，主要用于获取 neuron_matrix。
            max_length (int): 生成序列的最大长度。
            num_refinement_steps (int): 非自回归生成的迭代优化步数。

        Returns:
            list: 包含生成 token 序列张量的列表。
            None: Perplexity 不适用于 NAR 模型，返回 None。
        """
        device = input_ids.device
        batch_size = input_ids.shape[0]

        # 1. 和原版一样，首先获取 "thinking" 过程的中间表征 (neuron_matrix)
        neuron_matrixes, _, _ = self.forward(input_ids, attention_mask, targets, test_mode=False)
        extra_attention_mask = torch.ones((batch_size, self.neuron_dim_s), dtype=torch.long, device=device)

        # 2. 获取 tokenizer 的特殊 token ID
        eos_token_id = self.tokenizer.eos_token_id

        final_generated_sequences = []
        # 外层循环保持不变，为 N 次不同的生成任务 (num_iterations)
        for i in range(self.num_iterations):
            fixed_generate_length = targets[i].size(-1)
            # ========== 非自回归生成的核心逻辑开始 ==========

            # 3. 初始化：创建一个填满 [MASK] 的序列作为起点
            generated_ids = torch.full((batch_size, fixed_generate_length), eos_token_id, dtype=torch.long, device=device)

            # 4. 迭代优化循环 (Mask-Predict)
            for step in range(num_refinement_steps):
                # 调用解码器，一次性获得所有位置的 logits
                # logits shape: (batch_size, max_length, vocab_size)
                is_mask = (generated_ids != eos_token_id)
                nar_attention_mask, _ = self.concat_neuron_text(extra_attention_mask, extra_attention_mask, is_mask)  # (batch, 100 + decode_len, dim)
                logits = self.decode_with_NAR(neuron_matrixes[i], extra_attention_mask, generated_ids, mask_ratio=0, attention_mask=is_mask)

                # 贪心策略：直接选择每个位置概率最高的 token
                predicted_ids = torch.argmax(logits, dim=-1)

                if step == num_refinement_steps - 1:
                    generated_ids = predicted_ids
                    break

                # a. 计算置信度 (softmax 后取对应 token 的概率)
                probs = torch.softmax(logits, dim=-1)
                confidence_scores = torch.gather(probs, 2, predicted_ids.unsqueeze(-1)).squeeze(-1)

                # b. 根据余弦调度计算本次迭代需要遮盖的 token 数量
                mask_ratio = np.cos((step / num_refinement_steps) * (np.pi / 2))
                num_to_mask = int(mask_ratio * fixed_generate_length)

                # c. 找到置信度最低的 token 的索引
                low_confidence_indices = torch.topk(confidence_scores, k=num_to_mask, dim=1, largest=False).indices

                # d. 将当前预测结果中置信度低的位置重新设置为 [MASK]
                generated_ids = predicted_ids.clone()
                generated_ids.scatter_(1, low_confidence_indices, eos_token_id)

            # ========== 非自回归生成的核心逻辑结束 ==========
            final_generated_sequences.append(generated_ids)

        # Perplexity 计算不适用于此模型，返回 None
        return final_generated_sequences, None
    
    def sample_ratio(self, min_range=0.6, max_range=1):
        # --- 1. 设置参数 ---
        # # Beta分布的形状参数 (alpha 和 beta)
        # # alpha > beta 会使分布偏向1
        # # alpha 和 beta 的差值越大，偏向性越强
        alpha_param = 10  # alpha 参数，设为较大值
        beta_param = 2    # beta 参数，设为较小值

        # --- 2. 生成符合Beta分布的随机数 [0, 1] ---

        # 使用 beta.rvs() 函数生成随机变量 (Random Variates)
        # a=alpha, b=beta, size=样本数量
        random_values_0_1 = beta.rvs(a=alpha_param, b=beta_param, size=1)[0]

        # --- 3. 将 [0, 1] 区间的数映射到 [60%, 100%] ---

        # 使用线性变换公式: y = min + x * (max - min)
        final_value = min_range + random_values_0_1 * (max_range - min_range)
        return final_value