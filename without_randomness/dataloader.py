import json
import torch
import random
import re
from torch.utils.data import Dataset

# Data Loader
class ProblemAnswerDataset(Dataset):
    def __init__(self, file_path, tokenizer, num_splits=5, max_length=1024):
        """
        Args:
            file_path (str): Path to the dataset (JSONL file with {"problem": ..., "answer": ...}).
            tokenizer: A tokenizer (e.g., GPT tokenizer) for tokenizing input text.
            max_length (int): Maximum sequence length.
            eos_token_id (int): End-of-sequence token ID.
        """
        self.num_splits = num_splits
        # self.data = self.load_data(file_path)
        # with open(file_path,'r') as f:
        #     self.data = json.load(f)
        self.data = self.load_jsonl(file_path)
        # self.data.sort(key=lambda x: len(x["question"].split(' ')))
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.eos_token_id = tokenizer.eos_token_id
        
    def load_data(self, file_path):
        data = self.load_jsonl(file_path)
        filter_data = []
        for d in data:
            answer = d['answer']
            answer_splits = self.split_answer1(answer)
            if answer_splits[-1] != "" and answer_splits[-2] != "": 
                filter_data.append(d)
        return filter_data
            
    def load_jsonl(self, file):
        data = []
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        return data

    def is_number_like(self, text):
        # 简单判断是否是数字，包括小数、科学计数法等
        return bool(re.fullmatch(r"[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?", text))

    def get_number_mask(self, input_text, max_length=512, truncation=True):
        # 对文本进行 tokenization，保留 token string 和对应的 offset（位置）
        tokens = self.tokenizer.encode_plus(input_text, return_offsets_mapping=True, add_special_tokens=False, max_length=max_length, truncation=truncation)
        input_ids = tokens["input_ids"]
        offsets = tokens["offset_mapping"]
        
        mask = []
        for (start, end) in offsets:
            token_str = input_text[start:end]
            if self.is_number_like(token_str):
                mask.append(1)
            else:
                mask.append(0)
        return input_ids, mask
    
    def extract_last_number(self, text):
        # 匹配整数或小数（包括负数）
        numbers = re.findall(r'-?\d+(?:\.\d+)?', text)
        if numbers:
            return numbers[-1]  # 返回最后一个匹配的数字
        else:
            return None
    
    def split_answer(self, answer):
        answer = answer.replace('####','\n####').replace('\n\n', '\n')
        try:
            sentences = answer.split('\n')
        except:
            sentences = answer.split('. ')

        num_sentences = len(sentences)

        splits = sentences
        
        if num_sentences <= self.num_splits:
            splits = sentences
        else:
            # 计算每个 chunk 的大致大小
            avg_chunk_size = (num_sentences - 1) / (self.num_splits - 1)
            splits = []
            start_idx = 0

            for i in range(self.num_splits-1):
                # 计算当前 chunk 的结束索引
                end_idx = round((i + 1) * avg_chunk_size)  # 四舍五入取整，确保所有句子都分配
                splits.append(" ".join(sentences[start_idx:end_idx]).strip())  # 组合句子
                start_idx = end_idx  # 更新索引
            splits.append(sentences[-1])

        # if len(splits) > 1 and random.random() < 1:
        #     merge_idx = random.randint(0, len(splits) - 2)  # 随机选择一个位置合并 i 和 i+1
        #     merged = splits[merge_idx] + "\n" + splits[merge_idx + 1]
        #     new_split = splits[:merge_idx]
        #     new_split.append(merged)

        #     # 向前平移：i <- i+1 <- i+2 ...
        #     for j in range(merge_idx+2, len(splits)):
        #         new_split.append(splits[j])
        #     new_split.append("")  # 最后一个填充为换行符
        #     splits = new_split

        # 确保始终有num_splits段
        while len(splits) < self.num_splits:
            splits.append("")  # 如果不够num_splits组，填充空字符串
        return splits
    
    def split_answer_random(self, answer, max_len=16):
        answer = answer.replace('\n\n', '\n').replace('####', '\n####')
        try:
            sentences = answer.split('\n')
        except:
            sentences = answer.split('. ')

        n = len(sentences)
        splits = []
        i = 0
        while i < n and len(splits) < self.num_splits:
            # ✅ 纯随机等概率采样
            k = random.randint(1, max_len)
            # ✅ 防止越界
            k = min(k, n - i)

            merged = "\n".join(sentences[i:i + k])
            labeled = f"[{k}] {merged}"
            splits.append(labeled)

            i += k

        while len(splits) < self.num_splits:
            splits.append("")

        return splits

    def split_solution(self, solution):
        answer_id = random.randint(0, len(solution)-1)
        answer = solution[answer_id].strip()
        answer = answer.replace('\n\n', '\n')
        try:
            sentences = answer.split('\n')
        except:
            sentences = answer.split('. ')

        num_sentences = len(sentences)

        if num_sentences > self.num_splits:
            split_indices = sorted(random.sample(range(1, num_sentences), self.num_splits - 1))
            split_indices.append(num_sentences)  # 加上末尾索引
            start_idx = 0
            splits = []

            for idx in split_indices:
                chunk = " ".join(sentences[start_idx:idx]).strip()
                splits.append(chunk)
                start_idx = idx
        else:
            splits = sentences

        if len(splits) > 1:
            merge_idx = random.randint(0, len(splits) - 2)  # 随机选择一个位置合并 i 和 i+1
            merged = splits[merge_idx] + "\n" + splits[merge_idx + 1]
            new_split = splits[:merge_idx]
            new_split.append(merged)

            # 向前平移：i <- i+1 <- i+2 ...
            for j in range(merge_idx+2, len(splits)):
                new_split.append(splits[j])
            new_split.append("")  # 最后一个填充为换行符
            splits = new_split

        # 确保始终有num_splits段
        while len(splits) < self.num_splits:
            splits.append("")  # 如果不够num_splits组，填充空字符串
        return splits

    def split_answer1(self, answer):
        splits = [answer]
        while len(splits) < self.num_splits:
            splits.append("")  # 如果不够num_splits组，填充空字符串
        return splits

    def split_answer2(self, answer):
        # if type(answer)==list:
        #     return [''.join(answer)]
        # splits = [answer]
        # return splits
        if type(answer)==list:
            sentences = answer
        else:
            answer = answer.replace('\n\n', '\n')
            try:
                sentences = answer.split('\n')
                sentences = sentences[:-2] + [sentences[-2]+'\n'+sentences[-1]]  # 总结数值结果不算一步
            except:
                sentences = answer.split('. ')

        splits = sentences[:self.num_splits]
        # 确保始终有num_splits段
        while len(splits) < self.num_splits:
            splits.append("")  # 如果不够num_splits组，填充空字符串
            
        # splits = sentences[:1]
        # number = self.extract_last_number(splits[0])
        # if number:
        #     splits.append(number)
        # else:
        #     splits.append("")  
        return splits
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        problem = item["question"]
        answer = item["answer"] # final_ans
        if 'solutions' in item:
            solution = item['solutions']

        # Tokenize with instruct template (no BOS, since chat template adds it)
        messages = [{"role": "user", "content": problem}]
        formatted = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        problem_tokens = torch.tensor(
            self.tokenizer.encode(formatted, max_length=256, truncation=True, add_special_tokens=False),
            dtype=torch.long
        )

        if 'solutions' in item:
            answer_splits = self.split_solution(solution)
        else:
            answer_splits = self.split_answer(answer)
            # answer_splits = self.split_answer_random(answer, max_len=16)
        # 生成num_splits个 `targets`
        # Step 0: BOS + text (no \n prefix), Step i>0: \n + text
        # Concatenated result: [BOS](step1)\n(step2)\n(step3)...
        targets = []
        num_masks = []
        for i in range(self.num_splits):
            target_text = answer_splits[i]
            if i == 0:
                # First step: prepend BOS, no \n
                bos_id = self.tokenizer.bos_token_id
                target_ids, number_mask = self.get_number_mask(target_text, max_length=127, truncation=True)
                target_ids = [bos_id] + target_ids
                number_mask = [0] + number_mask
            else:
                # Subsequent steps: \n prefix, no BOS
                target_ids, number_mask = self.get_number_mask("\n" + target_text, max_length=128, truncation=True)
            target_tokens = torch.tensor(target_ids, dtype=torch.long)
            number_mask = torch.tensor(number_mask, dtype=torch.long)
            targets.append(target_tokens)
            num_masks.append(number_mask)

        return {
            "input_ids": problem_tokens,
            "targets": targets,
            "num_masks": num_masks
        }
    
class ProblemAnswerStepDataset(Dataset):
    def __init__(self, file_path, tokenizer, step_end_id, num_splits=5, max_length=1024):
        """
        Args:
            file_path (str): Path to the dataset (JSONL file with {"problem": ..., "answer": ...}).
            tokenizer: A tokenizer (e.g., GPT tokenizer) for tokenizing input text.
            max_length (int): Maximum sequence length.
            eos_token_id (int): End-of-sequence token ID.
        """
        self.num_splits = num_splits
        # self.data = self.load_data(file_path)
        self.data = self.load_jsonl(file_path)
        # self.data.sort(key=lambda x: len(x["question"].split(' ')))
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.eos_token_id = tokenizer.eos_token_id
        self.step_end_id = step_end_id
        
    def load_data(self, file_path):
        data = self.load_jsonl(file_path)
        filter_data = []
        for d in data:
            answer = d['answer']
            answer_splits = self.split_answer1(answer)
            if answer_splits[-1] != "" and answer_splits[-2] != "": 
                filter_data.append(d)
        return filter_data
            
    def load_jsonl(self, file):
        data = []
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        return data

    def is_number_like(self, text):
        # 简单判断是否是数字，包括小数、科学计数法等
        return bool(re.fullmatch(r"[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?", text))

    def get_number_mask(self, input_text, max_length=512, truncation=True):
        # 对文本进行 tokenization，保留 token string 和对应的 offset（位置）
        tokens = self.tokenizer.encode_plus(input_text, return_offsets_mapping=True, add_special_tokens=False, max_length=max_length, truncation=truncation)
        input_ids = tokens["input_ids"]
        offsets = tokens["offset_mapping"]
        
        mask = []
        for (start, end) in offsets:
            token_str = input_text[start:end]
            if self.is_number_like(token_str):
                mask.append(1)
            else:
                mask.append(0)
        return input_ids, mask
    
    def extract_last_number(self, text):
        # 匹配整数或小数（包括负数）
        numbers = re.findall(r'-?\d+(?:\.\d+)?', text)
        if numbers:
            return numbers[-1]  # 返回最后一个匹配的数字
        else:
            return None
    
    def split_answer(self, answer):
        answer = answer.replace('####','\n####').replace('\n\n', '\n')
        try:
            sentences = answer.split('\n')
        except:
            sentences = answer.split('. ')

        num_sentences = len(sentences)

        splits = sentences
        
        if num_sentences <= self.num_splits:
            splits = sentences
        else:
            # 计算每个 chunk 的大致大小
            avg_chunk_size = (num_sentences - 1) / (self.num_splits - 1)
            splits = []
            start_idx = 0

            for i in range(self.num_splits-1):
                # 计算当前 chunk 的结束索引
                end_idx = round((i + 1) * avg_chunk_size)  # 四舍五入取整，确保所有句子都分配
                splits.append(" ".join(sentences[start_idx:end_idx]).strip())  # 组合句子
                start_idx = end_idx  # 更新索引
            splits.append(sentences[-1])

        # if len(splits) > 1 and random.random() < 1:
        #     merge_idx = random.randint(0, len(splits) - 2)  # 随机选择一个位置合并 i 和 i+1
        #     merged = splits[merge_idx] + "\n" + splits[merge_idx + 1]
        #     new_split = splits[:merge_idx]
        #     new_split.append(merged)

        #     # 向前平移：i <- i+1 <- i+2 ...
        #     for j in range(merge_idx+2, len(splits)):
        #         new_split.append(splits[j])
        #     new_split.append("")  # 最后一个填充为换行符
        #     splits = new_split

        # 确保始终有num_splits段
        while len(splits) < self.num_splits:
            splits.append("")  # 如果不够num_splits组，填充空字符串
        return splits
    
    def split_answer_random(self, answer, max_len=16):
        answer = answer.replace('\n\n', '\n').replace('####', '\n####')
        try:
            sentences = answer.split('\n')
        except:
            sentences = answer.split('. ')

        n = len(sentences)
        splits = []
        i = 0
        while i < n and len(splits) < self.num_splits:
            # ✅ 纯随机等概率采样
            k = random.randint(1, max_len)
            # ✅ 防止越界
            k = min(k, n - i)

            merged = "\n".join(sentences[i:i + k])
            labeled = f"[{k}] {merged}"
            splits.append(labeled)

            i += k

        while len(splits) < self.num_splits:
            splits.append("")

        return splits

    def split_solution(self, solution):
        answer_id = random.randint(0, len(solution)-1)
        answer = solution[answer_id].strip()
        answer = answer.replace('\n\n', '\n')
        try:
            sentences = answer.split('\n')
        except:
            sentences = answer.split('. ')

        num_sentences = len(sentences)

        if num_sentences > self.num_splits:
            split_indices = sorted(random.sample(range(1, num_sentences), self.num_splits - 1))
            split_indices.append(num_sentences)  # 加上末尾索引
            start_idx = 0
            splits = []

            for idx in split_indices:
                chunk = " ".join(sentences[start_idx:idx]).strip()
                splits.append(chunk)
                start_idx = idx
        else:
            splits = sentences

        if len(splits) > 1:
            merge_idx = random.randint(0, len(splits) - 2)  # 随机选择一个位置合并 i 和 i+1
            merged = splits[merge_idx] + "\n" + splits[merge_idx + 1]
            new_split = splits[:merge_idx]
            new_split.append(merged)

            # 向前平移：i <- i+1 <- i+2 ...
            for j in range(merge_idx+2, len(splits)):
                new_split.append(splits[j])
            new_split.append("")  # 最后一个填充为换行符
            splits = new_split

        # 确保始终有num_splits段
        while len(splits) < self.num_splits:
            splits.append("")  # 如果不够num_splits组，填充空字符串
        return splits

    def split_answer1(self, answer):
        splits = [answer]
        while len(splits) < self.num_splits:
            splits.append("")  # 如果不够num_splits组，填充空字符串
        return splits

    def split_answer2(self, answer):
        # if type(answer)==list:
        #     return [''.join(answer)]
        # splits = [answer]
        # return splits
        if type(answer)==list:
            sentences = answer
        else:
            answer = answer.replace('\n\n', '\n')
            try:
                sentences = answer.split('\n')
                sentences = sentences[:-2] + [sentences[-2]+'\n'+sentences[-1]]  # 总结数值结果不算一步
            except:
                sentences = answer.split('. ')

        splits = sentences[:self.num_splits]
        # 确保始终有num_splits段
        while len(splits) < self.num_splits:
            splits.append("")  # 如果不够num_splits组，填充空字符串
            
        # splits = sentences[:1]
        # number = self.extract_last_number(splits[0])
        # if number:
        #     splits.append(number)
        # else:
        #     splits.append("")  
        return splits
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        problem = item["question"]
        answer = item["answer"]
        if 'solutions' in item:
            solution = item['solutions']
        
        # Tokenize
        problem_tokens = torch.tensor(self.tokenizer.encode(problem, max_length=256, truncation=True), dtype=torch.long)

        if 'solutions' in item:
            answer_splits = self.split_solution(solution)
        else:
            answer_splits = self.split_answer(answer)
            # answer_splits = self.split_answer_random(answer, max_len=16)
        # 生成num_splits个 `targets`
        last_valid_idx = 0
        for i in range(len(answer_splits) - 1, -1, -1):
            if answer_splits[i].strip() != "":
                last_valid_idx = i
                break
        
        targets = []
        num_masks = []
        for i in range(self.num_splits):
            target_text = answer_splits[i]
            # print(target_text)
            # target_tokens = torch.tensor(self.tokenizer.encode("\n" + target_text, max_length=512, truncation=True), dtype=torch.long)
            target_ids, number_mask = self.get_number_mask("\n" + target_text, max_length=128, truncation=True)
            if i == last_valid_idx:
                target_ids.append(self.eos_token_id)
            else:
                target_ids.append(self.step_end_id)

            target_tokens = torch.tensor(target_ids, dtype=torch.long)
            number_mask = torch.tensor(number_mask, dtype=torch.long)
            targets.append(target_tokens)
            num_masks.append(number_mask)
        
        return {
            "input_ids": problem_tokens,
            "targets": targets,
            "num_masks": num_masks
        }

class CollateFn:
    def __init__(self, pad_token_id, pack_len=512, target_append=True):
        self.pad_token_id = pad_token_id
        self.eos_token_id = pad_token_id
        self.pack_len = pack_len
        self.target_append = target_append
    
    def __call__(self, batch):
        """Collate function for dynamic padding."""
        max_input_len = max(len(item["input_ids"]) for item in batch)
        num_splits = len(batch[0]["targets"])
        eos_token = torch.tensor([self.eos_token_id], dtype=torch.long)

         # 计算 num_splits 个 `targets` 的最大长度
        # max_target_lens = [max(max(len(item["targets"][i]) for item in batch), 256) for i in range(num_splits)]
        if self.target_append:
            max_target_lens = [max(sum([len(item["targets"][j]) for j in range(i+1)]) for item in batch) for i in range(num_splits)]
        else:
            max_target_lens = [max([len(item["targets"][i]) for item in batch]) for i in range(num_splits)]

        # Padding
        input_ids = []
        targets = [[] for _ in range(num_splits)]
        loss_masks = [[] for _ in range(num_splits)]
        number_masks = [[] for _ in range(num_splits)]
        attention_masks = []

        for item in batch:
            pad_len = max_input_len - len(item["input_ids"])
            input_ids.append(torch.cat([item["input_ids"], torch.full((pad_len,), self.pad_token_id, dtype=torch.long)]))
            attention_masks.append(torch.cat([torch.ones(len(item["input_ids"]), dtype=torch.float), torch.zeros(pad_len, dtype=torch.float)]))
            
            previous_steps = []
            for i in range(num_splits):
                if self.target_append:
                    target_pad_len = max_target_lens[i] - sum([len(item["targets"][j]) for j in range(i+1)])
                else:
                    target_pad_len = max_target_lens[i] - len(item["targets"][i])
                if len(previous_steps) == 0 or not self.target_append:
                    targets[i].append(torch.cat([item["targets"][i], eos_token, torch.full((target_pad_len,), self.pad_token_id, dtype=torch.long)]))
                    loss_masks[i].append(torch.cat([torch.ones(len(item["targets"][i]), dtype=torch.float), torch.zeros(target_pad_len+1, dtype=torch.float)]))
                    ## number mask
                    number_masks[i].append(torch.cat([item["num_masks"][i], torch.zeros(target_pad_len+1, dtype=torch.float)]))

                else:
                    targets[i].append(torch.cat([torch.cat(previous_steps), item["targets"][i], eos_token, torch.full((target_pad_len,), self.pad_token_id, dtype=torch.long)]))
                    loss_masks[i].append(torch.cat([torch.zeros(len(torch.cat(previous_steps)), dtype=torch.float), torch.ones(len(item["targets"][i]), dtype=torch.float), torch.zeros(target_pad_len+1, dtype=torch.float)]))
                previous_steps.append(item["targets"][i])
    
        return {
            "input_ids": torch.stack(input_ids),  # (batch, max_input_len)
            "attention_mask": torch.stack(attention_masks),  # (batch, max_input_len)
            "targets": [torch.stack(t) for t in targets],  # 5 个 tensor，每个 (batch, max_target_len)
            "loss_masks": [torch.stack(m) for m in loss_masks],  # 5 个 tensor，每个 (batch, max_target_len)
            # "num_masks": [torch.stack(m) for m in number_masks]
        }

class CollateFn_pro_append:
    def __init__(self, pad_token_id, pack_len=512, target_append=True):
        self.pad_token_id = pad_token_id
        self.eos_token_id = pad_token_id
        self.pack_len = pack_len
        self.target_append = target_append
    
    def __call__(self, batch):
        """Collate function for dynamic padding."""
        max_input_len = max(len(item["input_ids"]) for item in batch)
        num_splits = len(batch[0]["targets"])
        eos_token = torch.tensor([self.eos_token_id], dtype=torch.long)

         # 计算 num_splits 个 `targets` 的最大长度
        # max_target_lens = [max(max(len(item["targets"][i]) for item in batch), 256) for i in range(num_splits)]
        if self.target_append:
            max_target_lens = [max(sum([len(item["targets"][j]) for j in range(i+1)]) + len(item["input_ids"]) for item in batch) for i in range(num_splits)]
        else:
            max_target_lens = [max([len(item["targets"][i]) for item in batch]) for i in range(num_splits)]

        # Padding
        input_ids = []
        targets = [[] for _ in range(num_splits)]
        loss_masks = [[] for _ in range(num_splits)]
        number_masks = [[] for _ in range(num_splits)]
        attention_masks = []

        for item in batch:
            pad_len = max_input_len - len(item["input_ids"])
            input_ids.append(torch.cat([item["input_ids"], torch.full((pad_len,), self.pad_token_id, dtype=torch.long)]))
            attention_masks.append(torch.cat([torch.ones(len(item["input_ids"]), dtype=torch.float), torch.zeros(pad_len, dtype=torch.float)]))
            
            previous_steps = []
            for i in range(num_splits):
                if self.target_append:
                    target_pad_len = max_target_lens[i] - sum([len(item["targets"][j]) for j in range(i+1)]) - len(item["input_ids"])
                else:
                    target_pad_len = max_target_lens[i] - len(item["targets"][i])
                if len(previous_steps) == 0 or not self.target_append:
                    targets[i].append(torch.cat([item["input_ids"], item["targets"][i], eos_token, torch.full((target_pad_len,), self.pad_token_id, dtype=torch.long)]))
                    loss_masks[i].append(torch.cat([torch.zeros(len(item["input_ids"]), dtype=torch.float), torch.ones(len(item["targets"][i]), dtype=torch.float), torch.zeros(target_pad_len+1, dtype=torch.float)]))
                    ## number mask
                    number_masks[i].append(torch.cat([item["num_masks"][i], torch.zeros(target_pad_len+1, dtype=torch.float)]))

                else:
                    targets[i].append(torch.cat([item["input_ids"], torch.cat(previous_steps), item["targets"][i], eos_token, torch.full((target_pad_len,), self.pad_token_id, dtype=torch.long)]))
                    loss_masks[i].append(torch.cat([torch.zeros(len(item["input_ids"]), dtype=torch.float), torch.zeros(len(torch.cat(previous_steps)), dtype=torch.float), torch.ones(len(item["targets"][i]), dtype=torch.float), torch.zeros(target_pad_len+1, dtype=torch.float)]))
                previous_steps.append(item["targets"][i])
    
        return {
            "input_ids": torch.stack(input_ids),  # (batch, max_input_len)
            "attention_mask": torch.stack(attention_masks),  # (batch, max_input_len)
            "targets": [torch.stack(t) for t in targets],  # 5 个 tensor，每个 (batch, max_target_len)
            "loss_masks": [torch.stack(m) for m in loss_masks],  # 5 个 tensor，每个 (batch, max_target_len)
            # "num_masks": [torch.stack(m) for m in number_masks]
        }
    
class CollateFn_step:
    def __init__(self, pad_token_id, pack_len=512, target_append=True):
        self.pad_token_id = pad_token_id
        self.eos_token_id = pad_token_id
        self.pack_len = pack_len
        self.target_append = target_append
    
    def __call__(self, batch):
        """Collate function for dynamic padding."""
        max_input_len = max(len(item["input_ids"]) for item in batch)
        num_splits = len(batch[0]["targets"])
        eos_token = torch.tensor([self.eos_token_id], dtype=torch.long)

         # 计算 num_splits 个 `targets` 的最大长度
        # max_target_lens = [max(max(len(item["targets"][i]) for item in batch), 256) for i in range(num_splits)]
        if self.target_append:
            max_target_lens = [max(sum([len(item["targets"][j]) for j in range(i+1)]) for item in batch) for i in range(num_splits)]
        else:
            max_target_lens = [max([len(item["targets"][i]) for item in batch]) for i in range(num_splits)]

        # Padding
        input_ids = []
        targets = [[] for _ in range(num_splits)]
        loss_masks = [[] for _ in range(num_splits)]
        number_masks = [[] for _ in range(num_splits)]
        attention_masks = []

        for item in batch:
            pad_len = max_input_len - len(item["input_ids"])
            input_ids.append(torch.cat([item["input_ids"], torch.full((pad_len,), self.pad_token_id, dtype=torch.long)]))
            attention_masks.append(torch.cat([torch.ones(len(item["input_ids"]), dtype=torch.float), torch.zeros(pad_len, dtype=torch.float)]))
            
            previous_steps = []
            for i in range(num_splits):
                if self.target_append:
                    target_pad_len = max_target_lens[i] - sum([len(item["targets"][j]) for j in range(i+1)])
                else:
                    target_pad_len = max_target_lens[i] - len(item["targets"][i])
                if len(previous_steps) == 0 or not self.target_append:
                    targets[i].append(torch.cat([item["targets"][i], torch.full((target_pad_len,), self.pad_token_id, dtype=torch.long)]))
                    loss_masks[i].append(torch.cat([torch.ones(len(item["targets"][i])-1, dtype=torch.float), torch.zeros(target_pad_len+1, dtype=torch.float)]))
                    ## number mask
                    number_masks[i].append(torch.cat([item["num_masks"][i], torch.zeros(target_pad_len+1, dtype=torch.float)]))

                else:
                    targets[i].append(torch.cat([torch.cat(previous_steps), item["targets"][i], torch.full((target_pad_len,), self.pad_token_id, dtype=torch.long)]))
                    loss_masks[i].append(torch.cat([torch.zeros(len(torch.cat(previous_steps)), dtype=torch.float), torch.ones(len(item["targets"][i])-1, dtype=torch.float), torch.zeros(target_pad_len+1, dtype=torch.float)]))
                previous_steps.append(item["targets"][i])
    
        return {
            "input_ids": torch.stack(input_ids),  # (batch, max_input_len)
            "attention_mask": torch.stack(attention_masks),  # (batch, max_input_len)
            "targets": [torch.stack(t) for t in targets],  # 5 个 tensor，每个 (batch, max_target_len)
            "loss_masks": [torch.stack(m) for m in loss_masks],  # 5 个 tensor，每个 (batch, max_target_len)
            # "num_masks": [torch.stack(m) for m in number_masks]
        }