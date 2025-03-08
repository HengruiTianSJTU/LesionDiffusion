from open_clip import create_model_from_pretrained, get_tokenizer, add_model_config
from functools import reduce
import torch.nn as nn
import torch, re, einops
import numpy as np

class BiomedCLIP(nn.Module):
    use_text_split = False
    bert_max_length = 512
    def __init__(self, open_clip_config, pretrained, device = "cuda",
                context_length = 256, context_dim = 512):
        super().__init__()
        self.device = device
        self.context_length = context_length
        self.context_dim = context_dim
        self.bert_max_length = 256
        assert self.context_length % self.bert_max_length == 0 or self.context_length < self.bert_max_length
        self.bert_encode_batch = self.context_length // self.bert_max_length
        add_model_config(open_clip_config)
        model, _ = create_model_from_pretrained(model_name='open_clip_config', pretrained=pretrained)
        self.tokenizer = get_tokenizer(model_name='open_clip_config')
        self.model = model.text.to(self.device)
        self.default_input = 'CT'

    @staticmethod
    def token_split(string, max_length = bert_max_length):
        if len(string) < max_length:
            return [string]
        split_pos = [0] + [m.start() for m in re.finditer(r"\\\\|{", string)] + [len(string)]
        split_text = [string[split_pos[i]: split_pos[i+1]] for i in range(len(split_pos)-1)]

        def huffman_grouping(*t):
            if len(t) == 1:
                return t
            pair_len = [len(t[_] + t[_+1]) for _ in range(len(t)-1)]
            if min(pair_len) > max_length:
                return t
            pair_idx = np.argmin(pair_len)
            pair_t = t[pair_idx] + t[pair_idx + 1]
            if pair_idx + 2 < len(t):
                return huffman_grouping(*t[:pair_idx], pair_t, *t[pair_idx+2:])
            return huffman_grouping(*t[:pair_idx], pair_t)

        result_ls = huffman_grouping(*split_text)

        if max([len(_) for _ in result_ls]) > max_length:  # sep by "。"
            split_pos = [0] + [m.start() for m in re.finditer(r"。", string)] + [len(string)]
            split_text = [string[split_pos[i]: split_pos[i+1]] for i in range(len(split_pos)-1)]
            result_ls = huffman_grouping(*split_text)

        return result_ls

    def _merge_text_list(self, *ls):
        ls_ = []
        for l in ls:
            ls_.append(l)
            if not isinstance(l, list):
                assert isinstance(l:= str(l), str), f"got type {type(l)} for {l}, attempted conversion to str failed"
                ls_[-1] = self.token_split(l)
            if len(ls_[-1]) < self.bert_encode_batch:
                ls_[-1].append("")
            if len(ls_[-1]) > self.bert_encode_batch:
                ls_[-1] = l[:self.bert_encode_batch]
        return reduce(lambda x, y: x + y, ls_, [])

    def forward(self, x):
        device = self.device
        input = []
        output = []
        tokenizer = self.tokenizer
        model = self.model
        
        # version for main.DataModuleFromConfig
        max_len = 0
        for i in range(len(x)):
            input = list(x[i].values())
            max_len = max(max_len, len(input))

        for i in range(len(x)):
            input = list(x[i].values())
            padding_needed = max_len - len(input)
            input = input + [''] * padding_needed
            input = tokenizer(input, context_length=self.context_length).to(device)
            output.append(model(input))
        return torch.stack(output)
    
        # version for ldm.data.Torchio_contrast_dataloader.DataModuleFromConfig
        # input = [value[0] for value in x.values()]
        # input = tokenizer(input, context_length=self.context_length).to(device)
        # output=model(input)
        # return output[None]


    def encode(self, text):
        return self(text)
    
    def _process_dict(self, d):
        if not isinstance(d, dict):
            return str(d)
            
        parts = ''
        for _, value in d.items():
            if isinstance(value, dict):
                value_str = self._process_dict(value)
            else:
                value_str = str(value)
            parts += f"{value_str},"
        
        return parts.rstrip(',')

