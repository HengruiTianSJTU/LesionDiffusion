import torch
import torch.nn as nn
from functools import partial, reduce
from einops import rearrange, repeat
import re
import numpy as np
# import clip
# import kornia

from transformers import AutoTokenizer, AutoModel
from ldm.modules.x_transformer import Encoder, TransformerWrapper  # TODO: can we directly rely on lucidrains code and simply add this as a reuirement? --> test


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError



class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=1000, key='class'):
        super().__init__()
        self.key = key
        self.embedding = nn.Embedding(n_classes, embed_dim)

    def forward(self, batch, key=None):
        if key is None:
            key = self.key
        # this is for use in crossattn
        c = batch[key][:, None]
        c = self.embedding(c)
        return c


class TransformerEmbedder(AbstractEncoder):
    """Some transformer encoder layers"""
    def __init__(self, n_embed, n_layer, vocab_size, max_seq_len=77, device="cuda"):
        super().__init__()
        self.device = device
        self.transformer = TransformerWrapper(num_tokens=vocab_size, max_seq_len=max_seq_len,
                                              attn_layers=Encoder(dim=n_embed, depth=n_layer))

    def forward(self, tokens):
        tokens = tokens.to(self.device)  # meh
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, x):
        return self(x)


class BERTTokenizer(AbstractEncoder):
    """ Uses a pretrained BERT tokenizer by huggingface. Vocab size: 30522 (?)"""
    def __init__(self, device="cuda", vq_interface=True, max_length=77):
        super().__init__()
        from transformers import BertTokenizerFast  # TODO: add to reuquirements
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.device = device
        self.vq_interface = vq_interface
        self.max_length = max_length

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        return tokens

    @torch.no_grad()
    def encode(self, text):
        tokens = self(text)
        if not self.vq_interface:
            return tokens
        return None, None, [None, None, tokens]

    def decode(self, text):
        return text


class BERTEmbedder(AbstractEncoder):
    """Uses the BERT tokenizr model and add some transformer encoder layers"""
    def __init__(self, n_embed, n_layer, vocab_size=30522, max_seq_len=77,
                 device="cuda",use_tokenizer=True, embedding_dropout=0.0):
        super().__init__()
        self.use_tknz_fn = use_tokenizer
        if self.use_tknz_fn:
            self.tknz_fn = BERTTokenizer(vq_interface=False, max_length=max_seq_len)
        self.device = device
        self.transformer = TransformerWrapper(num_tokens=vocab_size, max_seq_len=max_seq_len,
                                              attn_layers=Encoder(dim=n_embed, depth=n_layer),
                                              emb_dropout=embedding_dropout)

    def forward(self, text):
        if self.use_tknz_fn:
            tokens = self.tknz_fn(text)#.to(self.device)
        else:
            tokens = text
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, text):
        # output of length 77
        return self(text)


class SpatialRescaler(nn.Module):
    def __init__(self,
                 n_stages=1,
                 method='bilinear',
                 multiplier=0.5,
                 in_channels=3,
                 out_channels=None,
                 bias=False):
        super().__init__()
        self.n_stages = n_stages
        assert self.n_stages >= 0
        assert method in ['nearest','linear','bilinear','trilinear','bicubic','area']
        self.multiplier = multiplier
        self.interpolator = partial(torch.nn.functional.interpolate, mode=method)
        self.remap_output = out_channels is not None
        if self.remap_output:
            print(f'Spatial Rescaler mapping from {in_channels} to {out_channels} channels after resizing.')
            self.channel_mapper = nn.Conv2d(in_channels,out_channels,1,bias=bias)

    def forward(self,x):
        for stage in range(self.n_stages):
            x = self.interpolator(x, scale_factor=self.multiplier)


        if self.remap_output:
            x = self.channel_mapper(x)
        return x

    def encode(self, x):
        return self(x)


class FrozenCLIPTextEmbedder(nn.Module):
    """
    Uses the CLIP transformer encoder for text.
    """
    def __init__(self, version='ViT-L/14', device="cuda", max_length=77, n_repeat=1, normalize=True):
        super().__init__()
        self.model, _ = clip.load(version, jit=False, device="cpu")
        self.device = device
        self.max_length = max_length
        self.n_repeat = n_repeat
        self.normalize = normalize

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        tokens = clip.tokenize(text).to(self.device)
        z = self.model.encode_text(tokens)
        if self.normalize:
            z = z / torch.linalg.norm(z, dim=1, keepdim=True)
        return z

    def encode(self, text):
        z = self(text)
        if z.ndim==2:
            z = z[:, None, :]
        z = repeat(z, 'b 1 d -> b k d', k=self.n_repeat)
        return z


class FrozenClipImageEmbedder(nn.Module):
    """
        Uses the CLIP image encoder.
        """
    def __init__(
            self,
            model,
            jit=False,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            antialias=False,
        ):
        super().__init__()
        self.model, _ = clip.load(name=model, device=device, jit=jit)

        self.antialias = antialias

        self.register_buffer('mean', torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False)
        self.register_buffer('std', torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False)

    def preprocess(self, x):
        # normalize to [0,1]
        x = kornia.geometry.resize(x, (224, 224),
                                   interpolation='bicubic',align_corners=True,
                                   antialias=self.antialias)
        x = (x + 1.) / 2.
        # renormalize according to clip
        x = kornia.enhance.normalize(x, self.mean, self.std)
        return x

    def forward(self, x):
        # x is assumed to be in range [-1,1]
        return self.model.encode_image(self.preprocess(x))


class OneHotCategoricalBCHW(torch.distributions.OneHotCategorical):
    """Like OneHotCategorical, but the probabilities are along dim=1."""

    def __init__(
            self,
            probs=None,
            logits=None,
            validate_args=None):

        if probs is not None and probs.ndim < 2:
            raise ValueError("`probs.ndim` should be at least 2")

        if logits is not None and logits.ndim < 2:
            raise ValueError("`logits.ndim` should be at least 2")

        probs = self.channels_last(probs) if probs is not None else None
        logits = self.channels_last(logits) if logits is not None else None

        super().__init__(probs, logits, validate_args)

    def sample(self, sample_shape=torch.Size()):
        res = super().sample(sample_shape)
        return self.channels_second(res)

    @staticmethod
    def channels_last(arr: torch.Tensor) -> torch.Tensor:
        """Move the channel dimension from dim=1 to dim=-1"""
        dim_order = (0,) + tuple(range(2, arr.ndim)) + (1,)
        return arr.permute(dim_order)

    @staticmethod
    def channels_second(arr: torch.Tensor) -> torch.Tensor:
        """Move the channel dimension from dim=-1 to dim=1"""
        dim_order = (0, arr.ndim - 1) + tuple(range(1, arr.ndim - 1))
        return arr.permute(dim_order)

    def max_prob_sample(self):
        """Sample with maximum probability"""
        num_classes = self.probs.shape[-1]
        res = torch.nn.functional.one_hot(self.probs.argmax(dim=-1), num_classes)
        return self.channels_second(res)

    def prob_sample(self):
        """Sample with probabilities"""
        return self.channels_second(self.probs)

    
class CategoricalDiffusionWrapper(nn.Module):
    def __init__(self, num_classes, sample_scheme="majority"):
        super().__init__()
        self.dummy = nn.Identity()
        self.is_conditional = False
        self.num_classes = num_classes
        self.sample_scheme = sample_scheme
        
    def encode(self, x, c=None):
        x_onehot = nn.functional.one_hot((x * (self.num_classes - 1)).long(), self.num_classes)
        x_onehot = rearrange(x_onehot, "b 1 h w d x -> b x h w d")
        return x_onehot
    
    def decode(self, p, c=None, sample_scheme=None):
        sample_scheme = sample_scheme if sample_scheme is not None else self.sample_scheme
        distrib = OneHotCategoricalBCHW(logits=p)
        if sample_scheme == "majority":
            x = distrib.max_prob_sample()
        elif sample_scheme == "confidence":
            x = distrib.prob_sample()
        else:
            x = distrib.sample()
        return x.argmax(1, keepdim=True)
    
    
class IdentityFirstStage(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Identity()
        
    def encode(self, x, c=None):
        return self.dummy(x)
    
    def decode(self, p, c=None):
        return self.dummy(p)
    
    
class FrozenBERTEmbedder(AbstractEncoder):
    use_text_split = False
    bert_max_length = 512
    def __init__(self, ckpt_path="/ailab/user/dailinrui/data/dependency/bert-ernie-health",
                 device="cuda", freeze=True, max_length=512):
        super().__init__()
        self.device = device
        self.max_length = max_length
        self.bert_max_length = 512
        assert self.max_length % self.bert_max_length == 0 or self.max_length < self.bert_max_length
        self.bert_encode_batch = self.max_length // self.bert_max_length
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt_path, local_files_only=True)
        self.transformer = AutoModel.from_pretrained(ckpt_path, local_files_only=True).to(self.device)
        if freeze:
            self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        #self.train = disabled_train
        for param in self.parameters():
            param.requires_grad = False

    @staticmethod
    def token_split(string, max_length=bert_max_length):
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

    def forward(self, text):
        if isinstance(text, str): text = [text]
        b = len(text)
        if self.use_text_split:
            text = self._merge_text_list(*text)
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.bert_max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        mask = batch_encoding["attention_mask"].to(self.device)
        outputs = self.transformer(input_ids=tokens, attention_mask=mask, return_dict=True)

        z = outputs.last_hidden_state
        z = rearrange(z, "(b x) n l -> b (n x) l", b=b, x=self.bert_encode_batch, n=self.bert_max_length)
        return z

    def encode(self, text):
        return self(text)


class IdentityEncoder(nn.Module):
    def encode(self, tensor):
        return tensor
