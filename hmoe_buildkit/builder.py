import os
import copy
import torch
from typing import List
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
)
from .model import HMOE, HMOEConfig, register


register()


class HMOEBuilder:

    def __init__(
        self,
        model_names: List[str],
        experts_per_seq: int,
        torch_dtype: str = "bfloat16",
        attn_implementation: str = "flash_attention_2",
        trust_remote_code: bool = True,
    ):

        self.model_names = model_names
        self.experts_per_seq = experts_per_seq
        self.torch_dtype = torch_dtype
        self.attn_implementation = attn_implementation
        self.trust_remote_code = trust_remote_code

        self.all_tokenizers = [
            AutoTokenizer.from_pretrained(
                name, 
                use_fast=True, 
                trust_remote_code=self.trust_remote_code
            )
            for name in model_names
        ]

        is_vlms = [self.is_vlm(name) for name in self.model_names]
        assert sum(is_vlms) <= 1, "At most 1 VLM can be used to build a HMOE"
        self.vlm_idx = is_vlms.index(True) if True in is_vlms else -1

        common_args = dict(
            torch_dtype=self.torch_dtype,
            attn_implementation=self.attn_implementation,
            trust_remote_code=self.trust_remote_code,
        )
        self.all_causallms = [
            AutoModelForVision2Seq.from_pretrained(
                name,
                **common_args
            ) 
            if self.vlm_idx == idx
            else AutoModelForCausalLM.from_pretrained(
                name,
                **common_args
            )
            for idx, name in enumerate(self.model_names)
        ]


    @staticmethod
    def is_vlm(name: str) -> bool:
        try:
            proc = AutoProcessor.from_pretrained(name)
            return (
                hasattr(proc, "tokenizer")
                and hasattr(proc, "image_processor") or hasattr(proc, "feature_extractor")
            )
        except Exception:
            return False


    def mix_tokenizers(self):

        if self.vlm_idx >= 0:
            base = copy.deepcopy(self.all_tokenizers[self.vlm_idx])
            processor = AutoProcessor.from_pretrained(
                self.model_names[self.vlm_idx],
                trust_remote_code=self.trust_remote_code,
                use_fast=True,
            )
            processor.tokenizer = base
        else:
            base = copy.deepcopy(self.all_tokenizers[0])
            processor = base

        all_tokens = {t for tok in self.all_tokenizers for t in tok.get_vocab()}
        new_tokens = list(all_tokens - set(base.get_vocab()))
        if new_tokens:
            base.add_tokens(new_tokens)

        # specials = {}
        # for tok in self.all_tokenizers:
        #     for k, v in tok.special_tokens_map.items():
        #         if v not in base.all_special_tokens:
        #             specials.setdefault(k, v)
        # if specials:
        #     base.add_special_tokens(specials)

        return processor

    
    def split_modules(self):

        experts = [
            copy.deepcopy(m.model.text_model)
            if self.vlm_idx==idx
            else copy.deepcopy(m.model)
            for idx, m in enumerate(self.all_causallms)
        ]
        for m in experts:
            del m.embed_tokens

        if self.vlm_idx >= 0:
            hidden_size = self.all_causallms[self.vlm_idx].config.text_config.hidden_size
            vision_config = self.all_causallms[self.vlm_idx].config.vision_config.to_dict()
            vision_encoder = copy.deepcopy(self.all_causallms[self.vlm_idx].model) if self.vlm_idx >= 0 else None
            del vision_encoder.text_model
        else:
            hidden_size = self.all_causallms[0].config.hidden_size
            vision_config = None
            vision_encoder = None

        return experts, hidden_size, vision_config, vision_encoder


    @staticmethod
    def _average_weights(src_embs, tgt_emb, vocabs, processor):
        vocab = processor.tokenizer if hasattr(processor, "tokenizer") else processor
        with torch.no_grad():
            for tok, tid in vocab.get_vocab().items():
                accum = torch.zeros_like(tgt_emb.weight[tid])
                cnt = 0
                for v, emb in zip(vocabs, src_embs):
                    if tok in v:
                        sid = v[tok]
                        d = min(emb.weight.size(1), accum.size(0))
                        accum[:d] += emb.weight[sid][:d].to(accum)
                        cnt += 1
                if cnt:
                    tgt_emb.weight[tid] = accum / cnt


    def build(self):

        tokenizer = self.mix_tokenizers()

        experts, hidden_size, vision_config, vision_encoder = self.split_modules()

        config = HMOEConfig(
            vocab_size=len(tokenizer.tokenizer if self.vlm_idx >= 0 else tokenizer),
            hidden_size=hidden_size,
            num_experts=len(experts),
            experts_per_seq=self.experts_per_seq,
            vision_config=vision_config,
            torch_dtype=self.torch_dtype,
            attn_implementation=self.attn_implementation,
        )
        model = HMOE(config, experts, vision_encoder=vision_encoder)

        self._average_weights(
            [m.get_input_embeddings() for m in self.all_causallms],
            model.embed_tokens,
            [t.get_vocab() for t in self.all_tokenizers],
            tokenizer,
        )
        self._average_weights(
            [m.get_output_embeddings() for m in self.all_causallms],
            model.lm_head,
            [t.get_vocab() for t in self.all_tokenizers],
            tokenizer,
        )

        del self.all_tokenizers, self.all_causallms

        return tokenizer, model