import copy
import torch
from typing import List
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
)
from .modeling_hmoe import HMOE, HMOEConfig


class HMOEBuilder:
    def __init__(
        self,
        model_names: List[str],
        experts_per_seq: int,
        torch_dtype: str = 'bfloat16',
        router_aux_loss_coef: float = 1e-3,
    ):
        self.model_names = model_names
        self.experts_per_seq = experts_per_seq
        self.torch_dtype = torch_dtype
        self.router_aux_loss_coef = router_aux_loss_coef

        self.all_tokenizers = [
            AutoTokenizer.from_pretrained(name, use_fast=True, trust_remote_code=True)
            for name in model_names
        ]

        is_vlms = [self.is_vlm(name) for name in model_names]
        if sum(is_vlms) > 1:
            raise ValueError("At most one VLM can be used to build an HMOE.")
        self.vlm_idx = is_vlms.index(True) if any(is_vlms) else -1

        common = dict(
            torch_dtype=self.torch_dtype,
            trust_remote_code=True,
        )
        self.all_causallms = [
            AutoModelForVision2Seq.from_pretrained(name, **common)
            if self.vlm_idx == idx
            else AutoModelForCausalLM.from_pretrained(name, **common)
            for idx, name in enumerate(model_names)
        ]

    @staticmethod
    def is_vlm(name: str) -> bool:
        try:
            proc = AutoProcessor.from_pretrained(name, trust_remote_code=True)
            return (
                hasattr(proc, "tokenizer")
                and (hasattr(proc, "image_processor") or hasattr(proc, "feature_extractor"))
            )
        except Exception:
            return False

    def mix_tokenizers(self) -> AutoTokenizer | AutoProcessor:
        if self.vlm_idx >= 0:
            base = copy.deepcopy(self.all_tokenizers[self.vlm_idx])
            processor = AutoProcessor.from_pretrained(
                self.model_names[self.vlm_idx],
                trust_remote_code=True,
                use_fast=True,
            )
            processor.tokenizer = base
        else:
            base = copy.deepcopy(self.all_tokenizers[0])
            processor = base

        all_tokens = {tok for tokz in self.all_tokenizers for tok in tokz.get_vocab()}
        new_tokens = list(all_tokens - set(base.get_vocab()))
        if new_tokens:
            base.add_tokens(new_tokens)

        return processor

    @staticmethod
    def _average_weights(src_embs, tgt_emb, vocabs, processor):
        tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
        with torch.no_grad():
            for tok, tid in tokenizer.get_vocab().items():
                accum = torch.zeros_like(tgt_emb.weight[tid])
                cnt = 0
                for v, emb in zip(vocabs, src_embs):
                    if tok in v:
                        sid = v[tok]
                        d = min(emb.weight.size(1), accum.numel())
                        accum[:d] += emb.weight[sid][:d].to(accum.device, non_blocking=True)
                        cnt += 1
                if cnt:
                    tgt_emb.weight[tid] = accum / cnt
        return tgt_emb

    def build(self):
        tokenizer = self.mix_tokenizers()

        config = HMOEConfig(
            model_names=self.model_names,
            experts_per_seq=self.experts_per_seq,
            vocab_size=len(tokenizer.tokenizer if self.vlm_idx >= 0 else tokenizer),
            torch_dtype=self.torch_dtype,
            router_aux_loss_coef=self.router_aux_loss_coef
        )

        model = HMOE(config)

        model.set_input_embeddings(
            self._average_weights(
                [m.get_input_embeddings() for m in self.all_causallms],
                model.get_input_embeddings(),
                [t.get_vocab() for t in self.all_tokenizers],
                tokenizer,
            )
        )
        model.set_output_embeddings(
            self._average_weights(
                [m.get_output_embeddings() for m in self.all_causallms],
                model.get_output_embeddings(),
                [t.get_vocab() for t in self.all_tokenizers],
                tokenizer,
            )
        )

        del self.all_tokenizers, self.all_causallms
        return tokenizer, model