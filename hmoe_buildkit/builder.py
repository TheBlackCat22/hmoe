import copy
import torch
from tqdm import tqdm
from transformers import (
    AutoProcessor, 
    AutoTokenizer, 
    PreTrainedTokenizerBase,
    AutoModelForVision2Seq,
    AutoModelForCausalLM,
)
from .model import HMOE



class HMOEBuilder:
    
    def __init__(self, model_names):

        self.model_names = model_names

        print('Loading Tokenizers')
        self.all_tokenizers = [
            AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True, 
                use_fast=True
            ) 
            for model_name in self.model_names
        ]

        is_vlms = [self.is_vlm(model_name) for model_name in self.model_names]
        assert sum(is_vlms) <=1, "At most 1 VLM can be used to build a HMOE"
        self.vlm_idx = is_vlms.index(True) if True in is_vlms else -1
        if self.vlm_idx > -1 :
            print(f'Detected {self.model_names[self.vlm_idx]} is a VLM.')

        print('Loading Models')
        self.all_causallms = [
            AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype='bfloat16',
                attn_implementation='flash_attention_2',
                device_map=0
            ) 
            if ((self.vlm_idx == -1) or (idx != self.vlm_idx))
            else AutoModelForVision2Seq.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype='bfloat16',
                attn_implementation='flash_attention_2',
                device_map=0
            )
            for idx, model_name in enumerate(self.model_names)
        ]


    @staticmethod
    def is_vlm(model_name):
        try:
            processor = AutoProcessor.from_pretrained(model_name)
            has_tokenizer = hasattr(processor, 'tokenizer') and isinstance(processor.tokenizer, PreTrainedTokenizerBase)
            has_image = hasattr(processor, 'image_processor') or hasattr(processor, 'feature_extractor')
            return has_tokenizer and has_image
        except Exception:
            return False

    
    @staticmethod
    def merge_tokenizers(all_tokenizers, vlm_idx):

        base_tokenizer = copy.deepcopy(all_tokenizers[vlm_idx])

        all_tokens = set()
        all_special_tokens = set()
        for tokenizer in all_tokenizers:
            all_tokens.update(tokenizer.get_vocab().keys())
            all_special_tokens.update(tokenizer.all_special_tokens)
    
        new_tokens = list(all_tokens - set(base_tokenizer.get_vocab().keys()) - all_special_tokens)
        base_tokenizer.add_tokens(new_tokens)
    
        special_token_map = {}
        for tokenizer in all_tokenizers:
            for k, v in tokenizer.special_tokens_map.items():
                if v not in base_tokenizer.all_special_tokens and k not in special_token_map:
                    special_token_map[k] = v
        base_tokenizer.add_special_tokens(special_token_map)

        if vlm_idx > -1:
            base_processor = AutoProcessor.from_pretrained(base_tokenizer.name_or_path, use_fast=True)
            base_processor.tokenizer = base_tokenizer
            return base_processor
        else:
            return base_tokenizer

    
    @staticmethod
    def split_modules(all_causallms, vlm_idx):
        all_embed_tokens = [
            copy.deepcopy(causallm.model.embed_tokens)
            if ((vlm_idx == -1) or (idx != vlm_idx))
            else copy.deepcopy(causallm.model.text_model.embed_tokens)
            for idx, causallm in enumerate(all_causallms)
        ]
        all_lm_heads = [
            copy.deepcopy(causallm.lm_head)
            for causallm in all_causallms
        ]

        all_models = [
            copy.deepcopy(causallm.model)
            if ((vlm_idx == -1) or (idx != vlm_idx))
            else copy.deepcopy(causallm.model.text_model)
            for idx, causallm in enumerate(all_causallms)
        ]
        for model in all_models:
            del model.embed_tokens

        if vlm_idx > -1:
            hidden_size = all_causallms[vlm_idx].model.text_model.config.hidden_size
            vision_encoder = copy.deepcopy(all_causallms[vlm_idx].model)
            del vision_encoder.text_model
        else:
            hidden_size = all_causallms[vlm_idx].config.hidden_size
            vision_encoder = None

        return all_embed_tokens, all_lm_heads, all_models, vision_encoder, hidden_size


    @staticmethod
    def build_embed_tokens(all_embed_tokens, all_tokenizers, hmoe_tokenizer, hmoe_model):
        all_vocabs = [tokenizer.get_vocab() for tokenizer in all_tokenizers]
        with torch.no_grad():
            for token, token_id in tqdm(hmoe_tokenizer.get_vocab().items()):
                num_has_token = 0
                final_embedding = torch.zeros_like(hmoe_model.embed_tokens.weight[token_id])
                for vocab, embed_tokens in zip(all_vocabs, all_embed_tokens):
                    if token in vocab:
                        source_token_id = vocab[token]
                        source_embedding = embed_tokens.weight[source_token_id]
                        overlap_dim = min(source_embedding.shape[0], hmoe_model.embed_tokens.embedding_dim)
                        final_embedding[:overlap_dim] += source_embedding[:overlap_dim].to(final_embedding.device)
                        num_has_token += 1
                hmoe_model.embed_tokens.weight[token_id] = final_embedding / num_has_token
        return hmoe_model

    
    @staticmethod
    def build_lm_head(all_lm_heads, all_tokenizers, hmoe_tokenizer, hmoe_model):
        all_vocabs = [tokenizer.get_vocab() for tokenizer in all_tokenizers]
        with torch.no_grad():
            for token, token_id in tqdm(hmoe_tokenizer.get_vocab().items()):
                num_has_token = 0
                final_embedding = torch.zeros_like(hmoe_model.lm_head.weight[token_id])
                for vocab, lm_head in zip(all_vocabs, all_lm_heads):
                    if token in vocab:
                        source_token_id = vocab[token]
                        source_embedding = lm_head.weight[source_token_id]
                        overlap_dim = min(source_embedding.shape[0], hmoe_model.lm_head.in_features)
                        final_embedding[:overlap_dim] += source_embedding[:overlap_dim].to(final_embedding.device)
                        num_has_token += 1
                hmoe_model.lm_head.weight[token_id] = final_embedding / num_has_token
        return hmoe_model


    def build(self):

        print("Merging Tokenizers")
        hmoe_tokenizer = self.merge_tokenizers(self.all_tokenizers, self.vlm_idx)

        print("Preparing HMOE")
        all_embed_tokens, all_lm_heads, all_models, vision_encoder, hidden_size = self.split_modules(self.all_causallms, self.vlm_idx)

        hmoe_model = HMOE(
            vocab_size = len(hmoe_tokenizer.tokenizer) if self.vlm_idx > -1 else len(hmoe_tokenizer),
            hidden_size = hidden_size,
            all_models = all_models,
            vision_encoder = vision_encoder
        )

        print("Mixing Embeddings")
        hmoe_model = self.build_embed_tokens(
            all_embed_tokens, 
            self.all_tokenizers, 
            hmoe_tokenizer.tokenizer if self.vlm_idx > -1 else hmoe_tokenizer, 
            hmoe_model
        )

        print("Mixing LM Head")
        hmoe_model = self.build_lm_head(
            all_lm_heads, 
            self.all_tokenizers, 
            hmoe_tokenizer.tokenizer if self.vlm_idx > -1 else hmoe_tokenizer, 
            hmoe_model
        )

        del self.all_tokenizers, self.all_causallms
        return hmoe_tokenizer, hmoe_model


    @staticmethod
    def save(output_dir, tokenizer, model):
        pass


    @staticmethod
    def load(output_dir):
        pass