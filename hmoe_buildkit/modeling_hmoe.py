import copy
import torch
import transformers
from torch import nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import PreTrainedModel, PretrainedConfig, GenerationMixin


def register():
    transformers.AutoConfig.register("hmoe", HMOEConfig)
    transformers.AutoModelForCausalLM.register(HMOEConfig, HMOE)


class HMOEConfig(PretrainedConfig):
    model_type = "hmoe"

    def __init__(
        self,
        model_names: List[str] = None,
        experts_per_seq: int = None,
        vocab_size: int = None,
        torch_dtype: str = 'bfloat16',
        router_aux_loss_coef: float = 1e-2,
        **kwargs,
    ):
        if model_names is not None:
            hidden_size, num_experts = self._derive_meta(model_names)
        else:
            hidden_size = num_experts = 0

        super().__init__(**kwargs)
        self.model_names = model_names or []
        self.experts_per_seq = experts_per_seq
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.torch_dtype = torch_dtype
        self.router_aux_loss_coef = router_aux_loss_coef

    @staticmethod
    def _is_vlm(name: str) -> bool:
        try:
            proc = transformers.AutoProcessor.from_pretrained(name, trust_remote_code=True)
            return hasattr(proc, "tokenizer") and (
                hasattr(proc, "image_processor") or hasattr(proc, "feature_extractor")
            )
        except Exception:
            return False

    @classmethod
    def _derive_meta(cls, model_names: List[str]):
        is_vlms = [cls._is_vlm(n) for n in model_names]
        vlm_idx = next((i for i, v in enumerate(is_vlms) if v), -1)
        if vlm_idx >= 0:
            cfg = transformers.AutoConfig.from_pretrained(model_names[vlm_idx])
            hidden_size = getattr(cfg, "text_config", cfg).hidden_size
        else:
            cfg = transformers.AutoConfig.from_pretrained(model_names[0])
            hidden_size = cfg.hidden_size
        return hidden_size, len(model_names)


class Expert(nn.Module):
    def __init__(self, hidden_size: int, model: nn.Module):
        super().__init__()
        self.in_proj = (
            nn.Linear(hidden_size, model.config.hidden_size, bias=False, dtype=model.dtype)
            if hidden_size != model.config.hidden_size
            else None
        )
        self.model = model
        self.out_proj = (
            nn.Linear(model.config.hidden_size, hidden_size, bias=False, dtype=model.dtype)
            if hidden_size != model.config.hidden_size
            else None
        )

    def forward(self, inputs_embeds, attention_mask=None, **unused):
        if self.in_proj is not None:
            inputs_embeds = self.in_proj(inputs_embeds)

        if getattr(self, "gradient_checkpointing", False) and self.training:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs).last_hidden_state
                return custom_forward

            hidden_states = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.model),
                inputs_embeds,
                attention_mask,
                use_reentrant=False,
            )
        else:
            hidden_states = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
            ).last_hidden_state

        if self.out_proj is not None:
            hidden_states = self.out_proj(hidden_states)
        return hidden_states


class HMOE(PreTrainedModel, GenerationMixin):
    config_class = HMOEConfig
    base_model_prefix = "hmoe"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config: HMOEConfig):
        super().__init__(config)

        self.config = config

        expert_models, vision_encoder = self._build_experts_and_vision()
        self.vision_encoder = vision_encoder

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, dtype=expert_models[0].dtype
        )
        self.router = nn.Linear(
            config.hidden_size, config.num_experts, bias=False, dtype=expert_models[0].dtype
        )
        self.experts = nn.ModuleList(
            [Expert(config.hidden_size, m) for m in expert_models]
        )
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False, dtype=expert_models[0].dtype
        )

        self.post_init()

    def _build_experts_and_vision(self):
        names = self.config.model_names
        if not names:
            raise ValueError("model_names is empty")

        is_vlms = [self._is_vlm(n) for n in names]
        vlm_idx = next((i for i, v in enumerate(is_vlms) if v), -1)

        experts = []
        vision_encoder = None
        for idx, name in enumerate(names):
            if idx == vlm_idx:
                vlm = transformers.AutoModelForVision2Seq.from_pretrained(
                    name,
                    torch_dtype=self.config.torch_dtype,
                    attn_implementation='flash_attention_2',
                    trust_remote_code=True,
                )
                del vlm.model.text_model.embed_tokens
                experts.append(copy.deepcopy(vlm.model.text_model))
                del vlm.model.text_model
                vision_encoder = copy.deepcopy(vlm.model)
            else:
                lm = transformers.AutoModelForCausalLM.from_pretrained(
                    name,
                    torch_dtype=self.config.torch_dtype,
                    attn_implementation='flash_attention_2',
                    trust_remote_code=True,
                )
                del lm.model.embed_tokens
                experts.append(copy.deepcopy(lm.model))
        return experts, vision_encoder

    @staticmethod
    def _is_vlm(name: str) -> bool:
        return HMOEConfig._is_vlm(name)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def _init_weights(self, module):
        std = self.config.hidden_size**-0.5
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)

    def tie_weights(self):
        return

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value
        for child in module.children():
            self._set_gradient_checkpointing(child, value)

    def gradient_checkpointing_enable(self, **kwargs):
        self._set_gradient_checkpointing(self, value=True)

    def gradient_checkpointing_disable(self):
        self._set_gradient_checkpointing(self, value=False)

    def _compute_router_probs(self, hidden):
        logits = self.router(hidden)
        probs = F.softmax(logits, dim=-1)
        return logits, probs

    def _topk_routing(self, probs):
        K = self.config.experts_per_seq
        if self.training:
            noise = -torch.empty_like(probs).exponential_().log()
            vals, idx = torch.topk(probs + noise, K)
        else:
            vals, idx = torch.topk(probs, K)
        vals = vals / vals.sum(dim=-1, keepdim=True)
        return vals, idx

    def _load_balancing_loss(self, router_probs, expert_indices):
        E = self.config.num_experts
        B = router_probs.size(0)
        counts = torch.zeros(E, dtype=router_probs.dtype, device=router_probs.device)
        counts.scatter_add_(
            0,
            expert_indices.view(-1),
            torch.ones_like(expert_indices.view(-1), dtype=router_probs.dtype),
        )
        counts = counts / (B * self.config.experts_per_seq)
        return (counts * router_probs.mean(0)).sum() * E

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        **model_kwargs,
    ):
        model_inputs = {"input_ids": input_ids}
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        model_inputs.update(model_kwargs)
        return model_inputs

    def _reorder_cache(self, past_key_values, beam_idx):
        return tuple()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        **kwargs,
    ):
        inputs_embeds = self.embed_tokens(input_ids)

        if pixel_values is not None and self.vision_encoder is not None:
            image_states = self.vision_encoder.get_image_features(
                pixel_values, pixel_attention_mask
            )
            inputs_embeds = self.vision_encoder.inputs_merger(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                image_hidden_states=image_states,
            )

        if attention_mask is not None:
            pooled = (inputs_embeds * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1, keepdim=True)
        else:
            pooled = inputs_embeds.mean(1)

        router_logits, router_probs = self._compute_router_probs(pooled)
        routing_weights, expert_indices = self._topk_routing(router_probs)
        aux_loss = self._load_balancing_loss(router_probs, expert_indices)

        B, S, H = inputs_embeds.size()
        K = self.config.experts_per_seq
        flat_weights = routing_weights.reshape(-1, 1, 1)
        flat_inputs = inputs_embeds.unsqueeze(1).expand(-1, K, -1, -1).reshape(B * K, S, H)
        flat_mask = (
            attention_mask.unsqueeze(1).expand(-1, K, -1).reshape(B * K, S)
            if attention_mask is not None
            else None
        )

        chosen_exp = expert_indices.reshape(-1)
        batch_range = torch.arange(B * K, device=flat_inputs.device)

        final_hidden = torch.zeros_like(inputs_embeds)
        for eid in range(self.config.num_experts):
            mask = chosen_exp == eid
            if not mask.any():
                continue
            out = self.experts[eid](
                flat_inputs[mask], flat_mask[mask] if flat_mask is not None else None
            )
            out = out * flat_weights[mask]
            batch_pos = batch_range[mask] // K
            final_hidden.index_add_(0, batch_pos, out)

        logits = self.lm_head(final_hidden)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            loss += self.config.router_aux_loss_coef * aux_loss

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
        )