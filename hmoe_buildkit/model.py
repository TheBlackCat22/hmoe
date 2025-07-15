import torch
from torch import nn
from typing import List, Any
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    GenerationMixin,
    PreTrainedModel,
    PretrainedConfig,
    AutoModelForCausalLM,
)
from transformers.modeling_outputs import CausalLMOutputWithPast


def register():
    AutoConfig.register("hmoe", HMOEConfig)
    AutoModelForCausalLM.register(HMOEConfig, HMOE)


class HMOEConfig(PretrainedConfig):
    model_type = "hmoe"

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        num_experts=4,
        experts_per_seq=2,
        vision_config=None,
        torch_dtype="bfloat16",
        attn_implementation="flash_attention_2",
        router_aux_loss_coef=1e-2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.experts_per_seq = experts_per_seq
        self.vision_config = vision_config
        self.torch_dtype = torch_dtype
        self.attn_implementation = attn_implementation
        self.router_aux_loss_coef = router_aux_loss_coef


class Expert(nn.Module):
    def __init__(self, hidden_size: int, model: nn.Module):
        super().__init__()
        self.config = model.config
        dtype = next(model.parameters()).dtype
        self.in_proj = nn.Linear(hidden_size, self.config.hidden_size, bias=False, dtype=dtype) if hidden_size!=self.config.hidden_size else None
        self.model = model
        self.out_proj = nn.Linear(self.config.hidden_size, hidden_size, bias=False, dtype=dtype) if hidden_size!=self.config.hidden_size else None
        self.to(dtype)

    def forward(self, inputs_embeds, attention_mask=None, **unused):
        inputs_embeds = self.in_proj(inputs_embeds) if self.in_proj is not None else inputs_embeds
        hidden_states = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask).last_hidden_state
        hidden_states = self.out_proj(hidden_states) if self.out_proj is not None else hidden_states
        return hidden_states


class HMOE(PreTrainedModel, GenerationMixin):
    config_class = HMOEConfig
    base_model_prefix = "hmoe"

    def __init__(
        self,
        config: HMOEConfig,
        expert_models: List[nn.Module],
        vision_encoder=None,
    ):
        super().__init__(config)
        self.config = config
        self._dtype = getattr(torch, config.torch_dtype)

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, dtype=self._dtype)
        self.vision_encoder = vision_encoder
        self.router = nn.Linear(config.hidden_size, config.num_experts, bias=False, dtype=self._dtype)
        self.experts = nn.ModuleList(
            [Expert(config.hidden_size, m) for m in expert_models]
        )
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, dtype=self._dtype)

        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def _compute_router_probs(self, hidden):
        logits = self.router(hidden)
        probs = F.softmax(logits, dim=-1)
        return logits, probs

    def _topk_routing(self, probs):
        if self.training:
            noise = -torch.empty_like(probs).exponential_().log()
            topk_vals, topk_idx = torch.topk(probs + noise, self.config.experts_per_seq)
        else:
            topk_vals, topk_idx = torch.topk(probs, self.config.experts_per_seq)
        topk_vals = topk_vals / topk_vals.sum(dim=-1, keepdim=True)
        return topk_vals, topk_idx

    def _load_balancing_loss(self, router_probs, expert_indices):
        E = self.config.num_experts
        B = router_probs.size(0)
        expert_counts = torch.zeros(E, dtype=router_probs.dtype, device=router_probs.device)
        expert_counts.scatter_add_(
            0,
            expert_indices.view(-1),
            torch.ones_like(expert_indices.view(-1), dtype=router_probs.dtype),
        )
        expert_counts = expert_counts.div(B * self.config.experts_per_seq)
        router_mean = router_probs.mean(dim=0)
        return (expert_counts * router_mean).sum() * E

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        if past_key_values:
            input_ids = input_ids[:, -1:]
        model_inputs = {"input_ids": input_ids}
        model_inputs.update(kwargs)
        return model_inputs

    def _reorder_cache(self, past_key_values, beam_idx):
        return tuple()

    def forward(
        self,
        input_ids,
        attention_mask=None,
        pixel_values=None,
        pixel_attention_mask=None,
        labels=None,
        **kwargs,
    ):
        inputs_embeds = self.embed_tokens(input_ids)

        if pixel_values is not None and self.vision_encoder is not None:
            image_hidden_states = self.vision_encoder.get_image_features(
                pixel_values, 
                pixel_attention_mask
            )
            inputs_embeds = self.vision_encoder.inputs_merger(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                image_hidden_states=image_hidden_states,
            )

        if attention_mask is not None:
            pooled = (inputs_embeds * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1, keepdim=True)
        else:
            pooled = inputs_embeds.mean(1)

        router_logits, router_probs = self._compute_router_probs(pooled)
        routing_weights, expert_indices = self._topk_routing(router_probs)

        aux_loss = self._load_balancing_loss(router_probs, expert_indices)
        self.aux_loss = aux_loss

        B, S, H = inputs_embeds.shape
        K = self.config.experts_per_seq
        flat_inputs = inputs_embeds.unsqueeze(1).expand(-1, K, -1, -1).reshape(B * K, S, H)
        flat_mask = (
            attention_mask.unsqueeze(1).expand(-1, K, -1).reshape(B * K, S)
            if attention_mask is not None
            else None
        )
        flat_weights = routing_weights.reshape(-1, 1, 1)

        final_hidden = torch.zeros_like(inputs_embeds)
        for eid in range(self.config.num_experts):
            mask = expert_indices.eq(eid).any(-1)
            if not mask.any():
                continue
            rows = mask.nonzero(as_tuple=False).squeeze(1)
            cols = expert_indices[mask].eq(eid).nonzero(as_tuple=False)[:, 1]
            flat_idx = rows * K + cols
            out = self.experts[eid](flat_inputs[flat_idx], flat_mask[flat_idx] if flat_mask is not None else None)
            out = out * flat_weights[flat_idx]
            final_hidden.index_add_(0, rows, out)

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
            aux_loss=aux_loss
        )


register()