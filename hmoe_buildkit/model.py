# Need to build huggingface pretrained model with config from scratch
# Necessary for saving and loading using huggingface
import torch
from torch import nn


class HMOE(nn.Module):

    def __init__(self, vocab_size, hidden_size, all_models, vision_encoder=None):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size, dtype=torch.bfloat16)
        self.vision_encoder = vision_encoder
        # Router
        # Project IN
        self.all_models = nn.ModuleList(all_models)
        # Project OUT
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False, dtype=torch.bfloat16)

    def forward(
        self,
        input_ids,
        attention_mask = None,
        pixel_values = None,
        pixel_attention_mask = None
    ):

        inputs_embeds = self.embed_tokens(input_ids)
        if pixel_values is not None:
            image_hidden_states = self.vision_encoder.get_image_features(pixel_values, pixel_attention_mask)
            inputs_embeds = self.vision_encoder.inputs_merger(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                image_hidden_states=image_hidden_states,
            )