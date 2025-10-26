#privit_model.py

# NO NEED TO RUN, JUST REPLACE THE ORIGINAL FILE WITH THIS CODE

from transformers import ViTForImageClassification, ViTConfig, ViTModel
from transformers.models.vit.modeling_vit import ViTLayer, ViTEncoder, ViTAttention, ViTIntermediate, ViTOutput, ViTSelfAttention, ViTSelfOutput
import torch
import math
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from typing import Optional, Union, Tuple
from transformers.modeling_outputs import ImageClassifierOutput, BaseModelOutputWithPooling
import json

class CustomViTIntermediate(ViTIntermediate):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class CustomViTSelfAttention(ViTSelfAttention):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__(config)
        # === FIX APPLIED HERE ===
        # All necessary attributes, including self.dropout, have been restored. My apologies.
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self, hidden_states, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        mixed_query_layer = self.query(hidden_states)
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        return (context_layer, attention_probs) if output_attentions else (context_layer,)

class CustomViTAttention(ViTAttention):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__(config)
        self.attention = CustomViTSelfAttention(config)
        self.output = ViTSelfOutput(config)

    def forward(self, hidden_states, head_mask=None, output_attentions=False):
        self_outputs = self.attention(hidden_states, head_mask, output_attentions)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs

class CustomViTLayer(ViTLayer):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__(config)
        self.attention = CustomViTAttention(config)
        self.intermediate = CustomViTIntermediate(config)
        self.output = ViTOutput(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> torch.Tensor:
        layernorm_output = self.layernorm_before(hidden_states)

        attention_outputs = self.attention(
            layernorm_output,
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = attention_outputs[0]

        hidden_states = attention_output + hidden_states
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)
        layer_output = self.output(layer_output, hidden_states)

        return layer_output

class CustomViTEncoder(ViTEncoder):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__(config)
        self.layer = nn.ModuleList([CustomViTLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        for i, layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states, head_mask, output_attentions)
        return hidden_states

class CustomViTModel(ViTModel):
    def __init__(self, config: ViTConfig, add_pooling_layer: bool = True):
        super().__init__(config)
        self.encoder = CustomViTEncoder(config)

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        embedding_output = self.embeddings(pixel_values)

        sequence_output = self.encoder(embedding_output)
        sequence_output = self.layernorm(sequence_output)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output)

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
        )

class CustomViTForImageClassification(ViTForImageClassification):
    def __init__(self,config: ViTConfig):
        super().__init__(config)
        self.vit = CustomViTModel(config, add_pooling_layer=False)

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[tuple, ImageClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.vit(
            pixel_values,
            return_dict=return_dict,
        )

        sequence_output = outputs[0] if isinstance(outputs, tuple) else outputs.last_hidden_state
        logits = self.classifier(sequence_output[:, 0, :])
        loss = None
        if labels is not None:
            if self.num_labels == 1: self.config.problem_type = "regression"
            else: self.config.problem_type = "single_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                loss = loss_fct(logits.squeeze(), labels.squeeze())
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + (outputs[1:] if isinstance(outputs, tuple) else (outputs.pooler_output,))
            return ((loss,) + output) if loss is not None else output

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
        )