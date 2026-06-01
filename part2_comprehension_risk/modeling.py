"""XLM-RoBERTa classifier with optional numeric Part 2 features."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from transformers import XLMRobertaModel, XLMRobertaPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput


class XLMRobertaForTextAuxiliaryClassification(XLMRobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.aux_feature_dim = int(getattr(config, "aux_feature_dim", 0))
        self.aux_hidden_size = int(getattr(config, "aux_hidden_size", 64))
        self.roberta = XLMRobertaModel(config, add_pooling_layer=False)
        dropout_prob = config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        self.dropout = nn.Dropout(dropout_prob)

        classifier_input_dim = config.hidden_size
        if self.aux_feature_dim > 0:
            self.aux_encoder = nn.Sequential(
                nn.Linear(self.aux_feature_dim, self.aux_hidden_size),
                nn.GELU(),
                nn.Dropout(dropout_prob),
                nn.Linear(self.aux_hidden_size, self.aux_hidden_size),
                nn.GELU(),
            )
            classifier_input_dim += self.aux_hidden_size

        self.classifier = nn.Linear(classifier_input_dim, 1)
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        aux_features: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> SequenceClassifierOutput:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled = self.dropout(outputs[0][:, 0])
        if self.aux_feature_dim > 0:
            if aux_features is None:
                aux_features = pooled.new_zeros(pooled.shape[0], self.aux_feature_dim)
            aux_repr = self.aux_encoder(aux_features.to(pooled.dtype))
            pooled = torch.cat([pooled, aux_repr], dim=-1)
        logits = self.classifier(self.dropout(pooled)).squeeze(-1)

        loss = None
        if labels is not None:
            loss = nn.functional.binary_cross_entropy_with_logits(logits, labels.float())

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
