"""XLM-RoBERTa models for behavior-conditioned token-level TRT regression."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from transformers import XLMRobertaModel, XLMRobertaPreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput


class XLMRobertaForBehaviorConditionedTRT(XLMRobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = 1
        self.profile_dim = int(getattr(config, "profile_dim", 0))
        self.profile_hidden_size = int(getattr(config, "profile_hidden_size", 64))
        self.conditioning_type = getattr(config, "conditioning_type", "concat")
        self.num_experts = int(getattr(config, "num_experts", 4))
        self.expert_hidden_size = int(getattr(config, "expert_hidden_size", config.hidden_size))

        self.roberta = XLMRobertaModel(config, add_pooling_layer=False)
        dropout_prob = config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        self.dropout = nn.Dropout(dropout_prob)

        if self.conditioning_type not in {"none", "concat", "moe"}:
            raise ValueError(f"Unknown conditioning_type: {self.conditioning_type}")

        if self.conditioning_type == "none":
            self.regressor = nn.Linear(config.hidden_size, 1)
        else:
            self.profile_encoder = nn.Sequential(
                nn.Linear(self.profile_dim, self.profile_hidden_size),
                nn.GELU(),
                nn.Dropout(dropout_prob),
                nn.Linear(self.profile_hidden_size, self.profile_hidden_size),
                nn.GELU(),
            )
            if self.conditioning_type == "concat":
                self.regressor = nn.Linear(config.hidden_size + self.profile_hidden_size, 1)
            else:
                self.gate = nn.Linear(self.profile_hidden_size, self.num_experts)
                self.experts = nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Linear(config.hidden_size, self.expert_hidden_size),
                            nn.GELU(),
                            nn.Dropout(dropout_prob),
                            nn.Linear(self.expert_hidden_size, 1),
                        )
                        for _ in range(self.num_experts)
                    ]
                )

        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        reader_profile: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> TokenClassifierOutput:
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

        sequence_output = self.dropout(outputs[0])
        logits = self._regress(sequence_output, reader_profile).squeeze(-1)
        loss = None
        if labels is not None:
            active = torch.isfinite(labels) & (labels != -100.0)
            if active.any():
                loss = nn.functional.mse_loss(logits[active], labels[active])
            else:
                loss = logits.sum() * 0.0

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def _regress(self, sequence_output: torch.Tensor, reader_profile: Optional[torch.Tensor]) -> torch.Tensor:
        if self.conditioning_type == "none":
            return self.regressor(sequence_output)
        if reader_profile is None:
            reader_profile = sequence_output.new_zeros(sequence_output.shape[0], self.profile_dim)
        profile_repr = self.profile_encoder(reader_profile.to(sequence_output.dtype))
        if self.conditioning_type == "concat":
            expanded_profile = profile_repr.unsqueeze(1).expand(-1, sequence_output.shape[1], -1)
            conditioned = torch.cat([sequence_output, expanded_profile], dim=-1)
            return self.regressor(self.dropout(conditioned))

        gate_weights = torch.softmax(self.gate(profile_repr), dim=-1)
        expert_outputs = torch.stack([expert(sequence_output) for expert in self.experts], dim=-1)
        return (expert_outputs * gate_weights[:, None, None, :]).sum(dim=-1)

