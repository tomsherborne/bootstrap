from abc import ABC, abstractmethod
from typing import List, Dict
from copy import deepcopy
import torch
from torch.nn import Module, ModuleList
from torch.functional import F
from allennlp.common.registrable import Registrable
from allennlp.data import Vocabulary
from allennlp.modules.seq2seq_encoders.bidirectional_language_model_transformer import MultiHeadedAttention


class TransformerCombiner(Module, Registrable):
    """
    An abstract class to represent the parent implementation of multiple distributions.
    Such that any Combiner instance computes the comb() function to output some y_E of size
    equal to y_1 subject to y_1.shape == y_2.shape == ... == y_N.shape
    y_E = comb(y_1, y_2, ..., y_N)
    """
    def __init__(self, n_sources: int):
        super(TransformerCombiner, self).__init__()
        self.n_sources = n_sources

    def __call__(self, input_states):
        return self.forward(input_states)

    @abstractmethod
    def forward(self, input_states):
        raise NotImplementedError("Subclasses of the Combiner Class should define the forward method")


@TransformerCombiner.register("uniform")
class UniformCombiner(TransformerCombiner):
    """
    Combiner module returning the sum of all inputs
    """
    def __init__(self, n_sources: int):
        super().__init__(n_sources)

    def forward(self, input_states: torch.Tensor) -> torch.Tensor:
        """
        Compute the sum of the input states
        :param input_states: List of the y_i states over i sources
        :return: the sum of the input states
        """
        # input states is shape: (batch_size, self.n_sources, tgt_time_steps, embed_dim)
        assert input_states.size(1) == self.n_sources
        # print(input_states.shape)

        # (group_size, tgt_time_steps, embed_dim)
        summed_states = torch.sum(input_states, dim=1, keepdim=False)
        # print(summed_states.shape)

        return summed_states


@TransformerCombiner.register("gated")
class GatedCombiner(TransformerCombiner):
    def __init__(self, n_sources: int, embed_dim: int):
        super().__init__(n_sources)

        # linear projection is of size (embed_dim, num_sources * embed_dim)
        self._hidden_projection = torch.nn.Linear(self.n_sources * embed_dim, embed_dim, bias=True)
        # gate projection computes gating vector for each input state
        self._gate_projection = torch.nn.Linear(embed_dim, self.n_sources, bias=True)

    def forward(self, input_states: torch.Tensor) -> torch.Tensor:
        """
        Compute the sum of the input states
        :param input_states: List of the y_i states over i sources
        :return: the sum of the input states
        """
        # input states is shape: (batch_size, num_srcs, tgt_time_steps, embed_dim)
        assert input_states.size(1) == self.n_sources
        # print(input_states.shape)

        group_size, _, tgt_time_steps, _ = input_states.shape

        # (group_size, tgt_time_steps, embed_dim * num_srcs)
        # import pdb;pdb.set_trace()
        input_states_flat = input_states.transpose(1, 2).contiguous().view(group_size, tgt_time_steps, -1)

        # print(input_states_flat.shape)

        projection_out = torch.tanh(self._hidden_projection(input_states_flat))
        # print(projection_out.shape)

        # (group_size, tgt_time_steps, num_srcs)
        gate_values = F.softmax(self._gate_projection(projection_out), dim=-1)
        # print(gate_values.shape)

        # (batch_size, tgt_time_steps, num_srcs, embed_dim) * (group_size, tgt_time_steps, num_srcs, 1) ->
        # (group_size, tgt_time_steps, num_srcs, embed_dim)
        gated_inputs = input_states.transpose(1, 2) * gate_values.unsqueeze(-1)
        # print(gated_inputs.shape)

        # (group_size, tgt_time_steps, embed_dim)
        summed_inputs = torch.sum(gated_inputs, dim=-2, keepdim=False)
        # print(summed_inputs.shape)

        return summed_inputs


class AttentionCombiner(Module):
    """
    Combination module for combining N MultiHeadedAttention modules with the defined combiner
    """
    def __init__(self, n_sources: int, attn: MultiHeadedAttention, combiner: TransformerCombiner) -> None:
        super().__init__()
        self.n_sources = n_sources
        self.src_attention = _clones(attn, self.n_sources)
        self.comb = combiner

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None
    ) -> torch.Tensor:
        # query is shape: (batch_size, tgt_time_steps, embed_dim)
        # key   is shape: (batch_size, n_srcs, max_input_sequence_length, encoder_output_dim)
        # value is shape: (batch_size, n_srcs, max_input_sequence_length, encoder_output_dim)

        batch_size, tgt_time_steps, embed_dim = query.size()
        source_attn_states = torch.zeros(size=(batch_size, self.n_sources, tgt_time_steps, embed_dim)).type_as(query).to(query.device)

        # Run MHA for each source
        # (group_size, n_srcs, tgt_time_steps, embed_dim)
        for src_idx in range(self.n_sources):
            source_attn_states[:, src_idx, :, :] = \
                self.src_attention[src_idx](query, key[:, src_idx, :, :], value[:, src_idx, :, :])
        # source_attention_states = [src_attn(query, k, v) for src_attn, k, v in zip(self.src_attention, key, value)]

        # Combine using combiner
        # (group_size, tgt_time_steps, embed_dim)
        combined_attention = self.comb(source_attn_states)

        return combined_attention


def _clones(module: Module, num_layers: int):
    """Produce N identical layers."""
    return ModuleList([deepcopy(module) for _ in range(num_layers)])
