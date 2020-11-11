# pylint: disable=no-member
from typing import Dict, List, Tuple, Any, Optional
from copy import deepcopy
import math

import numpy
from overrides import overrides
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.linear import Linear

from allennlp.common.checks import ConfigurationError
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder
from allennlp.modules.seq2seq_decoders import DecoderNet
from allennlp.models.model import Model
from allennlp.modules.token_embedders import Embedding
from allennlp.nn.initializers import InitializerApplicator
from allennlp.nn import util
from allennlp.nn.beam_search import BeamSearch
from allennlp.modules.seq2seq_encoders.bidirectional_language_model_transformer import (
    subsequent_mask,
    PositionwiseFeedForward,
    SublayerConnection,
    PositionalEncoding,
    MultiHeadedAttention,
)

from xlwomt.metrics import TokenSequenceAccuracy
from xlwomt.models.combiner import TransformerCombiner, AttentionCombiner


@Model.register("ensemble_transformer")
class EnsembleSequenceTransformer(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 source_embedder: TextFieldEmbedder,
                 target_embedder: Embedding,
                 encoder: Seq2SeqEncoder,
                 max_decoding_steps: int,
                 decoding_dim: int,
                 feedforward_hidden_dim: int,
                 num_layers: int,
                 num_attention_heads: int,
                 combiner_module: TransformerCombiner,
                 use_positional_encoding: bool = True,
                 positional_encoding_max_steps: int = 5000,
                 dropout_prob: float = 0.1,
                 residual_dropout_prob: float = 0.2,
                 attention_dropout_prob: float = 0.2,
                 beam_size: int = 1,
                 target_namespace: str = "tokens",
                 label_smoothing_ratio: Optional[float] = None,
                 initializer: Optional[InitializerApplicator] = None) -> None:
        super(EnsembleSequenceTransformer, self).__init__(vocab)

        self._target_namespace = target_namespace
        self._label_smoothing_ratio = label_smoothing_ratio
        self._start_index = self.vocab.get_token_index(START_SYMBOL, self._target_namespace)
        self._end_index = self.vocab.get_token_index(END_SYMBOL, self._target_namespace)
        self._token_based_metric = TokenSequenceAccuracy()

        # Beam Search
        self._max_decoding_steps = max_decoding_steps
        self._beam_search = BeamSearch(self._end_index, max_steps=max_decoding_steps, beam_size=beam_size)

        # Encoder
        self._encoder = nn.ModuleDict({"enc_0": deepcopy(encoder),
                                       "enc_1": deepcopy(encoder),
                                       "enc_2": deepcopy(encoder)})
        # del encoder

        # Vocabulary and embedder
        self._source_embedder = source_embedder
        self._target_embedder = target_embedder

        target_vocab_size = self.vocab.get_vocab_size(self._target_namespace)
        assert target_vocab_size == self._target_embedder.num_embeddings

        target_embedding_dim = self._target_embedder.get_output_dim()

        self._decoding_dim = decoding_dim
        # Sequence Decoder Features
        self._output_projection_layer = Linear(
            self._decoding_dim, target_vocab_size
        )

        self._decoder = Decoder(
            num_layers=num_layers,
            decoding_dim=decoding_dim,
            target_embedding_dim=target_embedding_dim,
            feedforward_hidden_dim=feedforward_hidden_dim,
            num_attention_heads=num_attention_heads,
            use_positional_encoding=use_positional_encoding,
            positional_encoding_max_steps=positional_encoding_max_steps,
            dropout_prob=dropout_prob,
            residual_dropout_prob=residual_dropout_prob,
            attention_dropout_prob=attention_dropout_prob,
            combiner=combiner_module,
            num_sources=3
        )

        # Parameter checks and cleanup
        if self._target_embedder.get_output_dim() != self._decoder.target_embedding_dim:
            raise ConfigurationError(
                "Target Embedder output_dim doesn't match decoder module's input."
            )
        #
        if self._encoder["enc_0"].get_output_dim() != self._decoder.get_output_dim():
            raise ConfigurationError(
                f"Encoder output dimension {self._encoder['enc_0'].get_output_dim()} should be"
                f" equal to decoder dimension {self._self_attention.get_output_dim()}."
            )

        if initializer:
            initializer(self)

        # Print the model
        print(self)

    def take_step(self,
                  last_predictions: torch.Tensor,
                  state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Take a decoding step. This is called by the beam search class.
        # Parameters
        last_predictions : `torch.Tensor`
            A tensor of shape `(group_size,)`, which gives the indices of the predictions
            during the last time step.
        state : `Dict[str, torch.Tensor]`
            A dictionary of tensors that contain the current state information
            needed to predict the next step, which includes the encoder outputs,
            the source mask, and the decoder hidden state and context. Each of these
            tensors has shape `(group_size, *)`, where `*` can be any other number
            of dimensions.
        # Returns
        Tuple[torch.Tensor, Dict[str, torch.Tensor]]
            A tuple of `(log_probabilities, updated_state)`, where `log_probabilities`
            is a tensor of shape `(group_size, num_classes)` containing the predicted
            log probability of each class for the next step, for each item in the group,
            while `updated_state` is a dictionary of tensors containing the encoder outputs,
            source mask, and updated decoder hidden state and context.
        Notes
        -----
            We treat the inputs as a batch, even though `group_size` is not necessarily
            equal to `batch_size`, since the group may contain multiple states
            for each source sentence in the batch.
        """
        # shape: (group_size, num_classes)
        output_projections, state = self._decoder_step(last_predictions, state)

        # shape: (group_size, num_classes)
        class_log_probabilities = F.log_softmax(output_projections, dim=-1)

        return class_log_probabilities, state

    @overrides
    def forward(self,  # type: ignore
                source_tokens_0: Dict[str, torch.LongTensor],
                source_tokens_1: Dict[str, torch.LongTensor],
                source_tokens_2: Dict[str, torch.LongTensor],
                metadata: List[Dict[str, Any]],
                target_tokens: Dict[str, torch.LongTensor] = None) -> Dict[str, torch.Tensor]:
        """
        Make forward pass with decoder logic for producing the entire target sequence.
        Parameters
        ----------
        source_tokens_0 : ``Dict[str, torch.LongTensor]``
           The output of `TextField.as_array()` applied on the source `TextField`. This will be
           passed through a `TextFieldEmbedder` and then through an encoder.
        source_tokens_1 : ``Dict[str, torch.LongTensor]``
           The output of `TextField.as_array()` applied on the source `TextField`. This will be
           passed through a `TextFieldEmbedder` and then through an encoder.
        source_tokens_2 : ``Dict[str, torch.LongTensor]``
           The output of `TextField.as_array()` applied on the source `TextField`. This will be
           passed through a `TextFieldEmbedder` and then through an encoder.
        metadata: List[Dict[str, Any]]
            Additional information for prediction
        target_tokens : ``Dict[str, torch.LongTensor]``, optional (default = None)
           Output of `Textfield.as_array()` applied on target `TextField`. We assume that the
           target tokens are also represented as a `TextField`.
        Returns
        -------
        Dict[str, torch.Tensor]
        """
        state = self._encode(source_tokens=[source_tokens_0,
                                            source_tokens_1,
                                            source_tokens_2])

        if target_tokens:
            # state = self._decoder.init_decoder_state(state)
            # The `_forward_loop` decodes the input sequence and computes the loss during training
            # and validation.
            output_dict = self._forward_loop(state, target_tokens)
        else:
            output_dict = {}

        if not self.training:
            # state = self._init_decoder_state(state)
            predictions = self._forward_beam_search(state)
            output_dict.update(predictions)
            if target_tokens:
                # shape: (batch_size, max_predicted_sequence_length)
                predicted_tokens = self.decode(output_dict)["predicted_tokens"]

                self._token_based_metric(predicted_tokens, [x["target_tokens"] for x in metadata])

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Finalize predictions.
        This method overrides ``Model.decode``, which gets called after ``Model.forward``, at test
        time, to finalize predictions. The logic for the decoder part of the encoder-decoder lives
        within the ``forward`` method.
        This method trims the output predictions to the first end symbol, replaces indices with
        corresponding tokens, and adds a field called ``predicted_tokens`` to the ``output_dict``.
        """
        predicted_indices = output_dict["predictions"]
        if not isinstance(predicted_indices, numpy.ndarray):
            predicted_indices = predicted_indices.detach().cpu().numpy()
        all_predicted_tokens = []
        for indices in predicted_indices:
            # Beam search gives us the top k results for each source sentence in the batch
            # but we just want the single best.
            if len(indices.shape) > 1:
                indices = indices[0]
            indices = list(indices)
            # Collect indices till the first end_symbol
            if self._end_index in indices:
                indices = indices[:indices.index(self._end_index)]
            predicted_tokens = [self.vocab.get_token_from_index(x, namespace=self._target_namespace)
                                for x in indices]
            all_predicted_tokens.append(predicted_tokens)
        output_dict["predicted_tokens"] = all_predicted_tokens  # type: ignore
        return output_dict

    def _encode(self, source_tokens: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Make forward pass on the encoder.
        # Parameters
        source_tokens : `List[Dict[str, torch.Tensor]]`
           List of the output of `TextField.as_array()` applied on the source `TextField`. This will be
           passed through a `TextFieldEmbedder` and then through an encoder.
        # Returns
        Dict[str, torch.Tensor]
            Map consisting of the key `source_mask` with the mask over the
            `source_tokens` text field,
            and the key `encoder_outputs` with the output tensor from
            forward pass on the encoder.
        """
        # shape: n_srcs list of (batch_size, max_input_sequence_length, encoder_input_dim)
        embedded_inputs = [self._source_embedder(src_toks) for src_toks in source_tokens]

        # shape: (batch_size, n_srcs, max_input_sequence_length, encoder_input_dim)
        embedded_inputs = torch.nn.utils.rnn.pad_sequence([e.permute(1, 0, 2) for e in embedded_inputs]).permute(2, 1, 0, 3)

        # shape: n_src size list of (batch_size, max_input_sequence_length)
        source_masks = [util.get_text_field_mask(src_toks) for src_toks in source_tokens]

        # shape: (batch_size, n_srcs, max_input_sequence_length)
        source_masks = torch.nn.utils.rnn.pad_sequence([s.permute(1, 0) for s in source_masks]).permute(2, 1, 0)

        # shape: List(batch_size, max_input_sequence_length, encoder_output_dim)
        # encoder_outputs = [self._encoder(embedded_inputs[:, src_idx, :, :], source_masks[:, src_idx, :])
        #                    for src_idx in range(3)]
        encoder_outputs = [self._encoder[f"enc_{src_idx}"](embedded_inputs[:, src_idx, :, :], source_masks[:, src_idx, :])
                             for src_idx in range(3)]

        # shape: (batch_size, n_srcs, max_input_sequence_length, encoder_output_dim)
        encoder_outputs = torch.stack(encoder_outputs, dim=1)

        return {"source_mask": source_masks, "encoder_outputs": encoder_outputs}

    def _forward_loop(self,
                      state: Dict[str, torch.Tensor],
                      target_tokens: Dict[str, torch.LongTensor] = None) -> Dict[str, torch.Tensor]:

        # shape: (batch_size, n_srcs, max_input_sequence_length, encoder_output_dim)
        encoder_outputs = state["encoder_outputs"]

        # shape: (batch_size, n_srcs, max_input_sequence_length)
        source_mask = state["source_mask"]

        # shape: (batch_size, max_target_sequence_length)
        targets = target_tokens["tokens"]

        _, target_sequence_length = targets.size()

        # Prepare embeddings for targets. They will be used as gold embeddings during decoder training
        # shape: (batch_size, max_target_sequence_length, embedding_dim)
        target_embedding = self._target_embedder(targets)

        # shape: (batch_size, max_target_batch_sequence_length)
        target_mask = util.get_text_field_mask(target_tokens)

        _, decoder_output = self._decoder(
            previous_state=state,
            previous_steps_predictions=target_embedding[:, :-1, :],
            encoder_outputs=encoder_outputs,
            source_mask=source_mask,
            previous_steps_mask=target_mask[:, :-1]
        )

        # shape: (group_size, max_target_sequence_length, num_classes)
        logits = self._output_projection_layer(decoder_output).type(torch.FloatTensor)

        # Compute loss.
        loss = self._get_loss(logits, targets, target_mask)
        output_dict = {"loss": loss}

        return output_dict

    def _forward_beam_search(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Prepare inputs for the beam search, does beam search and returns beam search results.
        """
        batch_size = state["source_mask"].size(dim=0)
        start_predictions = state["source_mask"][:, 0, :].new_full((batch_size,), fill_value=self._start_index)

        # shape (all_top_k_predictions): (batch_size, beam_size, num_decoding_steps)
        # shape (log_probabilities): (batch_size, beam_size)
        all_top_k_predictions, log_probabilities = self._beam_search.search(
            start_predictions, state, self.take_step
        )

        output_dict = {
            "class_log_probabilities": log_probabilities,
            "predictions": all_top_k_predictions,
        }
        return output_dict

    def _decoder_step(
            self, last_predictions: torch.Tensor, state: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Decode current state and last prediction to produce produce projections
        into the target space, which can then be used to get probabilities of
        each target token for the next step.
        Inputs are the same as for `take_step()`.
        """
        # shape: (batch_size, n_srcs, max_input_sequence_length, encoder_output_dim)
        encoder_outputs = state["encoder_outputs"]

        # shape: (batch_size, n_srcs, max_input_sequence_length)
        source_mask = state["source_mask"]

        # shape: (group_size, steps_count, decoder_output_dim)
        previous_steps_predictions = state.get("previous_steps_predictions")

        # shape: (batch_size, 1, target_embedding_dim)
        last_predictions_embeddings = self._target_embedder(last_predictions).unsqueeze(1)

        if previous_steps_predictions is None or previous_steps_predictions.shape[-1] == 0:
            # There is no previous steps, except for start vectors in `last_predictions`
            # shape: (group_size, 1, target_embedding_dim)
            previous_steps_predictions = last_predictions_embeddings
        else:
            # shape: (group_size, steps_count, target_embedding_dim)
            previous_steps_predictions = torch.cat(
                [previous_steps_predictions, last_predictions_embeddings], 1
            )

        decoder_state, decoder_output = self._decoder(
            previous_state=state,
            encoder_outputs=encoder_outputs,
            source_mask=source_mask,
            previous_steps_predictions=previous_steps_predictions,
        )
        state["previous_steps_predictions"] = previous_steps_predictions

        # Update state with new decoder state, override previous state
        state.update(decoder_state)

        if self._decoder.decodes_parallel:
            decoder_output = decoder_output[:, -1, :]

        # shape: (group_size, num_classes)
        output_projections = self._output_projection_layer(decoder_output)

        return output_projections, state

    def _get_loss(self,
                  logits: torch.FloatTensor,
                  targets: torch.LongTensor,
                  target_mask: torch.LongTensor) -> torch.Tensor:
        """
        Compute loss.
        Takes logits (unnormalized outputs from the decoder) of size (batch_size,
        num_decoding_steps, num_classes), target indices of size (batch_size, num_decoding_steps+1)
        and corresponding masks of size (batch_size, num_decoding_steps+1) steps and computes cross
        entropy loss while taking the mask into account.
        The length of ``targets`` is expected to be greater than that of ``logits`` because the
        decoder does not need to compute the output corresponding to the last timestep of
        ``targets``. This method aligns the inputs appropriately to compute the loss.
        During training, we want the logit corresponding to timestep i to be similar to the target
        token from timestep i + 1. That is, the targets should be shifted by one timestep for
        appropriate comparison.  Consider a single example where the target has 3 words, and
        padding is to 7 tokens.
           The complete sequence would correspond to <S> w1  w2  w3  <E> <P> <P>
           and the mask would be                     1   1   1   1   1   0   0
           and let the logits be                     l1  l2  l3  l4  l5  l6
        We actually need to compare:
           the sequence           w1  w2  w3  <E> <P> <P>
           with masks             1   1   1   1   0   0
           against                l1  l2  l3  l4  l5  l6
           (where the input was)  <S> w1  w2  w3  <E> <P>
        """
        # shape: (batch_size, num_decoding_steps)
        relevant_targets = targets[:, 1:].contiguous().to(logits.device)

        # shape: (batch_size, num_decoding_steps)
        relevant_mask = target_mask[:, 1:].contiguous().to(logits.device)

        return util.sequence_cross_entropy_with_logits(logits,
                                                       relevant_targets,
                                                       relevant_mask)

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics: Dict[str, float] = {}
        if not self.training:
            all_metrics.update(self._token_based_metric.get_metric(reset=reset))
        return all_metrics


def _clones(module: nn.Module, num_layers: int):
    """Produce N identical layers."""
    return nn.ModuleList([deepcopy(module) for _ in range(num_layers)])


class Decoder(DecoderNet):
    """
    Transformer N layer decoder with masking.
    Code taken from http://nlp.seas.harvard.edu/2018/04/03/attention.html
    """

    def __init__(self,
                 num_layers: int,
                 decoding_dim: int,
                 target_embedding_dim: int,
                 feedforward_hidden_dim: int,
                 num_attention_heads: int,
                 combiner: TransformerCombiner,
                 num_sources: int,
                 use_positional_encoding: bool = True,
                 positional_encoding_max_steps: int = 5000,
                 dropout_prob: float = 0.1,
                 residual_dropout_prob: float = 0.2,
                 attention_dropout_prob: float = 0.2,
                 ) -> None:
        super().__init__(decoding_dim, target_embedding_dim, decodes_parallel=True)

        self._decoding_dim = decoding_dim
        self._embed_scale = math.sqrt(decoding_dim)

        self._positional_embedder = (
            PositionalEncoding(input_dim=decoding_dim, max_len=positional_encoding_max_steps)
            if use_positional_encoding
            else None
        )
        self._dropout = nn.Dropout(dropout_prob)

        generic_attn = MultiHeadedAttention(num_attention_heads, decoding_dim, attention_dropout_prob)
        combined_attn = AttentionCombiner(num_sources, generic_attn, combiner)
        feed_forward = PositionwiseFeedForward(decoding_dim, feedforward_hidden_dim, dropout_prob)

        layer = DecoderLayer(
            size=decoding_dim,
            self_attn=deepcopy(generic_attn),
            src_attn=deepcopy(combined_attn),
            feed_forward=feed_forward,
            dropout=residual_dropout_prob
        )

        self._self_attention_layers = _clones(layer, num_layers)
        self.norm = nn.LayerNorm(layer.size)

    def init_decoder_state(
        self, encoder_out: Dict[str, torch.LongTensor]
    ) -> Dict[str, torch.Tensor]:
        return {}

    @overrides
    def forward(
            self,
            previous_state: Dict[str, torch.Tensor],
            encoder_outputs: torch.Tensor,
            source_mask: torch.Tensor,
            previous_steps_predictions: torch.Tensor,
            previous_steps_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:

        # shape: (batch_size, n_srcs, max_input_sequence_length) ->
        #        (batch_size, n_srcs, 1, max_input_sequence_length)
        source_mask = source_mask.unsqueeze(-2)
        future_mask = Variable(subsequent_mask(previous_steps_predictions.size(-2),
                                               device=source_mask.device)
                               .type_as(source_mask.data))

        if previous_steps_mask is None:
            previous_steps_mask = future_mask
        else:
            previous_steps_mask = previous_steps_mask.unsqueeze(-2) & future_mask

        previous_steps_predictions = previous_steps_predictions * self._embed_scale
        if self._positional_embedder:
            previous_steps_predictions = self._positional_embedder(previous_steps_predictions)
        previous_steps_predictions = self._dropout(previous_steps_predictions)

        for layer in self._self_attention_layers:
            previous_steps_predictions = layer(previous_steps_predictions,
                                               encoder_outputs,
                                               source_mask,
                                               previous_steps_mask)

        decoded = self.norm(previous_steps_predictions)
        return {}, decoded


class DecoderLayer(nn.Module):
    """
    A single layer of transformer decoder.
    Code taken from http://nlp.seas.harvard.edu/2018/04/03/attention.html
    """
    def __init__(
        self,
        size: int,
        self_attn: MultiHeadedAttention,
        src_attn: AttentionCombiner,
        feed_forward: F,
        dropout: float,
    ) -> None:
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = _clones(SublayerConnection(size, dropout), 3)

    def forward(
        self, x: torch.Tensor, memory: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor
    ) -> torch.Tensor:

        """Follow Figure 1 (right) for connections."""
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, memory, memory, src_mask))
        return self.sublayer[2](x, self.feed_forward)
