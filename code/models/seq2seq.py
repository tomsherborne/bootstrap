# pylint: disable=no-member
from typing import Dict, List, Tuple, Any, Optional

import numpy
from overrides import overrides
import torch
import torch.nn.functional as F
from torch.nn.modules import Dropout
from torch.nn.modules.linear import Linear

from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules import Attention, TextFieldEmbedder, Seq2SeqEncoder
from allennlp.models.model import Model
from allennlp.modules.token_embedders import Embedding
from allennlp.nn.initializers import InitializerApplicator
from allennlp.nn import util
from allennlp.nn.beam_search import BeamSearch

from xlwomt.metrics import TokenSequenceAccuracy


class CellgateDropoutLSTMCell(torch.nn.Module):
    """
    Customised LSTM Cell with Cell-Gate dropout for Dong & Lapata 2016 Replication
    Tom Sherborne April 2019+

    We apply dropout to the tanh bounded cell gate as an additional regulariser
    """
    def __init__(self, input_size: int, hidden_size: int, rec_dropout: Optional[Dropout] = None) -> None:
        super().__init__()
        self.input2hidden = Linear(input_size, 4 * hidden_size)
        self.hidden2hidden = Linear(hidden_size, 4 * hidden_size)
        self.rec_dropout = rec_dropout

    def forward(self, k_input: torch.Tensor,
                hidden_state: (torch.Tensor, torch.Tensor)) -> (torch.Tensor, torch.Tensor):
        prev_h, prev_c = hidden_state
        gated_inputs = torch.add(self.input2hidden(k_input), self.hidden2hidden(prev_h))
        in_gate, forget_gate, cell_gate, out_gate = gated_inputs.chunk(4, 1)
        in_gate = torch.sigmoid(in_gate)
        forget_gate = torch.sigmoid(forget_gate)
        out_gate = torch.sigmoid(out_gate)
        cell_gate = torch.tanh(cell_gate)
        if self.rec_dropout:
            cell_gate = self.rec_dropout(cell_gate)
        next_c = (forget_gate * prev_c) + (in_gate * cell_gate)
        next_h = out_gate * torch.tanh(next_c)
        return next_h, next_c


@Model.register("seq2seq_pa")
class Seq2SeqPostAttention(Model):
    """
    Modification of the existing `simple_seq2seq implementation which doesn't use attention input-feeding, instead
    computing attention over the current decoder state. The attention is fixed to use `SummarizeDotProductAttention`
    as is done in Dong & Lapata '16.
    """

    def __init__(self,
                 vocab: Vocabulary,
                 source_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 max_decoding_steps: int,
                 attention: Attention = None,
                 beam_size: int = 1,
                 target_namespace: str = "tokens",
                 target_embedding_dim: int = None,
                 scheduled_sampling_ratio: float = 0.,
                 dropout: float = 0.5,
                 recurrent_dropout: float = 0.0,
                 num_decoder_layers: int = 1,
                 initializer: Optional[InitializerApplicator] = None) -> None:
        super(Seq2SeqPostAttention, self).__init__(vocab)
        self._target_namespace = target_namespace
        self._scheduled_sampling_ratio = scheduled_sampling_ratio

        # We need the start symbol to provide as the input at the first timestep of decoding, and
        # end symbol as a way to indicate the end of the decoded sequence.
        self._start_index = self.vocab.get_token_index(START_SYMBOL, self._target_namespace)
        self._end_index = self.vocab.get_token_index(END_SYMBOL, self._target_namespace)

        self._token_based_metric = TokenSequenceAccuracy()

        # At prediction time, we use a beam search to find the most likely sequence of target tokens.
        self._max_decoding_steps = max_decoding_steps
        self._beam_search = BeamSearch(self._end_index, max_steps=max_decoding_steps, beam_size=beam_size)

        # Dense embedding of source vocab tokens.
        self._source_embedder = source_embedder

        self._dropout_prob = dropout
        self._dropout_recurrent_prob = recurrent_dropout

        # Encodes the sequence of source embeddings into a sequence of hidden states.
        self._encoder = encoder

        # size of output domain
        num_classes = self.vocab.get_vocab_size(self._target_namespace)

        # Decoder output dim needs to be the same as the encoder output dim since we initialize the
        # hidden state of the decoder with the final hidden state of the encoder.
        target_embedding_dim = target_embedding_dim or source_embedder.get_output_dim()
        self._encoder_output_dim = self._encoder.get_output_dim()
        self._decoder_input_dim = target_embedding_dim
        self._decoder_output_dim = self._encoder_output_dim

        # Attention mechanism applied to the encoder output for each step.
        if attention:
            self._attention = attention
            self._attention_projection = Linear(self._decoder_output_dim + self._encoder_output_dim,
                                                self._decoder_output_dim)
        else:
            self._attention = None

        # Dense embedding of vocab words in the target space.
        self._target_embedder = Embedding(num_classes, target_embedding_dim)

        if num_decoder_layers < 1:
            raise ValueError(f"Cannot specify < 1 decoder layers. {num_decoder_layers} layers given")

        self._decoder = CellgateDropoutLSTMCell(input_size=self._decoder_input_dim,
                                                hidden_size=self._decoder_output_dim)

        # We project the hidden state from the decoder into the output vocabulary space
        # in order to get log probabilities of each target token, at each time step.
        self._output_projection_layer = Linear(self._decoder_output_dim, num_classes)

        if initializer:
            initializer(self)

        # print Model size
        print(self)

    def take_step(self,
                  last_predictions: torch.Tensor,
                  state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Take a decoding step. This is called by the beam search class.
        Parameters
        ----------
        last_predictions : ``torch.Tensor``
            A tensor of shape ``(group_size,)``, which gives the indices of the predictions
            during the last time step.
        state : ``Dict[str, torch.Tensor]``
            A dictionary of tensors that contain the current state information
            needed to predict the next step, which includes the encoder outputs,
            the source mask, and the decoder hidden state and context. Each of these
            tensors has shape ``(group_size, *)``, where ``*`` can be any other number
            of dimensions.
        Returns
        -------
        Tuple[torch.Tensor, Dict[str, torch.Tensor]]
            A tuple of ``(log_probabilities, updated_state)``, where ``log_probabilities``
            is a tensor of shape ``(group_size, num_classes)`` containing the predicted
            log probability of each class for the next step, for each item in the group,
            while ``updated_state`` is a dictionary of tensors containing the encoder outputs,
            source mask, and updated decoder hidden state and context.
        Notes
        -----
            We treat the inputs as a batch, even though ``group_size`` is not necessarily
            equal to ``batch_size``, since the group may contain multiple states
            for each source sentence in the batch.
        """
        # shape: (group_size, num_classes)
        output_projections, state = self._decoder_step(last_predictions, state, dropout_mask=None)

        # shape: (group_size, num_classes)
        class_log_probabilities = F.log_softmax(output_projections, dim=-1)

        return class_log_probabilities, state

    @overrides
    def forward(self,  # type: ignore
                source_tokens: Dict[str, torch.LongTensor],
                metadata: List[Dict[str, Any]],
                target_tokens: Dict[str, torch.LongTensor] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Make forward pass with decoder logic for producing the entire target sequence.
        Parameters
        ----------
        source_tokens : ``Dict[str, torch.LongTensor]``
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
        state = self._encode(source_tokens)

        if target_tokens:
            state = self._init_decoder_state(state)
            # The `_forward_loop` decodes the input sequence and computes the loss during training
            # and validation.
            output_dict = self._forward_loop(state, target_tokens)
        else:
            output_dict = {}

        if not self.training:
            state = self._init_decoder_state(state)
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
        output_dict["predicted_tokens"] = all_predicted_tokens
        return output_dict

    def _encode(self, source_tokens: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Encode a batch of input utterances using a Seq2SeqEncoder.
        Apply time-independent dropout mask for the input from Gal & Gharamani 2015 (arxiv.org/pdf/1512.05287.pdf)
        :param source_tokens:
        :return:
        """
        # shape: (batch_size, max_input_sequence_length, encoder_input_dim)
        embedded_input = self._source_embedder(source_tokens)

        if self.training:
            # shape: (batch_size, encoder_input_dim)
            encoder_dropout_mask = util.get_dropout_mask(self._dropout_prob, embedded_input[:, 0, :])

            # shape: (batch_size, max_input_sequence_length, encoder_input_dim)
            embedded_input = embedded_input * encoder_dropout_mask.unsqueeze(1)

        # shape: (batch_size, max_input_sequence_length)
        source_mask = util.get_text_field_mask(source_tokens)

        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs = self._encoder(embedded_input, source_mask)    # pylint: disable=not-callable

        return {
                "source_mask": source_mask,
                "encoder_outputs": encoder_outputs,
        }

    def _init_decoder_state(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch_size = state["source_mask"].size(0)

        # shape: (batch_size, encoder_output_dim)
        final_encoder_output = util.get_final_encoder_states(
                state["encoder_outputs"],
                state["source_mask"],
                self._encoder.is_bidirectional()
        )

        # shape: (batch_size, decoder_output_dim)
        state[f"decoder_hidden"] = final_encoder_output

        # shape: (batch_size, decoder_output_dim)
        state[f"decoder_context"] = state["encoder_outputs"].new_zeros(batch_size, self._decoder_output_dim)

        return state

    def _forward_loop(self,
                      state: Dict[str, torch.Tensor],
                      target_tokens: Dict[str, torch.LongTensor] = None) -> Dict[str, torch.Tensor]:
        """
        Make forward pass during training or do greedy search during prediction.
        Notes
        -----
        We really only use the predictions from the method to test that beam search
        with a beam size of 1 gives the same results.
        """
        # shape: (batch_size, max_input_sequence_length)
        source_mask = state["source_mask"]

        # get batch size dynamically
        batch_size = source_mask.size(0)

        if target_tokens:
            # shape: (batch_size, max_target_sequence_length)
            targets = target_tokens["tokens"]

            _, target_sequence_length = targets.size()

            # The last input from the target is either padding or the end symbol.
            # Either way, we don't have to process it.
            num_decoding_steps = target_sequence_length - 1
        else:
            num_decoding_steps = self._max_decoding_steps

        # Initialize target predictions with the start index.
        # shape: (batch_size,)
        last_predictions = source_mask.new_full((batch_size,), fill_value=self._start_index)

        step_logits: List[torch.Tensor] = []
        step_predictions: List[torch.Tensor] = []

        # shape: (batch_size, decoder_input_dim)
        dec_embedding_dropout_mask = util.get_dropout_mask(self._dropout_prob,
                                                 source_mask.new_ones((batch_size, self._decoder_input_dim)))
        # shape: (batch_size, decoder_output_dim)
        dec_dropout_mask = util.get_dropout_mask(self._dropout_prob,
                                                 source_mask.new_ones((batch_size, self._decoder_output_dim)))

        for timestep in range(num_decoding_steps):
            if self.training and torch.rand(1).item() < self._scheduled_sampling_ratio:
                # Use gold tokens at test time and at a rate of 1 - _scheduled_sampling_ratio
                # during training.
                # shape: (batch_size,)
                input_choices = last_predictions
            elif not target_tokens:
                # shape: (batch_size,)
                input_choices = last_predictions
            else:
                # shape: (batch_size,)
                input_choices = targets[:, timestep]

            # shape: (batch_size, num_classes)
            output_projections, state = self._decoder_step(input_choices,
                                                           state,
                                                           dec_dropout_mask,
                                                           dec_embedding_dropout_mask)

            # list of tensors, shape: (batch_size, 1, num_classes)
            step_logits.append(output_projections.unsqueeze(1))

            # shape: (batch_size, num_classes)
            class_probabilities = F.softmax(output_projections, dim=-1)

            # shape (predicted_classes): (batch_size,)
            _, predicted_classes = torch.max(class_probabilities, 1)

            # shape (predicted_classes): (batch_size,)
            last_predictions = predicted_classes

            step_predictions.append(last_predictions.unsqueeze(1))

        # shape: (batch_size, num_decoding_steps)
        predictions = torch.cat(step_predictions, 1)

        output_dict = {"predictions": predictions}

        if target_tokens:
            # shape: (batch_size, num_decoding_steps, num_classes)
            logits = torch.cat(step_logits, 1)

            # Compute loss.
            target_mask = util.get_text_field_mask(target_tokens)
            loss = self._get_loss(logits, targets, target_mask)

            output_dict["loss"] = loss

        return output_dict

    def _forward_beam_search(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Make forward pass during prediction using a beam search."""
        batch_size = state["source_mask"].size()[0]
        start_predictions = state["source_mask"].new_full(
            (batch_size,), fill_value=self._start_index
        )

        # shape (all_top_k_predictions): (batch_size, beam_size, num_decoding_steps)
        # shape (log_probabilities): (batch_size, beam_size)
        all_top_k_predictions, log_probabilities = self._beam_search.search(
                start_predictions, state, self.take_step)

        output_dict = {
                "class_log_probabilities": log_probabilities,
                "predictions": all_top_k_predictions,
        }
        return output_dict

    def _decoder_step(self,
                      last_predictions: torch.Tensor,
                      state: Dict[str, torch.Tensor],
                      dropout_mask: torch.Tensor = None,
                      embedding_dropout_mask: torch.Tensor = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:  # pylint: disable=line-too-long
        """
        Decode current state and last prediction to produce produce projections
        into the target space, which can then be used to get probabilities of
        each target token for the next step.
        Inputs are the same as for `take_step()`.
        """
        # shape: (group_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs = state["encoder_outputs"]

        # shape: (group_size, max_input_sequence_length)
        source_mask = state["source_mask"]

        # shape: (group_size, target_embedding_dim)
        embedded_input = self._target_embedder(last_predictions)    # pylint: disable=not-callable

        # Dropout mask is the same for each time step per instance in the batch
        if embedding_dropout_mask is not None and self.training:
            # shape: (group_size, target_embedding_dim)
            embedded_input = embedded_input * embedding_dropout_mask

        # Retrieve the previous time steps from each layer
        # shape: (group_size, decoder_output_dim)
        decoder_hidden = state[f"decoder_hidden"]

        # shape: (group_size, decoder_output_dim)
        decoder_context = state[f"decoder_context"]

        # shape (decoder_hidden): (batch_size, decoder_output_dim)
        # shape (decoder_context): (batch_size, decoder_output_dim)
        decoder_hidden, decoder_context = self._decoder(embedded_input, (decoder_hidden, decoder_context))

        if dropout_mask is not None and self.training:
            # shape: (group_size, decoder_output_dim)
            decoder_hidden = decoder_hidden * dropout_mask

        # Reassign the current time steps from each layer for the subsequent time step
        # shape: (group_size, decoder_output_dim)
        state[f"decoder_hidden"] = decoder_hidden

        # shape: (group_size, decoder_output_dim)
        state[f"decoder_context"] = decoder_context

        # Post decoding attention
        if self._attention:
            # shape (attended_output): (batch_size, decoder_output_dim)
            attended_output = self._prepare_attended_output(decoder_hidden, encoder_outputs, source_mask)
            step_output = attended_output
        else:
            # shape (attended_output): (batch_size, decoder_output_dim)
            step_output = decoder_hidden

        # shape: (group_size, num_classes)
        output_projections = self._output_projection_layer(step_output)

        return output_projections, state

    def _prepare_attended_output(self,
                                 decoder_hidden_state: torch.LongTensor = None,
                                 encoder_outputs: torch.LongTensor = None,
                                 encoder_outputs_mask: torch.LongTensor = None) -> torch.Tensor:
        """
        Apply attention over encoder outputs and decoder state, summarise and project through a Linear layer.
        Here we are replicating the attention mechanism from Dong & Lapata 2016
        For decoder_output `x` and encoder states `y` we first compute
        dot(Y, x)
        then bound by softmax
        S = softmax(dot(Y, x))
        then summarise as
        C = sum(S)
        and project to the same dimension as the decoder output
        H = tanh(W * [C;x]) UPDATE: tanh was removed as it didnt make sense to bound the output
        """
        # Ensure mask is also a FloatTensor. Or else the multiplication within
        # attention will complain.
        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs_mask = encoder_outputs_mask.float()

        # shape: (batch_size, max_input_sequence_length)
        output_weights = self._attention(
            decoder_hidden_state, encoder_outputs, encoder_outputs_mask)

        # shape: (batch_size, encoder_output_dim)
        attended_output = util.weighted_sum(encoder_outputs, output_weights)

        # shape: (batch_size, encoder_output_dim + decoder_output_dim)
        attended_output = torch.cat((attended_output, decoder_hidden_state), -1)

        # shape: (batch_size, decoder_output_dim)
        attended_output = self._attention_projection(attended_output)

        return attended_output

    @staticmethod
    def _get_loss(logits: torch.LongTensor,
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
        relevant_targets = targets[:, 1:].contiguous()

        # shape: (batch_size, num_decoding_steps)
        relevant_mask = target_mask[:, 1:].contiguous()

        return util.sequence_cross_entropy_with_logits(logits, relevant_targets, relevant_mask)

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics: Dict[str, float] = {}
        if not self.training:
            all_metrics.update(self._token_based_metric.get_metric(reset=reset))
        return all_metrics
