from typing import Dict, List
import logging

from overrides import overrides

import random
from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("multi_sample_seq2seq")
class MultiSourceSeq2SeqDatasetReader(DatasetReader):
    """
    Parameters
    ----------
    source_tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the input sequences into words or other kinds of tokens. Defaults
        to ``WordTokenizer()``.
    target_tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the output sequences (during training) into words or other kinds
        of tokens. Defaults to ``source_tokenizer``.
    source_token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define input (source side) token representations. Defaults to
        ``{"tokens": SingleIdTokenIndexer()}``.
    target_token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define output (target side) token representations. Defaults to
        ``source_token_indexers``.
    source_add_start_token : bool, (optional, default=True)
        Whether or not to add `START_SYMBOL` to the beginning of the source sequence.
    """
    def __init__(self,
                 source_tokenizer: Tokenizer = None,
                 target_tokenizer: Tokenizer = None,
                 source_token_indexers: Dict[str, TokenIndexer] = None,
                 target_token_indexers: Dict[str, TokenIndexer] = None,
                 shuffle_probability: float = 0.0,
                 source_add_start_token: bool = True,
                 lazy: bool = False,
                 num_sources: int = 1,
                 reverse_source_sequence: bool = False) -> None:
        super().__init__(lazy)
        self._source_tokenizer = source_tokenizer
        self._target_tokenizer = target_tokenizer or self._source_tokenizer
        self._source_token_indexers = source_token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._target_token_indexers = target_token_indexers or self._source_token_indexers
        self._num_sources = num_sources
        self._source_add_start_token = source_add_start_token
        self._reverse_source_sequence = reverse_source_sequence
        self._shuffle_probability = shuffle_probability

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line_num, line in enumerate(data_file):
                line = line.strip("\n")

                if not line:
                    continue
                line_parts = line.split('\t')
                source_sequences = line_parts[:-1]
                target_sequence = line_parts[-1]
                yield self.text_to_instance(source_sequences, target_sequence)

    @overrides
    def text_to_instance(self, source_sents: List[str], target_string: str = None) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ

        if type(source_sents) == str:
            # test mode from predictor passes a string
            source_sents = [source_sents]

        # Randomly shuffle the input batch (1-shuffle_prob) times
        if random.random() < self._shuffle_probability:
            random.shuffle(source_sents)

        tokenized_sources = [self._source_tokenizer.tokenize(src) for src in source_sents]

        if self._reverse_source_sequence:
            tokenized_sources = [src[::-1] for src in tokenized_sources]
        if self._source_add_start_token:
            for src in tokenized_sources:
                src.insert(0, Token(START_SYMBOL))

        # Append END symbol
        for src in tokenized_sources:
            src.insert(0, Token(END_SYMBOL))

        source_fields = [TextField(src, self._source_token_indexers) for src in tokenized_sources]

        meta_fields = {"source_tokens": [[x.text for x in src[1:-1]] for src in tokenized_sources]}

        # the dev data case, we broadcast the same development sentence into all enc-dec pairs.
        if len(source_sents) == 1:
            fields_dict = {"source_tokens_0": source_fields[0],
                           "source_tokens_1": source_fields[0],
                           "source_tokens_2": source_fields[0]}
        else:
            fields_dict = {"source_tokens_0": source_fields[0],
                           "source_tokens_1": source_fields[1],
                           "source_tokens_2": source_fields[2]}

        if target_string is not None:
            tokenized_target = self._target_tokenizer.tokenize(target_string)
            tokenized_target.insert(0, Token(START_SYMBOL))
            tokenized_target.append(Token(END_SYMBOL))
            target_field = TextField(tokenized_target, self._target_token_indexers)
            fields_dict["target_tokens"] = target_field
            meta_fields["target_tokens"] = [y.text for y in tokenized_target[1:-1]]
        fields_dict["metadata"] = MetadataField(meta_fields)

        return Instance(fields_dict)
