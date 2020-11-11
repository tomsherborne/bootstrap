from typing import Dict, List
import logging
import random
from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("sample_source_seq2seq")
class SampleSourceSeq2SeqDatasetReader(DatasetReader):
    """
    Reads a TSV of format NL1 \t NL2 \t ... \t NLn \t LF \n with n parallel utterances for 1 LF.
    Randomly samples a single NL and returns an NL-LF pair
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
                 source_add_start_token: bool = True,
                 lazy: bool = False,
                 reverse_source_sequence: bool = False) -> None:
        super().__init__(lazy)
        self._source_tokenizer = source_tokenizer
        self._target_tokenizer = target_tokenizer or self._source_tokenizer
        self._source_token_indexers = source_token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._target_token_indexers = target_token_indexers or self._source_token_indexers
        self._source_add_start_token = source_add_start_token
        self._reverse_source_sequence = reverse_source_sequence

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

        if len(source_sents) > 1:
            source_string = random.choice(source_sents)
        else:
            source_string = source_sents[0]

        tokenized_source = self._source_tokenizer.tokenize(source_string)
        if self._reverse_source_sequence:
            tokenized_source = tokenized_source[::-1]
        if self._source_add_start_token:
            tokenized_source.insert(0, Token(START_SYMBOL))
        tokenized_source.append(Token(END_SYMBOL))
        source_field = TextField(tokenized_source, self._source_token_indexers)
        meta_fields = {"source_tokens": [x.text for x in tokenized_source[1:-1]]}
        fields_dict = {"source_tokens": source_field}
        if target_string is not None:
            tokenized_target = self._target_tokenizer.tokenize(target_string)
            tokenized_target.insert(0, Token(START_SYMBOL))
            tokenized_target.append(Token(END_SYMBOL))
            target_field = TextField(tokenized_target, self._target_token_indexers)
            fields_dict["target_tokens"] = target_field
            meta_fields["target_tokens"] = [y.text for y in tokenized_target[1:-1]]
        fields_dict["metadata"] = MetadataField(meta_fields)

        return Instance(fields_dict)
