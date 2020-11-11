# Bootstrapping a Crosslingual Semantic Parser
### Tom Sherborne, Yumo Xu and Mirella Lapata
##### In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: Findings (pp. 499–517)

[Read our paper here](https://www.aclweb.org/anthology/2020.findings-emnlp.45)

This repository is structures as:
```
.
├── code 	
│   ├── metrics
│   ├── models
│   └── reader
├── config
└── data
    └── overnight

```
- `code`: Codebase models for the paper (AllenNLP models, dataset readers and sequence_accuracy metric for training)
- `config`: AllenNLP training commands for rerunning experiments from the paper (currently only for the MT-Ensemble Transformer model). They currently assume another top-level folder called `big` which is a local cache for BERT models (i.e. if your GPU machine does not have direct internet access).
- `data/overnight`: Chinese and German versions of Overnight (Wang et al. 2015). Training data is machine translated and test/dev data is collected using Turk (see paper for details)

## ATIS

The ATIS dataset is subject to LDC licensing requirements and we keep this in a separate repo called `bootstrap_atis`. If you require access to this then please email me (`tom.sherborne@ed.ac.uk`). We will ask for evidence of holding relevant LDC licenses (LDC93S5, LDC94S19, and LDC95S26). 

## Requirements

This project was completed using `Python 3.5` with `AllenNLP 0.9` (and associated dependencies) and `PyTorch 1.2`. Much of the relevant framework code is now contained within `allennlp-models` which can be downloaded as an additional dependency. 

## Citation

If you use any of the resources from this work - please cite us as follows:
```
@inproceedings{sherborne-etal-2020-bootstrapping,
address = {Online},
author = {Sherborne, Tom and Xu, Yumo and Lapata, Mirella},
booktitle = {Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: Findings},
month = {nov},
pages = {499--517},
publisher = {Association for Computational Linguistics},
title = {{Bootstrapping a Crosslingual Semantic Parser}},
url = {https://www.aclweb.org/anthology/2020.findings-emnlp.45},
year = {2020}
}

```