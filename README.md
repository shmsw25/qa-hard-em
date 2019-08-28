# A Discrete Hard EM Approach for Weakly Supervised Question Answering

This is the original implementation of the following paper.

Sewon Min, Danqi Chen, Hannaneh Hajishirzi, Luke Zettlemoyer. [A Discrete Hard EM Approach for Weakly Supervised Question Answering][paper-pdf-link]. In: Proceedings of EMNLP (long). 2019

```
@inproceedings{ min2019discrete,
  title={ A Discrete Hard EM Approach for Weakly Supervised Question Answering },
  author={ Min, Sewon and Chen, Danqi and Hajishirzi, Hannaneh and Zettlemoyer, Luke },
  booktitle={ EMNLP },
  year={ 2019 }
}

```

You can use hard EM updates for any weakly-supervised QA task where precomputed solution set can be obtained, and can use any model architecture. This is an example code for open-domain question answering using BERT QA model.

In the paper, we experiment on six QA datasets in three different categories.

- Multi-mention reading comprehension
    - Distantly-supervised reading comprehension: [TriviaQA][triviaqa-paper] 
    - Reading comrepehension with free-form answers: [NarrativeQA][narrativeqa-paper]
    - Open-domain QA: [TriviaQA-unfiltered][triviaqa-paper], [NaturalQuestions (open-domain version)][nq-paper]
- Discrete Reasoning Task: [DROP][drop-paper]
- Semantic Parsing: [WikiSQL][wikisql-paper]


Below is the results reported in the paper (all on the test set).


| Dataset | TriviaQA | NarrativeQA | TriviaQA-unfiltered | NaturalQuestions | DROP | WikiSQL |
|---|---|---|---|---|---|---|
| First-only | 64.9 | 57.4 | 48.1 | 23.6 | 42.9 | - |
| MML | 65.5 | 56.1 | 47.4 | 25.8 | 39.7 | 70.5 |
| Hard-EM (Ours) | 67.1 | **58.8** | **50.9** | **28.1** | **52.8** | **83.9** |
| SOTA | **71.4** | 54.7 | 47.1 | 26.5 | 43.8 | 74.8 |

SOTA from [Wang et al 2018][triviaqa-sota-paper], [Nishida et al 2019][narrativeqa-sota-paper], [Lee et al 2019][kenton-paper], [Lee et al 2019][kenton-paper], [Dua et al][drop-paper] and [Agarwal et al 2019][wikisql-sota-paper], respectively.



## Quick Run on NQ and TriviaQA-unfiltered

```
python 3.5
PyTorch 1.1.0
```

Download Data and BERT, and unzip them in the current directory.

- [BERT][bert-model-link]: BERT Base Uncased in PyTorch
- [Preprocessed Data][preprocessed-data-link]: Preprocessed open-domain QA datasets with paragraphs retrieved through TF-IDF and BM25 (details below).
- [Data][data-link]: Original data before preprocessing, which contains `id`, `question` and `answers` only (details below). This is not required for running the model, but just in case you need data before preprocessing.

Then, you can do
```
# NQ
./run.sh nq first-only
./run.sh nq mml
./run.sh nq hard-em 4000
# TriviaQA
./run.sh triviaqa first-only
./run.sh triviaqa mml
./run.sh triviaqa hard-em 4000
```

## Details about data
Here we release preprocessed data and source for our experiments on two open-domain QA datasets, NaturalQuestions open-domain version ([Kwiatkowski et al 2019][nq-paper]) and TriviaQA-unfiltered ([Joshi et al 2017][triviaqa-paper]).

For both datasets, we treat the dev set as the test set, and split the train set into 90/10 for training and development, following conventions that were also used in [Chen et al 2017][drqa-paper] and [Lee et al 2019][kenton-paper].
For NaturalQuestions, follwoing [Lee et al 2019][kenton-paper], we take a subset of questions with short answers up to 5 tokens.

You can download this data from [here][data-link]. Each datapoint contains
- `id` a string, example id matching with the original data
- `question`: a string
- `answers`: a list of string


For preprocessing, we retrieve paragraphs for each question through TF-IDF (for document retrieval; using DrQA from [Chen et al 2017][drqa-paper]) and BM25 (for further paragraph retrieval). We filter train examples where the retriever fails to retrieve any paragraph with the answer text. 
Preprocessed data with retrieved paragraphs can be downloaded from [here][preprocessed-data-link].

### How to use your own preprocessed data

To use your own data, each line of the data file should be a dictionary (can be decoded by json) containing
- `id`: example id
- `question`: question (string)
- `context`: a list where each item is a tokenized paragraph (a list of list of string)
- `answers`: a list where i-th item is an answer entry in i-th paragraph of `context`; answer entry is a dictionary containing
          (1) `text`: answer text
          (2) `word_start`/`word_end`: index of the first/last answer word in the paragraph
- `final_answers`: a list of answer texts for evaluation; `text` in `answers` are always included in `final_answers`

Example:
```
{
  'id': 'user-input-0',
  'question': 'Which city is University of Washington located in?',
  'context': [["The", "University", "of", "Washington", "is", "a", "public", "research", "university", "in", "Seattle", ",", "Washington", ...],
              ["University", "of", "Washington", "has", "been", "affiliated", "with", "many", "notable", "alumni", "and", "faculty", ",", "including", ...]],
  'answers': [[{'text': 'Seattle', 'word_start': 10, 'word_end': 10}, {'text': 'Seattle, Washington', 'word_start': 10, 'word_end': 12}],
              []],
  'final_answers': ["Seattle", "Seattle, Washington"]
}
```

## Details abou the model
The model architecture is exactly same as [Min et al 2019][acl-paper]. We only modify loss functions to have different variations.
You can check the exact command line for training and evaluating the model in `run.sh`. Some useful flags are as follows.

- `--train_batch_size`: batch size for training; experiments reported in the paper use batch size of 192
- `--predict_batch_size`: batch size for evaluating
- `--loss_type`: learning method, one of
            - `first-only`: only considering answer span appearing earliest in the paragraph
            - `mml`: maximum marginal likelihood objective
            - `hard-em`: hard em objective (our main objective)
- `--tau`: hyperparameters for hard-em objective; only matters when `loss_type` is `hard-em`; experiments reported in the paper use 4000 for TriviaQA-unfiltered and 8000 for NaturalQuestions
- `--init_checkpoint`: model checkpoint to load; for training, it should be BERT checkpoint; for evaluating, it should be trained model
- `--output_dir`: directory to store trained model and predictions
- `--debug`: running experiment with only first 50 examples; useful for making sure the code is running
- `--eval_period`: interval to evaluate the model on the dev data
- `--n_paragraphs`: number of paragraphs per a question for evaluation; you can specify multiple numbers (`"10,20,40,80"`) to see scores on different number of paragraphs
- `--prefix`: prefix when storing predictions during evaluation
- `--verbose`: specify to see progress bar for loading data, training and evaluating



[paper-pdf-link]: TODO
[bert-model-link]: https://drive.google.com/file/d/1XaMX-u5ZkWGH3f0gPrDtrBK1lKDU-QFk/view?usp=sharing
[data-link]: https://drive.google.com/file/d/1qsN5Oyi_OtT2LyaFZFH26vT8Sqjb89-s/view?usp=sharing
[preprocessed-data-link]: https://drive.google.com/file/d/1FqTr6NzZf0CQ3FmA2dxF9R-2X0--CmBf/view?usp=sharing
[nq-paper]: https://storage.googleapis.com/pub-tools-public-publication-data/pdf/1f7b46b5378d757553d3e92ead36bda2e4254244.pdf
[kenton-paper]: https://arxiv.org/pdf/1906.00300.pdf
[triviaqa-paper]: https://arxiv.org/pdf/1705.03551.pdf
[drqa-paper]: https://arxiv.org/pdf/1704.00051.pdf
[acl-paper]: https://arxiv.org/pdf/1906.02900.pdf
[triviaqa-sota-paper]: https://aclweb.org/anthology/P18-1158
[narrativeqa-sota-paper]: https://arxiv.org/pdf/1901.02262.pdf
[drop-paper]: https://arxiv.org/pdf/1903.00161.pdf
[wikisql-sota-paper]: https://arxiv.org/pdf/1902.07198.pdf
[narrativeqa-paper]: https://arxiv.org/pdf/1712.07040.pdf
[wikisql-paper]: https://arxiv.org/pdf/1709.00103.pdf


