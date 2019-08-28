# A Discrete Hard EM Approach for Weakly Supervised Question Answering






You can use our hard EM updates for any weakly-supervised QA task, using any model architecture. This is an example code for open-domain question answering using BERT QA model.


## Quick Run on NQ and TriviaQA-unfiltered

First, download Data and BERT, and unzip them in the current directory.

- [BERT][bert-model-link]: BERT Base Uncased in PyTorch
- [Data][data-link]: Preprocessed open-domain QA datasets with paragraphs retrieved through TF-IDF and BM25.

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
We experiment on two open-domain QA datasets, NaturalQuestions open-domain version ([Kwiatkowski et al 2019][nq-paper], filtered by [Lee et al 2019][kenton-paper]) and TriviaQA-unfiltered ([Joshi et al 2017][triviaqa-paper]). We retrieve paragraphs for each question through TF-IDF (using Document Retriever from [Chen et al 2017][drqa-paper]) and BM25. We tokenizer them to be up to 300 tokens when splitted by BERT Tokenizer, which can be done optionally.

To use your own data, each line of the data file is a dictionary (can be decoded by json) containing
- `id`: example id
- `question`: question (string)
- `context`: a list where each item is a tokenized paragraph (a list of list of string)
- `answers`: a list where i-th item is an answer entry in i-th paragraph of `context`; answer entry is a dictionary containing
          - `text`: answer text
          - `word_start`: index of the first answer word in the paragraph
          - `word_end`: index of the last answer word in the paragraph
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


## Summary of the results

In the paper, we experiment on six different QA tasks. Below is the results reported in the paper (all on the test set).


| Dataset | TriviaQA | NarrativeQA | TriviaQA-unfiltered | NaturalQuestions | DROP | WikiSQL |
|---|---|---|---|---|---|---|
| First-only | 64.9 | 57.4 | 48.1 | 23.6 | 42.9 | - |
| MML | 65.5 | 56.1 | 47.4 | 25.8 | 39.7 | 70.5 |
| Hard-EM (Ours) | 67.1 | **58.8** | **50.9** | **28.1** | **52.8** | **83.9** |
| SOTA | **71.4** | 54.7 | 47.1 | 26.5 | 43.8 | 74.8 |

(SOTA are from [Wang et al 2018][triviaqa-sota-paper], [Nishida et al 2019][narrativeqa-sota-paper], [Lee et al 2019][kenton-paper], [Lee et al 2019][kenton-paper], [Dua et al][drop-paper] and [Agarwal et al 2019][wikisql-sota-paper], respectively.

[bert-model-link]: https://drive.google.com/file/d/1XaMX-u5ZkWGH3f0gPrDtrBK1lKDU-QFk/view?usp=sharing
[data-link]()
[nq-paper]: []
[kenton-paper]: []
[triviaqa-paper]: []
[drqa-paper]:
[acl-paper]:
[triviaqa-sota-paper]:
[narrativeqa-sota-paper]:
[drop-paper]:
[wikisql-sota-paper]:

























