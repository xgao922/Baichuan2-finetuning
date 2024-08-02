# Finetuning Baichuan2 for Story Generation

We fine-tuned Baichuan2 on a corpus consisting of 14988 short stories from STORAL, 6500 news from THUCNews, 919 documents from WikiPedia, and 27 novels from modern Chinese literature.

## Installation

``` bash
$ git clone git@github.com:xgao922/Baichuan2-finetuning.git
$ pip install -r requirements.txt
```

## Data preprocess

The training data should be placed at `/data`, and we preprocess the corpus by removing the punctuation and carefully dealing with the blanks.

``` bash
$ cd /scripts
$ python preprocess_corpus.py
```

## Training

``` bash
$ cd /fine-tune
$ bash train.sh
```

## Inference

``` bash
$ cd /inference
$ bash run_predict.sh
```

## Evaluation

The metric used for evaluation is topk.

