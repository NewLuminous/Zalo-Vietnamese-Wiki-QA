# Vietnamese Wikipedia Question Answering

## Table of contents

* [Problem statement](#problem-statement)
* [Data acquisition](#data-acquisition)
* [Approaches](#approaches)
* [Evaluation](#evaluation)

## Problem statement

Given a question, related paragraphs from a Wikipedia article (in the shuffle order) in Vietnamese, the task is finding paragraph which answers the question of each test case.

This challenge is categorized into the set of basic **Question-Answering** problems. We need to build a system that automatically answer questions posed by humans in a natural language. But this challenge is simpler, we just need to detect the answer provided for the question is good (which can fully answer the question) or bad (which cannot satisfy the question).

This is the challenge provided by **Zalo** in the [Zalo AI Challenge 2019](https://challenge.zalo.ai/portal/question-answering).

## Data acquisition

We have found three datasets which are closed to the requirements of this challenge.
- First, we have the dataset provided by **Zalo** - the host of the contest. This dataset provides **18108** rows of question-answer pairs.

- Second, we have the dataset provided by **Stanford University** called [SQuAD 2.0 (the Stanford Question-Answering Dataset)](https://rajpurkar.github.io/SQuAD-explorer/). This is the standard dataset for the Question-Answering problems for English language. We intend to translate this English data into Vietnamese. This dataset provides **141979** rows of question-answer pairs.

- Finally, we have the dataset called **Vietnamese SQuAD**, which is the combination of two smaller datasets. The first one is [MLQA (MultiLingual Question Answering)](https://github.com/facebookresearch/MLQA) provided by **Facebook Research**, which contains a part of **SQuAD 2.0** that is manually translated into Vietnamese. The second one is the [Vietnamese dataset](https://github.com/mailong25/bert-vietnamese-question-answering/tree/master/dataset) provided by **Long Mai**. This combination dataset provides **9460** rows of question-answer pairs.

## Approaches

We have tried many different baseline models but most of models includes two steps: features extraction and classification.
- With feature extraction, we use: Bag of words, Bag of words + NGrams, Tf-Idf, Tf-Idf + NGrams, Word2Vec and Average Word2Vec.

- With classification, we use: Naive Bayes, KNN, Rocchio, Random Forest, Extra Trees, Linear SVC, LightGBM, XGBoost, CRNN + Attention.

## Evaluation

| Model                                            | Macro-F1 score                                   |
|--------------------------------------------------|--------------------------------------------------|
| KNN                                              | 0.6495627076335737                               |
| Rocchio                                          | 0.6524710046296189                               |
| Naive Bayes                                      | 0.6575807856090050                               |
| Extra Trees                                      | 0.7004486941326811                               |
| LinearSVC                                        | 0.7073252865333168                               |
| LightGBM                                         | 0.7181613875112443                               |
| CRNN + Attention                                 | 0.7668151799465259                               |
| BERT                                             | 0.79 and higher                                  |

The current optimal approach for this problem is [BERT](https://github.com/google-research/bert), which is a Transformer-based machine learning technique for NLP pre-training developed by **Google**. Referencing the Github repository [namnv1113](https://github.com/namnv1113/Nanibot_ZaloAIChallenge2019_VietnameseWikiQA), they use the BERT model and get the approximately **0.79** of **F1 score**. This is currently also the optimal approach for most of Question-Answering problems.

This year (2020), **VinAI** have released their new researcher based on **BERT** called [PhoBERT](https://github.com/VinAIResearch/PhoBERT). This is the pre-trained models for Vietnamese. We have not tried to used this pre-trained models but we believe that the performance can even higher than the previous one.