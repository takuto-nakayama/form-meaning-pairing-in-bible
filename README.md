# Form Meaning Pairing with BERT

## Overview
This project aims to measure the overall complexity of languages:
(i) by measuring the unpredictability of what meaning a given form represents in a certain context, and
(ii) based on a linguistic unit that does not depend on any given languages.
The project used [Multilingual Bible Corpus](https://christos-c.com/bible/) as the dataset, and bert [multilingual BERT model](https://huggingface.co/google-bert/bert-base-multilingual-cased).

## Procedure
1. Tokenizing an input dataset into subwords.
2. Embedding each subword.
3. Clustering<br>(default clustering method is DBSCAN)
4. Computing Shannon entropy.
5. Visualizing the result with a scatter plot.

## How to Run
The command below is the simplest form:
```
$ python main.py id data-directory
```
, in which `id` refers to the name of directory in which the result will be stored, and `data-directory` to the name of directory that contains a data you want to use.<br>
Data must be a plain text and each line corresponds to a sentence or another kinds of set of linguistic units that represents a context for each word or subword.

options are the below:
- --model_name (str): the name of BERT model you want to use.<br>(DEFAULT=bert-base-multilingual-cased)
- --minimum_frequency (int): the number of the minimum frequency at which a subword in question needs to occur.<br>(DEFAULT=10)
- --brake_trials (int): the number of trials for which you want to stop the processes if the number pf clusters does not vary.<br>(DEFAULT=10)
- --output (bool): If True, the probabilities and entropies of each subword of each language will be output.<br>(DEFAULT=True)
- --pca (bool):If True, embeddings will be compressed with PCA<br>(DEFAULT=False)


## Citation
```
@conference{nakayama2024,
    author      =   {Takuto Nakayama},
    year        =   {2024},
    title       =   {Linguistic complexity through form-meaning pairings: An information theoretical approach to equi-complexity of language},
    booktitle   =   {The 21st International Congress of Linguists},
    address      =   {Poznań, Poland},
    note        =   {Oral Presentation},
}
```