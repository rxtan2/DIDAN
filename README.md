# Detecting Cross-Modal Inconsistency to Defend Against Neural Fake News

![alt text](motivational.png)

This repository contains a PyTorch implementation of the paper [Detecting Cross-Modal Inconsistency to Defend Against Neural Fake News](https://arxiv.org/abs/2009.07698) accepted at EMNLP 2020. If you find this implementation or the paper helpful, please consider citing:

    @InProceedings{tanDIDAN2020,
         author={Reuben Tan and Bryan A. Plummer and Kate Saenko},
         title={Detecting Cross-Modal Inconsistency to Defend Against Neural Fake News},
         booktitle={Empirical Methods in Natural Language Processing (EMNLP)},
         year={2020} }
    
# Dependencies

1. Python 3.6
2. Pytorch version 1.2.0

# Preprocess Data

To convert the articles and captions into the required input format, please go to https://github.com/nlpyang/PreSumm/blob/master/README.md and carry out steps 3 to 5 of data preparation.

# Required Arguments

1. captioning_dataset_path: Path to GoodNews captioning dataset json file
2. fake_articles: Path to generated articles
3. image_representations_dir: Directory which contains the object representations of images
4. real_articles_dir: Directory which contains the preprocessed Torch text files for real articles
5. fake_articles_dir: Directory which contains the preprocessed Torch text files for generated articles
6. real_captions_dir: Directory which contains the preprocessed Torch text files for real captions
7. ner_dir: Directory which contains a dictionary of named entities for each article and caption
