# Transfer Learning for Low-Resource Sentiment Analysis

In this project ([arXiv](https://arxiv.org/abs/2304.04703)), we provide a gold standard benchmark dataset of central Kurdish tweets that have been annotated by humans, along with another silver standard dataset that is automatically annotated. The gold standard dataset contains 1189 samples with positive, negative, neutral, mixed, and a none class which indicates that the tweet is objective with no sentiments. The silver standard dataset contains 1500 samples for three classes of positive, negative, and neutral.


We experimented with several machine learning models to classify the tweets into their respective sentiment categories. We have used Logistirc Regression, Decision Trees, SVM, Random Forest, and BiLSTM models. We have trained these models in two different settings, one with emojis and the other without emojis.
Requirements



## How to use   
    Clone the repository to your local machine.
    pip install -r requirements.txt
    Open the Jupyter Notebook and run the code cells to train and evaluate the models.
    The trained models can be used to predict the sentiment of new tweets.

## Cite This Work

```
@article{hameed2023sentiment,
      author = {Hameed, Razhan and Ahmadi, Sina and Daneshfar, Fatemeh},
      title = {Transfer Learning for Low-Resource Sentiment Analysis},
      year = {2023},
      publisher = {Association for Computing Machinery},
      abstract = {Sentiment analysis is the process of identifying and extracting subjective information from text. Despite the advances to employ cross-lingual approaches in an automatic way, the implementation and evaluation of sentiment analysis systems require language-specific data to consider various sociocultural and linguistic peculiarities. In this paper, the collection and annotation of a dataset are described for sentiment analysis of Central Kurdish. We explore a few classical machine learning and neural network-based techniques for this task. Additionally, we employ an approach in transfer learning to leverage pretrained models for data augmentation. We demonstrate that data augmentation achieves a high F1 score and accuracy despite the difficulty of the task.},
      note = {Under review},
      journal = {ACM Trans. Asian Low-Resour. Lang. Inf. Process.}
}
```
