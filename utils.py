import re
import string
import pandas as pd
from bs4 import BeautifulSoup

from klpt.preprocess import Preprocess
from klpt.tokenize import Tokenize
from klpt.stem import Stem

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

import mplcairo
import matplotlib as mpl
mpl.use("module://mplcairo.qt")
import matplotlib.pyplot as plt



preprocessor_ckb = Preprocess("Sorani", "Arabic", numeral="Latin")
tokenizer_ckb = Tokenize("Sorani", "Arabic")
stemmer = Stem("Sorani", "Arabic")


def lemmatize(word):
    stem = stemmer.lemmatize(word)
    if stem:
        return stem[0]
    else:
        return word
    
    

def clean_text(text, lemma=False, emoji=True):
    soup = BeautifulSoup(text, "html.parser")  # remove html tags
    text = soup.get_text()
    text = preprocessor_ckb.preprocess(text)
    emoji_clean = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    if emoji:
        text = emoji_clean.sub(r"", text)
    # Replace emoji with the coresponding word
    text = re.sub(r"\.(?=\S)", ". ", text)  # add space after full stop
    text = re.sub(r"http\S+", "", text)  # remove urls
#     punctuations = re.compile(
#         rf"[{string.punctuation}\u060c\u060d\u061b\u061f\u00bb\u00ab\u06D6-\u06ED\u005c]+"
#     )
    punctuations = re.compile(
        rf"[{string.punctuation}،.؟\u06D6-\u06ED]+"
    )
    text = re.sub(punctuations, r" ", text)
    if lemma:
        text = " ".join(
            [
                lemmatize(word)
                for word in text.split()
#                 if word not in preprocessor_ckb.stopwords
            ]
        )  # lemmatize
    else:
        text = " ".join(
            [word for word in text.split()] # if word not in preprocessor_ckb.stopwords
        )
    return text

# https://towardsdatascience.com/multilabel-text-classification-done-right-using-scikit-learn-and-stacked-generalization-f5df2defc3b5#0ccf
class ClfSwitcher(BaseEstimator):
    
    def __init__(self, estimator=RandomForestClassifier()):
        """
        A Custom BaseEstimator that can switch between classifiers.
        
        Parameters
        ----------
        estimator: sklearn object, the classifier
        """
        self.estimator = estimator
    
    def fit(self, X, y):
        self.estimator.fit(X, y)
        return self
    
    def predict(self, X):
        return self.estimator.predict(X)
    
    def predict_proba(self, X):
        return self.estimator.predict_proba(X)
    
    def score(self, X, y):
        return self.estimator.score(X, y)
    

def score(y_true, y_pred, index):
    """Calculate precision, recall, and f1 score"""
    
    metrics = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    acc = accuracy_score(y_true, y_pred)
    performance = {'precision': metrics[0], 'recall': metrics[1], 'f1': metrics[2], 'accuracy': acc }
    return pd.DataFrame(performance, index=[index])

def plot_metrics(scores, title='Score of Highest Performing Models'):
    """Plot scores of best models in a barplot based on f1 score"""
    
    scores.sort_values('f1', ascending=False).plot(
        kind='bar',
        figsize=(16, 8),
        title=title,
        ylabel='score'
    )
    plt.ylim(bottom=0.0)
    plt.savefig("./figures/figure.pdf", format="pdf")
    plt.show()
    
