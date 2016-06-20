

```python
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import re

%matplotlib inline
```

# Read Training and Test Data


```python
train = pd.read_csv('train.csv').fillna('')
test = pd.read_csv('test.csv').fillna('')

cols = ['query', 'product_title', 'product_description']

from itertools import combinations

pairs = list(combinations(cols, 2))
```

# Preprocessing

Apply lowercasing and stemming for each string in the `query`, `product_title`, and `product_description` columns.


```python
stopwords = nltk.corpus.stopwords.words('english')
stemmer = nltk.stem.SnowballStemmer('english')
token_pattern = re.compile(r'(?u)\b\w\w+\b')

def preprocess(text):
    tokens = [token.lower() for token in token_pattern.findall(text)]
    return ' '.join([stemmer.stem(token) for token in tokens if token not in stopwords])

def preprocess_df(df):    
    for col in cols:
        df[col] = df[col].map(preprocess)

preprocess_df(train)
preprocess_df(test)
```

# Feature Extraction

## Generate $n$-grams, $n \in \left\{1, 2, 3\right\}$


```python
from nltk.util import ngrams
from functools import partial

def ngramize(col, n, row):
    # I join the ngrams to avoid w1w2w3 unigrams becoming [(w1,), (w2,), (w3,)]
    return [' '.join(ngram) for ngram in ngrams(row[col].split(), n)]

def generate_ngrams(df):
    for col in cols:
        df['%s_%s' % (col, 'unigram')] = df.apply(partial(ngramize, col, 1), axis='columns')
        df['%s_%s' % (col, 'bigram')] = df.apply(partial(ngramize, col, 2), axis='columns')
        df['%s_%s' % (col, 'trigram')] = df.apply(partial(ngramize, col, 3), axis='columns')
```

## Counting Features

Following some pointers from Owen Zhang (currently ranked 6th in Kaggle), I'm extracting "trivial" features such as the length of text, and number of $n$-grams.


```python
def extract_counting_features(df):
    for col in cols:
        for gram in ['unigram', 'bigram', 'trigram']:
            counted = '%s_%s' % (col, gram)
            df['count_%s' % counted] = df.apply(lambda row: len(row[counted]), axis='columns')
            df['count_dist_%s' % counted] = df.apply(lambda row: len(set(row[counted])), axis='columns')
            df['ratio_%s' % counted] = df['count_dist_%s' % counted] / df['count_%s' % counted]
            df['ratio_%s' % counted] = df['ratio_%s' % counted].fillna(0)  # in case of division by zero, above
            
```

## Distance Features

Following some Wikipedia-ing and asking around the halls of building 37, there are two distance metrics which are equivalent, yet not identical: [Jaccard](https://en.wikipedia.org/wiki/Jaccard_index) and [Dice](https://en.wikipedia.org/wiki/Dice's coefficient). They measure how similar two sets are, but this also works for $n$-grams, as shown in the Dice coefficient Wiki page.


```python
def jaccard(a, b):
    A, B = set(a), set(b)
    try:
        return len(A & B) / len(A | B)
    except ZeroDivisionError:
        return 0.

def dice(a, b):
    A, B = set(a), set(b)
    try:
        return 2 * len(A & B) / (len(A) + len(B))
    except ZeroDivisionError:
        return 0.

def extract_distance_features(df):
    for f in [jaccard, dice]:
        for gram in ['unigram', 'bigram', 'trigram']:
            for p in pairs:
                df['%s_%s_%s_%s' % (f.__name__, gram, p[0], p[1])] = df.apply(lambda row: f(row['%s_%s' % (p[0], gram)],
                                                                                            row['%s_%s' % (p[1], gram)]),
                                                                              axis='columns')
```

## TF-IDF Features

After some head-banging, I figured out that first, I need to `fit` my vectorizer on the combination of `query`, `product_title`, and `product_description`. This way, the vectorizer knows every possible word, and I can use cosine similarity between the `transform`-ed pairs.


```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def extract_tfidf_features(df):
    vectorizer = TfidfVectorizer(ngram_range=(1, 3))
    vectorizer.fit(df[cols[0]] + ' ' + df[cols[1]] + ' ' + df[cols[2]])
    for p in pairs:
        x = vectorizer.transform(df[p[0]])
        y = vectorizer.transform(df[p[1]])
        df['%s_%s_cosim' % p] = cosine_similarity(x, y)[0][0] or 0
```

# Combining Features

The following code is taken from the Python benchmark on Kaggle by Ben Hammer:


```python
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesRegressor

class FeatureMapper:
    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None):
        for column_name, extractor in self.features:
            extractor.fit(X[column_name], y)

    def transform(self, X):
        extracted = []
        for column_name, extractor in self.features:
            fea = extractor.transform(X[column_name])
            if hasattr(fea, "toarray"):
                extracted.append(fea.toarray())
            else:
                extracted.append(fea)
        if len(extracted) > 1:
            return np.concatenate(extracted, axis=1)
        else: 
            return extracted[0]

    def fit_transform(self, X, y=None):
        extracted = []
        for column_name, extractor in self.features:
            fea = extractor.fit_transform(X[column_name], y)
            if hasattr(fea, "toarray"):
                extracted.append(fea.toarray())
            else:
                extracted.append(fea)
        if len(extracted) > 1:
            return np.concatenate(extracted, axis=1)
        else: 
            return extracted[0]

def identity(x):
    return x

class SimpleTransform(BaseEstimator):
    def __init__(self, transformer=identity):
        self.transformer = transformer

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X, y=None):
        return np.array([self.transformer(x) for x in X], ndmin=2).T

features = FeatureMapper([('count_query_unigram', SimpleTransform()),
                          ('count_dist_query_unigram', SimpleTransform()),
                          ('ratio_query_unigram', SimpleTransform()),
                          ('count_query_bigram', SimpleTransform()),
                          ('count_dist_query_bigram', SimpleTransform()),
                          ('ratio_query_bigram', SimpleTransform()),
                          ('count_query_trigram', SimpleTransform()),
                          ('count_dist_query_trigram', SimpleTransform()),
                          ('ratio_query_trigram', SimpleTransform()),
                          ('count_product_title_unigram', SimpleTransform()),
                          ('count_dist_product_title_unigram', SimpleTransform()),
                          ('ratio_product_title_unigram', SimpleTransform()),
                          ('count_product_title_bigram', SimpleTransform()),
                          ('count_dist_product_title_bigram', SimpleTransform()),
                          ('ratio_product_title_bigram', SimpleTransform()),
                          ('count_product_title_trigram', SimpleTransform()),
                          ('count_dist_product_title_trigram', SimpleTransform()),
                          ('ratio_product_title_trigram', SimpleTransform()),
                          ('count_product_description_unigram', SimpleTransform()),
                          ('count_dist_product_description_unigram', SimpleTransform()),
                          ('ratio_product_description_unigram', SimpleTransform()),
                          ('count_product_description_bigram', SimpleTransform()),
                          ('count_dist_product_description_bigram', SimpleTransform()),
                          ('ratio_product_description_bigram', SimpleTransform()),
                          ('count_product_description_trigram', SimpleTransform()),
                          ('count_dist_product_description_trigram', SimpleTransform()),
                          ('ratio_product_description_trigram', SimpleTransform()),
                          ('jaccard_unigram_query_product_title', SimpleTransform()),
                          ('jaccard_unigram_query_product_description', SimpleTransform()),
                          ('jaccard_unigram_product_title_product_description', SimpleTransform()),
                          ('jaccard_bigram_query_product_title', SimpleTransform()),
                          ('jaccard_bigram_query_product_description', SimpleTransform()),
                          ('jaccard_bigram_product_title_product_description', SimpleTransform()),
                          ('jaccard_trigram_query_product_title', SimpleTransform()),
                          ('jaccard_trigram_query_product_description', SimpleTransform()),
                          ('jaccard_trigram_product_title_product_description', SimpleTransform()),
                          ('dice_unigram_query_product_title', SimpleTransform()),
                          ('dice_unigram_query_product_description', SimpleTransform()),
                          ('dice_unigram_product_title_product_description', SimpleTransform()),
                          ('dice_bigram_query_product_title', SimpleTransform()),
                          ('dice_bigram_query_product_description', SimpleTransform()),
                          ('dice_bigram_product_title_product_description', SimpleTransform()),
                          ('dice_trigram_query_product_title', SimpleTransform()),
                          ('dice_trigram_query_product_description', SimpleTransform()),
                          ('dice_trigram_product_title_product_description', SimpleTransform()),
                          ('query_product_title_cosim', SimpleTransform()),
                          ('query_product_description_cosim', SimpleTransform()),
                          ('product_title_product_description_cosim', SimpleTransform())])

def extract_features(df):
    generate_ngrams(df)
    extract_counting_features(df)
    extract_distance_features(df)
    extract_tfidf_features(df)

extract_features(train)
extract_features(test)

pipeline = Pipeline([("extract_features", features),
                     ("regress", ExtraTreesRegressor(n_estimators=100,n_jobs=-1))])

pipeline.fit(train, train["median_relevance"])

predictions = pipeline.predict(test).round().astype(int)

submission = pd.DataFrame({"id": test["id"], "prediction": predictions})
submission.to_csv("ben_daniel_ex4.csv", index=False)
```
