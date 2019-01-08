import re
from os import cpu_count
from typing import List, Tuple

import nltk
import pandas as pd
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from numpy import ndarray
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

csvfile = '../Semantic-Analysis-Regulations/regulations_data_analysis.csv'
doc2vec_model_name = 'doc2vec_model'

with open(csvfile, 'r') as f:
    df = pd.read_csv(f)
    long_titles = df['long_title']

nltk.download('punkt')
nltk.download('stopwords')


def tokenize(title: str) -> List[str]:
    # Tokenize
    tokenized_title: List[str] = nltk.word_tokenize(title)

    # Remove unwanted words (numbers and stopwords)
    stops: List[str] = stopwords.words('english')
    # TODO: Do we need to filter out numbers?
    tokenized_title: List[str] = [token for token in tokenized_title
                                  if re.match(r'.*[A-Za-z].*', token)
                                  and token not in stops]

    # Stemming
    # https://www.nltk.org/api/nltk.stem.html#module-nltk.stem.porter
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokenized_title]


def build_model(long_titles: List[str]) -> None:
    print('Building Doc2Vec model')

    # TODO: When switch to full text, can we change the tag to title? Does that improve anything?
    documents = [TaggedDocument(doc, [i])
                 for i, doc in enumerate([tokenize(title)
                                          for title in long_titles])]
    model = Doc2Vec(vector_size=400,
                    window=2,
                    min_count=2,
                    workers=cpu_count(),
                    epochs=40)
    model.build_vocab(documents)
    model.train(documents,
                total_examples=len(documents),
                epochs=model.iter)
    model.save(doc2vec_model_name)


def load_model() -> Doc2Vec:
    return Doc2Vec.load(doc2vec_model_name)


def cluster() -> Tuple[KMeans, ndarray]:
    kmeans_model = KMeans(n_clusters=10,
                          n_jobs=-1,
                          verbose=True)
    colors = kmeans_model.fit_predict(load_model().docvecs.vectors_docs)
    return (kmeans_model, colors)


if __name__ == '__main__':
    build_model(long_titles)
    kmeans_model, colors = cluster()
    centers = kmeans_model.cluster_centers_
    print(centers)
    print(kmeans_model.labels_)

    # Got multi-dimensional vectors in the result. Project to plot
    pca_2d = PCA(n_components=2).fit_transform(load_model().docvecs.vectors_docs)
    plt.scatter(pca_2d[:, 0], pca_2d[:, 1],
                c=colors)
    plt.scatter(centers[:, 0], centers[:, 1],
                c='black',
                s=100,
                alpha=0.3)
    plt.show()
