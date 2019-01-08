import re
import sys
from collections import defaultdict
from os import cpu_count, walk, path
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

import conf

csvfile = '../Semantic-Analysis-Regulations/regulations_data_analysis.csv'
doc2vec_model_name = 'doc2vec_model_{}'.format(conf.DATA_SOURCE)

nltk.download('punkt')
nltk.download('stopwords')


def tokenize(text: str) -> List[str]:
    # Tokenize
    tokenized_text: List[str] = nltk.word_tokenize(text)

    # Remove unwanted words (numbers and stopwords)
    stops: List[str] = stopwords.words('english')
    # TODO: Do we need to filter out numbers?
    tokenized_text: List[str] = [token for token in tokenized_text
                                 if re.match(r'.*[A-Za-z].*', token)
                                 and token not in stops]

    # Stemming
    # https://www.nltk.org/api/nltk.stem.html#module-nltk.stem.porter
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokenized_text]


def build_model(documents: List[str]) -> None:
    print('Building Doc2Vec model')

    model = Doc2Vec(vector_size=conf.VECTOR_SIZE,
                    window=conf.WINDOW,
                    min_count=conf.MIN_COUNT,
                    workers=cpu_count() - 1,
                    epochs=conf.EPOCHS)
    model.build_vocab(documents)
    model.train(documents,
                total_examples=len(documents),
                epochs=model.iter)
    model.save(doc2vec_model_name)


def load_model() -> Doc2Vec:
    return Doc2Vec.load(doc2vec_model_name)


def cluster() -> Tuple[KMeans, ndarray]:
    kmeans_model = KMeans(n_clusters=conf.N_CLUSTERS,
                          n_jobs=-1,
                          verbose=True
                          )
    colors = kmeans_model.fit_predict(load_model().docvecs.vectors_docs)
    return (kmeans_model, colors)


if __name__ == '__main__':
    documents: List[TaggedDocument] = []
    tags: List[str] = []
    if conf.DATA_SOURCE == 'long_title':
        with open(csvfile, 'r') as f:
            df = pd.read_csv(f)
            tags = long_titles = df['long_title']
        documents = [TaggedDocument(doc, [i])
                     for i, doc in enumerate([tokenize(title)
                                              for title in long_titles])]
    elif conf.DATA_SOURCE == 'full_text':
        for root, _, files in walk('txt'):
            tags = sorted(files)
            for i, name in enumerate(tags):
                print('Loading file {}'.format(name))
                with open(path.join(root, name), 'r') as f:
                    documents.append(TaggedDocument(tokenize(f.read()), [i]))
            break
    else:
        print('Invalid data source')
        sys.exit(1)

    build_model(documents)
    kmeans_model, colors = cluster()
    centers = kmeans_model.cluster_centers_

    clusters = defaultdict(list)
    df = pd.DataFrame(sorted(list(zip(colors, tags)), key=lambda x: x[0]))

    df.to_pickle('data_frame_{}'.format(conf.DATA_SOURCE))
    df.to_csv('cluster_result_{}.csv'.format(conf.DATA_SOURCE),
              index=False)

    # print(centers)
    # print(kmeans_model.labels_)

    # Got multi-dimensional vectors in the result. Project to plot
    pca_2d = PCA(n_components=2).fit_transform(load_model().docvecs.vectors_docs)
    plt.scatter(pca_2d[:, 0], pca_2d[:, 1],
                c=colors,
                picker=True)
    plt.scatter(centers[:, 0], centers[:, 1],
                c='black',
                s=100,
                alpha=0.3)

    plt.show()