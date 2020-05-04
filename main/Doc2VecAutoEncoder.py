from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from gensim.parsing.preprocessing import preprocess_string
from sklearn.base import BaseEstimator
from sklearn import utils as skl_utils
from tqdm import tqdm
from main import Fileparser
from sklearn.neural_network import MLPRegressor
import numpy as np
from scipy.spatial.distance import cosine

def key_index(tupple):
    return tupple[0]

def get_computed_similarities(vectors, predicted_vectors):
    # returns the cosine similarities between vectors and predicted vectors
    data_size = len(vectors)
    cosine_similarities = []
    for i in range(data_size):
        cosine_sim_val = (1 - cosine(vectors[i], predicted_vectors[i]))
        cosine_similarities.append((i, cosine_sim_val))

    return cosine_similarities


class Doc2VecTransformer(BaseEstimator):
    # the parameters of the Doc2Vec
    def __init__(self, vector_size=100, learning_rate=0.02, epochs=1):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self._model = None
        self.vector_size = vector_size

    def fit(self, corpus):
        taggedDocs=[]

        for i in range(len(corpus)):
            taggedDocs.append(TaggedDocument(corpus[i].split(),[i]))
        model = Doc2Vec(documents=taggedDocs, vector_size=self.vector_size, )
        dataset=[x for x in tqdm(taggedDocs)]
        for epoch in range(self.epochs):
            model.train(skl_utils.shuffle(dataset), total_examples=len(taggedDocs), epochs=10)
            model.alpha -= self.learning_rate
            model.min_alpha = model.alpha

        self._model = model
        return self

    def transform(self, corpus):
        return np.asmatrix(np.array([self._model.infer_vector(c.split())
                                     for c in corpus]))

def runDoc2VecAutoEncoder(idx):

    # creates a doc2vec encoding for each sentence of the corpus then trains the autoencoder to recognize the patterns
    # makes the prediction and calculates the cosine similarity between the inputs and the predicted outputs
    corpusPath="C:\DiplomaProject\OriginalCorpus\Corpus"+str(idx)+".txt"
    cosPath="C:\DiplomaProject\CosineSimilarities\Cosine"+str(idx)+".txt"
    corpus = Fileparser.get_corpus_from_file(corpusPath)
    corpus.pop()
    #corpus=Fileparser.stemCorpus(corpus)
    doc2vec_tr = Doc2VecTransformer(vector_size=100)
    doc2vec_tr.fit(corpus)
    doc2vec_vectors = doc2vec_tr.transform(corpus)



    auto_encoder = MLPRegressor(hidden_layer_sizes=(50,20, 50, ))
    auto_encoder.fit(doc2vec_vectors, doc2vec_vectors)
    predicted_vectors = auto_encoder.predict(doc2vec_vectors)



    cosine_similarities = get_computed_similarities(vectors=doc2vec_vectors, predicted_vectors=predicted_vectors)
    print(len(cosine_similarities))

    with open(cosPath, 'w', encoding="utf-8", errors='ignore') as f:
        f.truncate(0)
        sorted_cosine_similarities = sorted(cosine_similarities, key=key_index, reverse=False)
        for j in range(len(cosine_similarities)):
            index, cosSim = sorted_cosine_similarities[j]
            f.write(str(cosSim))
            f.write("|")
        f.close()
for i in range(0,40):
    runDoc2VecAutoEncoder(i)
