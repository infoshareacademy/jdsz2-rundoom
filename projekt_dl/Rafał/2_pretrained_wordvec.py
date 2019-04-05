import pandas as pd
import keras
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.models import word2vec
import gensim

data = pd.read_json('News_Category_Dataset_v2.json', orient='values', lines=True)

# Ograniczenie zakresu czasowego danych do roku 2012
data = data.loc[data['date'].dt.year == 2012]

num_categories = len(set(data.category))

X = data['headline'].values
y = data['category'].values



word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('vectors.txt', binary=False)

slownik = {
    'data' : word2vec_model
}

import pickle
with open('temp.pickle', 'wb') as f:
    pickle.dump(slownik, f)

with open('temp.pickle', 'rb') as f:
    loaded_pickle = pickle.load(f)

words = loaded_pickle['data']



word2index = {token: token_index for token_index, token in enumerate(words.index2word)}

embedding_matrix = words.wv.syn0