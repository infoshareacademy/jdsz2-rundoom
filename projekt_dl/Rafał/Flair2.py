from flair.data_fetcher import NLPTaskDataFetcher
from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentLSTMEmbeddings, DocumentRNNEmbeddings
from flair.trainers import ModelTrainer
from pathlib import Path


from flair.models import TextClassifier


corpus = NLPTaskDataFetcher.load_classification_corpus(Path('./'),
                                                       train_file='train.csv', test_file='test.csv', dev_file='dev.csv')

#embedding = [BertEmbeddings()]
#embedding = [BertEmbeddings('bert-base-multilingual-cased')]
#embedding = [WordEmbeddings('glove'), FlairEmbeddings('news-forward-fast'), FlairEmbeddings('news-backward-fast')]

embedding = [WordEmbeddings('glove')]


document_embeddings = DocumentRNNEmbeddings(embedding, hidden_size=512,
                                             reproject_words=True, reproject_words_dimension=256)


classifier = TextClassifier(document_embeddings, label_dictionary=corpus.make_label_dictionary(),
                            multi_label=False)

trainer = ModelTrainer(classifier, corpus)

trainer.train('./', max_epochs=10)
