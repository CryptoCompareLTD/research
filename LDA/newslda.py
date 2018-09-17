"""
Implement LDA on documents using guided classes informed by the tags used on the CryptoCompare news API. 

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import guidedlda
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text 

class NewsLDA():
    """
    Perform guided LDA using collapsed Gibbs sampling on the target data. 

    Parameters
    -----------
    data: pd.Series 
        Object containing the documents to extract topics from.
        Generate using the result of the getnews() pickle object.
        df['document'] = df['title']+ ' ' + df['body']
    
    n_iter: int, default = 500
        The number of Gibbs sampling sweeps to use. 

    eta: float, default = 0.2, eta > 0
        The Dirichlet prior on words to topics. The higher the value, 
        the greater the mixture of words per topic.

    alpha: float, default = 0.5, alpha > 0
        The Dirichlet prior on documents to topics. The higher the value,
        the more topics per document.

    seed_confidence: float, default = 10
        The additional value to add to the Dirichlet prior for the seed
        topics. 
    """

    def __init__(self, data, n_iter=500, eta=0.2, alpha=0.2, seed_confidence=10):
        # Generate sparse matrix representation of documents with stopwords removed. 
        self._stopwords = text.ENGLISH_STOP_WORDS.union(['appeared','8217', '8230', '000'])
        self._vectoriser = CountVectorizer(stop_words = self._stopwords, max_features = 1500)
        self._data = list(data.values)
        self._docs = self._vectoriser.fit_transform(self._data)
        self._features = self._vectoriser.get_feature_names()

        # Specify the guided topics 
        self._seed_topics = [['btc', 'bitcoin', 'satoshi'],
               ['eth', 'ethereum', 'vitalik', 'foundation'],
               ['altcoin', 'altcoins', 'ltc', 'litecoin', 'xmr', 'monero','zec', 'zcash','etc', 'classic', 'xrp', 'ripple', 'trx', 'tron', 'ada', 'cardano', 'dash', 'digitalcash', 'xtz', 'tezoz', 'usdt', 'tether'], 
               ['mining', 'hashrate', 'hashing', 'pools', 'reward'],
               ['exchange', 'bitfinex', 'poloniex', 'binance'],
               ['market', 'markets', 'analysis', 'index', 'prices'],
               ['asia', 'china', 'korea', 'japan', 'hong', 'singapore', 'taiwan'],
               ['icos', 'ico', 'offering', 'token', 'tokens', 'raise', 'raised'],
               ['regulation', 'legal', 'law', 'tax', 'taxes'],
               ['blockchain', 'protocol', 'scaling'],
               ['bull', 'bear', 'bullish', 'rally', 'bearish', 'trading'],
               ['technology', 'tech'],
               ['ledger', 'trezor', 'keepkey', 'coinomi', 'jaxx', 'myetherwallet'],
               ['fiat', 'reserve', 'gold', 'bank', 'dollar', 'pound', 'euro', 'yen'], 
               ['business', 'investor', 'investors', 'revenue', 'enterprise', 'commerce'],
               ['commodity', 'oil', 'oil-backed'], 
               ['sponsored', 'press', 'release'],
               ['theft', 'stolen', 'scam', 'criminal']]
        self.topic_names = ['btc', 'eth', 'altcoins', 'mining', 'exchange', 'market', 'asia', 'ico', 'regulation', 'blockchain', 'trading', 'technology', 'wallet', 'fiat', 'business', 'commodity', 'sponsored', 'criminal']
        self._n_topics = len(self.topic_names)
    
        # Define LDA model parameters 
        self.seed_confidence = seed_confidence
        self._model = guidedlda.GuidedLDA(self._n_topics, n_iter=n_iter, alpha=alpha, eta=eta)

    def display_topics(self, no_top_words):
        """
        Display the main topic keywords, after the model has been fit.
        Adapted from https://bit.ly/2QkGBcD
        """
        for topic_id, topic in enumerate(self._model.topic_word_):
            print("Topic {}:".format(self.topic_names[topic_id]))
            print(" ".join([self._features[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

    def example_documents(self, topic_name, threshold= 0.7):
        """
        Display example documents mainly belonging to a topic. 
        """
        topic = self.topic_names.index(topic_name)
        docs = []
        print("Topic: {}".format(topic_name))
        for i, doc_dist in enumerate(self._model.doc_topic_):
            if doc_dist[topic]>0.7:
                docs.append(self._data[i])
        for doc in docs:
            print(doc + '\n')

    def test_document(self, n):
        """
        Display the topics associated with a document and the document. 
        """
        print('Document: \n {}'.format(self._docs[n]))
        print('Topic Assignments: \n')
        for i, val in enumerate(self._model.doc_topic_[n]):
            print(self.topic_names[i] + ': ',  val)

    def display_assignment(self):
        assignments = [sum(m) for m in self._model.doc_topic_.T]
        plt.pie(assignments, labels = self.topic_names)
        plt.show()

    def _seed_dict(self):
        """
        Generate the seed list dict used for guided LDA. 
        Removes words from the seed that are not in the features.
        """
        seed_list={}
        for topic, row in enumerate(self._seed_topics):
            for word in row:
                try:
                    seed_list[self._features.index(word)] = topic
                except ValueError:
                    self._seed_topics[topic].remove(word)
                    pass
        self._seed_list = seed_list

    def guided_lda(self):
        self._seed_dict()
        self._model.fit(self._docs, seed_topics=self._seed_list, seed_confidence=self.seed_confidence)
        self.assignments = self._model.doc_topic_




