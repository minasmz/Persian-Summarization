#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

import math
from six import iteritems
from six.moves import xrange
import gensim
import gensim.models.doc2vec
#from gensim.summarization import summarize

documents = gensim.models.doc2vec.TaggedLineDocument('result2.txt')
model_loaded = gensim.models.doc2vec.Doc2Vec.load('my_model_sents_from_res2.doc2vec')



# BM25 parameters.
PARAM_K1 = 1.5
PARAM_B = 0.75
EPSILON = 0.25


class BM25(object):

    def __init__(self, corpus):
        self.corpus_size = len(corpus)
        self.avgdl = sum(float(len(x)) for x in corpus) / self.corpus_size
        self.corpus = corpus
        self.f = []
        self.df = {}
        self.idf = {}
        self.initialize()

    def initialize(self):
        for document in self.corpus:
            frequencies = {}
            for word in document:
                if word not in frequencies:
                    frequencies[word] = 0
                frequencies[word] += 1
            self.f.append(frequencies)

            for word, freq in iteritems(frequencies):
                if word not in self.df:
                    self.df[word] = 0
                self.df[word] += 1

        for word, freq in iteritems(self.df):
            self.idf[word] = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)

    def get_score(self, document, index, average_idf):
        score = 0
        for word in document:
            if word not in self.f[index]:
                continue
            idf = self.idf[word] if self.idf[word] >= 0 else EPSILON * average_idf
            score += (idf * self.f[index][word] * (PARAM_K1 + 1)
                      / (self.f[index][word] + PARAM_K1 * (1 - PARAM_B + PARAM_B * self.corpus_size / self.avgdl)))
        return score

    def get_scores(self, document, average_idf):
        scores = []
        for index in xrange(self.corpus_size):
            score = self.get_score(document, index, average_idf)
            scores.append(score)
        return scores


def _get_sentences(sentences, corpus):
    hashable_corpus = _build_hasheable_corpus(corpus)
    sentences_by_corpus = dict(zip(hashable_corpus, sentences))
    OrigProcessedSentences =[sentences_by_corpus[tuple(important_doc)] for important_doc in corpus]
    return OrigProcessedSentences[1]


def get_bm25_weights(corpus):
    bm25 = BM25(corpus)
    average_idf = sum(float(val) for val in bm25.idf.values()) / len(bm25.idf)

    weights = []
    for doc in corpus:
        scores = bm25.get_scores(doc, average_idf)
        weights.append(scores)
    return weights

def _build_hasheable_corpus(corpus):
    return [tuple(doc) for doc in corpus]



def get_w2v_scores(doc, sentences, corpus):
    corpus_size = len(corpus)
    scores = []
    sentences_by_corpus = dict(zip(corpus, sentences))
    OrigProcessedSentences =[sentences_by_corpus[tuple(each_doc)] for each_doc in corpus]
    OrigProcessedSentences1 =sentences_by_corpus[tuple(doc)]
    proc1 = OrigProcessedSentences1[1]
    for index in range (len(OrigProcessedSentences)):
        OrigProcessedSentences2 = OrigProcessedSentences[index]
        proc2 = OrigProcessedSentences2[1]
        score = model_loaded.docvecs.similarity_unseen_docs(model_loaded,proc1,proc2)
        scores.append(score)
    return scores

#corpus --> document
def get_w2v_weights(corpus,sentences):
    weights = []
    for doc in corpus:
        scores = get_w2v_scores(doc,sentences,corpus)
        weights.append(scores)
    return weights



