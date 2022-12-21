For more details you can refer to paper in the following [link](https://arxiv.org/abs/2212.09701)
If you find this repository helpful, please cite the [paper](https://arxiv.org/abs/2212.09701)


# Persian-Summarization

# Statistical and semantical text summarizer in Persian language

It’s a project for text summarization in Persian language. It uses text summarization of [Gensim python library](https://github.com/RaRe-Technologies/gensim) for implementing [TextRank algorithm](https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf). This algorithm assumes each sentence a node in a graph and returns nodes with highest relation with other nodes (sentences). In other words it returns most important nodes with some statistical calculation and does not include any semantics of the sentences. For instance if you use different words for the same meaning it won’t recognize and assumes they are different which in reality they are not. For solving this problem and including semantic in the result I trained a doc2vec model by doc2vec.py in Genism with [Hamshahri corpus](http://dbrg.ut.ac.ir/hamshahri/) as training set. The doc2vec model is included in the repository (my_model_sents_from_res2.doc2vec). I used this model for calculating similarity of two sentences for weighting the graph edges. (instead of weighting based on some tf-idf algorithm which is used in Gensim) and return the result by TextRank algorithm.

Some modification is made on Gensim library for making it compatible with Persian language, I used [Hazm library]() for text normalizing, sentence tokenizing and POS tagging.


## Python pagages versions you need to install on your device

`pip install six == 1.11.0`

`pip install gensim == 3.1.0`

`pip install numpy == 1.11.3`

`pip install scipy == 1.0.0`

`pip install hazm==0.5.2`

## How to start

copy summarization file and replace it with the one in Gensim library. In play.py you can see an example of text summarization with the command below:

`summarize(text, ratio, word_count)`

ratio is 0.2 and word_count is None by default. ratio returns the fraction of the input text you want to summarize and word_count specify minimum number of words you want in the result summarization.

You can train your own doc2vec model and load that in your project instead of the file included in project also POS tagger model in resource folder as well.
The stopwords in STOPWORD file is obtained from [persian-stopwords](https://github.com/kharazi/persian-stopwords)

### Thanks
I developed this project at [Irsapardaz Pasargad](http://www.irsapardaz.ir/1970/%D8%AE%D9%84%D8%A7%D8%B5%D9%87-%D8%B3%D8%A7%D8%B2-%D9%85%D8%AA%D9%86-%D9%81%D8%A7%D8%B1%D8%B3%DB%8C-%D8%A8%D8%A7-%D8%A7%D8%B3%D8%AA%D9%81%D8%A7%D8%AF%D9%87-%D8%A7%D8%B2-%D8%B1%D9%88%D8%A7%D8%A8%D8%B7). Thanks to Mr. [Amin Mozhgani](https://github.com/AminMozhgani) for his selfless helps during this project.

#### Contact
mina.smz2016@gmail.com
