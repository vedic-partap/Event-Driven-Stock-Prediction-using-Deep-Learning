# Event-Driven Stock Prediction using Deep-Learning
A deep learning method for event driven stock market prediction.  Deep learning is useful for event-driven stock price movement prediction by proposing a novel neural tensor network for learning event embedding, and using a deep convolutional neural network to model the combined influence of long-term events and short-term events on stock price movements

### Important concept need to be learnt : 
1. NLP - [http://web.stanford.edu/class/cs224n](http://web.stanford.edu/class/cs224n/)
2. GloVe - GloVe- [https://nlp.stanford.edu/pubs/glove.pdf](https://nlp.stanford.edu/pubs/glove.pdf)
3. NLT - [www.nltk.org](www.nltk.org)
4. Convolutional Neural Network - [http://www.aclweb.org/anthology/D14-1181](http://www.aclweb.org/anthology/D14-1181)


## Sentiment Analysis for Event-Driven Stock Prediction

Also try to visualize the data on smaller dimension using PCA to get the good idea of the problem

Use NLP to predict stock price movement based on news from Reuters, we need the following 5 steps:


1. Collection of Data ( essential and tricky task ) 

    1.1 get the whole ticker list

    1.2 crawl news 
    
    1.3 crawl prices using urllib2 (Yahoo Finance API is outdated)

2. Train the GloVe in corpus in NLTK

    2.1 build the word-word co-occurrence matrix
  
    2.2 factorizing the weighted log of the co-occurrence matrix
  
3. Feature Engineering
  
    3.2 Unify word format: unify tense, singular & plural, remove punctuations & stop words
  
    3.2 Extract feature using feature hashing based on the trained word vector (step 2)
  
    3.3 Pad word senquence (essentially a matrix) to keep the same dimension
  
4. Trained a ConvNet to predict the stock price movement based on a reasonable parameter selection
5. The result shows a significant 1-2% improve on the test set

Use the following script to crawl it and format it to our local file

#### 1 Crawling of Data

```python
./crawler_reuters.py # we can relate the news with company and date, this is more precise than Bloomberg News
```

Yahoo Finanace is also a great place to collect the data  

### 2 Word Embedding
To use our customized word vector, apply GloVe to train word vector from Reuters corpus in NLTK

```python
./embeddingWord.py
```
### 3. Feature Engineering

Projection of word to word vector
Seperate test set away from training+validation test, otherwise we would get a too optimistic result.

```python
./genFeatureMatrix.py
```
Here there is important point to note when we are separating the Cross Validation set and the Training Set. The shuffiling of data can create a very large mistake and untraceble. Consider we have news that are similar in contex but the language of news are slightly different and the got separated in training and the cross validation set. Then the error in cross validation set get biased as the model is already trained against that model so that example will be of no use and effective cross validation set reduces.

### 4. Train a ConvoNet to predict the stock price movement. 
```python
./model_cnn.py
```

### Other Reading Materials
1. H Lee, etc, [On the Importance of Text Analysis for Stock Price Prediction](http://nlp.stanford.edu/pubs/lrec2014-stock.pdf), LREC, 20145. Xiao Ding, [Deep Learning for Event-Driven Stock Prediction](http://ijcai.org/Proceedings/15/Papers/329.pdf), IJCAI2015
2. [IMPLEMENTING A CNN FOR TEXT CLASSIFICATION IN TENSORFLOW](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/)
3. [Keras predict sentiment-movie-reviews using deep learning](http://machinelearningmastery.com/predict-sentiment-movie-reviews-using-deep-learning/)
4. [Keras sequence-classification-lstm-recurrent-neural-networks](http://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/)
5. [tf-idf + t-sne](https://github.com/lazyprogrammer/machine_learning_examples/blob/master/nlp_class2/tfidf_tsne.py)
6. [Implementation of CNN in sequence classification](https://github.com/dennybritz/cnn-text-classification-tf)
7. [Getting Started with Word2Vec and GloVe in Python](http://textminingonline.com/getting-started-with-word2vec-and-glove-in-python)

