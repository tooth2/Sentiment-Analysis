# Sentiment-Analysis
Sentiment Analysis : PyTorch Implementation, Tensorflow Keras Implementation
The dataset is very popular in Natural Language Processing, usually referred to as the IMDb dataset. It consists of movie reviews from the website imdb.com, each labeled as either 'positive', if the reviewer enjoyed the film, or 'negative' otherwise.

## Implementation Approach

### Dataset
[IMDB Moview Review](https://keras.io/api/datasets/imdb/)

"This is a dataset of 25,000 movies reviews from IMDB, labeled by sentiment (positive/negative). Reviews have been preprocessed, and each review is encoded as a list of word indexes (integers). For convenience, words are indexed by overall frequency in the dataset, so that for instance the integer "3" encodes the 3rd most frequent word in the data. This allows for quick filtering operations such as: "only consider the top 10,000 most common words, but eliminate the top 20 most common words".As a convention, "0" does not stand for a specific word, but instead is used to encode any unknown word.

### Data Preprocessing 
`util.py`
IMDB movie review raw data is from html scrapping. Therefore there are HTML tags that need to be removed. Thus, util.py manages 
- html tag removal, 
- removal of non-letter characters, 
- normalizing uppercase letters by converting them to lowercase, 
- tokenizing, 
- removal of stop words, and 
- stemming the remaining words in each document by using nltk.stem.porte.PorterStemmer


### Classifier
Tree-based algorithms often work quite well on Bag-of-Words as their highly discontinuous and sparse nature is nicely matched by the structure of trees
- Naive Bayes classifier from scikit-learn (specifically, GaussianNB)
- GradientBoostingClassifier from scikit-learn

### RNN Model 
RNN network is designed : embedding layer + LSTM memory cell + Fully Connected + Sigmoid activation
![Network](sentiment_network.png)

|Layer(type)         |        Output Shape       |       Param #   |
|---------------------|------------------| ------------|
|embedding_1 (Embedding)   |   (None, 500, 32)   |        160000    |
|lstm_1 (LSTM)      |          (None, 100)      |         53200     |
|dense_1 (Dense)      |        (None, 1)       |       101      | 

* Total params: 213,301
* Trainable params: 213,301
* Non-trainable params: 0

### Hyperparameter tuning
* batch size 
* number of training epochs

### Deployment 



