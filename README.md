# chatbot
train a neural net to answer specific question about class schedule and information 

### A bit about word2vec

There are a number of parameter choices that affect the run time and the quality of the final model that is produced. For details on the algorithms below, see the word2vec API documentation as well as the Google documentation. 

* Architecture: Architecture options are skip-gram (default) or continuous bag of words. We found that skip-gram was very slightly slower but produced better results.
* Training algorithm: Hierarchical softmax (default) or negative sampling. For us, the default worked well.
* Downsampling of frequent words: The Google documentation recommends values between .00001 and .001.
* Word vector dimensionality: More features result in longer runtimes, and often, but not always, result in better models. Reasonable values can be in the tens to hundreds; we used 300.
* Context / window size: How many words of context should the training algorithm take into account? 10 seems to work well for hierarchical softmax (more is better, up to a point).
* Worker threads: Number of parallel processes to run. This is computer-specific, but between 4 and 6 should work on most systems.
* Minimum word count: This helps limit the size of the vocabulary to meaningful words. Any word that does not occur at least this many times across all documents is ignored. Reasonable values could be between 10 and 100.

### Preprocessing.py has methods to preprocess the texts, train word2vec and vectorize the train and test sets.

1.`text_to_wordlist(text, remove_stopwords=False)`
  returns a list of processed word in a sentence.
  processing includes: removing numbers and special characters, converting them into lowercase, 
  removing stopwords(optional), stemming and lemmatizing.

2.`text_to_sentences(text, tokenizer,remove_stopwords=False )`
  returns a list of processed sentences in a document. tokenizer is the tokenizer used in this case it is punkt tokenizer

3.`parse_and_clean_sentences(df)`
  returns all the processed sentences in all the documents. df is a pandas dataframe. df should contain df["text"] as the  document column name

4.`train_word2vec(df, num_features = 300, min_word_count = 1, num_workers = 4, context = 4, downsampling = 1e-3)`
  trains word to vec with df["text"] as input data. Saves and returns the model.
  the default parameter values are:

  num_features = 300
  min_word_count = 1
  num_workers = 4       
  context = 4            
  downsampling = 1e-3  

5.`makeFeatureVec(words, model, num_features)`
  counts the average word vectors of a document 

6.`getAvgFeatureVecs(text, model)`
  Given a set of documents (each one a list of words), calculate the average feature vector for each one and return a 2D numpy array 

7.`preprocess_data(df, model)`
  basically this function does everything :) just call this function to return processed and vectorized data.

Follow this order to train wordd2vec and then vectorize the train and test data
* import dataset
`df = pd.read_csv("datafile.csv", header=0, delimiter="\t", quoting=3)`

* train word2vec dataset
`model = train_word2vec(df, num_features, min_word_count, num_workers, context, downsampling)`

* preprocess and vectorize train and test data
`train_data = preprocess_data(trainDataFrame, model)`
`test_data = preprocess_data(testDataFrame, model)`

some useful calls

`from gensim.models import Word2Vec`   to import w2v

`model = train_word2vec(df, num_features, min_word_count, num_workers, context, downsampling)`  to train and save model

`model.most_similar("hi")`   to find similar words to a word

`model = Word2Vec.load("trainedWord2vecmodel")`  to load the saved model model

`model.wv.syn0.shape` to find shape of the model

`model.wv.index2word`  to all the words in the model

`model["word"]`    to features of the word

### Model used for intent classification
We replicated the model used by Yoon Kim from NYU for intent classification. The paper can be found here:
https://arxiv.org/pdf/1408.5882.pdf

### model.py has the code to train the model and save it as protobuf so that it can be deployed as a service.

### commands to train models
run command to train word2vec 
python3 run.py word2vec num_feature min_word_count num_workers context downsampling
e.g `>>python3 run.py word2vec 300 1 4 4 1e-3`

run command to train and save model
e.g `>>python3 run.py model`

### model_predict.py has the chat api
`from model_predict import get_response`
`response = get_response("text", threshold_value_for_fllback)`


