import pandas as pd
from nltk.corpus import stopwords # Import the stop word list
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
import re
import nltk.data
import logging
import numpy as np

stop = stopwords.words("english")
wordnet_lemmatizer = WordNetLemmatizer()
snowball_stemmer = SnowballStemmer("english")
# Load the punkt tokenizer
tokenizer = nltk.data.load('english.pickle')


def text_to_wordlist(text, remove_stopwords=False):
    # 1. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", text) 
    #
    # 2. Convert to lower case, split into individual words
    words = letters_only.lower().split() 
    #
    # 3. In Python, searching a set is much faster than searching
    #    a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))
    # 
    # 4. Remove stop words
    if remove_stopwords:
        meaningful_words = [w for w in words if not w in stops]  
    else:
        meaningful_words = words
    # 
    # 5. autocorrect spellings
    #auto_correct = [spell(w) for w in meaningful_words]
    # 
    # 6. use stemmer to stem
    stem_words = [snowball_stemmer.stem(w) for w in meaningful_words]
    #
    # 7. use lemmatizer to lemmatize the words
    lemma_words = [wordnet_lemmatizer.lemmatize(w) for w in stem_words]
    #
    # 8. Join the words back into one string separated by space, 
    #    and return the result.
    #return(lemma_words)
    return(meaningful_words)


# Define a function to split a review into parsed sentences
# we needd sentences because word2vec takes sentences as input. It leverages the SBD
def text_to_sentences(text, tokenizer,remove_stopwords=False ):
    # Function to split a review into parsed sentences. Returns a 
    # list of sentences, where each sentence is a list of words
    #
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(text.strip())
    #
    # 2. Loop over each sentence
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call text_to_wordlist to get a list of words
            sentences.append(text_to_wordlist(raw_sentence,remove_stopwords))
    #
    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists
    return sentences


def parse_and_clean_sentences(df):
    sentences = []  # Initialize an empty list of sentences

    print ("Parsing sentences from training set")
    for text in df["text"]:
        sentences += text_to_sentences(text, tokenizer)
    print ("parsing done!")   
    return sentences


def train_word2vec(df, num_features = 300, min_word_count = 1, num_workers = 4, context = 4, downsampling = 1e-3):
    
    from gensim.models import word2vec
    # parse and clean sentence
    sentences = parse_and_clean_sentences(df)
    
    #initialize logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)

    # Initialize and train the model (this will take some time)

    print ("Training model...")
    model = word2vec.Word2Vec(sentences, workers=num_workers,size=num_features, min_count = min_word_count,
                              window = context, sample = downsampling)

    # init_sims will make the model much more memory-efficient.
    model.init_sims(replace=True)

    # save the model for later use.can load it later using Word2Vec.load()
    model_name = "trainedWord2vecmodel"
    model.save(model_name)
    print ("model saved as", model_name)
    
    return model


##A simple way to assign a word2vec vector to a document is to take a mean of its words.
def makeFeatureVec(words, model, seq_len, num_features):
    # Function to average all of the word vectors in a given
    # paragraph
    #
    # Pre-initialize an empty 3D numpy array (for speed)
    featureVec = np.zeros((seq_len, num_features, 1),dtype="float32")
    #
    nwords = 0
    # 
    # Index2word is a list that contains the names of the words in 
    # the model's vocabulary. Convert it to a set, for speed 
    index2word_set = set(model.wv.index2word)
    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set: 
            featureVec[nwords] = np.array(model[word]).reshape(num_features,1)
            nwords = nwords + 1
            
    return featureVec


def getFeatureVecs(textlist, seq_len, model):
    # Given a set of documents (each one a list of words), calculate 
    # the average feature vector for each one and return a 2D numpy array 
    # 
    #find the num of features
    num_features = model.wv.syn0.shape[1]
    
    # Initialize a counter
    counter = 0
    # 
    # Preallocate a 4D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(textlist), seq_len, num_features, 1),dtype="float32")
    # 
    # Loop through the reviews
    for example in textlist:
        # 
        # Call the function (defined above) that makes average feature vectors
        reviewFeatureVecs[counter] = makeFeatureVec(example, model, seq_len, num_features)
        #
        # Increment the counter
        counter = counter + 1
        
    return reviewFeatureVecs


def preprocess_data(df, model):
    textlist = []
    padded_textlist = []
    print("Creating textlist")
    for text in df["text"]:
        textlist.append(text_to_wordlist(text, remove_stopwords=True ))
        
    #seq_len = max(len(x)for x in textlist)
    seq_len = 10
    print("max seq length = ",seq_len)
    print("padding textlist ")
    for i in range(len(textlist)):
        text = textlist[i]
        num_padding = seq_len - len(text)
        new_text = text + [""] * num_padding
        padded_textlist.append(new_text)
        
    textlist = padded_textlist
    print("creating feature vecs")    
    DataVecs = getFeatureVecs(textlist, seq_len, model)
    print ("done!!!")
    return DataVecs


def process_predict_data(df, model, seq_len):
    textlist = []
    padded_textlist = []
    for text in df["text"]:
        textlist.append(text_to_wordlist(text, remove_stopwords=True ))
        
    for i in range(len(textlist)):
        text = textlist[i]
        num_padding = seq_len - len(text)
        new_text = text + [""] * num_padding
        padded_textlist.append(new_text)
        
    textlist = padded_textlist    
    DataVecs = getFeatureVecs(textlist, seq_len, model)
    return DataVecs
