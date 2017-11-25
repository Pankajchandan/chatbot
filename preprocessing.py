import pandas as pd
import re
import logging
import numpy as np

def text_to_wordlist(text, remove_stopwords=False):
    # 1. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", text) 
    #
    # 2. Convert to lower case, split into individual words
    words = letters_only.lower().split() 
    #
    # 3. In Python, searching a set is much faster than searching
    #    a list, so convert the stop words to a set
    return(words)


def parse_and_clean_sentences(df):
    sentences = []  # Initialize an empty list of sentences

    print ("Parsing sentences from training set")
    for text in df["text"]:
        sentences += text_to_sentences(text, tokenizer)
    print ("parsing done!")   
    return sentences


##A simple way to assign a word2vec vector to a document is to take a mean of its words.
def makeFeatureVec(words, model, seq_len, num_features, word_set):
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
    index2word_set = word_set
    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set: 
            featureVec[nwords] = np.array(model[word]).reshape(num_features,1)
            nwords = nwords + 1
            
    return featureVec


def getFeatureVecs(textlist, seq_len, model, word_set):
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
        reviewFeatureVecs[counter] = makeFeatureVec(example, model, seq_len, num_features, word_set)
        #
        # Increment the counter
        counter = counter + 1
        
    return reviewFeatureVecs


def preprocess_data(df, model, word_set):
    textlist = []
    padded_textlist = []
    print("Creating textlist")
    for text in df["text"]:
        textlist.append(text_to_wordlist(text, remove_stopwords=True ))
        
    seq_len = max(len(x)for x in textlist)
    print("max seq length = ",seq_len)
    print("padding textlist ")
    for i in range(len(textlist)):
        text = textlist[i]
        num_padding = seq_len - len(text)
        new_text = text + [""] * num_padding
        padded_textlist.append(new_text)
        
    textlist = padded_textlist
    print("creating feature vecs")    
    DataVecs = getFeatureVecs(textlist, seq_len, model, word_set)
    print ("done!!!")
    return DataVecs


def process_predict_data(df, model, seq_len, word_set):
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
    DataVecs = getFeatureVecs(textlist, seq_len, model, word_set)
    return DataVecs
