
# coding: utf-8

# # Train Models

# In[1]:


import sys
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

#from keras.utils.np_utils import to_categorical
#from keras.layers import Embedding, LSTM, Dense, Conv1D, MaxPooling1D, Dropout, Activation
#from keras.models import Sequential
#from keras.preprocessing.text import Tokenizer
#from keras.preprocessing.sequence import pad_sequences
#from keras.callbacks import ModelCheckpoint


# ### Set model values

# In[2]:


default_training_file = "training_output_binary_filename.csv"
default_model_choice = "NB"

#NB specific setting
default_NB_features = "n-gram" #"bag of words" "tf-idf" "n-gram"

#-------------------------------------------------------

training_file = default_training_file
model_choice = default_model_choice
NB_features = default_NB_features


# ### Define train/test split function.

# In[3]:


# split dataset evenly based on labels
def split_test_train(total, stratify_col):
    transition_rows = total[total[stratify_col] != 0]
    non_transition_rows = total[total[stratify_col] == 0]
    
    # first split transitions into training/testing
    X_train1, X_test1, y_train1, y_test1 = train_test_split(transition_rows, 
                                                    transition_rows['transition_value'], 
                                                    test_size=0.30, random_state=42)
    
    # assert there are only transition labels in this dataframe
    assert len(X_train1[X_train1['transition_value'] == 0]) == 0
    assert len(X_test1[X_test1['transition_value'] == 0]) == 0
    
    train_len = len(X_train1) # number of non-transitions to add to training set
    test_len = len(X_test1) # number of non-transitions to add to testing set
    
    
    # next split non-transitions into training/testing
    X_train2, X_test2, y_train2, y_test2 = train_test_split(non_transition_rows, 
                                                    non_transition_rows['transition_value'], 
                                                    test_size=0.30, random_state=42)
    
    # pick train_len random rows from non-transition training set
    X_train2 = X_train2.sample(n = train_len, axis=0)
    
    # pick test_len random rows from non_transitions testing set
    X_test2 = X_test2.sample(n = test_len, axis=0)
    
    # assert there are no transition utterances in non-transition training and testing set
    assert len(X_train2[X_train2['transition_value'] != 0]) == 0
    assert len(X_test2[X_test2['transition_value'] != 0]) == 0
    
    # final result, concat the dataframe
    X_train_final = pd.concat([X_train1, X_train2])
    X_test_final = pd.concat([X_test1, X_test2])
    
    print (X_train_final.head())
    
    return X_train_final['text'], X_test_final['text'], X_train_final['transition_value'], X_test_final['transition_value']


# In[4]:


# assert training/testing split is balanced
def verify_train_test_split(train, x_train, y_train, x_test, y_test):
    transition_rows = train[train["transition_value"] != 0]
    assert len(x_train) == len(y_train)
    assert len(x_test) == len(y_test)
    assert len(x_train) == int(len(transition_rows) * 0.7) * 2
    assert len(x_test) == (len(transition_rows) * 2) - (int(len(transition_rows) * 0.7) * 2)
    assert len(y_train[y_train == 0]) == len(y_train[y_train != 0])
    assert len(y_test[y_test == 0]) == len(y_test[y_test != 0])
    print ("{0}% of utterances are transitions".format((sum(y_train) / len(x_train)) * 100))


# ### Define naive bayes model.

# In[5]:


# extract bag of words features from text for a model
def bag_of_words_features(x_train, x_test):
    count_vect = CountVectorizer()
    count_vect.fit(np.hstack((x_train)))
    X_train_counts = count_vect.transform(x_train)
    X_test_counts = count_vect.transform(x_test)
    
    assert X_train_counts.shape[1] == X_test_counts.shape[1]
    
    return X_train_counts, X_test_counts, count_vect


# In[6]:


def transform_tfidf(x_train, x_test):
    X_train_counts, X_test_counts, count_vect = bag_of_words_features(x_train, x_test)
    
    transformer = TfidfTransformer(smooth_idf=True)
    Xtrain_tfidf = transformer.fit_transform(X_train_counts)
    Xtest_tfidf = transformer.fit_transform(X_test_counts)
    
    assert Xtrain_tfidf.shape[1] == Xtest_tfidf.shape[1]
    
    return Xtrain_tfidf, Xtest_tfidf, count_vect


# In[7]:


def transform_ngram(start, stop, x_train, x_test):
    ngram_vectorizer = CountVectorizer(analyzer='word', ngram_range=(start, stop))
    counts = ngram_vectorizer.fit(np.hstack((x_train)))
    
    #print ("Number of transformed features {0}\n"
    # .format(len(ngram_vectorizer.get_feature_names())))
    
    #print ("First 10 features\n{0}"
    # .format('\n'.join(ngram_vectorizer.get_feature_names()[-10:])))
    
    X_train_counts = counts.transform(x_train)
    X_test_counts = counts.transform(x_test)
    
    assert X_train_counts.shape[1] == X_test_counts.shape[1]
    
    return X_train_counts, X_test_counts, ngram_vectorizer


# In[8]:


def features(x_train, x_test):
    if (NB_features == "bag of words"):
        return bag_of_words_features(x_train, x_test)
    
    elif (NB_features == "tf-idf"):
        return transform_tfidf(x_train, x_test)
    
    elif (NB_features == "n-gram"):
        return transform_ngram(1, 6, x_train, x_test)
    
    else:
        raise Exception("Feature set {0} it not supported"
         .format(NB_features))


# In[9]:


# output accuracy for a naive bayes model
# return the trained model
def create_naive_bayes_model(x_train, x_test, y_train, y_test):
    X_train_counts, X_test_counts, count_vect = features(x_train, x_test)
    
    clf = MultinomialNB()
    clf.fit(X_train_counts, y_train)
    
    assert X_test_counts.shape[0] == y_test.shape[0]
    
    acc = clf.score(X_test_counts, y_test, sample_weight=None)
    print("Model accuracy {0}".format(acc))
    
    return clf, count_vect


# ### Define neutral network model.

# In[10]:


def create_neural_network_model(x_train, x_test, y_train, y_test):
    #tokenize and pad word length
    tokenizer = Tokenizer(num_words=40000)
    tokenizer.fit_on_texts(x_train)
    sequences = tokenizer.texts_to_sequences(x_train)

    padded = pad_sequences(sequences, maxlen = 44)
    pred = to_categorical(y_train)
    
    model = Sequential()
    model.add(Embedding(40000, 150, input_length=44))
    model.add(Conv1D(64, 5, activation='relu'))
    model.add(MaxPooling1D(pool_size=4))
    model.add(LSTM(150, dropout=0.2, recurrent_dropout=0.5))
    model.add(Dense(2, activation='sigmoid')) #fully connected layer
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    
    filepath="weights.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    model.fit(padded, pred,validation_split=0.3, epochs = 50, callbacks = callbacks_list)


# ### Run main code.

# In[11]:


import pickle


# In[12]:


train = pd.read_table(training_file, sep="~")[['video_id', 'transition_value', 'text']]
train['transition_value'][train["transition_value"]==2] = 0

#print (train.head())
#print("Number of transitions in the dataset {0}".format(len(train[train['transition_value'] != 0])))
x_train, x_test, y_train, y_test = split_test_train(train[['text', 'transition_value']], "transition_value")
verify_train_test_split(train, x_train, y_train, x_test, y_test)

x_train.head()


# In[13]:


if (model_choice == "NN"):
    raise Exception("Neural network not supported yet.")
    create_neural_network_model(x_train, x_test, y_train, y_test)
elif (model_choice == "NB"):
    model, count_vect = create_naive_bayes_model(x_train, x_test, y_train, y_test)
    pickle.dump(model, open("nb_model.p", "wb"))
    pickle.dump(count_vect, open("nb_count_vect.p", "wb"))
else:
    raise Exception("No model provided.")

