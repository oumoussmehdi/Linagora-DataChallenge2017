
# coding: utf-8

# In[51]:

import operator
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity 


# In[52]:

from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, HashingVectorizer, CountVectorizer
import sklearn.preprocessing as pre


# ### load some of the files

# In[53]:

path_to_data = 'C:\\Users\\hbnp5049\\PycharmProjects\\untitled\\'# fill me!


# In[54]:

training = pd.read_csv(path_to_data + 'training_set.csv', sep=',', header=0)
training_info = pd.read_csv(path_to_data + 'training_info.csv', sep=',', header=0)


# In[55]:

testing = pd.read_csv(path_to_data + 'test_set.csv', sep=',', header=0)
testing_info = pd.read_csv(path_to_data + 'test_info.csv', sep=',', header=0)


# In[56]:

def email_sender_df(training):
    email_sender = {}
    for index, series in training.iterrows():
        row = series.tolist()
        sender = row[0]
        ids = row[1:][0].split(' ')
        for id in ids:
            email_sender[int(id)] = sender
    es_df = pd.DataFrame.from_dict(email_sender, orient='index').reset_index()
    es_df.rename(columns={'index': 'mid', 0: 'sender'}, inplace=True)
    print('df shape',es_df.shape)
    return es_df


# In[57]:

es_df = email_sender_df(training)
train_df = pd.merge(training_info,es_df,on='mid',how='inner')


# In[58]:

es_test_df = email_sender_df(testing)
test_df = pd.merge(testing_info,es_test_df,on='mid',how='inner')


# In[59]:

predictions_per_sender = {}


# In[68]:

def output_results(predictions_per_sender):
    #m.to_csv(r'C:\Users\hbnp5049\PycharmProjects\untitled\predictions.csv', header=mcols, index=None, sep=',')
    with open(path_to_data + 'predictions_tfidfClassifier.txt', 'w') as my_file:
        my_file.write('mid,recipients' + '\n')
        for sender, preds in predictions_per_sender.items():
            ids = preds[0]
            preds = preds[1]
            for index, my_preds in enumerate(preds):
                my_file.write(str(ids[index]) + ',' + ' '.join(my_preds) + '\n')


# In[69]:

def body_preprocessing(sender, X_train, X_test):
    vectorizer = TfidfVectorizer(sublinear_tf=True,
                             max_df=0.5,
                             stop_words='english',
                             analyzer = 'word',
                             max_features=103,
                            norm = 'l2')
    X_train_bin = vectorizer.fit_transform(X_train['body'].values)
    
    Centroids = {}
    for index, series in X_train.iterrows():
        row = series.tolist()
        recipients = row[3]
        for r in recipients:
            mu = X_train_bin[index]
            if r not in Centroids:
                Centroids[r] = mu
            else:
                Centroids[r] += mu
    
    #********Testing*****************
    X_test_bin = vectorizer.transform(X_test['body'])
    ids = []
    preds = []
    for index, series in X_test.iterrows():
        id = X_test['mid'][index]
        temp = {}
        for k in Centroids.keys():
            temp[k] = cosine_similarity(X_test_bin[index], Centroids[k])  
        sorted_temp = sorted(temp.items(), key = operator.itemgetter(1), reverse = True)
        predictions = sorted_temp[:10]
        predictions = [elt[0] for elt in predictions]
        ids.append(id)
        preds.append(predictions)
        print(id, predictions)
    predictions_per_sender[sender]=[ids, preds]
    #print(predictions_per_sender)
    output_results(predictions_per_sender)


# In[70]:

def recommendation(sender):
    # take the subset where x is the sender
    X_train = train_df[train_df['sender']==sender]
    X_train = X_train.reset_index(drop=True)
    # convert and clearn the column recipients
    for index, row in X_train.iterrows():
        recipients = row['recipients'].split() 
        recipients = [rec for rec in recipients if '@' in rec]
        X_train['recipients'][index] = recipients
    X_test = test_df[test_df['sender']==sender]
    X_test = X_test.reset_index(drop=True)
    # Focus only on a sender case
    body_preprocessing(sender, X_train, X_test)
    predictions_per_sender


# In[71]:
for sender in test_df['sender']:
    recommendation(sender)


# In[ ]:

'''
for sender in training['sender']:
    recommendation(sender)
    
'''

