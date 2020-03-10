#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 18:33:27 2020

@author: ignace
"""
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize 
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import nltk


data = pd.read_csv('sotu.csv')
data


#############################
#############################


porter = PorterStemmer()
lancaster=LancasterStemmer()
stop=list(set(stopwords.words('english')))
texts=data['Speech'].tolist()

#preprocess the texts of the data within CountVectorize
def mypreprocess(text):
    #text is a document of a corpus
    #lowercase of text
    text=text.lower()
    #remove all the irrelevant numbers and punctuation
    text=re.sub(r'[^a-z]+',' ',text)
    #tokenize the words
    token1=word_tokenize(text)
    #remove the meaningless stopping words
    token2=[t for t in token1 if t not in stopwords.words('english')]
    #stemming transformation
    Porter=1
    if Porter==1:
        token3=[porter.stem(t) for t in token2]
    else:
        token3=[lancaster.stem(t) for t in token2]
    return token3


token_new=[]
texts_new=[]
for text in texts:
    print(texts.index(text))
    token_new.append(mypreprocess(text))
    texts_new.append(' '.join(mypreprocess(text)))
    

#############################
#############################


# Initial Setup
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(min_df=0.0001) #feature extraction function buildup
result=cv.fit_transform(texts_new)  #transform the data into document-term matrix
name=cv.get_feature_names() #feature name
print(cv.get_feature_names()[1:100])
print(result.shape)
X=result.toarray() #prepare for the data in matrix
y=np.array(data['Party'].tolist()) #prepare the y response.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,random_state=1) #split the data into training sample and test sample


word_freq_df = pd.DataFrame(X_train, columns=cv.get_feature_names())
top_words_df = pd.DataFrame(word_freq_df.sum()).sort_values(0, ascending=False)


clf = MultinomialNB()
clf.fit(X, y)
print(clf.get_params())
print('Naive Bayes for in-sample and out-of-sample cases:')
print(accuracy_score(clf.predict(X_train),y_train))
print(accuracy_score(clf.predict(X_test),y_test))







#############################
#############################
#Lasso & Ridge Regression
    
class Model:
    def __init__(self,model,name):
        self.Model=model
        self.name=name
    
    def mysummary(self):
        result=self.Model
        try:
            coefs_=(((result.coef_).toarray())[0]).tolist()
        except AttributeError:
            coefs_=((result.coef_)[0]).tolist()
        coefs_abs=[abs(t) for t in coefs_]
        Coef=pd.DataFrame({'Term':name,'Coef':coefs_,'Abs':coefs_abs,})
        new=Coef.sort_values(by=['Coef'],ascending=False)
        print(new)
        self.new=new
        
    def mygraph(self):
        new=self.new
        new['positive']=new.Coef>0
        new['Coef'].plot.bar(color=new.positive.map({True: 'b', False: 'r'}))
        plt.xticks(rotation=50)
        plt.xlabel("Terms")
        plt.ylabel("Term Loading in Absolute Values")
        plt.show()
        
        
cv = CountVectorizer(min_df=0.0001)
result=cv.fit_transform(texts_new)
name=cv.get_feature_names()
print(cv.get_feature_names()[1:100])
print(result.shape)
X=result.toarray()
y=np.array(data['Party'].tolist())

    
tfidf_id=0
if tfidf_id==1:
    cv=TfidfVectorizer(min_df=0.001)
else:
    cv = CountVectorizer(min_df=0.001)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,random_state=1)
cs = np.arange(0.001,2,0.01)


clflasso = LogisticRegressionCV(Cs=cs,
                           cv=5,
                           solver='saga',
                           random_state=0,
                           n_jobs=-1,
                           penalty='l1',
                           )

    

clfridge = LogisticRegressionCV(Cs=cs,
                           cv=5,
                           solver='saga',
                           random_state=0,
                           n_jobs=-1,
                           penalty='l2'
                           )

result_lasso=clflasso.fit(X_train, y_train)
result_ridge=clfridge.fit(X_train, y_train)

print('LASSO regression for in-sample and out-of-sample cases:')
print(accuracy_score(result_lasso.predict(X_train),y_train))
print(accuracy_score(result_lasso.predict(X_test),y_test))


print('Ridge regression for in-sample and out-of-sample cases:')
print(accuracy_score(result_ridge.predict(X_train),y_train))
print(accuracy_score(result_ridge.predict(X_test),y_test))

import matplotlib.pyplot as plt

def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()


Ridge = clfridge.predict_proba(X_test)
Ridge = Ridge[:, 1]

auc = roc_auc_score(y_test, Ridge)
print('AUC: %.2f' % auc)

fpr, tpr, thresholds = roc_curve(y_test, Ridge)

plot_roc_curve(fpr, tpr)





x=Model(result_lasso,name)
x.mysummary()
x.mygraph()


#############################
class Model:
    def __init__(self,model,name):
        self.Model=model
        self.name=name

    def mysummary(self):
        result=self.Model
        try:
            coefs_=(((result.coef_).toarray())[0]).tolist()
        except AttributeError:
            coefs_=((result.coef_)[0]).tolist()
        coefs_abs=[abs(t) for t in coefs_]
        Coef=pd.DataFrame({'Term':name,'Coef':coefs_,'Abs':coefs_abs,})
        new=Coef.sort_values(by=['Coef'],ascending=False)
        print(new)
        self.new=new

    def mygraph(self):
        new=self.new
        new['positive']=new.Coef>0
        new['Coef'].plot.bar(color=new.positive.map({True: 'b', False: 'r'}))
        plt.xticks(rotation=50)
        plt.xlabel("Terms")
        plt.ylabel("Term Loading in Absolute Values")
        plt.show()


cv = CountVectorizer(min_df=0.0001)
result=cv.fit_transform(texts_new)
name=cv.get_feature_names()
print(cv.get_feature_names()[1:100])
print(result.shape)
X=result.toarray()
y=np.array(data['Party'].tolist())


tfidf_id=0
if tfidf_id==1:
    cv=TfidfVectorizer(min_df=0.001)
else:
    cv = CountVectorizer(min_df=0.001)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,random_state=1)
cs = np.arange(0.001,2,0.01)


clflasso = LogisticRegressionCV(Cs=cs,
                           cv=5,
                           solver='saga',
                           random_state=0,
                           n_jobs=-1,
                           penalty='l1',
                           )



clfridge = LogisticRegressionCV(Cs=cs,
                           cv=5,
                           solver='saga',
                           random_state=0,
                           n_jobs=-1,
                           penalty='l2'
                           )

result_lasso=clflasso.fit(X_train, y_train)
result_ridge=clfridge.fit(X_train, y_train)

print('LASSO regression for in-sample and out-of-sample cases:')
print(accuracy_score(result_lasso.predict(X_train),y_train))
print(accuracy_score(result_lasso.predict(X_test),y_test))


print('Ridge regression for in-sample and out-of-sample cases:')
print(accuracy_score(result_ridge.predict(X_train),y_train))
print(accuracy_score(result_ridge.predict(X_test),y_test))

import matplotlib.pyplot as plt

def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()


Ridge = clfridge.predict_proba(X_test)
Ridge = Ridge[:, 1]

auc = roc_auc_score(y_test, Ridge)
print('AUC: %.2f' % auc)

class Model:
    def __init__(self,model,name):
        self.Model=model
        self.name=name

    def mysummary(self):
        result=self.Model
        try:
            coefs_=(((result.coef_).toarray())[0]).tolist()
        except AttributeError:
            coefs_=((result.coef_)[0]).tolist()
        coefs_abs=[abs(t) for t in coefs_]
        Coef=pd.DataFrame({'Term':name,'Coef':coefs_,'Abs':coefs_abs,})
        new=Coef.sort_values(by=['Coef'],ascending=False)
        print(new)
        self.new=new

    def mygraph(self):
        new=self.new
        new['positive']=new.Coef>0
        new['Coef'].plot.bar(color=new.positive.map({True: 'b', False: 'r'}))
        plt.xticks(rotation=50)
        plt.xlabel("Terms")
        plt.ylabel("Term Loading in Absolute Values")
        plt.show()


cv = CountVectorizer(min_df=0.0001)
result=cv.fit_transform(texts_new)
name=cv.get_feature_names()
print(cv.get_feature_names()[1:100])
print(result.shape)
X=result.toarray()
y=np.array(data['Party'].tolist())


tfidf_id=0
if tfidf_id==1:
    cv=TfidfVectorizer(min_df=0.001)
else:
    cv = CountVectorizer(min_df=0.001)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,random_state=1)
cs = np.arange(0.001,2,0.01)


clflasso = LogisticRegressionCV(Cs=cs,
                           cv=5,
                           solver='saga',
                           random_state=0,
                           n_jobs=-1,
                           penalty='l1',
                           )



clfridge = LogisticRegressionCV(Cs=cs,
                           cv=5,
                           solver='saga',
                           random_state=0,
                           n_jobs=-1,
                           penalty='l2'
                           )

result_lasso=clflasso.fit(X_train, y_train)
result_ridge=clfridge.fit(X_train, y_train)

print('LASSO regression for in-sample and out-of-sample cases:')
print(accuracy_score(result_lasso.predict(X_train),y_train))
print(accuracy_score(result_lasso.predict(X_test),y_test))


print('Ridge regression for in-sample and out-of-sample cases:')
print(accuracy_score(result_ridge.predict(X_train),y_train))
print(accuracy_score(result_ridge.predict(X_test),y_test))

import matplotlib.pyplot as plt

def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()


Ridge = clfridge.predict_proba(X_test)
Ridge = Ridge[:, 1]

auc = roc_auc_score(y_test, Ridge)
print('AUC: %.2f' % auc)

fpr, tpr, thresholds = roc_curve(y_test, Ridge)

plot_roc_curve(fpr, tpr)

fpr, tpr, thresholds = roc_curve(y_test, Ridge)

plot_roc_curve(fpr, tpr)

#############################

from sklearn.neural_network import MLPClassifier


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=1)
MLP = MLPClassifier()

result_mlp = MLP.fit(X_train, y_train)
print('MLP regression for in-sample and out-of-sample cases:')
print(accuracy_score(result_mlp.predict(X_train),y_train))
print(accuracy_score(result_mlp.predict(X_test),y_test))


# lets add some hyperparameters


#############################
#############################

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(max_iter=100)

parameter_space = {
    'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],
}

from sklearn.model_selection import GridSearchCV

clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)
clf.fit(X_train, y_train)
y_true, y_pred = y_test , clf.predict(X_test)

from sklearn.metrics import classification_report
print('Results on the test set:')
print(classification_report(y_true, y_pred))


3


#############################
#############################


#Grid Search CV
from sklearn.ensemble import RandomForestClassifier


rfc = RandomForestClassifier(n_jobs=-1,max_features= 'sqrt' ,n_estimators=10, oob_score = True) 

param_grid = {
    'n_estimators': [200, 700],
    'max_features': ['auto', 'sqrt', 'log2']
}



CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid)
CV_rfc.fit(X, y)
CV_rfc.best_params_

print('Random Forest for in-sample and out-of-sample cases:')
print(accuracy_score(CV_rfc.predict(X_train),y_train))
print(accuracy_score(CV_rfc.predict(X_test),y_test))



OSP = CV_rfc.predict(X_test)
y_test

#############################
#############################





import gensim

# Word Embedding in Skip-Gram
SkipGram = gensim.models.Word2Vec(token_new, sg=1,
                                  min_count=3,  # Ignore words that appear less than this
                                  size=200,  # Dimensionality of word embeddings
                                  workers=3,  # Number of processors (parallelisation)
                                  window=5,  # Context window for words during train
                                  iter=30)

result_skipgram = SkipGram.most_similar('baghdadi')
inflat_SkipGram = SkipGram.wv['baghdadi']
print('Skip-Gram vector for inflat token')
print(inflat_SkipGram)
print('+++++++++++++++++++++++++++++++++++++++')
print('Skip-Gram top similar tokens:')
print(result_skipgram)

# CBOW model


# Word Embedding in CBOW
CBOW = gensim.models.Word2Vec(token_new, sg=0,
                              min_count=3,  # Ignore words that appear less than this
                              size=200,  # Dimensionality of word embeddings
                              workers=3,  # Number of processors (parallelisation)
                              window=5,  # Context window for words during train
                              iter=30)

result_CBOW = CBOW.most_similar('climat')
inflat_CBOW = CBOW.wv['climat']
print(inflat_CBOW)
print('+++++++++++++++++++++++++++++++++++++++')
print('CBOW top similar tokens:')
print(result_CBOW)

# Search for tokens
for x in token_new:
    for t in x:
        if 'nixon' in t:
            print(t)

# K-Means Clusterign

import nltk
from nltk.cluster import KMeansClusterer


def KM_Cluster(model, num):
    # num: the number of the terms is showed up
    # model: the model implemented in word_embedding
    Num_of_Cluster = 3
    km = KMeansClusterer(Num_of_Cluster, distance=nltk.cluster.util.cosine_distance, repeats=25)
    X = model[model.wv.vocab]
    assign_clusters1 = km.cluster(X, assign_clusters=True)
    words = list(model.wv.vocab)[:num]
    for i, word in enumerate(words):
        print(word + ":" + str(assign_clusters1[i]))


print('+++++++++++++++++++++++++++++++++++++++++++++++++++')
print('CBOW Clustering Result:')
KM_Cluster(CBOW, 10)
print('++++++++++++++++++++++++++++++++++++++++++++++++++++')
print('SkipGram result')
KM_Cluster(SkipGram, 10)
#############################
#############################
#############################
#############################

#############################
#############################

#############################
#############################

#############################
#############################

