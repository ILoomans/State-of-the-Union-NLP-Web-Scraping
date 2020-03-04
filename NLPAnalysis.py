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
import nltk
nltk.download('punkt')

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

naive_bayes = MultinomialNB() #setup Bayesian Model 
naive_bayes.fit(X_train, y_train) #train the model with data
predictions = naive_bayes.predict(X_test) #make predictions given test samples


print('In Sample prediction accuracy++++++++++++++++++++++++++++++++++++++++++')
print(accuracy_score(y_train, naive_bayes.predict(X_train))) #compare the prediction value with the true value in training sample

print('Out of Sample prediction accuracy++++++++++++++++++++++++++++++++++++++')
print(accuracy_score(y_test, predictions)) #compare the prediction with true value for out-of-sample


parameters=list(np.arange(0.0001,0.02,0.0001))
accuracy_insample=[]
accuracy_outsample=[]
cross_val=[]
#loop the parameter grids of the model:
for i in parameters:
    #the tunning parameters in this algorithm is the document frequency
    print(parameters.index(i))
    cv = CountVectorizer(min_df=i) #feature space selection and optimize it with cross-validation
    result=cv.fit_transform(texts_new)
    name=cv.get_feature_names()
    X=result.toarray()
    y=np.array(data['Party'].tolist())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,random_state=1)
    naive_bayes = MultinomialNB()
    naive_bayes.fit(X_train, y_train)
    cross_val.append(cross_val_score(naive_bayes, X, y,scoring='accuracy', cv=10))
    predictions = naive_bayes.predict(X_test)
    accuracy_insample.append(accuracy_score(y_train, naive_bayes.predict(X_train)))
    accuracy_outsample.append(accuracy_score(y_test, predictions))




#############################
#############################
#Lasso
    
class Model: #define a class object which is similar to function but should a set of objects (functions) in python
    def __init__(self,model,name):
        self.Model=model #define itsself property in the local environment
        self.name=name #define itself name property in the local environment
    
    #include two functions in this class of objects
    def mysummary(self): #first summary function
        result=self.Model
        try:
            coefs_=(((result.coef_).toarray())[0]).tolist() #try to obtain lasso coefficient 
        except AttributeError:
            coefs_=((result.coef_)[0]).tolist() #when lasso result doesn't work then, try ridge regression coefficient
        coefs_abs=[abs(t) for t in coefs_] #calculate the absolute values of the loadings (coefficients)
        Coef=pd.DataFrame({'Term':name,'Coef':coefs_,'Abs':coefs_abs,}) #generate a table to show the term importance 
        new=Coef.sort_values(by=['Coef'],ascending=False) #sort the dataframe based on absolute value
        print(new)
        self.new=new
        
    def mygraph(self): #second self defined function in this class to draw the graph
        new=self.new #new is the local attribute defined in the last function about the coefficient data frame
        #Then, find the postive coefficents and negative coefficients and draw the graph with different colors
        new['positive']=new.Coef>0 
        new['Coef'].plot.bar(color=new.positive.map({True: 'b', False: 'r'}))
        plt.xticks(rotation=50)
        plt.xlabel("Terms")
        plt.ylabel("Term Loading in Absolute Values")
        plt.show()
        
        
cv = CountVectorizer(min_df=0.0001) #feature extraction function buildup
result=cv.fit_transform(texts_new)  #transform the data into document-term matrix
name=cv.get_feature_names() #feature name
print(cv.get_feature_names()[1:100])
print(result.shape)
X=result.toarray() #prepare for the data in matrix
y=np.array(data['Party'].tolist()) #prepare the y response.

    
tfidf_id=1 #identifier whether you will select a tfidf vectorization or not
if tfidf_id==1:
    cv=TfidfVectorizer(min_df=0.001)
else:
    cv = CountVectorizer(min_df=0.001) #feature space selection and optimize it with cross-validation


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,random_state=1)
cs = np.arange(0.001,2,0.01) #tunning parameter of C set for cross-validation 


clflasso = LogisticRegressionCV(Cs=cs,
                           cv=5, #number of the cross validation
                           solver='saga', #optimization solver
                           random_state=0, #random state control
                           n_jobs=-1, #whether we should implement all the CPU processors or not
                           penalty='l1', #options on the penalization rules, the other two choices: l1 for lasso, l2 for ridge
                           )

    

clfridge = LogisticRegressionCV(Cs=cs,
                           cv=5, #number of the cross validation
                           solver='saga',
                           random_state=0,
                           n_jobs=-1, #whether we should implement all the CPU processors or not
                           penalty='l2', #options on the penalization rules, the other two choices:
                           )

result_lasso=clflasso.fit(X_train, y_train)
result_ridge=clfridge.fit(X_train, y_train)

print('LASSO regression for in-sample and out-of-sample cases:')
print(accuracy_score(result_lasso.predict(X_train),y_train))
print(accuracy_score(result_lasso.predict(X_test),y_test))


print('Ridge regression for in-sample and out-of-sample cases:')
print(accuracy_score(result_ridge.predict(X_train),y_train))
print(accuracy_score(result_ridge.predict(X_test),y_test))

x=Model(result_ridge,name)
x.mysummary()
x.mygraph()


#############################
#############################

from sklearn.neural_network import MLPClassifier


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,random_state=1)
MLP = MLPClassifier()

result_mlp = MLP.fit(X_train, y_train)
print('MLP regression for in-sample and out-of-sample cases:')
print(accuracy_score(result_mlp.predict(X_train),y_train))
print(accuracy_score(result_mlp.predict(X_test),y_test))




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




#############################
#############################
import gensim

#Word Embedding in Skip-Gram
SkipGram = gensim.models.Word2Vec(token_new,sg=1,
                              min_count=3,  # Ignore words that appear less than this
                              size=200,  # Dimensionality of word embeddings
                              workers=3,  # Number of processors (parallelisation)
                              window=5,  # Context window for words during train
                              iter=30)


result_skipgram=SkipGram.most_similar('regim')
inflat_SkipGram=SkipGram.wv['regim']
print('Skip-Gram vector for inflat token')
print(inflat_SkipGram)
print('+++++++++++++++++++++++++++++++++++++++')
print('Skip-Gram top similar tokens:')
print(result_skipgram)


#CBOW model


#Word Embedding in CBOW
CBOW= gensim.models.Word2Vec(token_new,sg=0,
                              min_count=3,  # Ignore words that appear less than this
                              size=200,  # Dimensionality of word embeddings
                              workers=3,  # Number of processors (parallelisation)
                              window=5,  # Context window for words during train
                              iter=30)


result_CBOW=CBOW.most_similar('nixon')
inflat_CBOW=CBOW.wv['nixon']
print(inflat_CBOW)
print('+++++++++++++++++++++++++++++++++++++++')
print('CBOW top similar tokens:')
print(result_CBOW)


for x in token_new:
  for t in x: 
      if 'nixon' in t:
          print(t)
          


# K-Means Clusterign 

import nltk
from nltk.cluster import KMeansClusterer
def KM_Cluster(model,num):
    #num: the number of the terms is showed up
    #model: the model implemented in word_embedding
    Num_of_Cluster=3
    km=KMeansClusterer(Num_of_Cluster, distance=nltk.cluster.util.cosine_distance, repeats=25)
    X=model[model.wv.vocab]
    assign_clusters1=km.cluster(X,assign_clusters=True)
    words = list(model.wv.vocab)[:num]
    for i, word in enumerate(words):  
        print (word + ":" + str(assign_clusters1[i]))
        
        
        
print('+++++++++++++++++++++++++++++++++++++++++++++++++++')
print('CBOW Clustering Result:')
KM_Cluster(CBOW,10)
print('++++++++++++++++++++++++++++++++++++++++++++++++++++')
print('SkipGram result')
KM_Cluster(SkipGram,10)
#############################
#############################

#Grid Search CV
from sklearn.ensemble import RandomForestClassifier


rfc = RandomForestClassifier(n_jobs=-1,max_features= 'sqrt' ,n_estimators=10, oob_score = True) 

param_grid = {
#    'n_estimators': [200, 700],
    'max_features': ['auto', 'sqrt', 'log2']
}

CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)


CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
CV_rfc.fit(X, y)
CV_rfc.best_params_

print('Random Forest for in-sample and out-of-sample cases:')
print(accuracy_score(CV_rfc.predict(X_train),y_train))
print(accuracy_score(CV_rfc.predict(X_test),y_test))


OSP = CV_rfc.predict(X_test)
y_test

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

