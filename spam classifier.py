#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import math
import glob
import os


# In[2]:


#reading data
df=pd.read_csv(r"spam.csv",encoding='latin-1')


# In[3]:


# preprocessing of data
df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)
df.rename(columns={'v1':'target','v2':'text'},inplace=True)
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()

df['target']=encoder.fit_transform(df['target'])
#df.isnull().sum()
df=df.drop_duplicates(keep='first')

df.duplicated().sum()

# EDA
plt.pie(df['target'].value_counts(),labels=['ham','spam'],autopct="%.3f")
import nltk

nltk.download('punkt')
nltk.download('stopwords')

df['num_characters']=df['text'].apply(len)

df['num_words']=df['text'].apply(lambda x: len(nltk.word_tokenize(x)))

df['num_sentences']=df['text'].apply(lambda x: len(nltk.sent_tokenize(x)))

plt.figure(figsize=(12,9))
sns.histplot(df[df['target']==0]['num_characters'],color='green')
sns.histplot(df[df['target']==1]['num_characters'],color='red')

sns.pairplot(df,hue='target')

#importing important library
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
from nltk.corpus import stopwords
import string
import nltk


# In[4]:


# function for clearing and transforming the data
def transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    temp=[]
    for i in text:
        if(i.isalnum()):
            temp.append(i)
    text=temp[:]
    temp.clear()
    for i in text:
        if(i not in stopwords.words('english') and i not in string.punctuation):
            temp.append(i)
    text=temp[:]
    temp.clear()
    for i in text:
        temp.append(ps.stem(i))
            
    return " ".join(temp)


# In[5]:


df['transform_text']=df['text'].apply(transform_text)

from sklearn.feature_extraction.text import CountVectorizer

cv=CountVectorizer()


# In[6]:


x=cv.fit_transform(df['transform_text']).toarray()


# In[7]:



for i in range(x.shape[0]):
    for j in range(x.shape[1]):
        if(x[i][j]>0):
            x[i][j]=1
            #sum+=x[i][j]

y=df['target'].values

# making data in vector form
cv=CountVectorizer()
cv.fit(df['transform_text'])
unique_word=list(cv.vocabulary_.keys())
unique_word.sort()
print(len(unique_word))
print(type(unique_word))

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.15,random_state=42)

total_train_spam=np.sum(y_train)
total_train_email=y_train.shape[0]
total_train_ham=total_train_email-total_train_spam

#parameter estimation
p0=total_train_ham/total_train_email
p1=total_train_spam/total_train_email
print("the percentage of ham data in dataset is ",p0)
print("the percentage of spam data in dataset is ",p1)


# In[8]:



spam_index_list=[]
ham_index_list=[]
for i in range(total_train_email):
    if(y_train[i]>0):
        spam_index_list.append(i)
    else:
        ham_index_list.append(i)

train_spam_email=x_train[spam_index_list]
train_ham_email=x_train[ham_index_list]

print(train_spam_email.shape)
print(train_ham_email.shape)


#estimating pij
p1j=np.sum(train_spam_email, axis = 0)
p0j=np.sum(train_ham_email, axis = 0)
p0j=p0j.astype(np.float64)
p1j=p1j.astype(np.float64)

from __future__ import division
for i in range(p0j.shape[0]):
    p0j[i]=p0j[i]/train_ham_email.shape[0]
for i in range(p1j.shape[0]):
    p1j[i]=p1j[i]/train_spam_email.shape[0]


# In[9]:


# function for classifying th email as spam or non spam
def check_class(test_email):
    p_ham_class=1
    for i in range(len(test_email)):
        if(test_email[i]==1):
            p_ham_class*=p0j[i]
        else:
            p_ham_class*=(1-p0j[i])
    p_ham_class*=p0
    
    p_spam_class=1
    for i in range(len(test_email)):
        if(test_email[i]==1):
            p_spam_class*=p1j[i]
        else:
            p_spam_class*=(1-p1j[i])
            
    p_spam_class*=p1
    
    if(p_ham_class>p_spam_class):
        return 0
    else:
        return 1
        


# In[10]:


# checking the accuracy on dataset
sum=0
accuracy_cnt=0
for i in range(x_test.shape[0]):
    test_email=x_test[i]
    clas=check_class(test_email)
    if(clas==y_test[i]):
        accuracy_cnt+=1
accuracy=accuracy_cnt/x_test.shape[0]
accuracy*=100

print("accuracy on test data is ",accuracy)
  


# In[11]:


# logic for reading emails from test folder and classifying if it is spam or not spam
for each in sorted(glob.glob('test/*.txt')):
    print(each," :",end=' ')
    
    with open(each) as f:
        text =f.read().replace('\n', '')
        text=transform_text(text)
        list1=[0]*x.shape[1]
        for word in text.split(" "):
            list1.append(word)
        list2=[]
        for word in unique_word:
            if(word in list1):
                list2.append(1)
            else:
                list2.append(0)
                
        ret_class=check_class(list2)
        if(ret_class==1):
            print(os.path.basename(each)," is Spam")
        else:
            print(os.path.basename(each)," is not Spam")
    
        
            
            
        


# In[ ]:




