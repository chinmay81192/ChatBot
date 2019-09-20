
# coding: utf-8

# In[ ]:


import nltk
import string
import numpy as np
import random


# In[2]:


f = open('sample.txt','r',errors='ignore')


# In[3]:


raw = f.read()


# In[4]:


raw = raw.lower()


# In[5]:


nltk.download('punkt')
nltk.download('wordnet')


# In[6]:


sent_tokens=nltk.sent_tokenize(raw)
print(type(sent_tokens))
word_tokens=nltk.word_tokenize(raw)


# In[7]:


sent_tokens[0]


# In[8]:


word_tokens[:2]


# In[9]:


lemmer=nltk.stem.WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]


# In[10]:


remove_punct_dict=dict((ord(punct),None) for punct in string.punctuation)


# In[ ]:


def LemNormalizeText(text):
    text=text.lower()
    return LemTokens(nltk.word_tokenize(text.translate(remove_punct_dict)))


# In[ ]:


GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up?","hey","hey there")
GREETING_RESPONSES = ["hi", "hey", "hi there", "hello", "I am glad! You are talking to me"]
def greeting(sentence):
 
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[38]:


def response(user_response):
    robo_response=''
    sent_tokens.append(user_response)
    TfidfVec=TfidfVectorizer(tokenizer=LemNormalizeText,stop_words='english')
    tfidf=TfidfVec.fit_transform(sent_tokens)
    vals=cosine_similarity(tfidf[-1],tfidf)
    idx=vals.argsort()[0][-2]
    flat=vals.flatten()
    flat.sort()
    req_tfidf=flat[-2]
    
    if(req_tfidf==0):
        robo_response="Sorry! Unable to understand the question."
        return robo_response
    else:
        robo_response=robo_response+sent_tokens[idx]
        return robo_response


# In[ ]:


flag=True
print("CASSIDY: Hi! My name is Cassidy and I will assist you with your queries regarding Masters in Applied Computer Science\nTo stop communicating with me type Bye")
while(flag==True):
    user_response=input()
    user_response=user_response.lower()
    if(user_response!='bye'):
        if(user_response=='thanks' or user_response=='thank you'):
            flag=False
            print("CASSIDY: You are welcome!")
        else:
            if(greeting(user_response)!=None):
                print("CASSIDY: "+greeting(user_response))
            else:
                print("CASSIDY: ",end="")
                print(response(user_response))
                sent_tokens.remove(user_response)
    else:
        flag=False
        print("CASSIDY: Bye! Take care!")

