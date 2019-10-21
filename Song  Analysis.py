#!/usr/bin/env python
# coding: utf-8

# *These are the lyrics for 57650 songs. They can be used for Natural Language Processing purposes, such as clustering of the words with similar meanings or predicting artist by the song. The dataset can be expanded with some more features for more advanced research like sentiment analysis. The data is not modified, only slightly cleaned, which gives a lot of freedom to devise your own applications*
# ** specially i analyze the only Queen artist like find out the trend the helps of NLP with nltk tool
# 

# In[1]:


import pandas as pd
import numpy as np
import os


# In[2]:


df = pd.read_csv('/home/parmar/Downloads/Datasets/songdata.csv')


# In[3]:


import os
for dirname, _, filenames in os.walk('//home/parmar/Downloads/Datasets/songdata.csv'):
    for filename in filenames:
        
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[4]:


df.info()


# In[5]:


queen_df = df[df.artist == 'Queen']
queen_df.head()


# In[6]:


import nltk
import plotly.graph_objs as go

def count_words(matrix):
    words_g = []
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    stop_words = set(nltk.corpus.stopwords.words('english'))
    stemmer = nltk.stem.SnowballStemmer('english')
    for lyric in matrix:
        for word in tokenizer.tokenize(lyric):
            if not word in stop_words:
                words_g.append(word)
                f = nltk.FreqDist(words_g)
                words = []
                values = []
                for word, value in f.most_common(20):
                    words.append(word)
                    values.append(value)
                    return words_g, words, values
                words_g, words, values = count_words(queen_df.text.values)
                plot = go.Figure(go.Bar(x=words, y=values, marker_color=['orange']*4+['#cc6600']*4+['#b35900']*4+['#994d00']*4+['#804000']*4))
                plot.update_layout(title='Palabras mas frecuentes')
                plot.show()


# In[7]:


from textblob import TextBlob
def sentence_sentiment(lyrics):
    sentiment = []
    means = []
    for lyric in lyrics:
        blob = TextBlob(lyric)
        sentiment.append(blob.sentiment.polarity)
    return sentiment


# In[8]:


get_ipython().system('pip install textblioi')


# In[9]:


means = sentence_sentiment(queen_df.text.values.tolist())


# In[10]:


sentiments = []
for i in means:
    if i > 0:
        sentiments.append('Positivo')
    elif i == 0:
        sentiments.append('Neutral')
    elif i < 0:
        sentiments.append('Negativo')


# In[11]:


queen_df['sentiment'] = sentiments
labels = queen_df.sentiment.value_counts().index.tolist()
values = queen_df.sentiment.value_counts().values.tolist()


# In[12]:


import plotly.graph_objs as go
plot = go.Figure(go.Pie(labels=labels, values=values, marker_colors=['lightgreen', 'darkred', 'darkorange']))
plot.update_layout(title='Distribución de sentimientos')

plot.show()


# In[17]:


import spacy
nlp = spacy.load('en_core_web_sm')
def verbs(text_matrix):
    verbs = []
    penalties = ['\'s', '\'m', '\'re', '\'ve']
    for liryc in text_matrix:
        doc = nlp(liryc)
        for token in doc:
            if token.pos_ == 'VERB':
                if not token.text in penalties:
                    if not token.is_stop:
                        verbs.append(token.text)
    return verbs


# In[18]:


get_ipython().system('pip install spacy')


# In[19]:


verbs = verbs(queen_df.text.values.tolist())


# In[20]:


get_ipython().system('pip install nltk')


# In[21]:


f = nltk.FreqDist(verbs)
wo = []
va = []
for w, v in f.most_common(20):
    wo.append(w)
    va.append(v)
    
plot = go.Figure(go.Bar(x=wo, y=va, marker_color=['orange']*4+['#cc6600']*4+['#b35900']*4+['#994d00']*4+['#804000']*4))
plot.update_layout(title='Verbos Mas Frecuentes')
plot.show()


# In[22]:


def persons_parser(text_matrix):
    persons = []
    organizations = []
    geo_political = []
    penalties = ['\'s', '\'m', '\'re', '\'ve']
    for liryc in text_matrix:
        doc = nlp(liryc)
        for token in doc.ents:
            if token.label_ == 'PERSON':
                if not token.text in penalties:
                    persons.append(token.text)
            elif token.label_ == 'ORG':
                organizations.append(token.text)
            elif token.label_== 'GPE':
                geo_political.append(token.text)
    return persons, organizations, geo_political


# In[23]:


persons, organizations, geo_political = persons_parser(queen_df.text.values.tolist())


# In[24]:


f = nltk.FreqDist(persons)
wo = []
va = []
for w, v in f.most_common(10):
    wo.append(w)
    va.append(v)
    
plot = go.Figure(go.Bar(x=wo, y=va, marker_color=['orange']*4+['#cc6600']*4+['#b35900']*4+['#994d00']*4+['#804000']*4))
plot.update_layout(title='Referencias a supuestas personas')

plot.show()


# In[26]:


f = nltk.FreqDist(organizations)
wo = []
va = []
for w, v in f.most_common(10):
    wo.append(w)
    va.append(v)
    
plot = go.Figure(go.Bar(x=wo, y=va, marker_color=['orange']*4+['#cc6600']*4+['#b35900']*4+['#994d00']*4+['#804000']*4))
plot.update_layout(title='Palabras reconocidas como organizaciones')
plot.show()


# In[27]:


f = nltk.FreqDist(geo_political)
wo = []
va = []
for w, v in f.most_common(10):
    wo.append(w)
    va.append(v)
    
plot = go.Figure(go.Bar(x=wo, y=va, marker_color=['orange']*4+['#cc6600']*4+['#b35900']*4+['#994d00']*4+['#804000']*4))
plot.update_layout(title='Palabras reconocidas como geopoliticas')
plot.show()


# In[28]:


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

tfv = TfidfVectorizer(stop_words='english')
cv = CountVectorizer(stop_words='english')


# In[29]:


words = tfv.fit_transform(queen_df.text.values)
lda = NMF(n_components=5).fit(words)


# In[30]:


feature_names = tfv.get_feature_names()


# In[33]:


for i_topico, topico in enumerate(lda.components_):
    print(f'Topico Numero #{i_topico+1}')
    print(" ".join([feature_names[i] for i in topico.argsort()[-20:-1]]))
    
topic_dict = {0:'Supuestamente Dolor', 1:'Supuestamente El Mismo', 2:'Supuestamente Disfrutar La Vida', 3:'Supuestamente Experiencias Adrenalinicas', 4:'Terceros'}


# In[34]:


import numpy as np
def classify_topic(document):
    return topic_dict[np.argmax(lda.transform(tfv.transform([document])))]


# In[35]:


queen_df['topic'] = queen_df.text.apply(classify_topic)


# In[36]:


labels = queen_df.topic.value_counts().index.tolist()
values = queen_df.topic.value_counts().values.tolist()


# In[37]:


plot = go.Figure(go.Bar(x=labels, y=values, marker_color=['orange']*4+['#cc6600']*4+['#b35900']*4+['#994d00']*4+['#804000']*4))
plot.update_layout(title='Frecuencia De Los Topicos')
plot.show()


# In[ ]:




