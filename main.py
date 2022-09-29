#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing libraries
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

import nltk
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from wordcloud import WordCloud
from nltk.tokenize import word_tokenize, sent_tokenize, wordpunct_tokenize, TreebankWordTokenizer, TweetTokenizer

import csv,sys
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV 
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import SelectKBest,chi2 
from sklearn.metrics import classification_report
from nltk.corpus import wordnet


# In[2]:


# Importing dataset
df = pd.read_table("project2_training_data.txt", names = ['doc'], header = None)
df1=pd.read_table('project2_training_data_labels.txt', names =['class'],header = None)
df['class'] = df1


# In[18]:


# Load Data in list

f1=open('project2_training_data.txt', 'r', encoding="utf8") 
data1 = list(csv.reader(f1,delimiter='\t'))
data = []
for i in data1:
    for j in i:
        data.append(j)
        
f2=open('project2_training_data_labels.txt', 'r', encoding="utf8") 
labels1 = list(csv.reader(f2,delimiter='\t'))
labels = []
for i in labels1:
    for j in i:
        labels.append(j)


# In[4]:


# Text Preprocessing

# Stopword removal
import re
from nltk.corpus import stopwords  ## stopwords from nltk corpus
import nltk

b = list(df["doc"])
nltk.download('wordnet')
import nltk.corpus
nltk.download('stopwords')

stop = stopwords.words('english')
import string
exclude = set(string.punctuation)
corpus = []
for i in range(len(b)):
    review =re.sub(r'http\S+', ' ', str(b[i]))
    review = re.sub("\d*\.\d+","",review)
    review =re.sub(r'@\S+', ' ', review)
    review = re.sub(r'[^\w\s]', '', review)
    review = re.sub('()', '', review)
    review = re.sub(r'[0-9]+', '', review)

    
    
    review = re.sub('\[[^]]*\]', ' ', review)
    
    review = review.lower()
    review = review.split()
  
    review = ' '.join(review)

    corpus.append(review)
df = df.assign(clean_doc = corpus)


df['doc_clean'] = df['clean_doc'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
df_clean = df[['doc_clean','class']]


# Stemming and Lemmatization

w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]


df_clean['text_lemmatized'] = df_clean.doc_clean.apply(lemmatize_text)


from nltk.stem.snowball import SnowballStemmer
# Use English stemmer.
stemmer = SnowballStemmer("english")
df_clean['stemmed'] = df_clean['text_lemmatized'].apply(lambda x: [stemmer.stem(y) for y in x]) # Stem every word.


for i in range(len(df_clean['text_lemmatized'])):
    df_clean['text_lemmatized'][i] = ' '.join(df_clean['text_lemmatized'][i])
    
# Stopword removal only
df_clean_1 = df_clean[['doc_clean','class']]
# Stopword removal + stemming and Lemmatization
df_clean_2 = df_clean[['text_lemmatized','class']]
df_clean_2 = df_clean_2.rename(columns={'text_lemmatized':'all_cleaned'})


# In[20]:


# TFIDF

vectorizer_1=TfidfVectorizer(stop_words='english',ngram_range=(1,3),token_pattern=r'\b\w+\b')
tfidf = vectorizer_1.fit_transform(data)
terms=vectorizer_1.get_feature_names()
tfidf = tfidf.toarray()

cv_dataframe=pd.DataFrame(tfidf,columns=vectorizer_1.get_feature_names())
print('**********TF-IDF Matrix**********\n')
print(cv_dataframe)


# In[6]:


# Train-Test Split

trn_data, tst_data, trn_cat, tst_cat = train_test_split(tfidf, labels, test_size=0.20, random_state=42,stratify=labels) 


# In[7]:


# Model training

# Simple run without parameter tuning

clf=MultinomialNB(alpha=0,fit_prior=True, class_prior=None)  

#Classificaion    
clf.fit(trn_data,trn_cat)
predicted = clf.predict(tst_data)
predicted =list(predicted)

print(f'the performance measure of the simple model (without parameter tuning): \n\n {classification_report(tst_cat, predicted)}\n')



# In[8]:


# Train-Test Split

trn_data_1, tst_data_1, trn_cat_1, tst_cat_1 = train_test_split(data, labels, test_size=0.20, random_state=42,stratify=labels) 


# In[9]:


# With parameter tuning (without any data cleaning)

classification_models = ['Logistic Regression', 'Multinomial Naive Bayes', 'Linear SVC', 'SVC', 'Decision Tree']

for i in classification_models:
    # Naive Bayes Classifier    
    if i=='Multinomial Naive Bayes':      
        clf=MultinomialNB(alpha=0,fit_prior=True, class_prior=None)  
        clf_parameters = {
        'clf__alpha':(0,1),
        }  
# SVM Classifier
    elif i=='Linear SVC': 
        clf = svm.LinearSVC(class_weight='balanced')  
        clf_parameters = {
        'clf__C':(0.1,1,2,10,50,100),
        }   
    elif i=='SVC':
        clf = svm.SVC(kernel='linear', class_weight='balanced')  
        clf_parameters = {
        'clf__C':(0.1,0.5,1,2,10,50,100),
        }   
# Logistic Regression Classifier    
    elif i=='Logistic Regression':    
        clf=LogisticRegression(class_weight='balanced') 
        clf_parameters = {
        'clf__solver':('newton-cg','lbfgs','liblinear'),
        }    
# Decision Tree Classifier
    elif i=='Decision Tree':
        clf = DecisionTreeClassifier(random_state=40)
        clf_parameters = {
        'clf__criterion':('gini', 'entropy'), 
        'clf__max_features':('auto', 'sqrt', 'log2'),
        'clf__ccp_alpha':(0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1),
        }  
                                
# Feature Extraction
    
    pipeline = Pipeline([
    ('vect', CountVectorizer(token_pattern=r'\b\w+\b')),
    ('feature_selector', SelectKBest(chi2, k=1000)),         
    ('tfidf', TfidfTransformer(use_idf=True,smooth_idf=True)),     
    ('clf', clf),]) 
        
    feature_parameters = {
    'vect__min_df': (2,3),
    'vect__ngram_range': ((1,1),(1, 2),(1,3),(2,3)),  # Unigrams, Bigrams or Trigrams
    }
    
# Classificaion
    parameters={**feature_parameters,**clf_parameters} 
    grid = GridSearchCV(pipeline,parameters,scoring='f1_micro',cv=10)          
    grid.fit(trn_data_1,trn_cat_1)     
    clf= grid.best_estimator_  
    print(f'\n\n********* Best Set of Parameters for {i}********* \n\n')
    print(clf)
    
    predicted = clf.predict(tst_data_1)
    predicted =list(predicted)

    # Evaluation
    print('\n Total documents in the training set: '+str(len(trn_data_1))+'\n')    
    print('\n Total documents in the test set: '+str(len(tst_data_1))+'\n')
    print ('\n Confusion Matrix \n')  
    print (confusion_matrix(tst_cat_1, predicted))  

    print(f'the performance measure of the {i} model with tuned parameters: \n\n {classification_report(tst_cat_1, predicted)}\n')

    pr=precision_score(tst_cat_1, predicted, average='micro') 
    print (f'\n Precision for {i}:'+str(pr)) 

    rl=recall_score(tst_cat_1, predicted, average='micro') 
    print (f'\n Recall for {i}:'+str(rl))

    fm=f1_score(tst_cat_1, predicted, average='micro') 
    print (f'\n Micro Averaged F1-Score for {i}:'+str(fm))


# In[10]:


'''The training dataset is trained and the model accuracy is observed for the following models: Logistic regression, Multinomial naive bayes, 
Random forest, Decision Tree, SVC, Linear SVC with their tuned parameters. The Macro averaged f1 score for Logitic regression is maximum (0.878) 
for parameters (class_weight='balanced',solver='newton-cg') indicating that the model is the best performing among all other for the given dataset (without any text-preprocessing).
We will use this model for further analysis.'''

clf_mod_unc = LogisticRegression(class_weight='balanced', solver='newton-cg')
clf_model_unc = clf_mod_unc.fit(trn_data,trn_cat)


# In[11]:


# Comparing the performance of Logistic regression model with and without text-preprocessing (stemming, lemmetization, stopword removal etc.)

# With Cleaned data (stopword removal, stemming and lemmatization)
data_clean = []
for i in df_clean.columns:
    for j in df_clean[i]:
        data_clean.append(j)
    break
# Vectorization
vectorizer=TfidfVectorizer(stop_words='english',ngram_range=(1,3),token_pattern=r'\b\w+\b')
tfidf_cln = vectorizer.fit_transform(data_clean)
terms_cln=vectorizer.get_feature_names()
tfidf_cln = tfidf_cln.toarray()
    
# Train-test split
trn_data_cln, tst_data_cln, trn_cat_cln, tst_cat_cln = train_test_split(tfidf_cln, labels, test_size=0.20, random_state=42,stratify=labels) 

# Fitting and testing of data
clf_mod_cln = LogisticRegression(class_weight='balanced', solver='newton-cg')
clf_model_cln = clf_mod_cln.fit(trn_data_cln,trn_cat_cln)

predicted_1 = clf_model_cln.predict(tst_data_cln)
predicted_1 =list(predicted_1)

# Evaluation
print('\n Total documents in the training set: '+str(len(trn_data))+'\n')    
print('\n Total documents in the test set: '+str(len(tst_data))+'\n')
print ('\n Confusion Matrix \n')  
print (confusion_matrix(tst_cat, predicted))  

print(f'the performance measure of Logistic Regression with processed data\n: {classification_report(tst_cat_cln, predicted_1)}\n')


# In[12]:


# Checking the performance of model based on the combination of Bag of Words and Part of speech tagging

tokenized_sents = [nltk.word_tokenize(i) for i in df_clean_1['doc_clean']]
pos_tags = [nltk.pos_tag(token) for token in tokenized_sents]
#print(pos_tags)

only_pos_tags = []

for i in pos_tags:
    one_sen = ''
    for j in i:
        one_sen += (j[1] + ', ')
    only_pos_tags.append(one_sen)
    
vectorizer=CountVectorizer()
vec_pos = vectorizer.fit_transform(only_pos_tags)
terms_pos=vectorizer.get_feature_names()
vec_pos = vec_pos.toarray()


lst = []
for i in vec_pos:
    wse = []
    for j in i:
        wse.append(j)
    lst.append(wse)
    
for i in lst:
    while len(i) != tfidf.shape[1]:
        i.append(0)
        
vec_pos_ar = np.array(lst)

add_vec = tfidf + vec_pos_ar
    


# In[13]:


# Train-test split
trn_data_pos, tst_data_pos, trn_cat_pos, tst_cat_pos = train_test_split(add_vec, labels, test_size=0.20, random_state=42,stratify=labels) 

# Fitting and testing of data
clf_mod_pos = LogisticRegression(class_weight='balanced',solver='newton-cg')
clf_model_pos = clf_mod_pos.fit(trn_data_pos,trn_cat_pos)

predicted_pos = clf_model_pos.predict(tst_data_pos)
predicted_pos =list(predicted_pos)

# Evaluation
print('\n Total documents in the training set: '+str(len(trn_data_pos))+'\n')    
print('\n Total documents in the test set: '+str(len(tst_data_pos))+'\n')
print ('\n Confusion Matrix \n')  
print (confusion_matrix(tst_cat_pos, predicted_pos))  

print(f'the performance measure of Logistic Regression with processed data\n: {classification_report(tst_cat_pos, predicted_pos)}\n')

pr=precision_score(tst_cat_pos, predicted_pos, average='micro') 
print (f'\n Precision :'+str(pr)) 

rl=recall_score(tst_cat_pos, predicted_pos, average='micro') 
print (f'\n Recall :'+str(rl))

fm=f1_score(tst_cat_pos, predicted_pos, average='micro') 
print (f'\n Micro Averaged F1-Score :'+str(fm))


# In[16]:


# Checking the performance of our model ("model_1") on the given dataset

pos_wrd_dict = {}
neg_wrd_dict = {}
neut_wrd_dict = {}
for i, j in zip(df_clean['doc_clean'], df_clean['class']):
    if j == 'positive':
        for k in i.split():
            if k not in pos_wrd_dict.keys():
                pos_wrd_dict.update({k:1})
            else:
                pos_wrd_dict[k] +=1
    elif j == 'negative':
        for k in i.split():
            if k not in neg_wrd_dict.keys():
                neg_wrd_dict.update({k:1})
            else:
                neg_wrd_dict[k] +=1
    elif j == 'neutral':
        for k in i.split():
            if k not in neut_wrd_dict.keys():
                neut_wrd_dict.update({k:1})
            else:
                neut_wrd_dict[k] +=1
                


pos_wrd_dict_rev = [[i, j] for j, i in pos_wrd_dict.items()]
neg_wrd_dict_rev = [[i, j] for j, i in neg_wrd_dict.items()]
neut_wrd_dict_rev = [[i, j] for j, i in neut_wrd_dict.items()]

a = []
b = []
c = []
for i in pos_wrd_dict.values():
    a.append(i)
for i in neg_wrd_dict.values():
    b.append(i)
for i in neut_wrd_dict.values():
    c.append(i)
    
a.sort(reverse=True)
b.sort(reverse=True)
c.sort(reverse=True)

pos_t50 = []
neg_t50 = []
neut_t50 = []

for i in [a, b, c]:
    if i == a:
        for j in a[0:50]:
            for k in pos_wrd_dict_rev:
                if k[0] == j:
                    pos_t50.append(k[1])
                    pos_wrd_dict_rev.remove(k)
    elif i == b:
        for j in b[0:50]:
            for k in neg_wrd_dict_rev:
                if k[0] == j:
                    neg_t50.append(k[1])
                    neg_wrd_dict_rev.remove(k)
    elif i == c:
        for j in c[0:50]:
            for k in neut_wrd_dict_rev:
                if k[0] == j:
                    neut_t50.append(k[1])
                    neut_wrd_dict_rev.remove(k)
                
    
    
values3 = [x for x in pos_t50[0:10]]
freq3 = a[0:10]

values4 = [x for x in neg_t50[0:10]]
freq4 = b[0:10]

values5 = [x for x in neut_t50[0:10]]
freq5 = c[0:10]


# In[17]:


whole_corpus = []
for i, j in zip(df_clean['doc_clean'], df_clean['class']):
    doc_vec = []
    for k in TweetTokenizer().tokenize(i):
        try:
            syn1 = wordnet.synsets(k)[0]
        except IndexError:
            syn1 = 0
        sim_val = 0
        sim_dict = []
        count = 0
        for l in pos_t50[0:50]+neg_t50[0:50]+neut_t50[0:50]:
            try:
                syn2 = wordnet.synsets(l)[0]
            except IndexError:
                syn2 = 0
            try:
                similar = syn1.wup_similarity(syn2)
            except:
                similar = 0
            count += 1
            #print(similar)
            if similar >= sim_val:
                sim_dict = [l, count]
                sim_val = similar
        if 0 <= sim_dict[1] <= 50:
            #print(sim_dict[l])
            k = 1
        elif 51 <= sim_dict[1] <= 100:
            #print(sim_dict[l])
            k = -1
        elif 101 <= sim_dict[1] <= 150:
            #print(sim_dict[l])
            k = 0
        doc_vec.append(k)
    whole_corpus.append(doc_vec)
            
        
labels_output = []
for i in whole_corpus:
    add = 0
    for j in i:
        add += j
    if add > +2:
        add = 'positive'
    elif add < -2:
        add = 'negative'
    else:
        add = 'neutral'
    labels_output.append(add)
    
precision_m1 = precision_score(labels, labels_output, average='micro')
print(f"\nThe precision for model_1 is : {precision_m1}\n")
    
recall_m1 = recall_score(labels, labels_output, average='micro')
print(f"The recall for model_1 is : {recall_m1}\n")

f1_score_m1 = f1_score(labels, labels_output, average='micro')
print(f"The micro averaged f1 score for model_1 is : {f1_score_m1}")
        


# In[1]:


'''We found that the tuned Logistic Regeression works better for unprocessed data among all other models and with different data processing techniques. It performs better than combining BOW and POS tagging 
vectors as well. It also perform better than our self built model "model_1". So we are going to produce result for the test 
dataset without processing on the Logistic Regression model with the parameters (class_weight='balanced', solver='newton-cg').'''


# In[23]:


# Loading test dataset

test_df = pd.read_table("project2_test_data.txt", names = ['doc'], header = None)

test_data = []
for i in test_df.columns:
    for j in test_df[i]:
        test_data.append(j)
    break
    
# Test Data vectorization
tfidf_test = vectorizer_1.transform(test_data)
terms_test=vectorizer_1.get_feature_names()
tfidf_test = tfidf_test.toarray()
    

# Prediction
predicted_test = clf_model_unc.predict(tfidf_test)
predicted_test =list(predicted_test)

test_df = pd.DataFrame(predicted_test)
test_df.to_csv('classes.txt', header=None,index=None, sep=' ', mode='a')


# In[ ]:




