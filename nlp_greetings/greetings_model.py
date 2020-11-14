

'''With Naïve Bayes'''

import streamlit as st 
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer

convo_df = pd.read_csv('tagged_selections_by_sentence.csv')
convo_df = convo_df.dropna()

comment_list = convo_df['Selected'].tolist()
greetings_list = np.array(convo_df['Greeting'].tolist())

count_vect = CountVectorizer()
x_train_counts = count_vect.fit_transform(comment_list)

tfidf_transformer = TfidfTransformer()
x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)

train_x, test_x, train_y, test_y = train_test_split(x_train_tfidf, greetings_list, test_size=0.3)

clf = MultinomialNB().fit(train_x, train_y)
y_score = clf.predict(test_x)

n_right = 0
for i in range(len(y_score)):
    if y_score[i] == test_y[i]:
        n_right += 1


st.write("### Accuracy: ", round((n_right/float(len(test_y)) * 100),2),"%")



