# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 01:18:11 2017

@author: user
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import euclidean_distances
import csv
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


data_rent= pd.read_json("C:/Users/user/Downloads/renthop/trainn.json")


import nltk
from nltk.tag import pos_tag
from nltk import word_tokenize
from nltk.corpus import stopwords
stop = stopwords.words('english')




data_high=data_rent.loc[(data_rent['interest_level']=='high')]
data_medium=data_rent.loc[(data_rent['interest_level']=='medium')]
data_low=data_rent.loc[(data_rent['interest_level']=='low')]


data_high['cleaned'].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis = 0)




from sklearn.model_selection import train_test_split
train, test = train_test_split(data_rent, test_size = 0.2)
print(len(train))
print(len(test))




binVectorizer = CountVectorizer(binary=True)
counts = binVectorizer.fit_transform(train['feat_cleaned'])



from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
targets = train['interest_level'].values
classifier.fit(counts, targets)





examples = test['feat_cleaned']
example_counts = binVectorizer.transform(examples)
predictions = classifier.predict(example_counts)
predictions_df=pd.DataFrame(predictions)
#predictions_df.head(10)
actual=test['interest_level'].values
from sklearn.metrics import confusion_matrix
matrix=pd.DataFrame(confusion_matrix(actual, predictions,labels=["low", "medium", "high"]))
print(matrix)

test_rent= pd.read_json("C:/Users/user/Downloads/renthop/testn.json")

#test_rent['predictions']=
examples = test_rent['feat_cleaned']
example_counts = binVectorizer.transform(examples)
predictions_test = classifier.predict(example_counts)
predictions_test_df=pd.DataFrame(predictions_test)


crossmatrix=pd.crosstab(test['interest_level'], predictions, rownames=['True'], colnames=['Predicted'], margins=True)
print(crossmatrix)

#result=test_rent['listing_id']
#result['predictions']=predictions_test_df

predictions_test_df.to_csv("C:/Users/user/Downloads/renthop/result.csv", index=True)


