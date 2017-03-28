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

from subprocess import check_output
#print (check_output(["ls","C:/Users/user/Downloads/renthop"]).decode("utf8"))
data_rent= pd.read_json("C:/Users/user/Downloads/renthop/test.json")
print(data_rent.head())

#f = open("C:/Users/user/Downloads/renthop/data1.csv")

#data_rent.to_csv("C:/Users/user/Downloads/renthop/data1.csv", index=False)#, cols=('A','B','sum'))

data_rent['features']=data_rent['features'].apply(lambda x: ', '.join(x))
import nltk
from nltk.tag import pos_tag
from nltk import word_tokenize
from nltk.corpus import stopwords
stop = stopwords.words('english')
def cleaning_text(sentence):
   sentence=sentence.lower()
   sentence=re.sub('[^\w\s]',' ', sentence)
   sentence=re.sub('\d+',' ', sentence)
   cleaned=' '.join([w for w in sentence.split() if not w in stop])
   cleaned=' '.join([w for w , pos in pos_tag(cleaned.split()) if (pos == 'NN' or pos=='JJ' or pos=='JJR' or pos=='JJS' )])
   cleaned=' '.join([w for w in cleaned.split() if not len(w)<=2 ]) #removes single lettered words and digits
   cleaned=cleaned.strip()
   return cleaned
	  
data_rent['cleaned']= data_rent['description'].apply(lambda x: cleaning_text(x))
del data_rent['description']

data_rent['feat_cleaned']= data_rent['features'].apply(lambda x: cleaning_text(x))
del data_rent['features']

data_rent["final_feat"] = data_rent["cleaned"].map(str) +" "+data_rent["feat_cleaned"]
#del data_rent['feat_cleaned']


data_rent.to_json("C:/Users/user/Downloads/renthop/testn.json")
'''
X=data_rent

del X['description']
del X['cleaned']
del X['features']
del X['feat_cleaned']

X.to_csv("C:/Users/user/Downloads/renthop/trainn.csv", index=False)
'''
