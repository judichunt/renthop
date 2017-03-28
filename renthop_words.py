# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 01:21:52 2017

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





data_rent['name'] =  data_rent['description'].apply(lambda x: sum([x.count('doorman')])>0)
data_rent['name'] =  data_rent['name'] | data_rent['features'].apply(lambda x: sum([x.count('doorman')])>0)
         
crossmatrix=pd.crosstab(data_rent['interest_level'], data_rent['name'] , rownames=['Ture'], colnames=['Predicted'], margins=True)
print(crossmatrix)


