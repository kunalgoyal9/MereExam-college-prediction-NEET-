#!/usr/bin/python3.5
import os
from flask import Flask, jsonify, request
#from bs4 import BeautifulSoup
#from operator import itemgetter
#from time import strptime, strftime, mktime, gmtime, localtime, time
import json
import requests
#import threading
import pickle

app = Flask(__name__)


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
import re
#import matplotlib.pyplot as plt

data = pd.read_csv('round_1_alloted.csv',header=None)


data = data.set_index(0)

   
for idx,row in data.iterrows():    
	if 'UR PH' in row[5]:
	    data[5][idx] = 'UR PH'
	if 'OBC PH' in row[5] :
	    data[5][idx] = 'OBC PH'
	if(row[5] == 'SC PH1'):
	    data[5][idx] = 'SC PH'
	if(row[5] == 'OBC PH2'):
	    data[5][idx] = 'OBC PH'
	if(row[5] == 'SC PH2'):
	    data[5][idx] = 'SC PH'
	if(row[5] == 'ST PH1' or row[5] == 'ST PH2'):
	    data[5][idx] = 'ST PH'



data = data.drop(6,axis=1)



from collections import defaultdict
clg_to_int = defaultdict(str)
idx = 0
for i in data[2]:
    if(i not in clg_to_int):
        clg_to_int[i] = idx
        idx+=1






# map int to college through previous clg_to_int mapping

int_to_clg = ['str']*218
for value in clg_to_int:
    #print(value)
    int_to_clg[clg_to_int[value]] = value




for i in range(len(int_to_clg)):
  int_to_clg[i]=re.sub('\r',' ',int_to_clg[i])
  int_to_clg[i]=re.sub('\n',' ',int_to_clg[i])


# In[12]:


# add extra coloumn to convert college name to previously mapped value

data[7] = pd.Series(np.random.randint(len(data)), index=data.index)

for i in range(len(data)):
    #print(datar)
    data.iloc[i,5]=int(clg_to_int[data.iloc[i,1]])


# In[13]:


# now we can drop college name 


# data = data.drop(2,axis = 1)

la = np.logical_and
# urdata = data[la(la(data[4]=='UR',data[3]=='MBBS'),la(60 < data[7],data[7]<70))]
# rank = urdata[1]
# clgid = urdata[7]
# plt.scatter(rank,clgid,s=1)
# int_to_clg[55]



# partition X and Y 

X = data[[1,3,4,5]]
Y = data[7]
Y = pd.DataFrame(Y)




# creating category of X

category = [3,4,5]
for cat in category:
    dumm = pd.get_dummies(X[cat],prefix=cat)
    X = pd.concat([X,dumm],axis=1)
    X.drop(cat,axis=1,inplace = True)




#X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)


dtree_model = DecisionTreeRegressor(random_state=42).fit(X,Y)


@app.route('/')
def index():
    rank = int(request.args.get('rank'))
    course = request.args.get('course')
    alloted_cat = request.args.get('alloted_cat')
    candidate_cat = request.args.get('candidate_cat')
    
    val = np.max(data[la(data[4]==alloted_cat,data[3]==course)][1])
    val += val*0.1
    if rank>val:
        strng = 'Sorry, according to previous counsellings you are not eligible to get any college for All India NEET couselling'
        resp = jsonify(result=strng)
        return resp

    clmns = ['3_BDS','3_MBBS','4_OBC','4_OBC PH','4_SC','4_SC PH','4_ST','4_ST PH','4_UR','4_UR PH','5_OBC','5_OBC PH','5_SC','5_SC PH','5_ST','5_ST PH','5_UR','5_UR PH']
    test = pd.DataFrame({0:[1],1:[rank]})
    for i in clmns:
        test[i] = pd.Series(np.random.randint(1))
    test = test.set_index(0)
    test['3_'+course] = 1
    test['4_'+alloted_cat] = 1
    test['5_'+candidate_cat] = 1
    #dtree_model = DecisionTreeRegressor(random_state=42).fit(X, Y)
    ans = dtree_model.predict(test)
    clgs = []
    if int(ans[0])<3:
        for i in range(6):
            clgs.append(int_to_clg[i])
    elif int(ans[0])>213:
        for i in range(211,216):
            clgs.append(int_to_clg[i])
    else:
        for i in range(int(ans[0]),int(ans[0])+3):
            clgs.append(int_to_clg[i])
        for i in range(int(ans[0])-3,int(ans[0])):
            clgs.append(int_to_clg[i])
    resp = jsonify(result = clgs)
    return resp
if __name__ == '__main__':
	app.run(port = 7000, debug = True)