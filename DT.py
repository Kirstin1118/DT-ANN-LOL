# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 10:37:27 2020

@author: Kirstin YU
"""

# Load libraries
import pandas as pd 
import numpy as np
from sklearn.metrics import accuracy_score #for accuracy calculation 
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier 

col_names = ['gameId','creationTime','gameDuration','seasonId','winner','firstBlood','firstTower','firstInhibitor','firstBaron','firstDragon','firstRiftHerald','t1_towerKills','t1_inhibitorKills','t1_baronKills','t1_dragonKills','t1_riftHeraldKills','t2_towerKills','t2_inhibitorKills','t2_baronKills','t2_dragonKills','t2_riftHeraldKills'] 
# load dataset 
train = pd.read_csv("C:\\Users\\user\\Desktop\\project\\new_data.csv", header=None, names=col_names) # Pay attention to the address where the data is stored
train = train.iloc[1:] # delete the first row of the dataframe 
train.head()  

test = pd.read_csv("C:\\Users\\user\\Desktop\\project\\test_set.csv", header=None, names=col_names) # Pay attention to the address where the data is stored 
test = test.iloc[1:] # delete the first row of the dataframe 
test.head()

#split dataset in features and target variable 
feature_cols = ['firstBlood','firstTower','firstInhibitor','firstBaron','firstDragon','firstRiftHerald','t1_towerKills','t1_inhibitorKills','t1_baronKills','t1_dragonKills','t1_riftHeraldKills','t2_towerKills','t2_inhibitorKills','t2_baronKills','t2_dragonKills','t2_riftHeraldKills'] 
X_train = train[feature_cols] # Features 
y_train = train.winner # Target variable 
X_test = test[feature_cols] # Features 
y_test = test.winner # Target variable 

# Create Decision Tree classifer object 
clf = DecisionTreeClassifier(criterion="entropy", max_depth=8) 
 
# Train Decision Tree Classifer 
clf = clf.fit(X_train,y_train) 
 
#Predict the response for test dataset 
y_pred = clf.predict(X_test) 
 
# Model Accuracy, how often is the classifier correct? 
print("Accuracy:",accuracy_score(y_test, y_pred)) 

from sklearn.tree import export_graphviz 
from six import StringIO   
from IPython.display import Image   
import pydotplus 
import os      
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'  
# Configure environment variables 
dot_data = StringIO() 
export_graphviz(clf, out_file=dot_data,   
                filled=True, rounded=True, 
                special_characters=True,feature_names = 
feature_cols,class_names=['1','2']) 
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())   
graph.write_png('DTb.png')
Image(graph.create_png()) 