
# Importing required libraries

from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split



# Read mouse data for Emotional readings
emo_MR = pd.read_csv('Group 12 Data/Emotional/MR.csv')
emo_MP = pd.read_csv('Group 12 Data/Emotional/MP.csv')
emo_MM = pd.read_csv('Group 12 Data/Emotional/Mouse_movement_emotional.csv')
emo_MC = pd.read_csv('Group 12 Data/Emotional/MC.csv')

# Read mouse data for Neutral readings
neu_MR = pd.read_csv('Group 12 Data/Neutral/MR.csv')
neu_MP = pd.read_csv('Group 12 Data/Neutral/MP.csv')
neu_MM = pd.read_csv('Group 12 Data/Neutral/Mouse_movement_neutral.csv')
neu_MC = pd.read_csv('Group 12 Data/Neutral/MC.csv')


# Separating Attributes and label

le = preprocessing.LabelEncoder()
emo_MM = emo_MM.to_numpy()
emo_MM = np.delete(emo_MM,0,1)
X_MME = np.delete(emo_MM,emo_MM.shape[1]-1,1)
y_MME = np.delete(emo_MM,slice(emo_MM.shape[1]-1),1)
y_MME = le.fit_transform(y_MME)

neu_MM = neu_MM.to_numpy()
neu_MM = np.delete(neu_MM,0,1)
X_MMN = np.delete(neu_MM,neu_MM.shape[1]-1,1)
y_MMN = np.delete(neu_MM,slice(neu_MM.shape[1]-1),1)
y_MMN = le.transform(y_MMN)

# Splitting train test

X_MME_train, X_MME_test, y_MME_train, y_MME_test = train_test_split(X_MME, y_MME, test_size=0.2, random_state=42)
X_MMN_train, X_MMN_test, y_MMN_train, y_MMN_test = train_test_split(X_MMN, y_MMN, test_size=0.2, random_state=42)



# Neural Net Classifiers for predicting emotional user

clf_E = MLPClassifier(hidden_layer_sizes=(4,4,5), max_iter=1000)
clf_E.fit(X_MME_train,y_MME_train)
print("Accuracy for Emotional :" + str(clf_E.score(X_MME_test,y_MME_test)))

clf_N = MLPClassifier(hidden_layer_sizes=(4,4,5), max_iter=1000)
clf_N.fit(X_MMN_train,y_MMN_train)
print("Accuracy for Neutral :" + str(clf_N.score(X_MMN_test,y_MMN_test)))





