#Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Import Dataset
dataset=pd.read_csv("Data_Preprocessing.csv")
dataset.isnull().sum()

#Dependent and Independent Variable
x=dataset.iloc[:,0:3].values
y=dataset.iloc[:,3:4].values

#Import imputer and fill the null values
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
imputer=imputer.fit(x[:,1:3])
x[:,1:3]=imputer.transform(x[:,1:3])

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
Label_x = LabelEncoder()
x[:,0]=Label_x.fit_transform(x[:,0])
ColumnTransformer = ColumnTransformer([('encoder', OneHotEncoder(),[0])],remainder = 'passthrough')
x=np.array(ColumnTransformer.fit_transform(x),dtype = np.str)
x=x[:,1:]

#Split the data into train and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=.20,random_state = 0)

#Features Scarlar
from sklearn.preprocessing import StandardScaler 
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
