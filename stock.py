import pandas as pd
import matplotlib.pyplot as plot
import sklearn
from numpy import shape

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#Dataset input
dataset = pd.read_csv(r'C:\Users\lenovo\Desktop\TataMotors.csv')
print("***************ORIGINAL DATASET***************")
print(dataset)

#For the "Date" feature to take it as a float type and not string.
dataset=dataset.replace('[^\d.]','',regex=True).astype(float)

#display ley elements
print("***************KEY ELEMENTS And Datatype OF THE DATASET***************")
print(dataset.keys())
print(dataset.dtypes)

#creation of dataframe and display 1st five entries with all attributes.
df = pd.DataFrame(dataset)
print("******************Original data set display all 7 features/columns and first 5 rows******************** ")
print(df.head(5))
print(shape(df))

#Load the Standardizer
import numpy as np
scaler= StandardScaler()

#Standardized the data
scaled_data = scaler.fit_transform(df)
print("***********The Data after Standardization using StandardScalar********** ")

#Standardised data had Nan(Not a number) values and infinite values which cannot be used for PCA.
#So nan_to_num is used in which Nan is replaced by zero and infinte values by largest finite value.
scaled_data = np.nan_to_num(scaled_data)

#Check  the results after this function.
print(np.any(np.isnan(scaled_data)))
print(np.all(np.isfinite(scaled_data)))

print(scaled_data)
print(shape(scaled_data))

#PCA
pca = PCA(n_components=2)
x = pca.fit_transform(scaled_data)
print("***************PCA Transformed Dataset***************")
print(x)
print(shape(x))

'''
Plot the graph for High values against Date.
plot.figure(figsize=(8,6))
plot.plot(scaled_data[:,0],scaled_data[:,1])
plot.xlabel('Date')
plot.ylabel('High')
plot.title('Principal Components')
plot.show()

'''

#PCA
plot.figure(figsize=(8,6))
plot.scatter(x[:,0],x[:,1],c=df['Date'])
plot.xlabel('PC1')
plot.ylabel('PC2')
plot.title('Principal Components')
plot.show()
