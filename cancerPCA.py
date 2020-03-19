import pandas as pd
import matplotlib.pyplot as plot
from numpy import shape

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
#Creating pandas dataframe
dataset = pd.DataFrame()
#loading the dataset
dataset = load_breast_cancer()
#display ley elements
print("***************KEY ELEMENTS OF THE DATASET***************")
print(dataset.keys())
#display description  the dataset
print("**********COMPLETE DESCRIPTION OF BREAST CANCER DATASET**************")
print(dataset.DESCR)

#creation of dataframe and display 1st five entries with all attributes.
df = pd.DataFrame(dataset['data'],columns=dataset['feature_names'])
print("******************Original data set display all 30 features/columns and first 5 rows******************** ")
print(df.head(5))

#Load the Standardizer
scaler = StandardScaler()

print(scaler.fit(df))
#Standardized the data
scaled_data = scaler.fit_transform(df)
print("***********The Data after Standardization using StandardScalar********** ")
print(scaled_data)


#USing PCA to decompose dataset features from 30 features to only 2 features in different vector space.
pca  = PCA(n_components=2)
pca.fit(scaled_data)
print("*********PCA************")
print(pca)
x_pca = pca.transform(scaled_data)
print("BEFORE PCA : 30 FEATRUES")
print(shape(scaled_data))
print("AFTER PCA : 2 FEATRUES")
print(shape(x_pca))
print("**********Scaled data after Standardization***********")
print(scaled_data)
print("************PCA Tranformed data*******************")
print(x_pca)

#plotting the two Principal components
plot.figure(figsize=(8,6))
#Plot all the rows of these two PC
plot.scatter(x_pca[:,0],x_pca[:,1],c = dataset['target'])
plot.xlabel('First Principal Component')
plot.ylabel('Second Principal Component')
plot.show()


'''
PLEASE  TAKE A NOTE :
In the Figure we have two principal components PC1 ad PC2 ,
and as shown the entries are very well classified into two classes..
here the two classes are ,
Class 1 = Malignant(Values = 0 , Colour = Yellow)
Class 2 = Benign (Value =1 , Colour = Purple)
Why to use PCA ?
1.Data Visualization
2.Speeding Machine Learning algorithm
'''

