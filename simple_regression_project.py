#This is simple linear regression project.that predict score basis of the hour.
#Importing Libraries
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  

#imports the CSV dataset using pandas

dataset = pd.read_csv('student_scores.csv')  

#explore the dataset
print (dataset.shape)
print (dataset.ndim)
print (dataset.head())
print (dataset.describe())

#check data types for each column
print (dataset.dtypes)

#Check if any NaN values in dataset
dataset.isnull().any(axis=0)

# Check for outlier values
plt.boxplot(dataset.values)

#let's plot our data points on 2-D graph to eyeball our dataset 
# and see if we can manually find any relationship between the data. 
dataset.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()

#prepare the data to train the model
features = dataset.iloc[:, :-1].values  
labels = dataset.iloc[:, 1].values 

"""
train the model now 
"""

from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(features, labels) 

#To see the value of the intercept and slop calculated by the linear regression algorithm for our dataset, execute the following code.
print(regressor.intercept_)  
print (regressor.coef_)

#what would be prediction of score if someone studies 9 hours
print (regressor.coef_*9 + regressor.intercept_)
print (regressor.predict(9))


#Visualize the best fit line
import matplotlib.pyplot as plt

# Visualising the  results
plt.scatter(features, labels, color = 'red')
plt.plot(features, regressor.predict(features), color = 'blue')
plt.title('Study Hours and Exam Score')
plt.xlabel('Study Hours')
plt.ylabel('Exam Score: Marks')
plt.show()

#and then the train test split
from sklearn.model_selection import train_test_split  
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=0)  

#train the algo
from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(features_train, labels_train) 

#making predictions
#To make pre-dictions on the test data, execute the following script:

labels_pred = regressor.predict(features_test) 

#To compare the actual output values for features_test with the predicted values, execute the following script 
df = pd.DataFrame({'Actual': labels_test, 'Predicted': labels_pred})  
print (df) 

from sklearn import metrics  
print('Mean Absolute Error:', metrics.mean_absolute_error(labels_test, labels_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(labels_test, labels_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(labels_test, labels_pred))) 

print (np.mean(dataset.values[:,1]))
 
#Visualize the best fit line
import matplotlib.pyplot as plt

# Visualising the Test set results
plt.scatter(features_test, labels_test, color = 'red')
plt.plot(features_train, regressor.predict(features_train), color = 'blue')
plt.title('Study Hours and Exam Score')
plt.xlabel('Study Hours')
plt.ylabel('Exam Score: Marks')
plt.show()


#Model accuracy
print (regressor.score(features_test, labels_test))
print (regressor.score(features_train, labels_train))


