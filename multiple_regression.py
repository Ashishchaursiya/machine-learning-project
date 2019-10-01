

# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Classification.csv')
#temp = dataset.values

#dataset= dataset.get_dummies(dataset)
features = dataset.iloc[:, :-1].values
labels = dataset.iloc[:, -1].values

#Check if any NaN values in dataset
dataset.isnull().any(axis=0)


#check data types for each column
print (dataset.dtypes)


# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
features[:, 0] = labelencoder.fit_transform(features[:, 0])


from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [0])
features = onehotencoder.fit_transform(features).toarray()

# Avoiding the Dummy Variable Trap
# dropping first column
features = features[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.2, random_state = 0)


# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(features_train, labels_train)

#To see the value of the intercept and slop calculated by the linear regression algorithm for our dataset, execute the following code.
print(regressor.intercept_)  
print (regressor.coef_)


# Predicting the Test set results
Pred = regressor.predict(features_test)

import pandas as pd

print (pd.DataFrame(Pred, labels_test))


# Prediction for a new values 
# make this accorifng to the data csv
# Development is replaced by 1,0,0 to 0,0 to remove dummy trap

import numpy as np
x = [0,0, 1500, 1, 2]
x = np.array(x)
x = x.reshape(1,5)
regressor.predict(x)



le = labelencoder.transform(['Development'])
ohe = onehotencoder.transform(le.reshape(1,1)).toarray()
x = [ohe[0][1],ohe[0][2],1150,3,4]
x = np.array(x)


regressor.predict(x.reshape(1, -1))
