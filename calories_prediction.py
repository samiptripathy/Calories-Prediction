import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
#importing datasets
dataset= pd.read_csv('calories.csv')
print(dataset.shape)
print(dataset.head()) #shows the first 5 entries of the dataset
dataset1= pd.read_csv('exercise.csv')
print(dataset1.shape)
print(dataset1.head())
#combining both the dataframes
calories_data = pd.concat([dataset1, dataset['Calories']], axis = 1)#column is being added therefore axis = 1
print(calories_data.head())
print(calories_data.shape)
#data analysis
calories_data.isnull().sum()
calories_data.describe()
sns.countplot(calories_data['Gender'])#plotting the gender column in count plot
sns.distplot(calories_data['Age'])    #distribution plot for age
sns.distplot(calories_data['Weight'])
#finding correlation in the dataset
correlation = calories_data.corr()
plt.figure(figsize= (10,10))
sns.heatmap(correlation, cbar = True, fmt = '.1f', annot = True, annot_kws={'size':8},cmap = 'Blues') #body_temp, heart_rate, duration are highly correlated to calories
#converting categorical values to numerical values
calories_data.replace({"Gender":{'male':0, 'female':1}}, inplace = True)
calories_data.head()
#separating features and target
X = calories_data.drop(columns = ['User_ID','Calories'], axis = 1)
Y = calories_data['Calories']
#splitting the dataset into training and test data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 2)
#model training (xgboost regressor)
from xgboost import XGBRegressor
model = XGBRegressor()
model.fit(X_train, Y_train)
#prediction on test data
test_data_prediction = model.predict(X_test)
#calculating mean absolute error
from sklearn import metrics
mae = metrics.mean_absolute_error(Y_test, test_data_prediction)
print("Mean Absolute Error = ", mae)

