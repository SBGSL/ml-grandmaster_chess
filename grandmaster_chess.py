import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# Read the data
df = pd.read_csv('Height_Weight.csv')
print (df.shape)
s_df = df.sample(n=20, random_state=11)
train, test = train_test_split(s_df, test_size=0.3, random_state=11)
train = train.sort_values('Height')
test = test.sort_values('Height')

# predictors
xtrain = train['Height'].to_frame()
xtest  = test['Height'].to_frame()

# output
ytrain = train['Weight']
ytest  = test['Weight']
lr = LinearRegression().fit(xtrain, ytrain)
print ('The equation is: Weight = {} + {}xHeight'.format(round(lr.intercept_,3),round(lr.coef_[0],3)))
