import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

ad_df = pd.read_csv("https://raw.githubusercontent.com/Arathi-J/DS-ML/main/advertising%20-%20advertising.csv")

x = ad_df['TV']
y = ad_df['Sales']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
x_train = x_train.values.reshape(-1,1)
x_test = x_test.values.reshape(-1,1)

model = LinearRegression()
model.fit(x_train.reshape(-1,1),y_train)

def sales_predicted(budget):
    return model.predict([[budget]])
print('Predicted value = ',sales_predicted(50))
slop = model.coef_[0]
intercept = model.intercept_
print('Slope = ',slop)
print('Intercept = ',intercept)


