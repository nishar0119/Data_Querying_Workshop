import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import random
age_data = pd.read_csv("demographics_by_age_group.csv", usecols = [0, 1, 2], nrows = 12)
df = pd.DataFrame(age_data)
X = df['age group']
X = pd.get_dummies(data=X, drop_first=True)

#x = df['age group']
y = df['cases']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = .30, random_state = 0)
mlr = LinearRegression()
model = mlr.fit(x_train, y_train)
y_predict= mlr.predict(x_test)
plt.scatter(y_test, y_predict)
plt.show()
#print(age_data)
plt.scatter(df['age group'], df['cases'], alpha=0.4)
#plt.show()
