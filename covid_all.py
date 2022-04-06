import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
n = 75816
s = 7000
skip = sorted(random.sample(range(n),n-s))
df = pd.read_csv("CSCI3360_Data copy.csv")
X = df[['Income', 'Age', 'Race', 'Gender', 'Metro-Atlanta']]
y = df['cases']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = .30, random_state = 0)
mlr = LinearRegression()
model = mlr.fit(x_train, y_train)
print(mlr.coef_)
y_predict= mlr.predict(x_test)
print(mlr.score(x_test, y_test))
#plt.scatter(y_test, y_predict)
#plt.show()

print("Income")
print(df['Income'].corr(df['cases']))
print("Age")
print(df['Age'].corr(df['cases']))
print("Race")
print(df['Race'].corr(df['cases']))
print("Gender")
print(df['Gender'].corr(df['cases']))
print("Metro Atlanta")
print(df['Metro-Atlanta'].corr(df['cases']))
#plt.scatter(df['Income'], df['cases'], alpha=0.4)
#plt.show()
#bplot = sns.boxplot(y='cases', x='Income', data=df, width=0.5, palette="colorblind")
#bplot = sns.boxplot(y='cases', x='Age', data=df, width=0.5, palette="colorblind")

plt.show()
