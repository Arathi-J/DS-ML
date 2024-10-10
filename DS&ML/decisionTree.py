import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from matplotlib import pyplot as plt
df = pd.read_csv('https://raw.githubusercontent.com/Arathi-J/DS-ML/refs/heads/main/iris.csv');
print(df.head())
print(df.shape)

x=df.drop(['species'], axis=1)
y=df['species']
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2)

dtc=DecisionTreeClassifier(criterion='entropy',min_samples_split=10)
dtc.fit(x_train,y_train)
pred=dtc.predict(x_test)
print('Accuracy', accuracy_score(y_test, pred))
print('\nConfusion Matrix = \n',confusion_matrix(y_test,pred))

tree.plot_tree(dtc)
plt.show()
