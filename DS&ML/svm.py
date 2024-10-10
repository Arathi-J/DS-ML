import pandas as pd
from sklearn import datasets


train_df = pd.read_csv('https://raw.githubusercontent.com/akshayr89/MNSIST_Handwritten_Digit_Recognition-SVM/master/train.csv')
test_df = pd.read_csv('https://raw.githubusercontent.com/akshayr89/MNSIST_Handwritten_Digit_Recognition-SVM/master/test.csv')

print(train_df.info())
print(test_df.info())

print(train_df.iloc[:,:5],'\n',train_df.iloc[:,-5:])


