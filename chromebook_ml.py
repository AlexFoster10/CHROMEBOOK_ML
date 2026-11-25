import pandas, sklearn

from sklearn.model_selection import train_test_split

df = pandas.read_csv('data.csv')

train, test = train_test_split(df, test_size=0.2)
print(train.head())
print(test.head())

