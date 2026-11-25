import pandas, sklearn

from sklearn.model_selection import train_test_split
from sklearn import tree, preprocessing
from sklearn.metrics import accuracy_score
df = pandas.read_csv('data.csv')

le = preprocessing.LabelEncoder()
df = df.apply(le.fit_transform)

X = df[['DAY', 'MODEL']]
y = df['ISSUE']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dtree = tree.DecisionTreeClassifier()
dtree = dtree.fit(X_train, y_train)

predictions = dtree.predict(X_train)


dtree_acc = accuracy_score(y_train, predictions)
print("Decision Tree Accuracy TRAIN: ", dtree_acc)

predictions = dtree.predict(X_test)


dtree_acc = accuracy_score(y_test, predictions)
print("Decision Tree Accuracy TEST: ", dtree_acc)
