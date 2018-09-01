import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# 1
data = pd.read_csv("C:/Python/ML/train.csv", index_col='PassengerId')

# 2
x_labels = ['Pclass', 'Fare', 'Age', 'Sex']
X = data.loc[:, x_labels]

# 3
X['Sex'] = X['Sex'].map(lambda sex: 1 if sex == 'male' else 0)

# 4
y = data['Survived']

# 5
X = X.dropna()
y = y[X.index.values]

# 6
clf = DecisionTreeClassifier(random_state=241)
clf.fit(np.array(X.values), np.array(y.values))

# 7
importances = pd.Series(clf.feature_importances_, index=x_labels)
model = np.array([importances.sort_values(ascending=False).head(2).index.values])
model.tofile("C:/Python/ML/decisionTree.txt", sep=" ")