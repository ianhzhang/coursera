import pandas as pd
import numpy as np

# 1. read data
df = pd.read_csv('drug200.csv')

# 2. Determine x, y with right columns
X = df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']] .values  # ndarray
y = df['Drug'].values

# 3. Standardlize: convert 'F','M' to 0,1.   'LOW','
from sklearn import preprocessing
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
X[:,1] = le_sex.transform(X[:,1]) 

le_BP = preprocessing.LabelEncoder()
le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
X[:,2] = le_BP.transform(X[:,2])

le_Chol = preprocessing.LabelEncoder()
le_Chol.fit([ 'NORMAL', 'HIGH'])
X[:,3] = le_Chol.transform(X[:,3]) 


# 4. Split data into training and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=3)

# 5. Create Model and training
from sklearn.tree import DecisionTreeClassifier
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)

drugTree.fit(X_train, y_train)
y_test_hat = drugTree.predict(X_test)

# 6. Evaluate results:
from sklearn import metrics
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_test, y_test_hat))

