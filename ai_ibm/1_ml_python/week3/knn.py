import pandas as pd
import numpy as np

# 1. read data
df = pd.read_csv('teleCust1000t.csv')

# 2. Determine x, y with right columns
X = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']] .values  #.astype(float)
y = df['custcat'].values

# 3. Standardlize X
from sklearn import preprocessing
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))

# 4. Split data into training and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)

# 5. Create Model and training
from sklearn.neighbors import KNeighborsClassifier
k = 9   
# We decide k
# We can loop through different k to determin which k has the highest accuracy
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)


# 6. Evaluate results:
from sklearn import metrics
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, neigh.predict(X_test)))

