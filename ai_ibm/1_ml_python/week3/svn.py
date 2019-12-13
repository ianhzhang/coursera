import pandas as pd
import numpy as np

# 1. read data
df = pd.read_csv('cell_samples.csv')
df = df[pd.to_numeric(df['BareNuc'], errors='coerce').notnull()]

# 2. Determine x, y with right columns
X = df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']] .values  #.astype(float)
y = df['Class'].values.astype('int')

# 3. Standardlize X
#from sklearn import preprocessing
#X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))

# 4. Split data into training and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print(X_train)
print("---------------")
print(y_train)
# 5. Create Model and training
from sklearn import svm
clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train) 
yhat = clf.predict(X_test)

# 6. Evaluate results:
#from sklearn import metrics
from sklearn.metrics import jaccard_similarity_score
print(jaccard_similarity_score(y_test, yhat))
