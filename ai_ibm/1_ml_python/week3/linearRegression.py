import pandas as pd
import numpy as np

# 1. read data
df = pd.read_csv("ChurnData.csv")
df['churn'] = df['churn'].astype('int')

# 2. Determine x, y with right columns
X = df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']] .values  #.astype(float)
y = df['churn'].values

# 3. Standardlize X
from sklearn import preprocessing
X = preprocessing.StandardScaler().fit(X).transform(X)

# 4. Split data into training and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)

# 5. Create Model and training
from sklearn.linear_model import LogisticRegression

LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)

yhat = LR.predict(X_test)

# 6. Evaluate results:
from sklearn.metrics import jaccard_similarity_score
print(jaccard_similarity_score(y_test, yhat))

