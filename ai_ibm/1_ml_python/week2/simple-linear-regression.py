import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score

file = "FuelConsumption.csv"

df = pd.read_csv(file)
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]


regr = linear_model.LinearRegression()

msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

################################################################################################
train_x = np.asanyarray(train[['ENGINESIZE']])      # array([ [2], [2.4],  ])
train_y = np.asanyarray(train[['CO2EMISSIONS']])    # tarin['ENGINESIZE'] -> array([2, 2.4, , ])
regr.fit (train_x, train_y)
################################################################################################

# The coefficients
print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_hat = regr.predict(test_x)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_hat - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_hat - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_hat , test_y) )