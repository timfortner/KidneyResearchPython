
# Importing the libraries
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import keras
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import StratifiedKFold
import os
import csv
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cuda0"
# Importing the dataset

dataset = pd.read_csv('FINAL.csv')
d = '\t'
f = open('FINAL.csv','r')

reader = csv.reader(f, delimiter=d)
ncol   = len(next(reader)) # Read first line and count columns
f.seek(0)
DY_GRFAIL = dataset.loc['DY_GRFAIL']
X = dataset.iloc[:, 2:ncol].values
y = dataset.iloc[:, DY_GRFAIL].values




#k-fold cross validation
def build_model():
        model = Sequential()
        model.add(Dense(units = 48, kernel_initializer = 'uniform', activation = 'relu', input_dim = 1021))
        model.add(Dense(units = 48, kernel_initializer = 'uniform', activation = 'relu'))
        model.add(Dense(units = 48, kernel_initializer = 'uniform', activation = 'relu'))
        model.add(Dense(units = 48, kernel_initializer = 'uniform', activation = 'relu'))
        model.add(Dense(units = 48, kernel_initializer = 'uniform', activation = 'relu'))
        model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'linear'))
        model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['mae'])
        
        # Plot
        #plt.scatter(y_test, y, c=['green','black'], alpha=0.5)
        #plt.xlabel('Actual')
        #plt.ylabel('Predicted')
        #plt.show()

        return model

model  = KerasRegressor(build_fn = build_model,batch_size = 250, epochs = 10)
#scores = cross_val_score(model, X_train, y_train, cv = skf)

skf = StratifiedKFold(n_splits=10)
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    # Feature Scaling
    sc          = StandardScaler()
    X_train     = sc.fit_transform(X_train)
    X_test      = sc.transform(X_test)
    history     = model.fit(X_train, y_train, epochs = 10, batch_size = 250)
    predictions = model.predict(X_train, y_train)
    #mean_absolute_error(y_train, predictions)

    plt.plot(y_train, predictions, color='black', linewidth=4)
