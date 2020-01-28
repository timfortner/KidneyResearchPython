
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cuda0"
# Importing the dataset
dataset = pd.read_csv('sample.csv')
X = dataset.iloc[:, 2:1070].values
y = dataset.iloc[:, 479].values

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

#k-fold cross validation
import keras
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(output_dim = 48, init = 'glorot_uniform', activation = 'relu',input_dim = 1068))
    classifier.add(Dense(output_dim = 48, init = 'glorot_uniform', activation = 'relu', input_dim=48))
    classifier.add(Dense(output_dim = 48, init = 'glorot_uniform', activation = 'relu', input_dim=48))
    classifier.add(Dense(output_dim = 48, init = 'glorot_uniform', activation = 'relu', input_dim=48))
    classifier.add(Dense(output_dim = 1, init = 'glorot_uniform', activation = 'linear'))
    classifier.compile(optimizer = optimizer, loss = 'mean_squared_error', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size':[4,6,8,16,24,32,48], 
              'nb_epoch':[150,200,400,600],
              'optimizer':['adam','rmsprop', 'sgd', 'adamax']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 5)
grid_search = grid_search.fit(X,y)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
print(best_parameters)
print(best_accuracy)
