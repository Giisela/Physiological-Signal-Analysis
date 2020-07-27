import numpy as np
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.constraints import maxnorm
from keras.utils import to_categorical
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, truncnorm, randint
from sklearn import metrics
from sklearn.metrics import confusion_matrix



def create_model(neurons=1,optimizer='adam',init_mode='uniform',activation='relu',dropout_rate=0.0, weight_constraint=0):
	model = Sequential()
	model.add(Dense(neurons, input_dim=6, kernel_initializer=init_mode,activation=activation, kernel_constraint=maxnorm(weight_constraint)))
	model.add(Dropout(dropout_rate))

	model.add(Dense(1, kernel_initializer=init_mode, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	return model


def label(data):
    final_y = to_categorical(data)
    return final_y

def nn_grid(X, y):

    model = KerasClassifier(build_fn=create_model, verbose=0)

    batch_size = [20, 60, 80]
    epochs = [10, 50, 100]
    optimizer = ['SGD', 'RMSprop', 'Adam', 'Nadam']
    init_mode = ['uniform']
    activation = ['relu', 'tanh', 'sigmoid', 'hard_sigmoid']
    weight_constraint = [1, 4]
    dropout_rate = [0.2, 0.4]
    neurons = [3, 6, 12, 24]
    #Best: 0.667010 using {'batch_size': 20, 'epochs': 10, 'init_mode': 'uniform', 'optimizer': 'Nadam', 'weight_constraint': 4, 'activation': 'relu', 'neurons': 125, 'dropout_rate': 0.2}

    param_grid = dict(batch_size=batch_size, epochs=epochs, optimizer=optimizer, init_mode=init_mode, 
    activation=activation, weight_constraint=weight_constraint, dropout_rate=dropout_rate,
    neurons=neurons)

    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, verbose=2)
    grid_result = grid.fit(X, np.asarray(y))

    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

def random_search(X_train, y_train, X_val, y_val, X_test, y_test):

    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    # Number of features to consider at every split
    max_features = ['auto','sqrt', 'log2', None]
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]

    class_weight = ['balanced', 'balanced_subsample', None]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                    'max_features': max_features,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf,
                    'class_weight': class_weight,
                    'bootstrap': bootstrap}

    # create random forest classifier model
    rf_model = RandomForestClassifier()

    # set up random search meta-estimator
    # this will train 100 models over 5 folds of cross validation (500 models total)
    clf = RandomizedSearchCV(rf_model, param_distributions=random_grid, n_iter=12, cv=3, n_jobs=8, random_state=42)

    # train the random search meta-estimator to find the best model out of 100 candidates
    model = clf.fit(X_train, y_train)

    predictions_val = model.predict(X_val)
    print("accuracy val Random:", metrics.accuracy_score(y_val, predictions_val))

    print(confusion_matrix(y_val,predictions_val))
    print(metrics.classification_report(y_val,predictions_val))

    predictions = model.predict(X_test)

    print("accuracy test Random:", metrics.accuracy_score(y_test, predictions))

    print(confusion_matrix(y_test,predictions))
    print(metrics.classification_report(y_test,predictions))

    from pprint import pprint
    pprint(model.best_estimator_.get_params())

    '''gr = GridSearchCV(rf_model, param_grid=random_grid, cv=5, n_jobs=-1)

    # train the random search meta-estimator to find the best model out of 100 candidates
    model_gr = gr.fit(X_train, y_train)
    
    predictions_gr_Val = model_gr.predict(X_val)

    print("accuracy val Grid:", metrics.accuracy_score(y_val, predictions_gr_Val))

    print(confusion_matrix(y_val,predictions_gr_Val))
    print(metrics.classification_report(y_val,predictions_gr_Val))
    
    
    predictions_gr = model_gr.predict(X_test)

    print("accuracy test Grid:", metrics.accuracy_score(y_test, predictions_gr))

    print(confusion_matrix(y_test,predictions_gr))
    print(metrics.classification_report(y_test,predictions_gr))

    from pprint import pprint
    pprint(model_gr.best_estimator_.get_params())'''



    