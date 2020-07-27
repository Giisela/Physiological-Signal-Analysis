import numpy as np
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.engine.saving import load_model
from keras.utils import to_categorical
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import ShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate


from sklearn import metrics
from WriteFiles import *
from performance import *
import os.path

def random_model(X_train, X_val, X_test, y_train, y_val, y_test, path,title, counter,i, n_estimators, max_depth):
    y_prediction = []
    y_prediction_value = []
    y_testing = []
    y_values = []

    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, n_jobs=8, verbose=2)
    #clf = KNeighborsClassifier(n_neighbors=1)
    
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    X_val = sc.transform(X_val)
    # Train the model using the training sets y_pred=clf.predict(X_test)
    clf.fit(X_train, y_train)

    # prediction on test set
    y_pred_val = clf.predict(X_val)
    y_pred = clf.predict(X_test)

    y_prediction.extend(y_pred)
    y_prediction_value.extend(y_pred_val)
    y_testing.extend(y_test)
    y_values.extend(y_val)


    ##### performance every shuffle iteration (flag=1)##########
    performance_global_everyIteration_shuffle(y_pred_val, y_pred, y_val, y_test, 'rf', path, title, counter, i)


    return y_prediction, y_prediction_value, y_testing, y_values



def rf_allMix(X, y, path,title,  n_estimators, max_depth):
    # Create a Gaussian Classifier
    acc_app = []
    precision_app = []
    recall_app = []
    f1_score_app = []
    mean_absolut = []

    y_testing = []

    y_prediction = []
    clf = RandomForestClassifier(n_estimators=n_estimators, max_features=len(X[0]), n_jobs=-1, max_depth=max_depth)

    rs = ShuffleSplit(n_splits=10, test_size=0.20, random_state=42)
    rs.get_n_splits(X)
    for train_index, test_index in rs.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        # Train the model using the training sets y_pred=clf.predict(X_test)
        clf.fit(X_train, y_train)
        # prediction on test set
        y_pred = clf.predict(X_test)
        y_prediction.extend(y_pred)
        y_testing.extend(y_test)


        accuracy = metrics.accuracy_score(y_test, y_pred)
        acc_app.append(accuracy)

        precision = metrics.precision_score(y_test, y_pred, average='micro')
        precision_app.append(precision)

        recall = metrics.recall_score(y_test, y_pred, average='micro')
        recall_app.append(recall)

        f1_score = metrics.f1_score(y_test, y_pred, average='micro')
        f1_score_app.append(f1_score)

        mean_absolut_error = metrics.mean_absolute_error(y_test, y_pred)
        mean_absolut.append(mean_absolut_error)

        performance_every_shuffler_allmix(y_pred, y_test, accuracy, precision, recall, f1_score, mean_absolut_error, path, title)

    ##################################################
    performance_global_shuffle_allmix(y_prediction, y_testing, acc_app, precision_app, recall_app, f1_score_app, mean_absolut, path, title)


def neural_network_keras_ecg(input_dim):
    model = Sequential()
    model.add(Dense(12, activation='relu', input_dim=input_dim))
    model.add(Dense(9, activation='relu'))
    model.add(Dense(6, activation='relu'))
    model.add(Dense(3, activation='sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(lr=0.009),
        metrics=['accuracy']
    )
    model.summary()
    return model

def neural_network_keras_eda(input_dim):
    model = Sequential()
    model.add(Dense(12, activation='relu', input_dim=input_dim))
    model.add(Dense(9, activation='relu'))
    model.add(Dense(6, activation='relu'))
    model.add(Dense(3, activation='sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(lr=0.009),
        metrics=['accuracy']
    )
    model.summary()
    return model

def neural_network_keras_emgz(input_dim):
    model = Sequential()
    model.add(Dense(12, activation='relu', input_dim=input_dim))
    model.add(Dense(9, activation='relu'))
    model.add(Dense(6, activation='relu'))
    model.add(Dense(3, activation='sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(lr=0.009),
        metrics=['accuracy']
    )
    model.summary()
    return model

def neural_network_keras_emgmf(input_dim):
    model = Sequential()
    model.add(Dense(8, activation='relu', input_dim=input_dim))
    model.add(Dense(9, activation='relu'))
    model.add(Dense(3, activation='sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(lr=0.009),
        metrics=['accuracy']
    )
    model.summary()
    return model

def neural_network_keras_emg(input_dim):
    model = Sequential()
    model.add(Dense(10, activation='relu', input_dim=input_dim))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(3, activation='sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(lr=0.009),
        metrics=['accuracy']
    )
    model.summary()
    return model

def neural_network_keras_all(input_dim):
    model = Sequential()
    model.add(Dense(28, activation='relu', input_dim=input_dim))
    model.add(Dense(3, activation='sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(lr=0.005),
        metrics=['accuracy']
    )
    model.summary()
    return model

def neural_network_keras_ecg_2(input_dim):
    model = Sequential()
    model.add(Dense(10, activation='relu', input_dim=input_dim))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(3, activation='sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(lr=0.003),
        metrics=['accuracy']
    )
    model.summary()
    return model

def load_trained_model(weights_path):
   model = load_model(weights_path)
   return model


def nn(X_train, X_val, X_test, y_train, y_val, y_test, input_dim, batch_size, path, title, counter, i, feature):
    
    y_prediction_value = []
    y_prediction = []
    y_testing = []
    y_values = []

    
    model_file = path+'best_model.h5'

    # convert class vectors to binary class matrices
    #sc = StandardScaler()
    sc = MinMaxScaler()

    X_train = sc.fit_transform(X_train)
    X_val = sc.transform(X_val)
    X_test = sc.transform(X_test)
    

    y_train = to_categorical(y_train)
    y_val = to_categorical(y_val)
    y_test = to_categorical(y_test)
    

    if os.path.isfile(model_file):
        model = load_trained_model(model_file)
    else:
        if "ecg" in feature:
            model = neural_network_keras_ecg(input_dim)
        elif "emgz" in feature or "eda" in feature:
            model = neural_network_keras_emgz(input_dim)
        elif "eda" in feature:
            model = neural_network_keras_eda(input_dim)
        elif 'emgmf' in feature:
            model = neural_network_keras_emgmf(input_dim)
        elif "emg" in feature:
            model = neural_network_keras_emg(input_dim)
        elif "ecg_2" in feature:
            model = neural_network_keras_ecg_2(input_dim)
        else:
            model = neural_network_keras_all(input_dim)

    # Set callback functions to early stop training and save the best model so far
    callbacks = [EarlyStopping(monitor='val_acc', patience=60, mode='max', verbose=1, restore_best_weights=True),
                    ModelCheckpoint(filepath=path + 'best_model.h5', monitor='val_acc', mode='max', verbose=1,
                                    save_best_only=True)]
    # Train neural network
    history = model.fit(X_train,  # Features
                        y_train,  # Target vector
                        epochs=1000,  # Number of epochs
                        callbacks=callbacks,  # Early stopping
                        verbose=2,  # Print description after each epoch
                        batch_size=batch_size,  # Number of observations per batch
                        validation_data=(X_val, y_val))  # Data for evaluation

    # evaluate the model
    score_train, train_acc = model.evaluate(X_train, y_train, verbose=2)
    score_val, val_acc = model.evaluate(X_val, y_val, verbose=2)
    score_test, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print('Train: %.3f, Validation: %.3f, Test: %.3f' % (train_acc,val_acc, test_acc))

    y_pred_val = model.predict_classes(X_val)
    y_val = [idx for values in y_val for idx, value in enumerate(values) if value == 1]
    print(y_pred_val)
    print(y_val)

    y_prediction_value.extend(y_pred_val)


    y_pred = model.predict_classes(X_test)
    y_test = [idx for values in y_test for idx, value in enumerate(values) if value == 1]
    #y_pred = np.argmax(y_pred, axis=-1)
    print(y_pred)
    #y_test = np.argmax(y_test, axis=-1)
    print(y_test)

    y_prediction.extend(y_pred)
    y_testing.extend(y_test)
    y_values.extend(y_val)

    '''# plot loss during training
    plt.subplot(211)
    plt.title('Loss')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    # plot accuracy during training
    plt.subplot(212)
    plt.title('Accuracy')
    plt.plot(history.history['acc'], label='train')
    plt.plot(history.history['val_acc'], label='test')
    plt.legend()
    plt.savefig(path+"loss_accuracy_"+counter+"_shuffle"+i+".png")'''

    
    ##### performance every shuffle iteration (flag=1)##########
    performance_global_everyIteration_shuffle(y_pred_val, y_pred, y_val, y_test, 'nn',  path ,title, counter, i)

    return y_prediction, y_prediction_value,y_testing, y_values




def nn_allMix(X, y, input_dim, batch_size, path, title, feature):
    acc_app = []
    precision_app = []
    recall_app = []
    f1_score_app = []
    mean_absolut = []

    y_testing = []

    y_prediction = []

    rs = ShuffleSplit(n_splits=10, test_size=0.20, random_state=42)
    rs.get_n_splits(X)
    for train_index, test_index in rs.split(X):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model_file = path+'best_model.h5'


        # convert class vectors to binary class matrices
        # sc = StandardScaler()
        sc = MinMaxScaler()

        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

        if os.path.isfile(model_file):
            model = load_trained_model(model_file)
        else:
            if "ecg" in feature:
                model = neural_network_keras_ecg(input_dim)
            elif "emgz" in feature or "emgmf" in feature or "eda" in feature:
                model = neural_network_keras_emgz(input_dim)
            elif "emg" in feature:
                model = neural_network_keras_emg(input_dim)
            elif "ecg_2" in feature:
                model = neural_network_keras_ecg_2(input_dim)
            else:
                model = neural_network_keras_all(input_dim)

        # Set callback functions to early stop training and save the best model so far
        callbacks = [EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1, restore_best_weights=True),
                     ModelCheckpoint(filepath=path+'best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)]

        # Train neural network
        history = model.fit(X_train,  # Features
                            y_train,  # Target vector
                            epochs=60,  # Number of epochs
                            callbacks=callbacks,  # Early stopping
                            verbose=2,  # Print description after each epoch
                            batch_size=15,  # Number of observations per batch
                            validation_data=(X_val, y_val),
                            shuffle=True)  # Data for evaluation

        # evaluate the model
        score_train, train_acc = model.evaluate(X_train, y_train, verbose=2)
        score_test, test_acc = model.evaluate(X_test, y_test, verbose=2)
        print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
        print('Score Train: %.3f, Score Test: %.3f' % (score_train, score_test))

        y_pred = model.predict_classes(X_test)
        y_test = [idx for values in y_test for idx, value in enumerate(values) if value == 1]
        # y_pred = np.argmax(y_pred, axis=-1)
        print(y_pred)
        # y_test = np.argmax(y_test, axis=-1)
        print(y_test)


        '''# plot loss during training
        plt.subplot(211)
        plt.title('Loss')
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.legend()
        # plot accuracy during training
        plt.subplot(212)
        plt.title('Accuracy')
        plt.plot(history.history['acc'], label='train')
        plt.plot(history.history['val_acc'], label='test')
        plt.legend()
        plt.savefig(path + "loss_accuracy.png")'''

        y_prediction.extend(y_pred)
        y_testing.extend(y_test)

        accuracy = metrics.accuracy_score(y_test, y_pred)
        acc_app.append(accuracy)

        precision = metrics.precision_score(y_test, y_pred, average='micro')
        precision_app.append(precision)

        recall = metrics.recall_score(y_test, y_pred, average='micro')
        recall_app.append(recall)

        f1_score = metrics.f1_score(y_test, y_pred, average='micro')
        f1_score_app.append(f1_score)

        mean_absolut_error = metrics.mean_absolute_error(y_test, y_pred)
        mean_absolut.append(mean_absolut_error)

        performance_every_shuffler_allmix(y_pred, y_test, accuracy, precision, recall, f1_score, mean_absolut_error, path,
                                   title)

    ##################################################
    performance_global_shuffle_allmix(y_prediction, y_testing, acc_app, precision_app, recall_app, f1_score_app, mean_absolut,
                               path, title)



