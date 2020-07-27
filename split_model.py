from sklearn.model_selection import LeavePOut
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit


def leaveMout(X, y):
    leaveout = LeavePOut(2)  # taking p=2
    leaveout.get_n_splits(X)  # Number of splits of X
    # Printing the Train & Test Indices of splits
    for train_index, test_index in leaveout.split(X):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    return X_train, X_test, y_train, y_test

def k_folders(X, y):
    kf = KFold(n_splits=5, random_state=None, shuffle=False)

    for train_index, test_index in kf.split(X):
        print("Train:", train_index, "Test:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    return X_train, X_test, y_train, y_test

def ShuffleData(X, y):
    rs = ShuffleSplit(n_splits=20, test_size=0.20, random_state=42)
    rs.get_n_splits(X)
    for train_index, test_index in rs.split(X):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    return X_train, X_test, y_train, y_test

def ShuffleData_ecg_2(X, y):
    rs = ShuffleSplit(n_splits=30, test_size=0.25, random_state=42)
    rs.get_n_splits(X)
    for train_index, test_index in rs.split(X):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    return X_train, X_test, y_train, y_test



