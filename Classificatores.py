import pickle
from sklearn.utils import shuffle
import uuid
from itertools import zip_longest
from random import sample, choices
from itertools import repeat
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_validate


#from GridSearchNN import *

import pandas as pd
#from split_model import *
from Models import *
#from GridSearchNN import *


def grouper(iterable, n, fillvalue=None):
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)

def read_3emotion_perPercentage(loadData, feature):
    df = pd.DataFrame(loadData)
    df1 = pd.DataFrame()
    dict_feat = pd.DataFrame([])
    dict_feat1 = pd.DataFrame([])
    test_dataset = []
    train_dataset = []
    X = np.array([])
    X_test = np.array([])
    X_trein_dataset = np.array([])


    for id in df.keys():
        df_test = pd.DataFrame(loadData[id])
        print(df_test)
    
        print(len(df_test))
    


    newl = list(set([i.split("_")[0] for i in df.keys()]))
    Number_of_test = int(0.22 * len(newl))
    print(Number_of_test)
    list_dict = sample(list(newl), k=Number_of_test)
    print(list_dict)

    df1 = df1.append(pd.DataFrame({list_dict[0]+"_N": df[list_dict[0]+"_N"], list_dict[1]+"_N": df[list_dict[1]+"_N"], list_dict[2]+"_N": df[list_dict[2]+"_N"], list_dict[3]+"_N":list_dict[3]+"_N"}))
    df1 = df1.append(pd.DataFrame({list_dict[4]+"_F": df[list_dict[4]+"_F"], list_dict[5]+"_F": df[list_dict[5]+"_F"],list_dict[6]+"_F": df[list_dict[6]+"_F"], list_dict[7]+"_F": df[list_dict[7]+"_F"]}))
    df1 = df1.append(pd.DataFrame({list_dict[8]+"_H": df[list_dict[8]+"_H"], list_dict[9]+"_H": df[list_dict[9]+"_H"],list_dict[10]+"_H": df[list_dict[10]+"_H"],list_dict[11]+"_H": df[list_dict[11]+"_H"]}))

    
    del df[list_dict[0]+"_N"]
    del df[list_dict[1]+"_N"]
    del df[list_dict[2]+"_N"]
    del df[list_dict[3]+"_N"]
    del df[list_dict[4]+"_F"]
    del df[list_dict[5]+"_F"]
    del df[list_dict[6]+"_F"]
    del df[list_dict[7]+"_F"]
    del df[list_dict[8]+"_H"]
    del df[list_dict[9]+"_H"]
    del df[list_dict[10]+"_H"]
    del df[list_dict[11]+"_H"]


    for k in df1.keys():
        test_dataset.append(k)
        df_feature1 = pd.DataFrame(loadData[k], columns=feature)
        df_feature1.fillna(np.nanmean(df_feature1), inplace=True)

        emotion = k.split("_")[1]

        if "N" in emotion:
            df_feature1["Class"] = 0
        elif "F" in emotion:
            df_feature1["Class"] = 1
        else:
            df_feature1["Class"] = 2

        dict_feat1 = dict_feat1.append(df_feature1, ignore_index=False)
        dict_feat1.fillna(np.nanmean(dict_feat1), inplace=True)

        print(dict_feat1)


    
    
    X_test = np.array(dict_feat1, dtype=float)
    print("Test:",len(X_test))
    idx_N = np.nonzero(X_test[:,-1] == 0)
    idx_F = np.nonzero(X_test[:,-1] == 1)
    idx_H = np.nonzero(X_test[:,-1] == 2)


    Number_of_train_N = int(0.30 * len(idx_N[0]))
    print(len(idx_N[0]))
    print(Number_of_train_N)
    list_dict_N = sample(list(idx_N[0]), k=Number_of_train_N)

    Number_of_train_F = int(0.30 * len(idx_F[0]))
    print(len(idx_F[0]))
    print(Number_of_train_F)
    list_dict_F = sample(list(idx_F[0]), k=Number_of_train_F)

    Number_of_train_H = int(0.30 * len(idx_H[0]))
    print(len(idx_H[0]))
    print(Number_of_train_H)
    list_dict_H = sample(list(idx_H[0]), k=Number_of_train_H)
 
    
    a = np.array(X_test[list_dict_N,:])
    b = np.array(X_test[list_dict_F,:])
    c = np.array(X_test[list_dict_H,:])
    X_trein_dataset = np.concatenate((a, b, c), axis=0)
    print("X_trein_dataset size:", len(X_trein_dataset))

    list_dict_eliminate = []
    list_dict_eliminate.append(list_dict_N + list_dict_F +list_dict_H)

    list_dict_eliminate[0].sort(reverse=True)
    for val in list_dict_eliminate[0]:
        X_test = np.delete(X_test,val,0)
            
    print("X_test size:", len(X_test))
    
    for k in df.keys():
        train_dataset.append(k)
        df_feature = pd.DataFrame(loadData[k], columns=feature)
        df_feature.fillna(np.nanmean(df_feature), inplace=True)


        emotion = k.split("_")[1]

        if "N" in emotion:
            df_feature["Class"] = 0
        elif "F" in emotion:
            df_feature["Class"] = 1
        else:
            df_feature["Class"] = 2
        
        
        dict_feat = dict_feat.append(df_feature, ignore_index=True)
        dict_feat.fillna(np.nanmean(dict_feat), inplace=True)

    
    print("Trein:", dict_feat)
    
    
    X_trein = np.array(dict_feat, dtype=float)
    X = np.concatenate((X_trein_dataset,X_trein), axis=0)
    print("Trein: ", len(X))

    return X, X_test, train_dataset, test_dataset

def read_3emotion_per_participante(loadData, feature):
    df = pd.DataFrame(loadData)
    df1 = pd.DataFrame()
    dict_feat = pd.DataFrame([])
    dict_feat1 = pd.DataFrame([])
    test_dataset = []
    train_dataset = []
    X = np.array([])
    X_test = np.array([])

    newl = list(set([i.split("_")[0] for i in df.keys()]))
    Number_of_test = int(0.22 * len(newl))
    list_dict = sample(list(newl), k=Number_of_test)

    df1 = df1.append(pd.DataFrame({list_dict[0]+"_N": df[list_dict[0]+"_N"], list_dict[1]+"_N": df[list_dict[1]+"_N"], list_dict[2]+"_N": df[list_dict[2]+"_N"], list_dict[3]+"_N":list_dict[3]+"_N"}))
    df1 = df1.append(pd.DataFrame({list_dict[4]+"_F": df[list_dict[4]+"_F"], list_dict[5]+"_F": df[list_dict[5]+"_F"],list_dict[6]+"_F": df[list_dict[6]+"_F"], list_dict[7]+"_F": df[list_dict[7]+"_F"]}))
    df1 = df1.append(pd.DataFrame({list_dict[8]+"_H": df[list_dict[8]+"_H"], list_dict[9]+"_H": df[list_dict[9]+"_H"],list_dict[10]+"_H": df[list_dict[10]+"_H"],list_dict[11]+"_H": df[list_dict[11]+"_H"]}))

    
    del df[list_dict[0]+"_N"]
    del df[list_dict[1]+"_N"]
    del df[list_dict[2]+"_N"]
    del df[list_dict[3]+"_N"]
    del df[list_dict[4]+"_F"]
    del df[list_dict[5]+"_F"]
    del df[list_dict[6]+"_F"]
    del df[list_dict[7]+"_F"]
    del df[list_dict[8]+"_H"]
    del df[list_dict[9]+"_H"]
    del df[list_dict[10]+"_H"]
    del df[list_dict[11]+"_H"]


    for k in df1.keys():
        test_dataset.append(k)
        df_feature1 = pd.DataFrame(loadData[k], columns=feature)
        df_feature1.fillna(np.nanmean(df_feature1), inplace=True)
        emotion = k.split("_")[1]

        if "N" in emotion:
            df_feature1["Class"] = 0
        elif "F" in emotion:
            df_feature1["Class"] = 1
        else:
            df_feature1["Class"] = 2

        dict_feat1 = dict_feat1.append(df_feature1, ignore_index=True)
        dict_feat1.fillna(np.nanmean(dict_feat1), inplace=True)

    print("Test:", dict_feat)

    print(len(X_test))
    X_test = np.array(dict_feat1, dtype=float)
    


    for k in df.keys():
        train_dataset.append(k)
        df_feature = pd.DataFrame(loadData[k], columns=feature)
        df_feature.fillna(np.nanmean(df_feature), inplace=True)
        emotion = k.split("_")[1]

        if "N" in emotion:
            df_feature["Class"] = 0
        elif "F" in emotion:
            df_feature["Class"] = 1
        else:
            df_feature["Class"] = 2


        dict_feat = dict_feat.append(df_feature, ignore_index=True)
        dict_feat.fillna(np.nanmean(dict_feat), inplace=True)

    print("Trein:", dict_feat)
    X = np.array(dict_feat, dtype=float)
    print(len(X))
    return X, X_test, train_dataset, test_dataset


def read_per_participante(loadData, feature):
    df = pd.DataFrame(loadData)
    df1 = pd.DataFrame()
    dict_feat = pd.DataFrame([])
    dict_feat1 = pd.DataFrame([])
    test_dataset=[]
    train_dataset=[]
    X = np.array([])
    X_test = np.array([])

    newl = list(set([i.split("_")[0] for i in df.keys()]))
    Number_of_test = int(0.22 * len(newl))
    list_dict = sample(list(newl), k=Number_of_test)

    for key in list_dict:
        df1 = df1.append(pd.DataFrame({key+"_N": df[key+"_N"], key+"_F": df[key+"_F"], key+"_H": df[key+"_H"]}))
        del df[key + "_N"]
        del df[key + "_F"]
        del df[key + "_H"]


    for k in df1.keys():
        test_dataset.append(k)
        df_feature1 = pd.DataFrame(loadData[k], columns=feature)
        df_feature1.fillna(np.nanmean(df_feature1), inplace=True)

        emotion = k.split("_")[1]

        if "N" in emotion:
            df_feature1["Class"] = 0
        elif "F" in emotion:
            df_feature1["Class"] = 1
        else:
            df_feature1["Class"] = 2

        
        dict_feat1 = dict_feat1.append(df_feature1, ignore_index=True)
        dict_feat1.fillna(np.nanmean(dict_feat1), inplace=True)
        #print(dict_feat1)

    X_test = np.array(dict_feat1)

    for k in df.keys():
        train_dataset.append(k)
        df_feature = pd.DataFrame(loadData[k], columns=feature)
        df_feature.fillna(np.nanmean(df_feature), inplace=True)

        emotion = k.split("_")[1]

        if "N" in emotion:
            df_feature["Class"] = 0
        elif "F" in emotion:
            df_feature["Class"] = 1
        else:
            df_feature["Class"] = 2


        dict_feat = dict_feat.append(df_feature, ignore_index=True)
        dict_feat1.fillna(np.nanmean(dict_feat1), inplace=True)
    X = np.array(dict_feat, dtype=float)


    return X, X_test, train_dataset, test_dataset


def read_data(loadData, feature):
    df = pd.DataFrame(loadData)
    dict_feat = pd.DataFrame([])
    X = np.array([])
    participante = []
    y = []

    for k in df.keys():
        participante.append(k)
        df_feature = pd.DataFrame(loadData[k], columns=feature)
        df_feature.fillna(df_feature.mean(), inplace=True)
        emotion = k.split("_")[1]

        if "N" in emotion:
            df_feature["Class"] = 0
        elif "F" in emotion:
            df_feature["Class"] = 1
        else:
            df_feature["Class"] = 2


        dict_feat = dict_feat.append(df_feature, ignore_index=True)
        dict_feat.fillna(dict_feat.mean(), inplace=True)
        X = np.array(dict_feat)

        #keys = np.array(df_feature.keys())
        #key.append(keys)


    return X, participante


def classifier_ecg(flag, save_path):
    y_values = []
    y_testing = []

    y_pred_global_nn = []
    y_pred_global_rf = []
    y_test_global = []
    y_test_global_nn = []

    ECG_df = ['Heart_Rate', 'ECG_RR_Interval', 'ECG_HRV_HF', 'ECG_HRV_LF', 'ECG_HRV_ULF', 'T_Waves']
    loadData = pickle.load(open("Data_inc/split_person/All_30s.pkl", "rb"))

    '''##########with all data in test and train dataset############
    X_read, key = read_data(loadData, ECG_df)

    X = X_read[:, 0:6]
    y = np.array(X_read[:, 6], dtype=int)

    rf_allMix(X, y, "Classification/ECG_6/Random_Forest/", "ECG Random Forest", 500, 200)
    nn_allMix(X, y, 6, 400, "Classification/ECG_6/Neural_Network/", "ECG Neural Network", "ecg")'''

    path_rf = save_path+"Random_Forest/"
    path_nn = save_path+"Neural_Network/"

    for counter in range(10):

        if flag==0:
            ##########9 participants out for testing#####################
            X_read, X_read_test, train_dataset, test_dataset = read_per_participante(loadData, ECG_df)
        elif flag==1:
            ######### 3 emotion each, 3H, 3N, 3F ########################
            X_read, X_read_test, train_dataset, test_dataset = read_3emotion_per_participante(loadData, ECG_df)
        else:
            ######## 3 emotion each, 3H, 3N, 3F without 30% of information in test ########################
            X_read, X_read_test, train_dataset, test_dataset = read_3emotion_perPercentage(loadData, ECG_df)
        
        #print(X_read)
        #print(X_read_test)

        X = X_read[:, 0:6]
        y = np.array(X_read[:,6], dtype=int)
        
        X_test = X_read_test[:, 0:6]
        y_test = np.array(X_read_test[:,6], dtype=int)
        
                
        file = open(save_path+"ECG.txt", "a+")
        file.write("Counter: %s \t\t\t Train Dataset: %s \t\t\t Test Dataset: %s\n"%(counter, train_dataset, test_dataset))
        file.close()
        

        #X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42, shuffle=True)

        #random_search(X_train, y_train, X_val, y_val, X_test, y_test)
        
        
        rs = ShuffleSplit(n_splits=10, test_size=0.20, random_state=42)
        rs.get_n_splits(X)
        i=0
        for train_index, test_index in rs.split(X):
            X_train, X_val = X[train_index], X[test_index]
            y_train, y_val = y[train_index], y[test_index]
        

            y_pred_rf,y_pred_val_rf, y_testing_rf, y_values_rf = random_model(X_train, X_val, X_test, y_train, y_val, y_test, path_rf, "ECG Random Forest_"+str(counter), str(counter), str(i), 8000, None)            
            y_pred_nn, y_pred_val_nn, y_testing_nn, y_values_nn = nn(X_train, X_val, X_test, y_train, y_val, y_test, 6,200, path_nn, "ECG Neural Network_"+str(counter), str(counter), str(i), "ecg")
            #nn_grid(X,y)
            i=i+1
        
        ##### performance global shuffle (flag=0)##########
        performe_global_shuffle(y_pred_rf,y_testing_rf, y_pred_val_rf, y_values_rf,'rf', path_rf,"ECG Random Forest_"+str(counter), counter)
        ##### performance global shuffle (flag=0)##########
        performe_global_shuffle(y_pred_nn, y_testing_nn,y_pred_val_nn, y_values_nn, 'nn', path_nn,"ECG Random Forest_"+str(counter), counter)

        y_pred_global_rf.extend(y_pred_rf)
        y_pred_global_nn.extend(y_pred_nn)
        y_test_global.extend(y_testing_rf)
        y_test_global_nn.extend(y_testing_nn)

    ##### performance global iteration (flag=0)##########
    performance_global_finaly(y_pred_global_rf, y_test_global,'rf', path_rf, "ECG Random Forest")

    ##### performance global iteration (flag=0)##########
    performance_global_finaly(y_pred_global_nn, y_test_global_nn,'nn', path_nn, "ECG Neural Network")

    clean_lists()
    

def classifier_eda(flag, save_path):
    y_values = []
    y_testing = []

    y_pred_global_nn = []
    y_pred_global_rf = []
    y_test_global = []
    y_test_global_nn = []

    Eda_df = ['EDA_Tonic','SCR_Peaks_Indexes']
    loadData = pickle.load(open("Data_inc/split_person/All_30s.pkl", "rb"))

    '''##########with all data in test and train dataset############
    X_read, key = read_data(loadData, Eda_df)

    X = X_read[:, 0:2]
    y = np.array(X_read[:, 2], dtype=int)

    #rf_allMix(X, y, "Classification/EDA/Random_Forest/", "EDA Random Forest",200, 50)
    nn_allMix(X, y, 2, 400, "Classification/EDA/Neural_Network/", "EDA Neural Network", "eda")
    '''

    path_rf = save_path+"Random_Forest/"
    path_nn = save_path+"Neural_Network/"
    
    for counter in range(10):
        if flag==0:
            ##########9 participants out for testing#####################
            X_read, X_read_test, train_dataset, test_dataset = read_per_participante(loadData, Eda_df)
        elif flag==1:
            ######### 3 emotion each, 3H, 3N, 3F ########################
            X_read, X_read_test, train_dataset, test_dataset = read_3emotion_per_participante(loadData, Eda_df)
        else:
            ######## 3 emotion each, 3H, 3N, 3F without 30% of information in test ########################
            X_read, X_read_test, train_dataset, test_dataset = read_3emotion_perPercentage(loadData, Eda_df)
        
        #print(X_read)
        #print(X_read_test)
        
        X = X_read[:, 0:2]
        y = np.array(X_read[:, 2], dtype=int)

        X_test = X_read_test[:, 0:2]
        y_test = np.array(X_read_test[:, 2], dtype=int)
        
        
        file = open(save_path+"EDA.txt", "a+")
        file.write("Counter: %s \t\t\t Train Dataset: %s \t\t\t Test Dataset: %s\n"%(counter, train_dataset, test_dataset))
        file.close()

        rs = ShuffleSplit(n_splits=10, test_size=0.20, random_state=42)
        rs.get_n_splits(X)
        i=0
        for train_index, test_index in rs.split(X):
            X_train, X_val = X[train_index], X[test_index]
            y_train, y_val = y[train_index], y[test_index]


            y_pred_rf,y_pred_val_rf, y_testing_rf, y_values_rf = random_model(X_train, X_val, X_test, y_train, y_val, y_test, path_rf, "EDA Random Forest_"+str(counter), str(counter),str(i), 8000, None)
            
            y_pred_nn, y_pred_val_nn, y_testing_nn, y_values_nn = nn(X_train, X_val, X_test, y_train, y_val, y_test, 2,200, path_nn, "EDA Neural Network_"+str(counter), str(counter), str(i), "eda")
            #nn_grid(X,y)
            i=i+1
        
        ##### performance global shuffle (flag=0)##########
        performe_global_shuffle(y_pred_rf,y_testing_rf, y_pred_val_rf, y_values_rf,'rf', path_rf,"EDA Random Forest_"+str(counter), counter)
        ##### performance global shuffle (flag=0)##########
        performe_global_shuffle(y_pred_nn, y_testing_nn,y_pred_val_nn, y_values_nn, 'nn', path_nn,"EDA Random Forest_"+str(counter), counter)

        y_pred_global_rf.extend(y_pred_rf)
        y_pred_global_nn.extend(y_pred_nn)
        y_test_global.extend(y_testing_rf)
        y_test_global_nn.extend(y_testing_nn)

    ##### performance global iteration (flag=0)##########
    performance_global_finaly(y_pred_global_rf, y_test_global,'rf', path_rf, "EDA Random Forest")

    ##### performance global iteration (flag=0)##########
    performance_global_finaly(y_pred_global_nn, y_test_global_nn,'nn', path_nn, "EDA Neural Network")

    clean_lists()

    
def classifier_emgz(flag, save_path):
    y_values = []
    y_testing = []

    y_pred_global_nn = []
    y_pred_global_rf = []
    y_test_global = []
    y_test_global_nn = []

    EMG_df = ['EMG_Envelope_Z', 'EMG_Pulse_Onsets_Z']
    loadData = pickle.load(open("Data_inc/split_person/All_30s.pkl", "rb"))

    
    path_rf = save_path+"Random_Forest/"
    path_nn = save_path+"Neural_Network/"
    for counter in range(10):
        if flag==0:
            ##########9 participants out for testing#####################
            X_read, X_read_test, train_dataset, test_dataset = read_per_participante(loadData, EMG_df)
        elif flag==1:
            ######### 3 emotion each, 3H, 3N, 3F ########################
            X_read, X_read_test, train_dataset, test_dataset = read_3emotion_per_participante(loadData, EMG_df)
        else:
            ######## 3 emotion each, 3H, 3N, 3F without 30% of information in test ########################
            X_read, X_read_test, train_dataset, test_dataset = read_3emotion_perPercentage(loadData, EMG_df)
        
        #print(X_read)
        #print(X_read_test)
        
        X = X_read[:, 0:2]
        y = np.array(X_read[:, 2], dtype=int)

        X_test = X_read_test[:, 0:2]
        y_test = np.array(X_read_test[:, 2], dtype=int)
        
        
        file = open(save_path+"EMGZ.txt", "a+")
        file.write("Counter: %s \t\t\t Train Dataset: %s \t\t\t Test Dataset: %s\n"%(counter, train_dataset, test_dataset))
        file.close()

        rs = ShuffleSplit(n_splits=10, test_size=0.20, random_state=42)
        rs.get_n_splits(X)
        i=0
        for train_index, test_index in rs.split(X):
            X_train, X_val = X[train_index], X[test_index]
            y_train, y_val = y[train_index], y[test_index]


            y_pred_rf,y_pred_val_rf, y_testing_rf, y_values_rf = random_model(X_train, X_val, X_test, y_train, y_val, y_test, path_rf, "EMGZ Random Forest_"+str(counter), str(counter), str(i), 8000, None)
            
            y_pred_nn, y_pred_val_nn, y_testing_nn, y_values_nn = nn(X_train, X_val, X_test, y_train, y_val, y_test, 2,200, path_nn, "EMGZ Neural Network_"+str(counter), str(counter),str(i), "emgz")
            #nn_grid(X,y)
            i=i+1
        
        ##### performance global shuffle (flag=0)##########
        performe_global_shuffle(y_pred_rf,y_testing_rf, y_pred_val_rf, y_values_rf,'rf', path_rf,"EMGZ Random Forest_"+str(counter), counter)
        ##### performance global shuffle (flag=0)##########
        performe_global_shuffle(y_pred_nn, y_testing_nn,y_pred_val_nn, y_values_nn, 'nn', path_nn,"EMGZ Random Forest_"+str(counter), counter)

        y_pred_global_rf.extend(y_pred_rf)
        y_pred_global_nn.extend(y_pred_nn)
        y_test_global.extend(y_testing_rf)
        y_test_global_nn.extend(y_testing_nn)

    ##### performance global iteration (flag=0)##########
    performance_global_finaly(y_pred_global_rf, y_test_global,'rf', path_rf, "EMGZ Random Forest")

    ##### performance global iteration (flag=0)##########
    performance_global_finaly(y_pred_global_nn, y_test_global_nn,'nn', path_nn, "EMGZ Neural Network")

    clean_lists()
       
  
def classifier_emgmf(flag, save_path):
    y_values = []
    y_testing = []

    y_pred_global_nn = []
    y_pred_global_rf = []
    y_test_global = []
    y_test_global_nn = []

    EMG_df_MF = ['EMG_Envelope_MF', 'EMG_Pulse_Onsets_MF']
    loadData = pickle.load(open("Data_inc/split_person/All_30s.pkl", "rb"))

    '''##########with all data in test and train dataset############
    X_read, key = read_data(loadData, EMG_df_MF)
    X = X_read[:, 0:2]
    y = np.array(X_read[:, 2], dtype=int)

    rf_allMix(X, y, "Classification/EMGMF/Random_Forest/", "EMGMF Random Forest", 200, 50)
    nn_allMix(X, y, 2, 400, "Classification/EMGMF/Neural_Network/", "EMGMF Neural Network", "eda")'''

    path_rf = save_path+"Random_Forest/"
    path_nn = save_path+"Neural_Network/"
    for counter in range(10):
        if flag==0:
            ##########9 participants out for testing#####################
            X_read, X_read_test, train_dataset, test_dataset = read_per_participante(loadData, EMG_df_MF)
        elif flag==1:
            ######### 3 emotion each, 3H, 3N, 3F ########################
            X_read, X_read_test, train_dataset, test_dataset = read_3emotion_per_participante(loadData, EMG_df_MF)
        else:
            ######## 3 emotion each, 3H, 3N, 3F without 30% of information in test ########################
            X_read, X_read_test, train_dataset, test_dataset = read_3emotion_perPercentage(loadData, EMG_df_MF)
        
        #print(X_read)
        #print(X_read_test)
        
        X = X_read[:, 0:2]
        y = np.array(X_read[:, 2], dtype=int)

        X_test = X_read_test[:, 0:2]
        y_test = np.array(X_read_test[:, 2], dtype=int)
        
        
        file = open(save_path+"EMGMF.txt", "a+")
        file.write("Counter: %s \t\t\t Train Dataset: %s \t\t\t Test Dataset: %s\n"%(counter, train_dataset, test_dataset))
        file.close()

        rs = ShuffleSplit(n_splits=10, test_size=0.20, random_state=42)
        rs.get_n_splits(X)
        i=0
        for train_index, test_index in rs.split(X):
            X_train, X_val = X[train_index], X[test_index]
            y_train, y_val = y[train_index], y[test_index]


            y_pred_rf,y_pred_val_rf, y_testing_rf, y_values_rf = random_model(X_train, X_val, X_test, y_train, y_val, y_test, path_rf, "EMGMF Random Forest_"+str(counter), str(counter), str(i), 8000, None)
            
            y_pred_nn, y_pred_val_nn, y_testing_nn, y_values_nn = nn(X_train, X_val, X_test, y_train, y_val, y_test, 2,200, path_nn, "EMGMF Neural Network_"+str(counter), str(counter), str(i), "emgmf")
            #nn_grid(X,y)
            i=i+1
        
        ##### performance global shuffle (flag=0)##########
        performe_global_shuffle(y_pred_rf,y_testing_rf, y_pred_val_rf, y_values_rf,'rf', path_rf,"EMGMF Random Forest_"+str(counter), counter)
        ##### performance global shuffle (flag=0)##########
        performe_global_shuffle(y_pred_nn, y_testing_nn,y_pred_val_nn, y_values_nn, 'nn', path_nn,"EMGMF Random Forest_"+str(counter), counter)

        y_pred_global_rf.extend(y_pred_rf)
        y_pred_global_nn.extend(y_pred_nn)
        y_test_global.extend(y_testing_rf)
        y_test_global_nn.extend(y_testing_nn)

    ##### performance global iteration (flag=0)##########
    performance_global_finaly(y_pred_global_rf, y_test_global,'rf', path_rf, "EMGMF Random Forest")

    ##### performance global iteration (flag=0)##########
    performance_global_finaly(y_pred_global_nn, y_test_global_nn,'nn', path_nn, "EMGMF Neural Network")

    clean_lists()
    


def classifier_all(flag, save_path):
    y_values = []
    y_testing = []

    y_pred_global_nn = []
    y_pred_global_rf = []
    y_test_global = []
    y_test_global_nn = []


    All_df = ['Heart_Rate', 'ECG_RR_Interval', 'ECG_HRV_HF', 'ECG_HRV_LF', 'ECG_HRV_ULF', 'T_Waves', 'EMG_Activation_Z', 'EMG_Pulse_Onsets_Z','EMG_Activation_MF', 'EMG_Pulse_Onsets_MF', 'EDA_Tonic', 'SCR_Peaks_Indexes']
    #All_df = ['Heart_Rate', 'ECG_HRV_LF', 'ECG_HRV_HF', 'ECG_HRV_VLF', 'ECG_HRV_ULF','P_Waves', 'EMG_Envelope_Z', 'EMG_Pulse_Onsets_Z',
              #'EMG_Envelope_MF', 'EMG_Pulse_Onsets_MF', 'EDA_Tonic', 'SCR_Onsets']
    loadData = pickle.load(open("Data_inc/split_person/All_30s.pkl", "rb"))
    
    '''##########with all data in test and train dataset############
    X_read, key = read_data(loadData, All_df)
    X = X_read[:, 0:12]
    y = np.array(X_read[:, 12], dtype=int)

    rf_allMix(X, y, "Classification/All/Random_Forest/", "All Random Forest", 100, 50)
    nn_allMix(X, y, 12, 800, "Classification/All/Neural_Network/", "All Neural Network", "eda")'''
    

    path_rf = save_path+"Random_Forest/"
    path_nn = save_path+"Neural_Network/"
    
    for counter in range(10):
        if flag==0:
            ##########9 participants out for testing#####################
            X_read, X_read_test, train_dataset, test_dataset = read_per_participante(loadData, All_df)
        elif flag == 1:
            ######### 3 emotion each, 3H, 3N, 3F ########################
            X_read, X_read_test, train_dataset, test_dataset = read_3emotion_per_participante(loadData, All_df)
        else:
            ######## 3 emotion each, 3H, 3N, 3F without 30% of information in test ########################
            X_read, X_read_test, train_dataset, test_dataset = read_3emotion_perPercentage(loadData, All_df)
        
        print("Trein:",X_read)

        print("Test:",X_read_test)
        #print(X_read)
        #print(X_read_test)
        
        X = X_read[:, 0:12]
        y = np.array(X_read[:, 12], dtype=int)

        X_test = X_read_test[:, 0:12]
        y_test = np.array(X_read_test[:, 12], dtype=int)
        
        #X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42, shuffle=True)

        #random_search(X_train, y_train, X_val, y_val, X_test, y_test)    

        file = open(save_path+"All.txt", "a+")
        file.write("Counter: %s \t\t\t Train Dataset: %s \t\t\t Test Dataset: %s\n"%(counter, train_dataset, test_dataset))
        file.close()


        rs = ShuffleSplit(n_splits=10, test_size=0.20, random_state=42)
        rs.get_n_splits(X)
        i=0
        for train_index, test_index in rs.split(X):
            X_train, X_val = X[train_index], X[test_index]
            y_train, y_val = y[train_index], y[test_index]
            
            y_pred_rf,y_pred_val_rf, y_testing_rf, y_values_rf = random_model(X_train, X_val, X_test, y_train, y_val, y_test, path_rf, "All Random Forest_"+str(counter), str(counter), str(i), 10000, None) 
            y_pred_nn, y_pred_val_nn, y_testing_nn, y_values_nn = nn(X_train, X_val, X_test, y_train, y_val, y_test, 12, 200, path_nn, "All Neural Network_"+str(counter), str(counter), str(i), "all")
            i=i+1

        ##### performance global shuffle (flag=0)##########
        performe_global_shuffle(y_pred_rf,y_testing_rf, y_pred_val_rf, y_values_rf,'rf', path_rf,"All Random Forest_"+str(counter), counter)
        ##### performance global shuffle (flag=0)##########
        performe_global_shuffle(y_pred_nn, y_testing_nn,y_pred_val_nn, y_values_nn, 'nn', path_nn,"All Random Forest_"+str(counter), counter)

        y_pred_global_rf.extend(y_pred_rf)
        y_pred_global_nn.extend(y_pred_nn)
        y_test_global.extend(y_testing_rf)
        y_test_global_nn.extend(y_testing_nn)

    ##### performance global iteration (flag=0)##########
    performance_global_finaly(y_pred_global_rf, y_test_global,'rf', path_rf, "All Random Forest")

    ##### performance global iteration (flag=0)##########
    performance_global_finaly(y_pred_global_nn, y_test_global_nn,'nn', path_nn, "All Neural Network")

    clean_lists()




def classifier_emg(flag, save_path):
    y_values = []
    y_testing = []

    y_pred_global_nn = []
    y_pred_global_rf = []
    y_test_global = []
    y_test_global_nn = []

    EMG_df = ['EMG_Activation_Z','EMG_Envelope_Z', 'EMG_Pulse_Onsets_Z','EMG_Activation_MF','EMG_Envelope_MF', 'EMG_Pulse_Onsets_MF']
    loadData = pickle.load(open("Data_inc/split_person/All_30s.pkl", "rb"))

    '''##########with all data in test and train dataset############
    X_read, key = read_data(loadData, EMG_df)
    X = X_read[:, 0:4]
    y = np.array(X_read[:, 4], dtype=int)

    #rf_allMix(X, y, "Classification/EMG/Random_Forest/", "EMG Random Forest", 100, 50)
    nn_allMix(X, y, 4, 600, "Classification/EMG/Neural_Network/", "EMG Neural Network", "eda")'''


    path_rf = save_path+"Random_Forest/"
    path_nn = save_path+"Neural_Network/"
    for counter in range(10):
        if flag==0:
            ##########9 participants out for testing#####################
            X_read, X_read_test, train_dataset, test_dataset = read_per_participante(loadData, EMG_df)
        elif flag==1:
            ######### 3 emotion each, 3H, 3N, 3F ########################
            X_read, X_read_test, train_dataset, test_dataset = read_3emotion_per_participante(loadData, EMG_df)
        else:
            ######## 3 emotion each, 3H, 3N, 3F without 30% of information in test ########################
            X_read, X_read_test, train_dataset, test_dataset = read_3emotion_perPercentage(loadData, EMG_df)
        
        #print(X_read)
        #print(X_read_test)
        
        X = X_read[:, 0:6]
        y = np.array(X_read[:, 6], dtype=int)

        X_test = X_read_test[:, 0:6]
        y_test = np.array(X_read_test[:, 6], dtype=int)
        
        
        file = open(save_path+"EMG.txt", "a+")
        file.write("Counter: %s \t\t\t Train Dataset: %s \t\t\t Test Dataset: %s\n"%(counter, train_dataset, test_dataset))
        file.close()

        rs = ShuffleSplit(n_splits=10, test_size=0.20, random_state=42)
        rs.get_n_splits(X)
        i=0
        for train_index, test_index in rs.split(X):
            X_train, X_val = X[train_index], X[test_index]
            y_train, y_val = y[train_index], y[test_index]


            y_pred_rf,y_pred_val_rf, y_testing_rf, y_values_rf = random_model(X_train, X_val, X_test, y_train, y_val, y_test, path_rf, "EMG Random Forest_"+str(counter), str(counter),str(i), 8000, None)
            
            y_pred_nn, y_pred_val_nn, y_testing_nn, y_values_nn = nn(X_train, X_val, X_test, y_train, y_val, y_test, 6,200, path_nn, "EMG Neural Network_"+str(counter), str(counter),str(i), "emg")
            #nn_grid(X,y)
            i=i+1
        
        ##### performance global shuffle (flag=0)##########
        performe_global_shuffle(y_pred_rf,y_testing_rf, y_pred_val_rf, y_values_rf,'rf', path_rf,"EMG Random Forest_"+str(counter), counter)
        ##### performance global shuffle (flag=0)##########
        performe_global_shuffle(y_pred_nn, y_testing_nn,y_pred_val_nn, y_values_nn, 'nn', path_nn,"EMG Random Forest_"+str(counter), counter)

        y_pred_global_rf.extend(y_pred_rf)
        y_pred_global_nn.extend(y_pred_nn)
        y_test_global.extend(y_testing_rf)
        y_test_global_nn.extend(y_testing_nn)

    ##### performance global iteration (flag=0)##########
    performance_global_finaly(y_pred_global_rf, y_test_global,'rf', path_rf, "EMG Random Forest")

    ##### performance global iteration (flag=0)##########
    performance_global_finaly(y_pred_global_nn, y_test_global_nn,'nn', path_nn, "EMG Neural Network")

    clean_lists()

'''
def classifier_ecg_2():
    y_values = []
    y_testing = []

    y_pred_global_nn = []
    y_pred_global_rf = []
    y_test_global = []
    y_test_global_nn = []

    ECG_df = ['P_Waves', 'ECG_HRV_ULF']
    loadData = pickle.load(open("Data_inc/split_person/All_60s.pkl", "rb"))

    ##########with all data in test and train dataset############
    X_read, key = read_data(loadData, ECG_df)
    X = X_read[:, 0:2]
    y = np.array(X_read[:, 2], dtype=int)

    rf_allMix(X, y, "Classification/ECG_2/Random_Forest/", "ECG_2 Random Forest", 200, 50)
    nn_allMix(X, y, 2, 400, "Classification/ECG_2/Neural_Network/", "ECG_2 Neural Network", "eda")

    path_rf = "Classification/Emotion_30_60s/ECG_2/Random_Forest/"
    path_nn = "Classification/Emotion_30_60s/ECG_2/Neural_Network/"
    for counter in range(10):
        ##########9 participants out for testing#####################
        #X_read, X_read_test, train_dataset, test_dataset = read_per_participante(loadData, ECG_df)

        ######### 3 emotion each, 3H, 3N, 3F ########################
        #X_read, X_read_test, train_dataset, test_dataset = read_3emotion_per_participante(loadData, ECG_df)

        ######## 3 emotion each, 3H, 3N, 3F without 30% of information in test ########################
        X_read, X_read_test, train_dataset, test_dataset = read_3emotion_perPercentage(loadData, ECG_df)

        #print(X_read)
        #print(X_read_test)
        
        X = X_read[:, 0:2]
        y = np.array(X_read[:, 2], dtype=int)

        X_test = X_read_test[:, 0:2]
        y_test = np.array(X_read_test[:, 2], dtype=int)
        
        
        file = open("Classification/Emotion_30_60s/ECG_2/ECG_2.txt", "a+")
        file.write("Counter: %s \t\t\t Train Dataset: %s \t\t\t Test Dataset: %s\n"%(counter, train_dataset, test_dataset))
        file.close()

        rs = ShuffleSplit(n_splits=10, test_size=0.20, random_state=42)
        rs.get_n_splits(X)
        i=0
        for train_index, test_index in rs.split(X):
            X_train, X_val = X[train_index], X[test_index]
            y_train, y_val = y[train_index], y[test_index]


            y_pred_rf,y_pred_val_rf, y_testing_rf, y_values_rf = random_model(X_train, X_val, X_test, y_train, y_val, y_test, path_rf, "ECG_2 Random Forest_"+str(counter), str(counter), str(i), 2000, 20)
            
            y_pred_nn, y_pred_val_nn, y_testing_nn, y_values_nn = nn(X_train, X_val, X_test, y_train, y_val, y_test, 2,50, path_nn, "ECG_2 Neural Network_"+str(counter), str(counter),str(i), "ecg_2")
            #nn_grid(X,y)
            i=i+1
        
        ##### performance global shuffle (flag=0)##########
        performe_global_shuffle(y_pred_rf,y_testing_rf, y_pred_val_rf, y_values_rf,'rf', path_rf,"ECG_2 Random Forest_"+str(counter), counter)
        ##### performance global shuffle (flag=0)##########
        performe_global_shuffle(y_pred_nn, y_testing_nn,y_pred_val_nn, y_values_nn, 'nn', path_nn,"ECG_2 Random Forest_"+str(counter), counter)

        y_pred_global_rf.extend(y_pred_rf)
        y_pred_global_nn.extend(y_pred_nn)
        y_test_global.extend(y_testing_rf)
        y_test_global_nn.extend(y_testing_nn)

    ##### performance global iteration (flag=0)##########
    performance_global_finaly(y_pred_global_rf, y_test_global,'rf', path_rf, "ECG_2 Random Forest")

    ##### performance global iteration (flag=0)##########
    performance_global_finaly(y_pred_global_nn, y_test_global_nn,'nn', path_nn, "ECG_2 Neural Network")

    clean_lists()
    '''
