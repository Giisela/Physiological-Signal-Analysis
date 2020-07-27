############# library ##########
import sys
import warnings
import numpy as np
from matplotlib.pyplot import xcorr
from scipy.ndimage.interpolation import shift


import statistics

from natsort import natsort
from scipy.stats.stats import pearsonr, kendalltau, spearmanr
import matplotlib.pyplot as plt

############# files ###########
from Global_var import *
from WriteFiles import *
from Ploting_data import *

def window_MaxValue():
    print("window_maxValue")
    for key in keysRead_J:

        ECG_EMGZ[key] = lag_Max(ECG_J[key], EMGZ_J[key], "ecg", "emg_zyggo", key)
        #ECG_EMGMF[key] = lag_Max(ECG_J[key], EMGMF_J[key], "ecg", "emg_mfront", key)
        #ECG_EDA[key] = lag_Max(ECG_J[key], EDA_J[key],"ecg", "eda", key)

        #EMGZ_ECG[key] = lag_Max(EMGZ_J[key], ECG_J[key], "emg_zyggo", "ecg", key)
        #EMGZ_EMGMF[key] = lag_Max(EMGZ_J[key], EMGMF_J[key],"emg_zyggo", "emg_mfront", key)
        #EMGZ_EDA[key] = lag_Max(EMGZ_J[key], EDA_J[key], "emg_zyggo", "eda", key)

        #EMGMF_ECG[key] = lag_Max(EMGMF_J[key], ECG_J[key],"emg_mfront", "ecg", key)
        #EMGMF_EMGZ[key] = lag_Max(EMGMF_J[key], EMGZ_J[key], "emg_mfront", "emg_zyggo", key)
        #EMGMF_EDA[key] = lag_Max(EMGMF_J[key], EDA_J[key],"emg_mfront", "eda", key)

        #EDA_ECG[key] = lag_Max(EDA_J[key], ECG_J[key],"eda", "ecg", key)
        #EDA_EMGZ[key] = lag_Max(EDA_J[key], EMGZ_J[key], "eda", "emg_zyggo", key)
        #EDA_EMGMF[key] = lag_Max(EDA_J[key], EMGMF_J[key], "eda", "emg_mfront", key)

def lag_Max(signal1, signal2, label1, label2, id):
    x = xcorr(signal1, signal2, maxlags=15)
    
    lags = x[0]
    c = x[1]

    print("lags", lags)
    print("c", c)

    plt.title("Cross-Correlation")
    plt.stem(lags, c , use_line_collection = True)
    plt.grid()
    plt.xlabel("Lag")
    plt.ylim(0,1.1)
    plt.savefig("Plot_Participant/Lag/"+label1+"_"+label2+"_"+id+".png")
    plt.close()
    
    i = np.argmax(np.abs(c))
    lagDiff = lags[i]
    print("lag otimo:",lagDiff)

    # plot loss during training
    
    return 0


def extract_window_max(lag, signal1, signal2, label1, label2, id):

    max1_list = []
    max2_list = []
    if lag == 0:
        max1 = max(signal1)
        max_index1 = signal1.index(max1)
        print("valor lag 0:", max1)
        print("indice lag 0:", max_index1*1000)
        positiv_value = (len(ECG[id])-(max_index1*1000))
        negativ_value = ((max_index1*1000)-len(ECG[id]))
        if positiv_value<150000:
            for i in range(positiv_value-300000, positiv_value):
                max1_list.append((max_index1*1000)+i)

        elif negativ_value>-150000:
            for i in range(negativ_value, 300000-negativ_value):
                max1_list.append((max_index1*1000)+i)

        else:  
            for i in range(-150000, 150000):
                max1_list.append((max_index1*1000)+i)
        
        max2 = max(signal2)
        max_index2 = signal2.index(max2)
        
        positiv_value2 = (len(ECG[id])-(max_index2*1000))
        negativ_value2 = ((max_index2*1000)-len(ECG[id]))
        if positiv_value2<150000:
            for i in range(positiv_value2-300000, positiv_value2):
                max2_list.append((max_index2*1000)+i)

        elif negativ_value2>-150000:
            for i in range(negativ_value2, 300000-negativ_value2):
                max2_list.append((max_index2*1000)+i)

        else:  
            for i in range(-150000, 150000):
                max2_list.append((max_index2*1000)+i)

    else:
        max1 = max(signal1)
        max_index1 = signal1.index(max1)
        print("valor lag diff de 0:", max1)
        print("indice lag diff de 0:", max_index1*1000)
        positiv_value = (len(ECG[id])-(max_index1*1000))
        negativ_value = ((max_index1*1000)-len(ECG[id]))
        if positiv_value<150000:
            for i in range(positiv_value-300000, positiv_value):
                max1_list.append((max_index1*1000)+i)

        elif negativ_value>-150000:
            for i in range(negativ_value, 300000-negativ_value):
                max1_list.append((max_index1*1000)+i)

        else:  
            for i in range(-150000, 150000):
                max1_list.append((max_index1*1000)+i)

        # alignthesignals
        signal2 = list(shift(signal2, lag))
        max2 = max(signal2)
        max_index2 = signal2.index(max2)

        positiv_value2 = (len(ECG[id])-(max_index2*1000))
        negativ_value2 = ((max_index2*1000)-len(ECG[id]))
        if positiv_value2<150000:
            for i in range(positiv_value2-300000, positiv_value2):
                max2_list.append((max_index2*1000)+i)

        elif negativ_value2>-150000:
            for i in range(negativ_value2, 300000-negativ_value2):
                max2_list.append((max_index2*1000)+i)

        else:  
            for i in range(-150000, 150000):
                max2_list.append((max_index2*1000)+i)
    
    return max1_list, max2_list

def extract_window_min(lag, signal1, signal2, label1, label2, id):

    min1_list = []
    min2_list = []
    if lag == 0:
        min1 = min(signal1)
        min_index1 = signal1.index(min1)
        for i in range(-150000, 150000):
            min1_list.append(min_index1+i)
        min2 = min(signal2)
        min_index2 = signal2.index(min2)
        for i in range(-150000, 150000):
            min2_list.append(min_index2+i)

    else:
        min1 = min(signal1)
        min_index1 = signal1.index(min1)
        for i in range(-150000, 150000):
            min1_list.append(min_index1 + i)

        # alignthesignals
        signal2 = list(shift(signal2, lag))
        min2 = min(signal2)
        min_index2 = signal2.index(min2)
        for i in range(-150000, 150000):
            min2_list.append(min_index2 + i)

    return min1_list, min2_list



def extract_WindowMaxValue():
    print("extract_WindowMaxValue")
    for key in keysRead_J:
        print("extract_WindowMaxValue: ", key)
        storeInside["ECG_EMGZ",key] = extract_window_max(ECG_EMGZ[key], ECG_J[key], EMGZ_J[key], "ecg", "emg_zyggo", key)[0]
        #storeInside["ECG_EMGMF", key] = extract_window_max(ECG_EMGMF[key], ECG_J[key], EMGMF_J[key], "ecg", "emg_mfront", key)[0]
        #storeInside["ECG_EDA", key] = extract_window_max(ECG_EDA[key], ECG_J[key], EDA_J[key], "ecg", "eda", key)[0]

        storeInside["EMGZ_ECG", key] = extract_window_max(EMGZ_ECG[key], EMGZ_J[key], ECG_J[key], "emg_zyggo", "ecg", key)[0]
        #storeInside["EMGZ_EMGMF",key] = extract_window_max(EMGZ_EMGMF[key], EMGZ_J[key], EMGMF_J[key], "emg_zyggo", "emg_mfront", key)[0]
        #storeInside["EMGZ_EDA", key] = extract_window_max(EMGZ_EDA[key], EMGZ_J[key], EDA_J[key], "emg_zyggo", "eda", key)[0]

        storeInside["EMGMF_ECG", key] = extract_window_max(EMGMF_ECG[key], EMGMF_J[key], ECG_J[key], "emg_mfront", "ecg", key)[0]
        #storeInside["EMGMF_EMGZ", key] = extract_window_max(EMGMF_EMGZ[key], EMGMF_J[key], EMGZ_J[key], "emg_mfront", "emg_zyggo", key)[0]
        #storeInside["EMGMF_EDA", key] = extract_window_max(EMGMF_EDA[key], EMGMF_J[key], EDA_J[key], "emg_mfront", "eda", key)[0]

        storeInside["EDA_ECG", key] = extract_window_max(EDA_ECG[key], EDA_J[key], ECG_J[key], "eda", "ecg", key)[0]
        #storeInside["EDA_EMGZ", key] = extract_window_max(EDA_EMGZ[key], EDA_J[key], EMGZ_J[key], "eda", "emg_zyggo", key)[0]
        #storeInside["EDA_EMGMF", key] = extract_window_max(EDA_EMGMF[key], EDA_J[key], EMGMF_J[key], "eda", "emg_mfront", key)[0]
        
        '''print("ECG:", len(ECG[key]))
        print("EMGZ:", len(EMGZ[key]))
        print("EMGMF:", len(EMGMF[key]))
        print("EDA:", len(EDA[key]))
        
        print("ECG store:", len(storeInside["ECG_EMGZ", key]))
        print("EMGZ store:", len(storeInside["EMGZ_ECG", key]))
        print("EMGMF store:", len(storeInside["EMGMF_ECG", key]))
        print("EDA store:", len(storeInside["EDA_ECG", key]))'''

        
        for key_window in keysRead:
            if key == key_window:
                print(key)

                window_info(key,storeInside["ECG_EMGZ", key], "ECG-windowMaxIndex.csv")
                window_info(key,storeInside["EMGZ_ECG", key], "EMGZ-windowMaxIndex.csv")
                window_info(key,storeInside["EMGMF_ECG", key], "EMGMF-windowMaxIndex.csv")
                window_info(key,storeInside["EDA_ECG", key], "EDA-windowMaxIndex.csv")

                '''window_ECG[key] = ECG[key][storeInside["ECG_EMGZ", key]]
                window_EMGZ[key] = EMGZ[key][storeInside["EMGZ_ECG", key]]
                window_EMGMF[key] = EMGMF[key][storeInside["EMGMF_ECG", key]]
                window_EDA[key] = EDA[key][storeInside["EDA_ECG", key]]

                del ECG[key]
                del EMGZ[key]
                del EMGMF[key]
                del EDA[key]'''

                del storeInside["ECG_EMGZ", key]
                #del storeInside["ECG_EMGMF", key]
                #del storeInside["ECG_EDA", key]

                del storeInside["EMGZ_ECG", key]
                #del storeInside["EMGZ_EMGMF", key]
                #del storeInside["EMGZ_EDA", key]

                del storeInside["EMGMF_ECG", key]
                #del storeInside["EMGMF_EMGZ", key]
                #del storeInside["EMGMF_EDA", key]

                del storeInside["EDA_ECG", key]
                #del storeInside["EDA_EMGZ", key]
                #del storeInside["EDA_EMGMF", key]

        '''print("Window_info key: ", key)
        print("Window_info value: ", window_ECG[key])
        print("Window_info value: ", window_EMGZ[key])
        print("Window_info value: ", window_EMGMF[key])
        print("Window_info value: ", window_EDA[key])

        
        #Transfer to files
        window_info(key, window_ECG[key], "ECG-windowMaxValue.csv")
        window_info(key, window_EMGZ[key], "EMGZ-windowMaxValue.csv")
        window_info(key, window_EMGMF[key], "EMGMF-windowMaxValue.csv")
        window_info(key, window_EDA[key], "EDA-windowMaxValue.csv")'''

def extract_WindowMinValue():
    for key in keysRead_J:
        storeInside["ECG_EMGZ",key] = extract_window_min(ECG_EMGZ[key], ECG_J[key], EMGZ_J[key], "ecg", "emg_zyggo", key)[0]
        storeInside["ECG_EMGMF", key] = extract_window_min(ECG_EMGMF[key], ECG_J[key], EMGMF_J[key], "ecg", "emg_mfront", key)[0]
        storeInside["ECG_EDA", key] = extract_window_min(ECG_EDA[key], ECG_J[key], EDA_J[key], "ecg", "eda", key)[0]

        storeInside["EMGZ_ECG", key] = extract_window_min(EMGZ_ECG[key], EMGZ_J[key], ECG_J[key], "emg_zyggo", "ecg", key)[0]
        storeInside["EMGZ_EMGMF",key] = extract_window_min(EMGZ_EMGMF[key], EMGZ_J[key], EMGMF_J[key], "emg_zyggo", "emg_mfront", key)[0]
        storeInside["EMGZ_EDA", key] = extract_window_min(EMGZ_EDA[key], EMGZ_J[key], EDA_J[key], "emg_zyggo", "eda", key)[0]

        storeInside["EMGMF_ECG", key] = extract_window_min(EMGMF_ECG[key], EMGMF_J[key], ECG_J[key], "emg_mfront", "ecg", key)[0]
        storeInside["EMGMF_EMGZ", key] = extract_window_min(EMGMF_EMGZ[key], EMGMF_J[key], EMGZ_J[key], "emg_mfront", "emg_zyggo", key)[0]
        storeInside["EMGMF_EDA", key] = extract_window_min(EMGMF_EDA[key], EMGMF_J[key], EDA_J[key], "emg_mfront", "eda", key)[0]

        storeInside["EDA_ECG", key] = extract_window_min(EDA_ECG[key], EDA_J[key], ECG_J[key], "eda", "ecg", key)[0]
        storeInside["EDA_EMGZ", key] = extract_window_min(EDA_EMGZ[key], EDA_J[key], EMGZ_J[key], "eda", "emg_zyggo", key)[0]
        storeInside["EDA_EMGMF", key] = extract_window_min(EDA_EMGMF[key], EDA_J[key], EMGMF_J[key], "eda", "emg_mfront", key)[0]

        for key_window in keysRead:
            if key == key_window:
                window_ECG[key] = ECG[key][storeInside["ECG_EMGZ", key]]
                window_EMGZ[key] = EMGZ[key][storeInside["ECG_EMGZ", key]]
                window_EMGMF[key] = EMGMF[key][storeInside["ECG_EMGZ", key]]
                window_EDA[key] = EDA[key][storeInside["ECG_EMGZ", key]]

                del ECG[key]
                del EMGZ[key]
                del EMGMF[key]
                del EDA[key]

                del storeInside["ECG_EMGZ", key]
                del storeInside["ECG_EMGMF", key]
                del storeInside["ECG_EDA", key]

                del storeInside["EMGZ_ECG", key]
                del storeInside["EMGZ_EMGMF", key]
                del storeInside["EMGZ_EDA", key]

                del storeInside["EMGMF_ECG", key]
                del storeInside["EMGMF_EMGZ", key]
                del storeInside["EMGMF_EDA", key]

                del storeInside["EDA_ECG", key]
                del storeInside["EDA_EMGZ", key]
                del storeInside["EDA_EMGMF", key]


        print("Window_info key: ", key)
        print("Window_info value: ", window_ECG[key])
        print("Window_info value: ", window_EMGZ[key])
        print("Window_info value: ", window_EMGMF[key])
        print("Window_info value: ", window_EDA[key])
        #Transfer to files
        window_info(key, window_ECG[key], "ECG-windowMinValue.csv")
        window_info(key, window_EMGZ[key], "EMGZ-windowMinValue.csv")
        window_info(key, window_EMGMF[key], "EMGMF-windowMinValue.csv")
        window_info(key, window_EDA[key], "EDA-windowMinValue.csv")

def calcs():
    file = open("resultados.txt", "a+")
    file.write(
        "nome;type;mean;standard deviation;meadian;percentile 25;percentile 75\n")
    for key in keysRead:
        file.write(key + ';' + "ECG" + ';' + str(np.mean(window_ECG[key]))
                   + ';' + str(np.std(window_ECG[key]))
                   + ';' + str(statistics.median(window_ECG[key]))
                   + ';' + str(np.percentile(window_ECG[key], 25))
                   + ';' + str(np.percentile(window_ECG[key], 75))
                   + '\n' +
                   " " + ';' + "EMGZ" + ';' + str(np.mean(window_EMGZ[key]))
                   + ';' + str(np.std(window_EMGZ[key]))
                   + ';' + str(statistics.median(window_EMGZ[key]))
                   + ';' + str(np.percentile(window_EMGZ[key], 25))
                   + ';' + str(np.percentile(window_EMGZ[key], 75))
                   + '\n' +
                   " " + ';' + "EMGMF" + ';' + str(np.mean(window_EMGMF[key]))
                   + ';' + str(np.std(window_EMGMF[key]))
                   + ';' + str(statistics.median(window_EMGMF[key]))
                   + ';' + str(np.percentile(window_EMGMF[key], 25))
                   + ';' + str(np.percentile(window_EMGMF[key], 75))
                   + '\n' +
                   " " + ';' + "EDA" + ';' + str(np.mean(window_EDA[key]))
                   + ';' + str(np.std(window_EDA[key]))
                   + ';' + str(statistics.median(window_EDA[key]))
                   + ';' + str(np.percentile(window_EDA[key], 25))
                   + ';' + str(np.percentile(window_EDA[key], 75))
                   + '\n')