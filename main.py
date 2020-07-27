############# library ##########
import sys
import warnings
import matplotlib.pyplot as plt
#import neurokit as nk
from multiprocessing import Pool

############# files ###########
from Global_var import *
from read_files import *
from FilterData import *
from Statistic import *
from WriteFiles import *
from Feature_Extraction import *
#from feature_Selection import *
#from Ploting_data import *
#from Classificatores import *
#from performance import *
#from itertools import zip_longest
#from scipy import signal

'''
def graphics(signal1, signal2, time, label, key):
    # f, ((ax1, ax2, ax3)) = plt.subplots(3, sharex=True, sharey=True)
    # signal2 = signal.resample(signal2,  len(signal2)*5)
    plt.title(key)
    plt.plot(signal1, "b-", label=label+"raw")
    plt.plot(signal2, "g-", label=label+"filtered")
    # print(len(signal2))
    # plt.plot(filterEMG(signal2))

    #plt.plot(nk.rsp_process(signal2, sampling_rate=200)["df"]["RSP_Filtered"], label="RSP Filtered")
    plt.xlabel("Samples")
    plt.ylabel("ECG (Bpm)")
    # ax1.set(ylabel="ECG")
    # ax2.set(ylabel="ECG")
    # ax3.set(xlabel="Samples", ylabel="ECG")
    # ax1.set_title("ECG raw from 33 Neutral State")
    # ax2.set_title("ECG filtered from 33 Neutral State")
    # ax3.set_title("ECG library from 33 Neutral State")
    # ax1.legend()
    # ax2.legend()
    # ax3.legend()
    plt.legend()
    plt.xlim((0, len(signal2)))
    plt.show()
'''
 

def extract_dataJ():
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    f_path = os.path.join(desktop, "5°ano/Tese/ApiSignals/Dados_Joao/")
    #f_path = "/home/gisela.pinto/Tese/Dados_Joao/"
    extract_ECG_J(f_path, "ECG.csv")
    extract_EMGZygo_J(f_path, "EMG_ZIGGO.csv")
    #extract_EMGMFront_J(f_path, "EMG_MFRONT.csv")
    #extract_EDA_J(f_path, "EDA.csv")
    extract_id()


def extract_realData():
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    f_path = os.path.join(desktop, "5°ano/Tese/ApiSignals/Data_CSV/")

    #f_path = "/home/gisela.pinto/Tese/Data_CSV/"
    extract_Triggers(f_path, "Triggers.csv")
    extract_ECG(f_path, "ECG.csv")
    #extract_EMGZygo(f_path, "EMG-Zygo.csv")
    #extract_EMGMFront(f_path, "EMG-MFRONT.csv")
    #extract_EDA(f_path, "EDA.csv")
    extract_id()
   

'''

def extract_WindowMax():
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    f_path = os.path.join(desktop, "5°ano/Tese/ApiSignals/Info/")

    #f_path = "/home/gisela.pinto/Tese/Info/"
    extract_ECG_window(f_path, "ECG-windowMaxValue.csv")
    extract_EDA_window(f_path, "EDA-windowMaxValue.csv")
    extract_EMGZygo_window(f_path, "EMGZ-windowMaxValue.csv")
    extract_EMGMFront_window(f_path, "EMGMF-windowMaxValue.csv")
    extract_id_window_max()



def extract_WindowMin():
    #desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    #f_path = os.path.join(desktop, "5°ano/Tese/ApiSignals/Info/")

    f_path = "/home/gisela.pinto/Tese/Info/"
    extract_ECG_window(f_path, "ECG-windowMinValue.csv")
    extract_EMGZygo_window(f_path, "EMGZ-windowMinValue.csv")
    extract_EMGMFront_window(f_path, "EMGMF-windowMinValue.csv")
    extract_EDA_window(f_path, "EDA-windowMinValue.csv")
    extract_id_window_min()


def extract_emotion():
    for key in keysRead:
        if "N" in key:
            key_N.append(key)
        elif "F" in key:
            key_F.append(key)
        else:
            key_H.append(key)
'''

def classifier1():
    ##########Classifier  unkonw participant##################
    #classifier_all(0, "Classification/Participant_30s/All/")
    #classifier_ecg(0, "Classification/Participant_30s/ECG_6/")
    classifier_emgmf(0, "Classification/Participant_30s/EMGMF/")
    classifier_emgz(0, "Classification/Participant_30s/EMGZ/")
    classifier_eda(0, "Classification/Participant_30s/EDA/")
    classifier_emg(0, "Classification/Participant_30s/EMG/")

def classifier2():
    ##########Classifier  4 emotions ##################
    classifier_all(1, "Classification/Emotion_30s/All/")
    classifier_ecg(1, "Classification/Emotion_30s/ECG_6/")
    classifier_emgmf(1, "Classification/Emotion_30s/EMGMF/")
    classifier_emgz(1, "Classification/Emotion_30s/EMGZ/")
    classifier_eda(1, "Classification/Emotion_30s/EDA/")
    classifier_emg(1, "Classification/Emotion_30s/EMG/")

def classifier3():
    ##########Classifier , 30% of emotion in trein 4 emotions ##################
    classifier_all(2, "Classification/Emotion_30_30s/All/")
    classifier_ecg(2, "Classification/Emotion_30_30s/ECG_6/")
    classifier_emgmf(2, "Classification/Emotion_30_30s/EMGMF/")
    classifier_emgz(2, "Classification/Emotion_30_30s/EMGZ/")
    classifier_eda(2, "Classification/Emotion_30_30s/EDA/")
    classifier_emg(2, "Classification/Emotion_30_30s/EMG/")

def classifier1_60():
    ##########Classifier  unkonw participant##################
    classifier_all(0, "Classification/Participant_60s/All/")
    classifier_ecg(0, "Classification/Participant_60s/ECG_6/")
    classifier_emgmf(0, "Classification/Participant_60s/EMGMF/")
    classifier_emgz(0, "Classification/Participant_60s/EMGZ/")
    classifier_eda(0, "Classification/Participant_60s/EDA/")
    classifier_emg(0, "Classification/Participant_60s/EMG/")

def classifier2_60():
    ##########Classifier  4 emotions ##################
    classifier_all(1, "Classification/Emotion_60s/All/")
    classifier_ecg(1, "Classification/Emotion_60s/ECG_6/")
    classifier_emgmf(1, "Classification/Emotion_60s/EMGMF/")
    classifier_emgz(1, "Classification/Emotion_60s/EMGZ/")
    classifier_eda(1, "Classification/Emotion_60s/EDA/")
    classifier_emg(1, "Classification/Emotion_60s/EMG/")

def classifier3_60():
    ##########Classifier , 30% of emotion in trein 4 emotions ##################
    classifier_all(2, "Classification/Emotion_30_60s/All/")
    classifier_ecg(2, "Classification/Emotion_30_60s/ECG_6/")
    classifier_emgmf(2, "Classification/Emotion_30_60s/EMGMF/")
    classifier_emgz(2, "Classification/Emotion_30_60s/EMGZ/")
    classifier_eda(2, "Classification/Emotion_30_60s/EDA/")
    classifier_emg(2, "Classification/Emotion_30_60s/EMG/")




def main():
    if not sys.warnoptions:
        warnings.simplefilter("ignore")


    #Read .csv files to load data
    #print("Extract joao data")
    #extract_dataJ()
    #print("Extract real data")
    #extract_realData()


    #Filter Real Data
    #print("Filter Data")
    #filterByLibrary()
    #print("Filter Data")
    #filter_signal()


    #"Lag max value and his window"
    #Lag max value and his window
    #print("Max value")
    #window_MaxValue()
    #extract_WindowMaxValue()
    #print("extract Max value")
    #print("extract Max value from CSV")
    #extract_WindowMax()
    '''
    lags=[-15, -14, -13, -12, -11, -10,  -9,  -8,  -7 , -6 , -5 , -4 , -3,  -2 , -1 ,  0 ,  1  , 2,
            3,   4,   5 ,  6 ,  7 ,  8 ,  9 , 10 , 11,  12 , 13 , 14 , 15]

    c = [0.9450, 0.9489, 0.9503, 0.9545, 0.9587, 0.9601,
        0.9646, 0.9679, 0.9739, 0.9780, 0.9802,  0.9835,
        0.9870, 0.9905, 0.9950, 
        0.9999, 
        0.9951, 0.9905,
        0.9870,  0.9835, 0.9802, 0.9780, 0.9739, 0.9679,
        0.9646, 0.9601, 0.9587, 0.9545, 0.9503, 0.9489,
        0.9450]

    lags = [-15, -14, -13, -12, -11, -10,  -9,  -8,
    -7,  -6,  -5,  -4,  -3,  -2,  -1,   0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,
    11,  12,  13,  14,  15]
        #c = x[1]
    c = [0.9090, 0.9105,
        0.9149,0.9195, 0.9229, 0.9255, 0.9299,0.9333, 
        0.9374, 0.9405, 0.9449, 0.9485,  0.9509, 0.9550,  
        0.9595, 0.9649, 0.9685, 0.9729, 0.9755, 0.9809, 
        0.9855, 0.9899, 
        
        0.9959,
        
        0.9899, 0.9855, 0.9809, 0.9755,
        0.9729, 0.9685, 0.9649, 0.9595]
    print("lag:",lags)
    print("c:",c)

    plt.title("Cross-Correlation")
    plt.stem(lags, c , use_line_collection = True)
    plt.grid()
    plt.xlabel("Lag")
    plt.ylim(0,1.1)
    plt.savefig("Plot_Participant/Lag/ecg_emgz_ID1_H.png")
    plt.close()'''


    # "Lag min value and his window"
    # Lag min value and his window
    #print("Min value", flush=True)
    #window_MaxValue()
    #print("extract Min value", flush=True)
    #extract_WindowMinValue()
    #extract_WindowMin()

    loadData = pickle.load(open("Info/key_participants_max.pkl", "rb"))

    for key in loadData:
        keysRead.append(key)
    
    
    ##Feature extraction
    #print("extract feature", flush=True)
    #extract_features()
    #print("select feature", flush=True)
    #selectAll()
    #print("extract feature all metrics", flush=True)
    #extract_features_all_metrics()
    featureExtraction_person_all()
    #featureExtraction_emotion_all()


    #classifier1()
    #classifier2()
    #classifier3()

    #classifier1_60()
    #classifier2_60()
    #classifier3_60()



    print("THE END")





if __name__=="__main__":
    main()
  