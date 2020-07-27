import csv
import os.path
from os import path
import  matplotlib.pyplot as plt
from matplotlib.pyplot import xcorr
import numpy as np
from scipy.signal import find_peaks
from pandas import DataFrame


def extract_ECG():
      desktop = os.path.join(os.path.expanduser("~"), "Desktop")
      f_path = os.path.join(desktop, "5°ano/Tese/ApiSignals/Dados_Joao/ECG.csv")
      f_name = path.basename(f_path)  # get the filename
      f_ext = f_name.split(".")[-2]
      rows = []
      with open(f_path, 'r') as csvfile:
          print("Path: ", f_path)
          readCSV = csv.reader(csvfile, delimiter=',')
          label = next(readCSV, None)
          id =[]
          for row in readCSV:
            id.append(row[0])
            replacestr = [0.0 if x == '' else x for x in row[1:]]
            stripped = ([float(data) for data in replacestr])
            rows.append(stripped)
          return id, rows


def extract_EMGZygo():
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    f_path = os.path.join(desktop, "5°ano/Tese/ApiSignals/Dados_Joao/EMG_ZIGGO.csv")
    f_name = path.basename(f_path)  # get the filename
    f_ext = f_name.split(".")[-2]
    rows = []
    id = []
    with open(f_path, 'r') as csvfile:
        print("Path: ", f_path)
        readCSV = csv.reader(csvfile, delimiter=',')
        label = next(readCSV, None)
        for row in readCSV:
            id.append(row[0])
            replacestr = [0.0 if x == '' else x for x in row[1:]]
            stripped = ([float(data) for data in replacestr])
            rows.append(stripped)
        return rows


def extract_EMGMFront():
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    f_path = os.path.join(desktop, "5°ano/Tese/ApiSignals/Dados_Joao/EMG_MFRONT.csv")
    f_name = path.basename(f_path)  # get the filename
    f_ext = f_name.split(".")[-2]
    rows = []
    id = []
    with open(f_path, 'r') as csvfile:
        print("Path: ", f_path)
        readCSV = csv.reader(csvfile, delimiter=',')
        label = next(readCSV, None)
        for row in readCSV:
            id.append(row[0])
            replacestr = [0.0 if x == '' else x for x in row[1:]]
            stripped = ([float(data) for data in replacestr])
            rows.append(stripped)
        return rows

def extract_EDA():
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    f_path = os.path.join(desktop, "5°ano/Tese/ApiSignals/Dados_Joao/EDA.csv")
    f_name = path.basename(f_path)  # get the filename
    f_ext = f_name.split(".")[-2]
    rows = []
    id = []
    with open(f_path, 'r') as csvfile:
        print("Path: ", f_path)
        readCSV = csv.reader(csvfile, delimiter=',')
        label = next(readCSV, None)
        for row in readCSV:
            id.append(row[0])
            replacestr = [0.0 if x == '' else x for x in row[1:]]
            stripped = ([float(data) for data in replacestr])
            rows.append(stripped)
        return rows



def lag_Max(signal1, signal2, label1, label2, id):
    x = xcorr(signal1, signal2, maxlags=10)
    lags = x[0]
    c = x[1]

    i = np.argmax(np.abs(c))
    lagDiff = lags[i]

    extract_peak(lagDiff, signal1, signal2,label1, label2, id)
    #plot_data(signal1, signal2,label1, label2, lagDiff, id)


def extract_peak(lag, signal1, signal2, label1, label2, id):
    if lag == 0:
        max1 = max(signal1)
        max_index1 = signal1.index(max1)
        max2 = max(signal2)
        max_index2 = signal2.index(max2)

    else:
        max1 = max(signal1)
        print("Max1: ",max1)
        max_index1 = signal1.index(max1)
        signal2 = [x+lag for x in signal2]
        print("signal2: ",signal2)
        max2 = max(signal2)
        print("Max2: ",max2)
        max_index2 = signal2.index(max2)



    plt.title(id)
    plt.plot(signal1, 'b-', linewidth=2, label=label1)
    plt.plot(signal2, 'g-', linewidth=2, label=label2)
    plt.plot(max1, signal1[max_index1], 'x')
    plt.plot(max2, signal2[max_index2], 'x')
    plt.ylabel('amplitude')
    plt.xlabel('segundos')
    plt.ylim(1.5, 5)
    plt.xlim(max1-2, max1+2)
    plt.grid()
    plt.legend()
    plt.subplots_adjust(hspace=0.50)
    plt.show()


def plot_data(signal1, signal2, label1, label2, lag, id):
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    f_path = os.path.join(desktop, "5°ano/Tese/ApiSignals/Combination/")
    plt.title(id)
    if lag == 0:
        plt.plot(signal1, 'b-', linewidth=2, label=label1)
        plt.plot(signal2, 'g-', linewidth=2, label=label2)
    else:
        signal2 = [x + lag for x in signal2]
        plt.plot(signal1, 'b-', linewidth=2, label=label1)
        plt.plot(signal2, 'g-', linewidth=2, label=label2)
    plt.xlabel('lag')
    plt.ylim(0, 5)
    plt.xlim(-100, 3000, 2000)
    plt.grid()
    plt.legend()
    plt.subplots_adjust(hspace=0.50)
    #plt.savefig(os.path.join(f_path, id+"_"+label1+"-"+label2))
    #plt.close()
    plt.show()




def main():
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    f_path = os.path.join(desktop, "5°ano/Tese/ApiSignals/Data_CSV/")

    id, data_ecg = extract_ECG()
    data_emgZ = extract_EMGZygo()
    data_emgM = extract_EMGMFront()
    data_eda = extract_EDA()

    for ecg, emg_Z, emg_M, eda, id_participant in zip(data_ecg, data_emgZ, data_emgM, data_eda, id):
        #print("ecg: ", ecg)

        lag_Max(ecg, emg_Z, "ecg", "emg_zyggo", id_participant)
        lag_Max(ecg, emg_M, "ecg", "emg_mfront", id_participant)
        lag_Max(ecg, eda, "ecg", "eda", id_participant)

        lag_Max(emg_Z, ecg, "emg_zyggo", "ecg", id_participant)
        lag_Max(emg_Z, emg_M, "emg_zyggo", "emg_mfront", id_participant)
        lag_Max(emg_Z, eda, "emg_zyggo", "eda", id_participant)

        lag_Max(emg_M, ecg, "emg_mfront", "ecg", id_participant)
        lag_Max(emg_M, emg_Z, "emg_mfront", "emg_zyggo", id_participant)
        lag_Max(emg_M, eda, "emg_mfront", "eda", id_participant)

        lag_Max(eda, ecg, "eda", "ecg", id_participant)
        lag_Max(eda, emg_Z, "eda", "emg_zyggo", id_participant)
        lag_Max(eda, emg_M, "eda", "emg_mfront", id_participant)





if __name__ == "__main__":
    main()