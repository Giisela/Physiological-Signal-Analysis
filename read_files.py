############# library ##########
import glob
import os
import scipy.io
import os.path
from os import path
import csv
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd


############# files ###########
from Global_var import *

def write_extractedTriggersignal(f_ext, value, signal):
    print("writing")
    df = pd.DataFrame([value], index=[f_ext]) # create a dataframe from match list
    df.to_csv("Data_emotionCSV/"+signal, sep=',', index=f_ext, header=False,  mode='a')  # merge

def extract_Triggers(f_path, filename):
    f_path = os.path.join(f_path, filename)
    with open(f_path, 'r') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            if not row[0] in Trigger:
                Trigger[row[0]] = []
            for line_number, data in enumerate(row[1:]):
                #stripped = ([float(data1) for data1 in data])
                Trigger[row[0]].append(int(data))


def extract_ECG(f_path, filename):
    print("ECG")
    f_path = os.path.join(f_path, filename)
    with open(f_path, 'r') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            if row[0] in Trigger:
                if not row[0] in ECG:
                    ECG[row[0]] = []
                
                start = Trigger[row[0]][0]
                end = Trigger[row[0]][1]

                for line_number, data in enumerate(row[start:end]):
                    #stripped = ([float(data1) for data1 in data])
                    ECG[row[0]].append(float(data))
    

                
                

def extract_EMGZygo(f_path, filename):
    print("EMGZ")

    f_path = os.path.join(f_path, filename)
    with open(f_path, 'r') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            if row[0] in Trigger:
                if not row[0] in EMGZ:
                    EMGZ[row[0]] = []

                start = Trigger[row[0]][0]
                end = Trigger[row[0]][1]

                for line_number, data in enumerate(row[start:end]):
                    #stripped = ([float(data1) for data1 in data])
                    EMGZ[row[0]].append(float(data))



def extract_EMGMFront(f_path, filename):
    print("EMGMF")

    f_path = os.path.join(f_path, filename)
    with open(f_path, 'r') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            if row[0] in Trigger:
                if not row[0] in EMGMF:
                    EMGMF[row[0]] = []

                start = Trigger[row[0]][0]
                end = Trigger[row[0]][1]

                for line_number, data in enumerate(row[start:end]):
                    #stripped = ([float(data1) for data1 in data])
                    EMGMF[row[0]].append(float(data))


def extract_EDA(f_path, filename):
    print("EDA")

    f_path = os.path.join(f_path, filename)
    with open(f_path, 'r') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            if row[0] in Trigger:
                if not row[0] in EDA:
                    EDA[row[0]] = []

                start = Trigger[row[0]][0]
                end = Trigger[row[0]][1]

                for line_number, data in enumerate(row[start:end]):
                    #stripped = ([float(data1) for data1 in data])
                    EDA[row[0]].append(float(data))



def extract_ECG_J(f_path, filename):
    f_path = os.path.join(f_path, filename)

    with open(f_path, 'r') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        label = next(readCSV, None)
        for row in readCSV:
            
            if not row[0] in ECG_J:
                ECG_J[row[0]] = []

            replacestr = [0 if x == '' else x for x in row[1:]]
            stripped = [float(data) for data in replacestr]
            for line_number, data in enumerate(stripped):
                ECG_J[row[0]].append(data)



def extract_EMGZygo_J(f_path, filename):
    f_path = os.path.join(f_path, filename)
    with open(f_path, 'r') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        label = next(readCSV, None)
        for row in readCSV:
            if not row[0] in EMGZ_J:
                EMGZ_J[row[0]] = []

            replacestr = [0 if x == '' else x for x in row[1:]]
            stripped = [float(data) for data in replacestr]
            for  line_number, data in enumerate(stripped):
                EMGZ_J[row[0]].append(data)




def extract_EMGMFront_J(f_path, filename):
    f_path = os.path.join(f_path, filename)

    with open(f_path, 'r') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        label = next(readCSV, None)
        for row in readCSV:
            if not row[0] in EMGMF_J:
                EMGMF_J[row[0]] = []

            replacestr = [0 if x == '' else x for x in row[1:]]
            stripped = [float(data) for data in replacestr]
            for line_number, data in enumerate(stripped):
                EMGMF_J[row[0]].append(data)



def extract_EDA_J(f_path, filename):
    f_path = os.path.join(f_path, filename)
    with open(f_path, 'r') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        label = next(readCSV, None)
        for row in readCSV:
            if not row[0] in EDA_J:
                EDA_J[row[0]] = []

            replacestr = [0 if x == '' else x for x in row[1:]]
            stripped = [float(data) for data in replacestr]
            for line_number, data in enumerate(stripped):
                EDA_J[row[0]].append(data)


def extract_ECG_window(f_path, filename):
    f_path = os.path.join(f_path, filename)
    with open(f_path, 'r') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=",")
        for row in readCSV:
            if not row[0] in window_ECG:
                window_ECG[row[0]] = []

            for line_number, data in enumerate(row[1:]):
                window_ECG[row[0]].append(float(data))


def extract_EMGZygo_window(f_path, filename):
    f_path = os.path.join(f_path, filename)
    with open(f_path, 'r') as csvfile:
        #line = (line.replace('[', '').replace(']', '').replace(' ', '').replace('\U00002013', '-') for line in csvfile)
        readCSV = csv.reader(csvfile, delimiter=",")
        for row in readCSV:
            if not row[0] in window_EMGZ:
                window_EMGZ[row[0]] = []

            for line_number, data in enumerate(row[1:]):
                # stripped = ([float(data1) for data1 in data])
                window_EMGZ[row[0]].append(float(data))



def extract_EMGMFront_window(f_path, filename):
    f_path = os.path.join(f_path, filename)
    with open(f_path, 'r') as csvfile:
        #line = (line.replace('[', '').replace(']', '').replace(' ', '').replace('\U00002013', '-') for line in csvfile)
        readCSV = csv.reader(csvfile, delimiter=",")
        for row in readCSV:
            if not row[0] in window_EMGMF:
                window_EMGMF[row[0]] = []

            for line_number, data in enumerate(row[1:]):
                # stripped = ([float(data1) for data1 in data])
                window_EMGMF[row[0]].append(float(data))



def extract_EDA_window(f_path, filename):
    f_path = os.path.join(f_path, filename)
    with open(f_path, 'r') as csvfile:
        #line = (line.replace('[', '').replace(']', '').replace(' ', '').replace('\U00002013', '-') for line in csvfile)
        readCSV = csv.reader(csvfile, delimiter=",")
        for row in readCSV:
            if not row[0] in window_EDA:
                window_EDA[row[0]] = []

            for line_number, data in enumerate(row[1:]):
                # stripped = ([float(data1) for data1 in data])
                window_EDA[row[0]].append(float(data))




def extract_id():
    for ecg in ECG_J.keys():
        keysRead_J.append(ecg)

'''def extract_id():
    for ecg in ECG.keys():
        keysRead.append(ecg)
        for ecg_j in ECG_J.keys():
            if ecg == ecg_j:
                keysRead_J.append(ecg_j)
'''


def extract_id_window_max():
    for ecg in window_ECG.keys():
        print(ecg)
        keysRead.append(ecg)
        keysRead_max.append(ecg)
        

    output = open('Info/key_participants_max.pkl', 'wb')
    pickle.dump(keysRead_max, output, protocol=4)


