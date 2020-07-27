############# library ##########
import os.path
from os import path
import scipy.io
import glob
import pandas as pd
import natsort

"""
def extract_mat(f_path, f_ext):
    mat = scipy.io.loadmat(f_path)
    mat = {k: v for k, v in mat.items() if k[0] != '_'}
    data = pd.DataFrame({k: pd.Series(v[0]) for k, v in mat.items()})
    data.to_csv("~/Desktop/5°ano/Tese/ApiSignals/Data_CSV/"+f_ext+".csv")
"""

def extract_matTriggers(f_path, f_ext):
    mat = scipy.io.loadmat(f_path)
    print(mat)
    start = mat['mat_triggers'][3]
    end = mat['mat_triggers'][4]
    result = []
    for value in zip(start, end):
        result.append(value)
    print(start)
    print(end)
    matTriggerToCSV(f_ext, result)
    print("CSV is Finish with triggers data!", flush=True)


def matTriggerToCSV(f_ext, indexStartEnd):
    df = pd.DataFrame(indexStartEnd, index=[f_ext]) # create a dataframe from match list
    df.to_csv("Data_CSV/Triggers.csv", sep=',', index=f_ext,header=False,  mode='a')  # merge


def extract_matECG(f_path, f_ext):
    print(f_path, flush=True)
    mat = scipy.io.loadmat(f_path)
    data1 = mat['channels'][0]
    data2 = data1[0][0][0][0][0]  # ECG
    result = []
    for line_number, line in enumerate(data2):
        result.append(line)

    mat2csv(result, f_ext, "ECG")
    print("CSV is Finish with ECG data!", flush=True)


def extract_matEMG_Zygo(f_path, f_ext):
    print(f_path, flush=True)
    mat = scipy.io.loadmat(f_path)
    data1 = mat['channels'][0]
    data2 = data1[1][0][0][0][0]  # EMG-Zygo
    result = []
    for line_number, line in enumerate(data2):
        result.append(line)


    mat2csv(result, f_ext, "EMG-Zygo")
    print("CSV is Finish with EMG-Zygo data!", flush=True)

def extract_matEMG_MFRONT(f_path, f_ext):
    print(f_path, flush=True)
    mat = scipy.io.loadmat(f_path)
    data1 = mat['channels'][0]
    data2 = data1[2][0][0][0][0]  # EMG_MFRONT
    result = []
    for line_number, line in enumerate(data2):
        result.append(line)

    mat2csv(result, f_ext, "EMG-MFRONT")

    print("CSV is Finish with EMG-MFRONT data!", flush=True)

def extract_matEDA(f_path, f_ext):
    print(f_path, flush=True)
    mat = scipy.io.loadmat(f_path)
    data1 = mat['channels'][0]
    data2 = data1[3][0][0][0][0]  # EDA
    result = []
    for line_number, line in enumerate(data2):
        result.append(line)

    mat2csv(result, f_ext, "EDA")

    print("CSV is Finish with EDA data!", flush=True)


def mat2csv(channel, f_ext,  type):
    f_path = "/home/gisela.pinto/Data_CSV/"
    print("ID:", f_ext, flush=True)
    if type == "ECG":
        if os.path.exists(os.path.join(f_path, "ECG.csv")):
            df = pd.read_csv("/home/gisela.pinto/Data_CSV/ECG.csv")
            df = pd.DataFrame([channel], index=[f_ext]) # create a dataframe from match list
            df.to_csv("/home/gisela.pinto/Data_CSV/ECG.csv", sep=',', index=f_ext, header=False, mode='a')  # merge

        else:
            df = pd.DataFrame([channel], index=[f_ext]) # create a dataframe from match list
            df.to_csv("/home/gisela.pinto/Data_CSV/ECG.csv", sep=',', index=f_ext, mode='a')  # create csv from df

    elif type == "EMG-Zygo":
        if os.path.exists(os.path.join(f_path, "EMG-Zygo.csv")):
            df = pd.read_csv("/home/gisela.pinto/Data_CSV/EMG-Zygo.csv")
            df = pd.DataFrame([channel], index=[f_ext])  # create a dataframe from match list
            df.to_csv("/home/gisela.pinto/Data_CSV/EMG-Zygo.csv", sep=',', index=f_ext, header=False, mode='a')  # merge

        else:
            df = pd.DataFrame([channel], index=[f_ext])  # create a dataframe from match list
            df.to_csv("/home/gisela.pinto/Data_CSV/EMG-Zygo.csv", sep=',', index=f_ext, mode='a')  # create csv from df

    elif type == "EMG-MFRONT":
        if os.path.exists(os.path.join(f_path, "EDA.csv")):
            df = pd.read_csv("/home/gisela.pinto/Data_CSV/EDA.csv")
            df = pd.DataFrame([channel], index=[f_ext])  # create a dataframe from match list
            df.to_csv("/home/gisela.pinto/Data_CSV/EDA.csv", sep=',', index=f_ext, header=False, mode='a')  # merge

        else:
            df = pd.DataFrame([channel], index=[f_ext])  # create a dataframe from match list
            df.to_csv("/home/gisela.pinto/Data_CSV/EDA.csv", sep=',', index=f_ext, mode='a')  # create csv from df

    elif type == "EDA":
        if os.path.exists(os.path.join(f_path, "EMG-MFRONT.csv")):
            df = pd.read_csv("/home/gisela.pinto/Data_CSV/EMG-MFRONT.csv")
            df = pd.DataFrame([channel], index=[f_ext])  # create a dataframe from match list
            df.to_csv("/home/gisela.pinto/Data_CSV/EMG-MFRONT.csv", sep=',', index=f_ext, header=False, mode='a')  # merge

        else:
            df = pd.DataFrame([channel], index=[f_ext])  # create a dataframe from match list
            df.to_csv("/home/gisela.pinto/Data_CSV/EMG-MFRONT.csv", sep=',', index=f_ext, mode='a')  # create csv from df


def main():
    
    #desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    #fpath = os.path.join(desktop, "5°ano/Tese/ApiSignals/Data/")
    
    '''fpath = "/home/gisela.pinto/Data/"
    for f_path in glob.glob(os.path.join(fpath, '*.mat')):  # , '*.mat'
        f_name = path.basename(f_path)  # get the filename
        f_ext = f_name.split("_")[0] + "_" + f_name.split("_")[2]
        f_ext = f_ext.split(".")[-2]

        extract_matECG(f_path, f_ext)
        extract_matEMG_Zygo(f_path, f_ext)
        extract_matEMG_MFRONT(f_path, f_ext)
        extract_matEDA(f_path, f_ext)'''

    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    fpath = os.path.join(desktop, "5°ano/Tese/ApiSignals/Triggers/")

    for f_path in natsort.natsorted(glob.glob(os.path.join(fpath, '*.mat'))) :  # , '*.mat'
        print(f_path)
        f_name = path.basename(f_path)  # get the filename
        print(f_name)
        f_ext = f_name.split("_")[0] + "_" + f_name.split("_")[2]
        print(f_ext)


        extract_matTriggers(f_path, f_ext)
        




if __name__ == "__main__":
    main()