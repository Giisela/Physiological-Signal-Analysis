############# library ##########
from biosppy import signals
from scipy import signal
from scipy.signal import butter, filtfilt, freqz
import numpy as np
import neurokit as nk
import matplotlib.pyplot as plt
############# files ###########
from Global_var import *


def impulse_response_filter(b, a, fs, fc):
    w, h = freqz(b, a, worN=8000)
    plt.plot(0.5 * fs * w / np.pi, np.abs(h), 'b')
    plt.plot(fc, 0.5 * np.sqrt(2), 'ko')
    plt.axvline(fc, color='k')
    plt.xlim(0, 0.5 * fs)
    plt.title("Lowpass Filter Frequency Response")
    plt.xlabel('Frequency [Hz]')
    plt.show()

def impz(b,a=1):
    l = len(b)
    impulse = np.repeat(0., l); impulse[0] = 1.0
    x = np.arange(0, l)
    response = signal.lfilter(b,a,impulse)
    plt.subplot(211)
    plt.stem(x, response)
    plt.ylabel('Amplitude')
    plt.xlabel(r'n (samples)')
    plt.title(r'Impulse response')
    plt.subplot(212)
    step = np.cumsum(response)
    plt.stem(x, step)
    plt.ylabel('Amplitude')
    plt.xlabel(r'n (samples)')
    plt.title(r'Step response')
    _ = plt.subplots_adjust(hspace=1)
    plt.show()

def filterECG_40Hz(ecg):

    fs = 1000
    fc = 40
    order = 10
    nyq = fs/2
    normal_cutoff = fc / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)

    ecgf = filtfilt(b, a, ecg)

    return ecgf

def filterECG_100Hz(ecg):

    fs = 1000
    fc = 100
    order = 10
    nyq = fs/2
    normal_cutoff = fc / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)

    ecgf = filtfilt(b, a, ecg)

    return ecgf

def filterEMG(emg):
    fs = 1000  # sample frequency (Hz)
    fc = 100
    nyq = 0.5*fs
    # create bandpass filter for EMG
    high = 45 / nyq
    low = 250 / nyq
    b, a = butter(4, [high, low], btype='bandpass')

    emg_filtered = filtfilt(b, a, emg)


    return emg_filtered


def filterEDA(eda):
    # create bandpass filter for eda
    fs =1000  # sample frequency (Hz)
    fc = 5
    nyq = 0.5*fs
    low = fc / nyq
    b, a = butter(4, [low], btype='lowpass')

    eda_filtered = filtfilt(b, a, eda)

    # process EMG signal: rectify
    # emg_rectified = abs(emg_filtered)
    return eda_filtered

def filterByLibrary():
    #for key in keysRead:
    key="ID1_F"
    #Filter ECG
    #original_ECG[key] = ECG[key]
    ECG[key] = signals.ecg.ecg(signal=ECG[key], sampling_rate=1000.0, show=True)[1]
    feature_ECG[key] = signals.ecg.ecg(signal=ECG[key], sampling_rate=1000.0, show=False)


    #Filter EDA
    #original_EDA[key] = EDA[key]
    EDA[key] = signals.eda.eda(signal=EDA[key], sampling_rate=1000.0, show=True)[1]
    feature_EDA[key] = signals.eda.eda(signal=EDA[key], sampling_rate=1000.0, show=False)

    #Filter EMG
    #original_EMGMF[key] = EMGMF[key]
    EMGMF[key] = signals.emg.emg(signal=EMGMF[key], sampling_rate=1000, show = True)[1]
    feature_EMGMF[key] = signals.emg.emg(signal=EMGMF[key], sampling_rate=1000, show=False)


    #original_EMGZ[key] = EMGZ[key]
    EMGZ[key] = signals.emg.emg(signal=EMGZ[key], sampling_rate=1000, show=True)[1]
    feature_EMGZ[key] = signals.emg.emg(signal=EMGZ[key], sampling_rate=1000, show=False)


def filter_signal():
    for key in keysRead:
        #Store Original value
        '''original_ECG[key] = ECG[key]
        original_EMGZ[key] = EMGZ[key]
        original_EMGMF[key] = EMGMF[key]
        original_EDA[key] = EDA[key]'''

        #Filter ECG
        ECG[key] = filterECG_40Hz(ECG[key])
        '''plt.title("ECG")
        plt.plot(original_ECG[key],"blue", label="Raw")
        plt.plot(ECG[key], "orange", label="filtered")
        plt.xlabel('Samples')
        plt.ylabel('ECG (V)')
        plt.xlim([0, len(ECG[key])])
        plt.legend()
        plt.show()
        #ECG_100[key] = filterECG_100Hz(ECG[key])'''

        #Filter EDA
        EDA[key] = filterEDA(EDA[key])
        '''plt.title("EDA")
        plt.plot(original_EDA[key],"blue", label="Raw")
        plt.plot(EDA[key], "orange", label="filtered")
        plt.xlabel('Samples')
        plt.ylabel('EDA (µS)')
        plt.xlim((0, len(EDA[key])))
        plt.legend()
        plt.show()'''
        
        #Filter EMG
        EMGZ[key] = filterEMG(EMGZ[key])
        '''plt.title("EMGZ")
        plt.plot(original_EMGZ[key],"blue", label="Raw")
        plt.plot(EMGZ[key], "orange", label="filtered")
        plt.xlabel('Samples')
        plt.ylabel('EMGZ (mV)')
        plt.xlim((0, len(EMGZ[key])))
        plt.legend()
        plt.show()'''
        
        EMGMF[key] = filterEMG(EMGMF[key])
        '''plt.title("EMGMF")
        plt.plot(original_EMGMF[key],"blue", label="Raw")
        plt.plot(EMGMF[key], "orange", label="filtered")
        plt.xlabel('Samples')
        plt.ylabel('EMGMF (mV)')
        plt.xlim((0, len(EMGMF[key])))
        plt.legend()
        plt.show()'''
        
