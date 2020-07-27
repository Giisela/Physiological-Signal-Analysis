import scipy.io
import os.path
from os import path
from glob import glob
import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, filtfilt
from natsort import natsorted



def butter_lowpass(cutoff, fs, order=5):
	nyq = 0.5 * fs
	normal_cutoff = cutoff / nyq
	b, a = butter(order, normal_cutoff, btype='low', analog=False)
	return b, a


def butter_highpass(cutoff, fs, order=5):
	nyq = 0.5 * fs
	normal_cutoff = cutoff / nyq
	b, a = butter(order, normal_cutoff, btype='high', analog=False)
	return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
	T = 10  # seconds
	n = 1000  # total number of samples
	t = np.linspace(0, T, n, endpoint=False)
	b, a = butter_lowpass(cutoff, fs, order=order)
	y = lfilter(b, a, data)
	return y,t


def butter_highpass_filter(data, cutoff, fs, order=5):
	T = 10  # seconds
	n = 1000  # total number of samples
	t = np.linspace(0, T, n, endpoint=False)
	b, a = butter_highpass(cutoff, fs, order=order)
	y = filtfilt(b, a, data)
	return y, t


def plot_data(result, y, t, f_ext, type):
	desktop = os.path.join(os.path.expanduser("~"), "Desktop")
	f_path = os.path.join(desktop, "5°ano/Tese/ApiSignals/Graficos/Graficos individuais/")
	plt.title(f_ext)

	plt.plot(t, result, 'y-', label='data')
	plt.plot(t, y, 'b-', linewidth=2, label='filtered data')
	plt.xlabel('Time [sec]')
	plt.grid()
	plt.legend()

	plt.subplots_adjust(hspace=0.50)
	plt.savefig(os.path.join(f_path, f_ext+"_"+type))
	plt.close()
	#plt.show()


def extract_matECG(f_path):
	print(f_path)
	f_name = path.basename(f_path)  # get the filename

	f_ext = f_name.split(".")[-2]
	mat = scipy.io.loadmat(f_path)
	data1 = mat['channels'][0]
	data2 = data1[0][0][0][0][0] #ECG

	result = []
	for line_number, line in enumerate(data2):
		if line_number > 999:
			break
		result.append(line)

	# Filter requirements.
	order = 3
	fs = 1000  # sample rate, Hz
	cutoff = 40  # desired cutoff frequency of the filter, Hz

	# "Noisy" data.
	# Filter the data, and plot both the original and filtered signals.
	y,t = butter_lowpass_filter(result, cutoff, fs, order)

	# plot_data(result, y, t, f_ext, "ECG") #plot individual
	'''
	plt.subplot(211)
	plt.title(f_ext)
	plt.plot(t, y, label='filtered ECG', color='black')
	plt.xlabel('Time [sec]')
	plt.grid()
	plt.legend()
	plt.subplots_adjust(hspace=0.50)
	
	#mat2csv(y, f_ext, t, "ECG")
	'''
	print("CSV is Finish with ECG data!")

def extract_matEMG_Zygo(f_path):
	print(f_path)
	f_name = path.basename(f_path)  # get the filename

	f_ext = f_name.split(".")[-2]
	mat = scipy.io.loadmat(f_path)
	data1 = mat['channels'][0]
	data2 = data1[1][0][0][0][0]  # EMG-Zygo

	result = []
	for line_number, line in enumerate(data2):
		if line_number > 999:
			break
		result.append(line)

	# Filter requirements.
	order = 3
	fs = 1000  # sample rate, Hz
	cutoff = 40  # desired cutoff frequency of the filter, Hz

	# "Noisy" data.
	# Filter the data, and plot both the original and filtered signals.
	y, t = butter_lowpass_filter(result, cutoff, fs, order)
	'''
	# plot_data(result, y, t, f_ext, "EMG-Zygo") #plot individual

	plt.subplot(212)
	plt.title(f_ext)
	plt.plot(t, y, label='filtered EMG-Zygo', color='magenta')
	plt.xlabel('Time [sec]')
	plt.grid()
	plt.legend()
	plt.subplots_adjust(hspace=0.50)
	
	#mat2csv(y, f_ext, t, "EMG-Zygo")
	'''
	print("CSV is Finish with EMG-Zygo data!")


def extract_matEMG_MFRONT(f_path):
	print(f_path)
	f_name = path.basename(f_path)  # get the filename

	f_ext = f_name.split(".")[-2]
	mat = scipy.io.loadmat(f_path)
	data1 = mat['channels'][0]
	data2 = data1[2][0][0][0][0]  # EMG_MFRONT

	result = []
	for line_number, line in enumerate(data2):
		if line_number > 999:
			break
		result.append(line)

	# Filter requirements.
	order = 3
	fs = 1000  # sample rate, Hz
	cutoff = 40  # desired cutoff frequency of the filter, Hz

	# "Noisy" data.
	# Filter the data, and plot both the original and filtered signals.
	y, t = butter_lowpass_filter(result, cutoff, fs, order)

	#plot_data(result, y, t, f_ext, "EMG-MFRONT")#plot individual
	'''
	plt.title(f_ext)
	plt.plot(t, y, label='filtered EMG-MFront', color='green')
	plt.xlabel('Time [sec]')
	plt.grid()
	plt.legend()

	plt.subplots_adjust(hspace=0.50)
	
	#mat2csv(y, f_ext, t, "EMG-MFRONT")
	'''
	print("CSV is Finish with EMG-MFRONT data!")



def extract_matEDA(f_path):
	print(f_path)
	f_name = path.basename(f_path)  # get the filename
	f_ext = f_name.split(".")[-2]
	mat = scipy.io.loadmat(f_path)
	data1 = mat['channels'][0]
	data2 = data1[3][0][0][0][0]  #EDA


	result = []
	for line_number, line in enumerate(data2):
		if line_number > 999:
			break
		result.append(line)

	# Filter requirements.
	order = 3
	fs = 1000  # sample rate, Hz
	cutoff = 40  # desired cutoff frequency of the filter, Hz

	# "Noisy" data.
	# Filter the data, and plot both the original and filtered signals.
	y, t = butter_highpass_filter(result, cutoff, fs, order)

	#plot_data(result, y, t, f_ext, "EDA") #plot individual

	plt.title(f_ext)
	plt.plot(t, result, label='EDA', color='blue')
	#plt.plot(t, y*10, label='filtered EDA', color='blue')
	plt.xlabel('Time [sec]')
	plt.grid()
	plt.legend()
	#plt.subplots_adjust(hspace=0.50)
	plt.show()
	#mat2csv(y, f_ext, t, "EDA")

	print("CSV is Finish with EDA data!")


def mat2csv(channel, f_ext, t,  type):
	desktop = os.path.join(os.path.expanduser("~"), "Desktop")
	f_path = os.path.join(desktop, "5°ano/Tese/ApiSignals/Data_CSV/")
	if type == "ECG":
		if os.path.exists(os.path.join(f_path, "ECG.csv")):
			df = pd.read_csv("~/Desktop/5°ano/Tese/ApiSignals/Data_CSV/ECG.csv")
			df = pd.DataFrame([channel], index=[f_ext],columns=list(t)) # create a dataframe from match list
			df.to_csv("~/Desktop/5°ano/Tese/ApiSignals/Data_CSV/ECG.csv", sep='\t', index=f_ext, header=False, mode='a')  # merge

		else:
			df = pd.DataFrame([channel], index=[f_ext], columns=list(t)) # create a dataframe from match list
			df.to_csv("~/Desktop/5°ano/Tese/ApiSignals/Data_CSV/ECG.csv", sep='\t', index=f_ext, mode='a')  # create csv from df

	elif type == "EMG-Zygo":
		if os.path.exists(os.path.join(f_path, "EMG-Zygo.csv")):
			df = pd.read_csv("~/Desktop/5°ano/Tese/ApiSignals/Data_CSV/EMG-Zygo.csv")
			df = pd.DataFrame([channel], index=[f_ext], columns=list(t))  # create a dataframe from match list
			df.to_csv("~/Desktop/5°ano/Tese/ApiSignals/Data_CSV/EMG-Zygo.csv", sep='\t', index=f_ext, header=False, mode='a')  # merge

		else:
			df = pd.DataFrame([channel], index=[f_ext], columns=list(t))  # create a dataframe from match list
			df.to_csv("~/Desktop/5°ano/Tese/ApiSignals/Data_CSV/EMG-Zygo.csv", sep='\t', index=f_ext, mode='a')  # create csv from df

	elif type == "EMG-MFRONT":
		if os.path.exists(os.path.join(f_path, "EMG-MFRONT.csv")):
			df = pd.read_csv("~/Desktop/5°ano/Tese/ApiSignals/Data_CSV/EMG-MFRONT.csv")
			df = pd.DataFrame([channel], index=[f_ext], columns=list(t))  # create a dataframe from match list
			df.to_csv("~/Desktop/5°ano/Tese/ApiSignals/Data_CSV/EMG-MFRONT.csv", sep='\t', index=f_ext, header=False, mode='a')  # merge

		else:
			df = pd.DataFrame([channel], index=[f_ext], columns=list(t))  # create a dataframe from match list
			df.to_csv("~/Desktop/5°ano/Tese/ApiSignals/Data_CSV/EMG-MFRONT.csv", sep='\t', index=f_ext, mode='a')  # create csv from df

	elif type == "EDA":
		if os.path.exists(os.path.join(f_path, "EDA.csv")):
			df = pd.read_csv("~/Desktop/5°ano/Tese/ApiSignals/Data_CSV/EDA.csv")
			df = pd.DataFrame([channel], index=[f_ext], columns=list(t))  # create a dataframe from match list
			df.to_csv("~/Desktop/5°ano/Tese/ApiSignals/Data_CSV/EDA.csv", sep='\t', index=f_ext, header=False, mode='a')  # merge

		else:
			df = pd.DataFrame([channel], index=[f_ext], columns=list(t))  # create a dataframe from match list
			df.to_csv("~/Desktop/5°ano/Tese/ApiSignals/Data_CSV/EDA.csv", sep='\t', index=f_ext, mode='a')  # create csv from df


def save_plot(f_ext):
	desktop = os.path.join(os.path.expanduser("~"), "Desktop")
	f_path = os.path.join(desktop, "5°ano/Tese/ApiSignals/Graficos/")
	plt.savefig(os.path.join(f_path, f_ext))
	plt.close()


def main():
	desktop = os.path.join(os.path.expanduser("~"), "Desktop")
	for f_path in natsorted(glob(os.path.join(desktop, "5°ano/Tese/ApiSignals/Data/*.mat"),recursive=True)):  # loop directory recursively
		f_name = path.basename(f_path)  # get the filename
		f_ext = f_name.split(".")[-2]
		#extract_matECG(f_path)
		#extract_matEMG_Zygo(f_path)
		#extract_matEMG_MFRONT(f_path)
		extract_matEDA(f_path)
		#save_plot(f_ext)
		#plt.show()


if __name__ == "__main__":
	main()