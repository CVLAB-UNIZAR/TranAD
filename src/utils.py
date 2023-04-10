import matplotlib.pyplot as plt
import os
from src.constants import *
import pandas as pd
import numpy as np
import torch
from scipy.signal import hilbert, butter, filtfilt
from scipy.fftpack import fft,fftfreq,rfft,irfft,ifft

class color:
	HEADER = '\033[95m'
	BLUE = '\033[94m'
	GREEN = '\033[92m'
	RED = '\033[93m'
	FAIL = '\033[91m'
	ENDC = '\033[0m'
	BOLD = '\033[1m'
	UNDERLINE = '\033[4m'

def plot_accuracies(accuracy_list, folder):
	os.makedirs(f'plots/{folder}/', exist_ok=True)
	trainAcc = [i[0] for i in accuracy_list]
	lrs = [i[1] for i in accuracy_list]
	plt.xlabel('Epochs')
	plt.ylabel('Average Training Loss')
	plt.plot(range(len(trainAcc)), trainAcc, label='Average Training Loss', linewidth=1, linestyle='-', marker='.')
	plt.twinx()
	plt.plot(range(len(lrs)), lrs, label='Learning Rate', color='r', linewidth=1, linestyle='--', marker='.')
	plt.savefig(f'plots/{folder}/training-graph.pdf')
	plt.clf()

def cut_array(percentage, arr):
	print(f'{color.BOLD}Slicing dataset to {int(percentage*100)}%{color.ENDC}')
	mid = round(arr.shape[0] / 2)
	window = round(arr.shape[0] * percentage * 0.5)
	return arr[mid - window : mid + window, :]

def getresults2(df, result):
	results2, df1, df2 = {}, df.sum(), df.mean()
	for a in ['FN', 'FP', 'TP', 'TN']:
		results2[a] = df1[a]
	for a in ['precision', 'recall']:
		results2[a] = df2[a]
	results2['f1*'] = 2 * results2['precision'] * results2['recall'] / (results2['precision'] + results2['recall'])
	return results2

def pearson_corr(prefalta, falta):
	pdPrefalta = pd.DataFrame(prefalta[0,:,:].detach().numpy())
	pdFalta = pd.DataFrame(falta.detach().numpy())

	r_window_size = 120
	# Compute rolling window synchrony
	pd_rolling_r = pdPrefalta.rolling(window=r_window_size, center=True).corr(pdFalta)
	pd_rolling_r = pd_rolling_r.fillna(1)
	#pd_rolling_r.plot()
	# plt.xlabel='Frame'
	# plt.ylabel='Pearson r'
	# plt.suptitle("Phases data and rolling window correlation")
	#plt.show()
	#print(pd_rolling_r.head(10))
	rolling_r = torch.tensor(pd_rolling_r.values)
	return rolling_r

def crosscorr(datax, datay, lag=0, wrap=False):
	""" Lag-N cross correlation.
    Shifted data filled with NaNs

    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length
    Returns
    ----------
    crosscorr : float
    """
	if wrap:
		shiftedy = datay.shift(lag)
		shiftedy.iloc[:lag] = datay.iloc[-lag:].values
		return datax.corr(shiftedy)
	else:
		return datax.corr(datay.shift(lag), method='pearson')

def time_lagged_cross_correlation(prefalta, falta, lag=0, wrap=False):
	pdPrefalta = pd.DataFrame(prefalta[0,:,:].detach().numpy())
	pdFalta = pd.DataFrame(falta.detach().numpy())
	seconds = 5
	fps = 30
	rs = [crosscorr(pdPrefalta[0],pdFalta[0], lag) for lag in range(-int(seconds*fps),int(seconds*fps+1))]
	offset = np.floor(len(rs)/2)-np.argmax(rs)
	f,ax=plt.subplots(figsize=(14,3))
	ax.plot(rs)
	ax.axvline(np.ceil(len(rs)/2),color='k',linestyle='--',label='Center')
	ax.axvline(np.argmax(rs),color='r',linestyle='--',label='Peak synchrony')
	ax.set(title=f'Offset = {offset} frames\nS1 leads <> S2 leads',ylim=[.1,.31],xlim=[0,301], xlabel='Offset',ylabel='Pearson r')
	ax.set_xticks([0, 50, 100, 151, 201, 251, 301])
	ax.set_xticklabels([-150, -100, -50, 0, 50, 100, 150]);
	plt.legend()
	plt.show()

def butter_bandpass(lowcut, highcut, fs, order=5):
	nyq = 0.5 * fs
	low = lowcut / nyq
	high = highcut / nyq
	b, a = butter(order, [low, high], btype='band')
	return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
	b, a = butter_bandpass(lowcut, highcut, fs, order=order)
	y = filtfilt(b, a, data)
	return y

def phase_syncrony(prefalta, falta):
	pdPrefalta = pd.DataFrame(prefalta[0,:,:].detach().numpy())
	pdFalta = pd.DataFrame(falta[0,:,:].detach().numpy())

	lowcut  = .001
	highcut = 0.05
	fs = 30.
	order = 1
	phase = np.zeros((4000,3))
	for i in range(3):
		d1 = pdPrefalta[i].interpolate().values
		d2 = pdFalta[i].interpolate().values
		y1 = butter_bandpass_filter(d1,lowcut=lowcut,highcut=highcut,fs=fs,order=order)
		y2 = butter_bandpass_filter(d2,lowcut=lowcut,highcut=highcut,fs=fs,order=order)

		al1 = np.angle(hilbert(y1),deg=False)
		al2 = np.angle(hilbert(y2),deg=False)
		phase[:,i] = 1-np.sin(np.abs(al1-al2)/2)

	return torch.tensor(phase)