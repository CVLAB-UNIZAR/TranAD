import matplotlib.pyplot as plt
import os
from src.constants import *
import pandas as pd
import numpy as np
import torch
from scipy.signal import hilbert, butter, filtfilt
import scipy as sp
from scipy.spatial import distance
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
	pdFalta = pd.DataFrame(falta.detach().numpy())

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

def dtw(s, t):
	n, m = len(s), len(t)
	dtw_matrix = np.zeros((n+1, m+1))
	for i in range(n+1):
		for j in range(m+1):
			dtw_matrix[i, j] = np.inf
	dtw_matrix[0, 0] = 0

	for i in range(1, n+1):
		for j in range(1, m+1):
			cost = abs(s[i-1] - t[j-1])
			# take last min from a square box
			last_min = np.min([dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1]])
			dtw_matrix[i, j] = cost + last_min
	return dtw_matrix

def energy(prefalta, falta, s):
	pdPrefalta = pd.DataFrame(prefalta[0,:,:].detach().numpy())
	pdFalta = pd.DataFrame(falta[0,:,:].detach().numpy())
	energy = np.zeros((4000,3))
	d1w = np.zeros(s)
	d2w = np.zeros(s)
	p1 = np.zeros((4000,3))
	p2 = np.zeros((4000,3))
	diffenergy = np.zeros((1,3))

	for m in range(4000-s):

		d1w = pdPrefalta[m:m+s-1]
		d2w = pdFalta[m:m+s-1]

		p1 = sp.sum(d1w*d1w)/d1w.size
		p2 = sp.sum(d2w*d2w)/d2w.size
		e1 = p1*d1w.size
		e2 = p2*d2w.size
		diffenergy = np.abs(e1-e2)
		energy[m] = diffenergy

	return torch.tensor(energy)

def diference_ponderate(prefalta, falta):
	pdPrefalta = pd.DataFrame(prefalta[0,:,:].detach().numpy())
	pdFalta = pd.DataFrame(falta[0,:,:].detach().numpy())
	Prefaltadesplazmax = np.zeros((4000,3))
	Prefaltadesplazmax = pdPrefalta
	Faltadesplazmax = np.zeros((4000,3))
	Faltadesplazmax = pdFalta
	diff = np.zeros((4000,3))
	media = np.zeros((4000,3))
	maximum = np.zeros((4000,3))
	diffponderate = np.zeros((4000,3))
	diffponderateumbral = np.zeros((4000,3))
	umbral = 0.5
	Prefaltadesplazmax.loc[Prefaltadesplazmax[0] < umbral, 0] = (umbral - Prefaltadesplazmax[0])+umbral
	Prefaltadesplazmax.loc[Prefaltadesplazmax[1] < umbral, 1] = (umbral - Prefaltadesplazmax[1])+umbral
	Prefaltadesplazmax.loc[Prefaltadesplazmax[2] < umbral, 2] = (umbral - Prefaltadesplazmax[2])+umbral

	Faltadesplazmax.loc[Faltadesplazmax[0] < umbral, 0] = (umbral - Faltadesplazmax[0])+umbral
	Faltadesplazmax.loc[Faltadesplazmax[1] < umbral, 1] = (umbral - Faltadesplazmax[1])+umbral
	Faltadesplazmax.loc[Faltadesplazmax[2] < umbral, 2] = (umbral - Faltadesplazmax[2])+umbral

	diff = np.abs((np.abs(Prefaltadesplazmax)) - (np.abs(Faltadesplazmax)))
	diffu = np.abs((np.abs(Prefaltadesplazmax)) - (np.abs(Faltadesplazmax)))
	umbraldiff = 0.04

	diff.loc[diff[0] < umbraldiff, 0] = 0
	diff.loc[diff[1] < umbraldiff, 1] = 0
	diff.loc[diff[2] < umbraldiff, 2] = 0

	diff.loc[diff[0] >= umbraldiff, 0] = 1
	diff.loc[diff[1] >= umbraldiff, 1] = 1
	diff.loc[diff[2] >= umbraldiff, 2] = 1

	media = np.abs(np.abs(Prefaltadesplazmax) + np.abs(Faltadesplazmax))/2

	diffponderate = (diff / media)
	diffponderateoriginal = torch.tensor(diffponderate.values)
	diffponderate = torch.tensor(diffponderate.values)
	s = 1
	p = 1
	umbralpond = 0.3

	for m0 in range(4000-s):
		if diffponderate[m0,0] < umbralpond:
			diffponderate[m0:m0+s,0] = 0
		else:
			diffponderate[m0:m0+p,0] = 1
	for m1 in range(4000-s):
		if diffponderate[m1,1] < umbralpond:
			diffponderate[m1:m1+s,1] = 0
		else:
			diffponderate[m1:m1+p,1] = 1
	for m2 in range(4000-s):
		if diffponderate[m2,2] < umbralpond:
			diffponderate[m2:m2+s,2] = 0
		else:
			diffponderate[m2:m2+p,2] = 1
	#return torch.tensor(diffponderate)
	return torch.tensor(diffponderateoriginal)

def compute_distance (prefalta, falta, metric):
	pdPrefalta = (pd.DataFrame(prefalta[0,:,:].detach().numpy()))
	pdFalta = pd.DataFrame(falta[0,:,:].detach().numpy())
	#eucliddist = np.zeros((4000,3))
	eucliddist = distance.cdist(pdPrefalta, pdFalta, 'metric', p=10)
	#eucliddist[1] = distance.pdist(pdPrefalta[1], pdFalta[1])
	#eucliddist[2] = distance.pdist(pdPrefalta[2], pdFalta[2])
	return torch.tensor(eucliddist)