import matplotlib.pyplot as plt
import os
from src.constants import *
import pandas as pd
import numpy as np
import torch
import scipy as sp
from scipy.spatial import distance
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

def plot_losses(loss_list, folder):
	os.makedirs(f'plots/{folder}/', exist_ok=True)
	plt.xlabel('Epochs')
	plt.ylabel('Average Training Loss')
	plt.plot(range(len(loss_list)), loss_list, label='Average Training Loss', linewidth=1, linestyle='-', marker='.')
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

def calc_correlation(actual, predic):
	actual = actual[0,:,:].data.cpu().numpy()
	predic = predic.data.cpu().numpy()

	a_diff = actual - np.mean(actual)
	p_diff = predic - np.mean(predic)
	numerator = (a_diff * p_diff)
	denominator = np.sqrt(np.sum(a_diff ** 2)) * np.sqrt(np.sum(p_diff ** 2))
	return numerator / denominator

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
	pdPrefalta = pd.DataFrame(prefalta[0,:,:].data.cpu().numpy())
	pdFalta = pd.DataFrame(falta[0,:,:].data.cpu().numpy())

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
	pdPrefalta = pd.DataFrame(prefalta[0, :, :].detach().numpy())
	pdFalta = pd.DataFrame(falta[0, :, :].detach().numpy())

	size = pdFalta.shape[0]

	energy = np.abs(pdFalta.pow(2).rolling(window=s).sum() - pdPrefalta.pow(2).rolling(window=s).sum())
	energy = energy.fillna(0).to_numpy().reshape([1, size, 3])

	return torch.tensor(energy)

def energy_pond(prefalta, falta, s, prod:10):
	pdPrefalta = pd.DataFrame(prefalta[0, :, :].detach().numpy())
	pdFalta = pd.DataFrame(falta[0, :, :].detach().numpy())

	size = pdFalta.shape[0]

	# Calculamos el máximo
	maximo = (pdPrefalta-0.5).abs().rolling(window=s*prod, min_periods=1).max()

	energy = np.abs(pdFalta.pow(2).rolling(window=s, min_periods=1).sum() -
					pdPrefalta.pow(2).rolling(window=s, min_periods=1).sum()) / maximo
	energy = energy.to_numpy().reshape([1, size, 3])

	return torch.tensor(energy)

def diference_ponderate(prefalta, falta):
	pdPrefalta = pd.DataFrame(prefalta[0,:,:].detach().numpy())
	pdFalta = pd.DataFrame(falta[0,:,:].detach().numpy())

	Prefaltadesplazmax = pdPrefalta

	Faltadesplazmax = pdFalta

	#Desplazamos los valores que están por debajo de 0.5 para que todos esten por encima
	umbral = 0.5
	Prefaltadesplazmax.loc[Prefaltadesplazmax[0] < umbral, 0] = (umbral - Prefaltadesplazmax[0])+umbral
	Prefaltadesplazmax.loc[Prefaltadesplazmax[1] < umbral, 1] = (umbral - Prefaltadesplazmax[1])+umbral
	Prefaltadesplazmax.loc[Prefaltadesplazmax[2] < umbral, 2] = (umbral - Prefaltadesplazmax[2])+umbral

	Faltadesplazmax.loc[Faltadesplazmax[0] < umbral, 0] = (umbral - Faltadesplazmax[0])+umbral
	Faltadesplazmax.loc[Faltadesplazmax[1] < umbral, 1] = (umbral - Faltadesplazmax[1])+umbral
	Faltadesplazmax.loc[Faltadesplazmax[2] < umbral, 2] = (umbral - Faltadesplazmax[2])+umbral

	#hacemos la diferencia entre las dos señales
	diff = np.abs(Prefaltadesplazmax - Faltadesplazmax)
	#calculamos la amplitud media entre las dos señales - Se hace la media elemento a elemento
	media = np.abs(Prefaltadesplazmax + Faltadesplazmax)/2
	#realzamos las medias a partir de un cierto umbral y realzamos la diferencia
	umbralmedia = 0.65
	media.loc[media[0] < umbralmedia, 0] = media[0]*100
	media.loc[media[1] < umbralmedia, 1] = media[1]*100
	media.loc[media[2] < umbralmedia, 2] = media[2]*100

	media.loc[media[0] > umbralmedia, 0] = media[0]*0.01
	media.loc[media[1] > umbralmedia, 1] = media[1]*0.01
	media.loc[media[2] > umbralmedia, 2] = media[2]*0.01

	diffponderate = (diff / media)

	umbraldiff = 7
	#Ponemos el resultado entre "0" y "1"
	diffponderate.loc[diffponderate[0] < umbraldiff, 0] = 0
	diffponderate.loc[diffponderate[1] < umbraldiff, 1] = 0
	diffponderate.loc[diffponderate[2] < umbraldiff, 2] = 0

	diffponderate.loc[diffponderate[0] >= umbraldiff, 0] = 1
	diffponderate.loc[diffponderate[1] >= umbraldiff, 1] = 1
	diffponderate.loc[diffponderate[2] >= umbraldiff, 2] = 1

	diffponderate = torch.tensor(diffponderate.values)
	return torch.tensor(diffponderate)

def compute_distance (prefalta, falta, metric):
	pdPrefalta = (pd.DataFrame(prefalta[0,:,:].detach().numpy()))
	pdFalta = pd.DataFrame(falta[0,:,:].detach().numpy())
	#eucliddist = np.zeros((4000,3))
	eucliddist = distance.cdist(pdPrefalta, pdFalta, 'metric', p=10)
	#eucliddist[1] = distance.pdist(pdPrefalta[1], pdFalta[1])
	#eucliddist[2] = distance.pdist(pdPrefalta[2], pdFalta[2])
	return torch.tensor(eucliddist)