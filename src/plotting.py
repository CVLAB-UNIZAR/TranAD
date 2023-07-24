import matplotlib.pyplot as plt
import scienceplots
from matplotlib.backends.backend_pdf import PdfPages
import statistics
import os, torch
import numpy as np
from tqdm import tqdm

plt.style.use(['science', 'ieee'])
plt.rcParams["text.usetex"] = False
plt.rcParams['figure.figsize'] = 6, 2

os.makedirs('plots', exist_ok=True)

def smooth(y, box_pts=1):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def plotter(name, y_true, y_pred, ascore, labels):
	if 'TranAD' or 'TranCIRCE' in name: y_true = torch.roll(y_true, 1, 0)
	os.makedirs(os.path.join('plots', name), exist_ok=True)
	pdf = PdfPages(f'plots/{name}/{name}.pdf')
	time = np.arange(4000)/100
	for dim in tqdm(range(y_true.shape[1])):
	#for dim in range(y_true.shape[1]):
		y_t, y_p, l, a_s = y_true[:, dim].detach().numpy(), y_pred[:, dim].detach().numpy(), labels[:, dim], ascore[:, dim]
		fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
		ax1.set_ylabel('Value')
		ax1.set_title(f'Dimension = {dim}')
		# if dim == 0: np.save(f'true{dim}.npy', y_t); np.save(f'pred{dim}.npy', y_p); np.save(f'ascore{dim}.npy', a_s)
		ax1.plot(smooth(y_t), linewidth=0.2, label='True')
		ax1.plot(smooth(y_p), '-', alpha=0.6, linewidth=0.3, label='Predicted')
		ax3 = ax1.twinx()
		ax3.plot(l, '--', linewidth=0.3, alpha=0.5)
		ax3.fill_between(np.arange(l.shape[0]), l, color='blue', alpha=0.3)
		if dim == 0: ax1.legend(ncol=2, bbox_to_anchor=(0.6, 1.02))
		ax2.plot(smooth(a_s), linewidth=0.2, color='g')
		ax2.set_xlabel('Timestamp')
		ax2.set_ylabel('Anomaly Score')
		pdf.savefig(fig)
		plt.close()
	pdf.close()

def plotEspectrogramas(s1, s2):
	fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
	ax1.set_ylabel(f'Prefalta Data')
	ax1.set_title(f'Time')
	ax1.specgram((s1-s2)[:,0].data.cpu().numpy(), Fs=20, cmap="rainbow")
	ax2.set_ylabel(f'Falta')
	ax2.set_title(f'Time')
	ax2.specgram(s2[:,0].data.cpu().numpy(), Fs=20, cmap="rainbow")
	plt.show()

def plotterSiamese(name, y_true, y_pred, ascore, labels, score, umbral, code):
	if 'TranAD' or 'TranCIRCE' in name: y_true = torch.roll(y_true, 1, 0)
	pdf = PdfPages(f'plots/TransformerSiamesCirce_CIRCE/{name}.pdf')
	time = np.arange(4000)/100
	for dim in tqdm(range(y_true.shape[1])):
		#for dim in range(y_true.shape[1]):
		y_t, y_p, l, a_s, s = y_true[:, dim].data.cpu().numpy(), \
									 y_pred[:, dim].data.cpu().numpy(), \
									 labels[:, dim], \
									 ascore[:, dim].data.cpu().numpy(), \
									 score[:, dim].data.cpu().numpy()
		vUmbral = np.ones_like(s)*umbral[dim].numpy()
		fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
		ax1.set_ylabel('Value')
		ax1.set_title(f'Fault = {code} / Dimension = {dim} / Th = {umbral[dim].numpy()}')
		# if dim == 0: np.save(f'true{dim}.npy', y_t); np.save(f'pred{dim}.npy', y_p); np.save(f'ascore{dim}.npy', a_s)
		ax1.plot(smooth(y_t), linewidth=0.2, label='True')
		ax1.plot(smooth(y_p), '-', alpha=0.6, linewidth=0.3, label='Predicted')
		ax3 = ax1.twinx()
		ax3.plot(l, '--', linewidth=0.3, alpha=0.5)
		ax3.fill_between(np.arange(l.shape[0]), l, color='blue', alpha=0.3)
		if dim == 0: ax1.legend(ncol=2, bbox_to_anchor=(0.6, 1.02))
		#ax2.plot(smooth(a_s1), linewidth=0.2, color='g')
		ax2.plot(smooth(a_s), linewidth=0.2, color='b')
		ax2.plot(vUmbral, linewidth=0.2, color='r')
		ax2.set_xlabel('Timestamp')
		ax2.set_ylabel('Anomaly Score')
		ax4 = ax2.twinx()
		ax4.plot(s, '--', linewidth=0.3, alpha=0.5)
		ax4.fill_between(np.arange(s.shape[0]), s, color='blue', alpha=0.3)
		pdf.savefig(fig)
		plt.close()
	pdf.close()

def plotDiff(name, falta, prefalta, labels, idx = 0):
	if 'TranAD' or 'TranCIRCE' or 'OSContrastiveTransformer' in name: y_true = torch.roll(falta, 1, 0)
	os.makedirs(os.path.join('plots', name), exist_ok=True)
	pdf = PdfPages(f'plots/{name}/diff.pdf')
	for dim in tqdm(range(y_true.shape[1])):
		y_t, y_p, l = falta[:, dim], prefalta[:, dim], labels[:, dim]
		fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
		ax1.set_ylabel('Value')
		ax1.set_title(f'Dimension = {dim}')
		# if dim == 0: np.save(f'true{dim}.npy', y_t); np.save(f'pred{dim}.npy', y_p); np.save(f'ascore{dim}.npy', a_s)
		ax1.plot(y_t.detach().numpy(), linewidth=0.2, label='Falta')
		ax1.plot(y_p.detach().numpy(), '-', alpha=0.6, linewidth=0.3, label='Prefalta')
		ax3 = ax1.twinx()
		ax3.plot(l, '--', linewidth=0.3, alpha=0.5)
		ax3.fill_between(np.arange(l.shape[0]), l, color='blue', alpha=0.3)
		e_t = y_t - y_p
		ax2.plot(e_t.detach().numpy(), linewidth=0.2, color='g')
		ax2.set_xlabel('Timestamp')
		ax2.set_ylabel('Difference')
		pdf.savefig(fig)
		plt.close()
	pdf.close()
