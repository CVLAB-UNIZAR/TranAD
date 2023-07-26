import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class SiameseDataset(Dataset):
    def __init__(self, csv_file, csv_resumen_faltas, root_dir, transform=None, mode='train'):
        if mode == 'train':
            file = csv_file + '.csv'
        else:
            file = csv_file + mode + '.csv'

        self.data = pd.read_csv(file)
        self.resumen = pd.read_csv(csv_resumen_faltas)
        if mode == 'train':
            self.data.columns = ["signal 1", "signal 2", "label"]  # Label=0 cuando son iguales
        else:
            self.data.columns = ["signal 1", "signal 2", "label", "threshold"]  # Label=0 cuando son iguales
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode

        self.pre_falta = np.load('processed/CIRCE/CIRCE_train.npy')
        self.faltas = np.load('processed/CIRCE/CIRCE_test.npy')
        self.labels = np.load('processed/CIRCE/CIRCE_labels.npy')

        self.trainDL = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        s1 = self.data.iat[index, 0]
        s2 = self.data.iat[index, 1]
        l = self.data.iat[index, 2]
        if self.mode != 'train':
            th = self.data.iat[index, 3]

        # Lectura de prefalta y faltas
        PF = self.pre_falta
        if s2 == 0:
            F = self.pre_falta
        else:
            F = self.faltas[s2 - 1]

        # Lectura de etiquetas
        if s2 == 0:
            # Generamos una etiqueta con ceros
            labels = np.zeros_like(self.labels[0])
        else:
            # Generamos una etiqueta con un escalón en el momento de la falta
            labels = self.labels[s2 - 1]

        if self.mode != 'train':
            return PF, F, labels, l, th
        else:
            return PF, F, labels, l

    def getDataLoader(self, batch_size):
        self.trainDL = DataLoader(self,
                                  shuffle=True,
                                  batch_size=batch_size)
        return self.trainDL

class StandardDataSet(Dataset):
    def __init__(self, csv_file, csv_resumen_faltas, root_dir, transform=None, mode='train'):
        if mode == 'train':
            file = csv_file + '.csv'
        else:
            file = csv_file + mode + '.csv'

        self.data = pd.read_csv(file)
        self.resumen = pd.read_csv(csv_resumen_faltas)
        if mode == 'train':
            self.data.columns = ["signal 1", "signal 2", "label"]  # Label=0 cuando son iguales
        else:
            self.data.columns = ["signal 1", "signal 2", "label", "threshold"]  # Label=0 cuando son iguales

        if mode == 'train':
            self.data = self.data.loc[self.data['label'] == 1]['signal 2'].iloc[0] # Guardamos sólo la columna de la segunda señal
        # cuando son señales distintas a PF
        else:
            self.data = self.data.loc[self.data['label'] == 1]['signal 2']
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode

        self.pre_falta = np.load('processed/CIRCE/CIRCE_train.npy')
        self.faltas = np.load('processed/CIRCE/CIRCE_test.npy')
        self.faltas = self.faltas[self.data]
        self.labels = np.load('processed/CIRCE/CIRCE_labels.npy')
        self.labels = self.labels[self.data]

        if mode == "train":
            self.dims = self.faltas.shape[1]
        else:
            self.dims = self.faltas.shape[2]

        self.trainDL = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        s1 = self.data.iat[index, 0]
        s2 = self.data.iat[index, 1]
        l = self.data.iat[index, 2]
        if self.mode != 'train':
            th = self.data.iat[index, 3]

        # Lectura de prefalta y faltas
        PF = self.pre_falta
        if s2 == 0:
            F = self.pre_falta
        else:
            F = self.faltas[s2 - 1]

        # Lectura de etiquetas
        if s2 == 0:
            # Generamos una etiqueta con ceros
            labels = np.zeros_like(self.labels[0])
        else:
            # Generamos una etiqueta con un escalón en el momento de la falta
            labels = self.labels[s2 - 1]

        if self.mode != 'train':
            return PF, F, labels, l, th
        else:
            return PF, F, labels, l

    def getDataLoader(self, batch_size):
        self.trainDL = DataLoader(self,
                                  shuffle=True,
                                  batch_size=batch_size)
        return self.trainDL