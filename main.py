import pickle
import os

import numpy as np
import pandas as pd
import torch.onnx
from tqdm import tqdm
from src.models import *
from src.constants import *
from src.plotting import *
from src.pot import *
from src.utils import *
from src.diagnosis import *
from src.merlin import *
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn
from time import time
from pprint import pprint
# from beepy import beep
from torchviz import make_dot
import dagshub
import random

from src.data import SiameseDataset
from src.contrastiveLoss import ContrastiveLoss, ContrastiveLossFF
import torch.nn.functional as Funct
import scipy.integrate as it

import os

os.environ["PATH"] += os.pathsep + 'C:/Users/jelia/anaconda3/envs/GANs/Library/bin/graphviz/'


def convert_to_windows(data, model):
    windows = [];
    w_size = model.n_window
    for i, g in enumerate(data):
        if i >= w_size:
            w = data[i - w_size:i]
        else:
            w = torch.cat([data[0].repeat(w_size - i, 1), data[0:i]])
        windows.append(
            w if 'TranAD' or 'TranCIRCE' or 'OSContrastiveTransformer' in args.model or 'Attention' in args.model else w.view(
                -1))
    return torch.stack(windows)


def load_dataset(dataset, idx):
    folder = os.path.join(output_folder, dataset)
    if not os.path.exists(folder):
        raise Exception('Processed Data not found.')
    loader = []
    for file in ['train', 'test', 'labels']:
        if dataset == 'SMD': file = 'machine-1-1_' + file
        if dataset == 'SMAP': file = 'P-1_' + file
        if dataset == 'MSL': file = 'C-1_' + file
        if dataset == 'UCR': file = '136_' + file
        if dataset == 'NAB': file = 'ec2_request_latency_system_failure_' + file
        if dataset == 'CIRCE': file = 'CIRCE_' + file
        loader.append(np.load(os.path.join(folder, f'{file}.npy')))
    # loader = [i[:, debug:debug+1] for i in loader]
    if args.less: loader[0] = cut_array(0.2, loader[0])
    train_loader = DataLoader(loader[0], batch_size=loader[0].shape[0])
    if dataset == 'CIRCE':
        random_idx = random.randint(0, 199)
        test_loader = DataLoader(loader[1][random_idx, :, :], batch_size=loader[1].shape[1])
        labels = loader[2][random_idx, :, :]

        test_loader_test = DataLoader(loader[1][idx, :, :], batch_size=loader[1].shape[1])
        labels_test = loader[2][idx, :, :]
    else:
        test_loader = DataLoader(loader[1][idx, :, :], batch_size=loader[1].shape[0])
        labels = loader[2]

    return train_loader, test_loader, labels, test_loader_test, labels_test


def save_model(model, optimizer, scheduler, epoch, accuracy_list, optimizer2=None, scheduler2=None):
    folder = f'checkpoints/{args.model}_{args.dataset}/'
    os.makedirs(folder, exist_ok=True)
    file_path = f'{folder}/model.ckpt'
    if optimizer2 is not None:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer1.state_dict(),
            'optimizer2_state_dict': optimizer2.state_dict(),
            'scheduler_state_dict': scheduler1.state_dict(),
            'scheduler2_state_dict': scheduler2.state_dict(),
            'accuracy_list': accuracy_list}, file_path)
    else:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'accuracy_list': accuracy_list}, file_path)


def load_model(modelname, dims):
    import src.models
    model_class = getattr(src.models, modelname)
    model = model_class(dims).double()
    optimizer = torch.optim.AdamW(model.parameters(), lr=model.lr, weight_decay=1e-5)
    #optimContrastive = torch.optim.RMSprop(model.parameters, lr=model.lr, alpha=0.99, eps=1e-8, weight_decay=0.0005, momentum=0.9)
    optimContrastive = optim.RMSprop(model.parameters(), lr=1e-4, alpha=0.99, eps=1e-8, weight_decay=0.0005, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
    fname = f'checkpoints/{args.model}_{args.dataset}/model.ckpt'
    if os.path.exists(fname) and (not args.retrain or args.test):
        print(f"{color.GREEN}Loading pre-trained model: {model.name}{color.ENDC}")
        checkpoint = torch.load(fname)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
        accuracy_list = checkpoint['accuracy_list']
    else:
        print(f"{color.GREEN}Creating new model: {model.name}{color.ENDC}")
        epoch = -1;
        accuracy_list = []
    return model, optimizer, optimContrastive, scheduler, epoch, accuracy_list


def backprop(epoch, model, data, dataO,
             optimizer,
             scheduler,
             training=True,
             dataTest=None,
             fase=1,
             optimizer2=None,
             scheduler2=None):
    l = nn.MSELoss(reduction='mean' if training else 'none')
    feats = dataO.shape[1]
    if 'DAGMM' in model.name:
        l = nn.MSELoss(reduction='none')
        compute = ComputeLoss(model, 0.1, 0.005, 'cpu', model.n_gmm)
        n = epoch + 1;
        w_size = model.n_window
        l1s = [];
        l2s = []
        if training:
            for d in data:
                _, x_hat, z, gamma = model(d)
                l1, l2 = l(x_hat, d), l(gamma, d)
                l1s.append(torch.mean(l1).item());
                l2s.append(torch.mean(l2).item())
                loss = torch.mean(l1) + torch.mean(l2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
            tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)},\tL2 = {np.mean(l2s)}')
            return np.mean(l1s) + np.mean(l2s), optimizer.param_groups[0]['lr']
        else:
            ae1s = []
            for d in data:
                _, x_hat, _, _ = model(d)
                ae1s.append(x_hat)
            ae1s = torch.stack(ae1s)
            y_pred = ae1s[:, data.shape[1] - feats:data.shape[1]].view(-1, feats)
            loss = l(ae1s, data)[:, data.shape[1] - feats:data.shape[1]].view(-1, feats)
            return loss.detach().numpy(), y_pred.detach().numpy()
    if 'Attention' in model.name:
        l = nn.MSELoss(reduction='none')
        n = epoch + 1;
        w_size = model.n_window
        l1s = [];
        res = []
        if training:
            for d in data:
                ae, ats = model(d)
                # res.append(torch.mean(ats, axis=0).view(-1))
                l1 = l(ae, d)
                l1s.append(torch.mean(l1).item())
                loss = torch.mean(l1)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # res = torch.stack(res); np.save('ascores.npy', res.detach().numpy())
            scheduler.step()
            tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
            return np.mean(l1s), optimizer.param_groups[0]['lr']
        else:
            ae1s, y_pred = [], []
            for d in data:
                ae1 = model(d)
                y_pred.append(ae1[-1])
                ae1s.append(ae1)
            ae1s, y_pred = torch.stack(ae1s), torch.stack(y_pred)
            loss = torch.mean(l(ae1s, data), axis=1)
            return loss.detach().numpy(), y_pred.detach().numpy()
    elif 'OmniAnomaly' in model.name:
        if training:
            mses, klds = [], []
            for i, d in enumerate(data):
                y_pred, mu, logvar, hidden = model(d, hidden if i else None)
                MSE = l(y_pred, d)
                KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=0)
                loss = MSE + model.beta * KLD
                mses.append(torch.mean(MSE).item());
                klds.append(model.beta * torch.mean(KLD).item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            tqdm.write(f'Epoch {epoch},\tMSE = {np.mean(mses)},\tKLD = {np.mean(klds)}')
            scheduler.step()
            return loss.item(), optimizer.param_groups[0]['lr']
        else:
            y_preds = []
            for i, d in enumerate(data):
                y_pred, _, _, hidden = model(d, hidden if i else None)
                y_preds.append(y_pred)
            y_pred = torch.stack(y_preds)
            MSE = l(y_pred, data)
            return MSE.detach().numpy(), y_pred.detach().numpy()
    elif 'USAD' in model.name:
        l = nn.MSELoss(reduction='none')
        n = epoch + 1;
        w_size = model.n_window
        l1s, l2s = [], []
        if training:
            for d in data:
                ae1s, ae2s, ae2ae1s = model(d)
                l1 = (1 / n) * l(ae1s, d) + (1 - 1 / n) * l(ae2ae1s, d)
                l2 = (1 / n) * l(ae2s, d) - (1 - 1 / n) * l(ae2ae1s, d)
                l1s.append(torch.mean(l1).item());
                l2s.append(torch.mean(l2).item())
                loss = torch.mean(l1 + l2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
            tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)},\tL2 = {np.mean(l2s)}')
            return np.mean(l1s) + np.mean(l2s), optimizer.param_groups[0]['lr']
        else:
            ae1s, ae2s, ae2ae1s = [], [], []
            for d in data:
                ae1, ae2, ae2ae1 = model(d)
                ae1s.append(ae1);
                ae2s.append(ae2);
                ae2ae1s.append(ae2ae1)
            ae1s, ae2s, ae2ae1s = torch.stack(ae1s), torch.stack(ae2s), torch.stack(ae2ae1s)
            y_pred = ae1s[:, data.shape[1] - feats:data.shape[1]].view(-1, feats)
            loss = 0.1 * l(ae1s, data) + 0.9 * l(ae2ae1s, data)
            loss = loss[:, data.shape[1] - feats:data.shape[1]].view(-1, feats)
            return loss.detach().numpy(), y_pred.detach().numpy()
    elif model.name in ['GDN', 'MTAD_GAT', 'MSCRED', 'CAE_M']:
        l = nn.MSELoss(reduction='none')
        n = epoch + 1;
        w_size = model.n_window
        l1s = []
        if training:
            for i, d in enumerate(data):
                if 'MTAD_GAT' in model.name:
                    x, h = model(d, h if i else None)
                else:
                    x = model(d)
                loss = torch.mean(l(x, d))
                l1s.append(torch.mean(loss).item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            tqdm.write(f'Epoch {epoch},\tMSE = {np.mean(l1s)}')
            return np.mean(l1s), optimizer.param_groups[0]['lr']
        else:
            xs = []
            for d in data:
                if 'MTAD_GAT' in model.name:
                    x, h = model(d, None)
                else:
                    x = model(d)
                xs.append(x)
            xs = torch.stack(xs)
            y_pred = xs[:, data.shape[1] - feats:data.shape[1]].view(-1, feats)
            loss = l(xs, data)
            loss = loss[:, data.shape[1] - feats:data.shape[1]].view(-1, feats)
            return loss.detach().numpy(), y_pred.detach().numpy()
    elif 'GAN' in model.name:
        l = nn.MSELoss(reduction='none')
        bcel = nn.BCELoss(reduction='mean')
        msel = nn.MSELoss(reduction='mean')
        real_label, fake_label = torch.tensor([0.9]), torch.tensor([0.1])  # label smoothing
        real_label, fake_label = real_label.type(torch.DoubleTensor), fake_label.type(torch.DoubleTensor)
        n = epoch + 1;
        w_size = model.n_window
        mses, gls, dls = [], [], []
        if training:
            for d in data:
                # training discriminator
                model.discriminator.zero_grad()
                _, real, fake = model(d)
                dl = bcel(real, real_label) + bcel(fake, fake_label)
                dl.backward()
                model.generator.zero_grad()
                optimizer.step()
                # training generator
                z, _, fake = model(d)
                mse = msel(z, d)
                gl = bcel(fake, real_label)
                tl = gl + mse
                tl.backward()
                model.discriminator.zero_grad()
                optimizer.step()
                mses.append(mse.item());
                gls.append(gl.item());
                dls.append(dl.item())
            # tqdm.write(f'Epoch {epoch},\tMSE = {mse},\tG = {gl},\tD = {dl}')
            tqdm.write(f'Epoch {epoch},\tMSE = {np.mean(mses)},\tG = {np.mean(gls)},\tD = {np.mean(dls)}')
            return np.mean(gls) + np.mean(dls), optimizer.param_groups[0]['lr']
        else:
            outputs = []
            for d in data:
                z, _, _ = model(d)
                outputs.append(z)
            outputs = torch.stack(outputs)
            y_pred = outputs[:, data.shape[1] - feats:data.shape[1]].view(-1, feats)
            loss = l(outputs, data)
            loss = loss[:, data.shape[1] - feats:data.shape[1]].view(-1, feats)
            return loss.detach().numpy(), y_pred.detach().numpy()
    elif 'TranAD' in model.name:
        l = nn.MSELoss(reduction='none')
        data_x = torch.DoubleTensor(data);
        dataset = TensorDataset(data_x, data_x)
        bs = model.batch if training else len(data)
        dataloader = DataLoader(dataset, batch_size=bs)
        n = epoch + 1;
        w_size = model.n_window
        l1s, l2s = [], []
        if training:
            for d, _ in dataloader:
                local_bs = d.shape[0]
                window = d.permute(1, 0, 2)
                elem = window[-1, :, :].view(1, local_bs, feats)
                z = model(window, elem)
                # g = make_dot(z, params=dict(model.named_parameters()), show_saved=True).render("tranAD", format="png")
                l1 = l(z, elem) if not isinstance(z, tuple) else (1 / n) * l(z[0], elem) + (1 - 1 / n) * l(z[1], elem)
                if isinstance(z, tuple): z = z[1]
                l1s.append(torch.mean(l1).item())
                loss = torch.mean(l1)
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
            scheduler.step()
            tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
            return np.mean(l1s), optimizer.param_groups[0]['lr']
        else:
            for d, _ in dataloader:
                window = d.permute(1, 0, 2)
                elem = window[-1, :, :].view(1, bs, feats)
                z = model(window, elem)
                if isinstance(z, tuple): z = z[1]
            loss = l(z, elem)[0]
            return loss.detach().numpy(), z.detach().numpy()[0]
    elif 'TranCIRCE' in model.name:
        l = nn.MSELoss(reduction='none')
        data_x = torch.DoubleTensor(data);
        dataset = TensorDataset(data_x, data_x)
        if dataTest is not None:
            data_test_x = torch.DoubleTensor(dataTest);
            dataset_test = TensorDataset(data_test_x, data_test_x)
        bs = model.batch if training else len(data)
        dataloader = DataLoader(dataset, batch_size=bs)
        if dataTest is not None:
            dataloader_test = DataLoader(dataset_test, batch_size=bs)
        n = epoch + 1;
        w_size = model.n_window
        l1s, l2s = [], []
        if training:
            if dataTest is not None:
                for d1, d2 in zip(dataloader, dataloader_test):
                    d1 = d1[0]

                    local_bs = d1.shape[0]
                    window = d1.permute(1, 0, 2)
                    elem = window[-1, :, :].view(1, local_bs, feats)
                    z = model(window, elem, 0)
                    l1 = l(z, elem) if not isinstance(z, tuple) else (1 / n) * l(z[0], elem) + (1 - 1 / n) * l(z[1],
                                                                                                               elem)
                    if isinstance(z, tuple): z = z[1]
                    l1s.append(torch.mean(l1).item())
                    loss = torch.mean(l1)
                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer.step()

                    d1 = d2[0]
                    local_bs = d1.shape[0]
                    window = d1.permute(1, 0, 2)
                    elem = window[-1, :, :].view(1, local_bs, feats)
                    z = model(window, elem, 1)
                    l1 = l(z, elem) if not isinstance(z, tuple) else (1 / n) * l(z[0], elem) + (1 - 1 / n) * l(z[1],
                                                                                                               elem)
                    if isinstance(z, tuple): z = z[1]
                    l1s.append(torch.mean(l1).item())
                    loss = torch.mean(l1)
                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer.step()
            else:
                for d1, _ in dataloader:
                    local_bs = d1.shape[0]
                    window = d1.permute(1, 0, 2)
                    elem = window[-1, :, :].view(1, local_bs, feats)
                    z = model(window, elem, 0)
                    l1 = l(z, elem) if not isinstance(z, tuple) else (1 / n) * l(z[0], elem) + (1 - 1 / n) * l(z[1],
                                                                                                               elem)
                    if isinstance(z, tuple): z = z[1]
                    l1s.append(torch.mean(l1).item())
                    loss = torch.mean(l1)
                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer.step()

            scheduler.step()
            tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
            return np.mean(l1s), optimizer.param_groups[0]['lr']
        else:
            for d, _ in dataloader:
                window = d.permute(1, 0, 2)
                elem = window[-1, :, :].view(1, bs, feats)
                z = model(window, elem)
                if isinstance(z, tuple): z = z[1]

            with torch.no_grad():
                plotDiff(f'.', torch.abs(z - 0.5)[0, :, :], torch.abs(dataO - 0.5), labels)
            loss = l(z, dataO)[0]  # 0.5*l(z[0], dataO)[0] + 0.5*l(z[1], dataO)[0]
            return loss.detach().numpy(), z.detach().numpy()[0]
    elif 'OSContrastiveTransformer' in model.name:
        loss1 = nn.MSELoss(reduction='none')
        loss2 = nn.MSELoss(reduction='none')
        data_x = torch.DoubleTensor(data);
        dataset = TensorDataset(data_x, data_x)
        if dataTest is not None:
            data_test_x = torch.DoubleTensor(dataTest);
            dataset_test = TensorDataset(data_test_x, data_test_x)
        bs = model.batch if training else len(data)
        dataloader = DataLoader(dataset, batch_size=bs)
        if dataTest is not None:
            dataloader_test = DataLoader(dataset_test, batch_size=bs)
        n = epoch + 1;
        w_size = model.n_window
        l1s, l2s = [], []
        if training:
            if dataTest is not None:
                if fase == 1:
                    # entrenamos la fase con datos de prefalta
                    for d in dataloader:
                        local_bs = d[0].shape[0]
                        window = d[0].permute(1, 0, 2)
                        elem = window[-1, :, :].view(1, local_bs, feats)
                        z = model(window, elem, 1)
                        l1 = torch.mean(loss1(z, elem)[0])
                        l1s.append(l1.item())
                        optimizer1.zero_grad()
                        l1.backward(retain_graph=True)
                        optimizer1.step()
                    scheduler1.step()
                    tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
                    return np.mean(l1s), optimizer.param_groups[0]['lr']
                else:
                    for d in dataloader_test:
                        local_bs = d[0].shape[0]
                        vF = d[0].permute(1, 0, 2)
                        F = vF[-1, :, :].view(1, local_bs, feats)
                        z = model(vF, F, 2)
                        l2 = torch.mean(loss2(z, F)[0])
                        l2s.append(l2.item())
                        optimizer2.zero_grad()
                        l2.backward(retain_graph=True)
                        optimizer2.step()

                    scheduler2.step()
                    tqdm.write(f'Epoch {epoch}, L2 = {np.mean(l2s)}')
                    return np.mean(l2s), optimizer.param_groups[0]['lr']
            else:
                for d1, _ in dataloader:
                    local_bs = d1.shape[0]
                    window = d1.permute(1, 0, 2)
                    elem = window[-1, :, :].view(1, local_bs, feats)
                    z = model(window, elem, 0)
                    l1 = l(z, elem) if not isinstance(z, tuple) else (1 / n) * l(z[0], elem) + (1 - 1 / n) * l(z[1],
                                                                                                               elem)
                    if isinstance(z, tuple): z = z[1]
                    l1s.append(torch.mean(l1).item())
                    loss = torch.mean(l1)
                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer.step()

            scheduler1.step()
            scheduler2.step()
            tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}, \tL2 = {np.mean(l2s)}')
            return np.mean(l1s), optimizer.param_groups[0]['lr']
        else:
            for d1, d2 in zip(dataloader, dataloader_test):
                d1 = d1[0]
                window1 = d1.permute(1, 0, 2)
                vPF = window1[-1, :, :].view(1, bs, feats)
                z1 = model(window1, vPF, 0)
                d2 = d2[0]
                window2 = d2.permute(1, 0, 2)
                elem = window2[-1, :, :].view(1, bs, feats)
                z2 = model(window2, elem, 1)
            # with torch.no_grad():
            #	plotDiff(f'.', z1[0], z1[1], labels)

            loss = phase_syncrony(z1, z2)
            # loss = l(z, z1[0,:,:])[0]
            return loss.detach().numpy(), (z1, z2)

    elif 'TransformerSiamesCirce' in model.name:
        loss = nn.MSELoss(reduction='none')
        data_x = torch.DoubleTensor(data);
        dataset = TensorDataset(data_x, data_x)
        if dataTest is not None:
            data_test_x = torch.DoubleTensor(dataTest);
            dataset_test = TensorDataset(data_test_x, data_test_x)
        bs = model.batch if training else len(data)
        dataloader = DataLoader(dataset, batch_size=bs)
        if dataTest is not None:
            dataloader_test = DataLoader(dataset_test, batch_size=bs)
        n = epoch + 1;
        w_size = model.n_window
        l1s, l2s = [], []
        if training:

            scheduler1.step()
            scheduler2.step()
            tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}, \tL2 = {np.mean(l2s)}')
            return np.mean(l1s), optimizer.param_groups[0]['lr']
        else:
            for d1, d2 in zip(dataloader, dataloader_test):
                d1 = d1[0]
                window1 = d1.permute(1, 0, 2)
                vPF = window1[-1, :, :].view(1, bs, feats)
                z1 = model(window1, vPF, 0)
                d2 = d2[0]
                window2 = d2.permute(1, 0, 2)
                elem = window2[-1, :, :].view(1, bs, feats)
                z2 = model(window2, elem, 1)
            # with torch.no_grad():
            #	plotDiff(f'.', z1[0], z1[1], labels)

            loss = phase_syncrony(z1, z2)
            # loss = l(z, z1[0,:,:])[0]
            return loss.detach().numpy(), (z1, z2)
    else:
        y_pred = model(data)
        loss = l(y_pred, data)
        if training:
            tqdm.write(f'Epoch {epoch},\tMSE = {loss}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            return loss.item(), optimizer.param_groups[0]['lr']
        else:
            return loss.detach().numpy(), y_pred.detach().numpy()


def train_siamese(epoch, model, data, optimizer, scheduler, device='cuda'):
    # dataLD[0..2]
    # dataLD[0][batch, 4000, 3]
    loss = ContrastiveLoss(gamma=1.2, margin=2).to(device)

    dataLD = DataLoader(data, batch_size=8, shuffle=True)

    ls=[]
    for d in dataLD:
        for i in range(d[0].shape[0]):
            pF = d[0][i].to(device)
            F = d[1][i].to(device)
            l = d[2][i].to(device)
            similar = d[3][i].to(device)

            vPF, vF = convert_to_windows(pF, model), convert_to_windows(F, model)

            local_bs1 = vPF.shape[0]
            window1 = vPF.permute(1, 0, 2)
            elem1 = window1[-1, :, :].view(1, local_bs1, 3)

            local_bs2 = vF.shape[0]
            window2 = vF.permute(1, 0, 2)
            elem2 = window2[-1, :, :].view(1, local_bs2, 3)

            optimizer.zero_grad()
            output1,output2 = model(window1, elem1, window2, elem2)

            loss_contrastive = loss(output1, output2, l, similar)
            ls.append(loss_contrastive.item())
            loss_contrastive.backward()
            optimizer.step()

    tqdm.write(f'Epoch {epoch}, L = {np.mean(ls)}')
    return np.mean(ls)

def inference_siamese(epoch, model, data, optimizer, scheduler, threshold=0.004, device='cuda', unique=True, prod=10):
    # dataLD[0..2]
    # dataLD[0][batch, 4000, 3]
    loss = ContrastiveLoss(gamma=1, margin=1)

    pF = torch.tensor(data[epoch][0]).to(device)
    F = torch.tensor(data[epoch][1]).to(device)
    l = torch.tensor(data[epoch][2]).to(device)
    similar = torch.tensor(data[epoch][3]).to(device)

    vPF, vF = convert_to_windows(pF, model), convert_to_windows(F, model)

    local_bs1 = vPF.shape[0]
    window1 = vPF.permute(1, 0, 2)
    elem1 = window1[-1, :, :].view(1, local_bs1, 3)

    local_bs2 = vF.shape[0]
    window2 = vF.permute(1, 0, 2)
    elem2 = window2[-1, :, :].view(1, local_bs2, 3)

    optimizer.zero_grad()
    output1, output2 = model(window1, elem1, window2, elem2)

    #loss = calc_correlation(output1, output2)

    #loss1 = 1 - phase_syncrony(output1, output2)
    #loss2 = torch.abs((output1 - output2) / output1) + torch.sqrt((output1 - output2)**2)

    loss2 = energy_pond(output1.cpu(), output2.cpu(), 10, prod=prod)

    #loss2 = it.cumtrapz(loss2.data.cpu().numpy(), initial=0.0)

    if unique:
        score = (loss2 > threshold)*1.0
    else:
        score = torch.from_numpy(np.zeros_like(output1.cpu().detach().numpy()))
        for i in range(threshold.shape[0]):
            if loss2.is_cuda:
                score[0, :, i] = loss2[:, :, i].cpu() > threshold[i]
            else:
                score[0, :, i] = loss2[:, :, i] > threshold[i]

    return loss2[0], (output1, output2), score[0]


def CIRCE_mode():
    # def backprop(epoch, model, data, dataO,
    # 			 optimizer, optimizer2,
    # 			 scheduler1, scheduler2,
    # 			 training = True,
    # 			 dataTest = None,
    # 			 fase = 1):

    # 1. Prepare data
    data = SiameseDataset('processed/CIRCE/faltas_1', 'data/CIRCE/ResumenBloqueSimulaciones1-200.csv', './',
                          mode='train')
    data_test = SiameseDataset('processed/CIRCE/faltas_1', 'data/CIRCE/ResumenBloqueSimulaciones1-200.csv', './',
                               mode='_test_E')
    model, optimizer, optimContrastive, scheduler, epoch, accuracy_list = load_model(args.model, data.faltas.shape[2])

    model = model.to('cuda')

    # 2. Training phase
    loss_list = []
    if not args.test:
        epoch = 0
        print(f'{color.HEADER}Training {args.model} on {args.dataset}{color.ENDC}')
        num_epochs = args.epochs;
        e = epoch + 1;
        start = time()
        for e in tqdm(list(range(epoch + 1, epoch + num_epochs + 1))):
            loss = train_siamese(e, model, data, optimizer=optimContrastive, scheduler=scheduler)
            loss_list.append(loss)
        save_model(model,optimizer,scheduler,e, loss_list)
        plot_losses(loss_list, f'{args.model}_{args.dataset}/TrainWithTest')

    # 3. Testing phase
    if args.test:
        torch.zero_grad = True
        model.eval()
        df = pd.DataFrame()
        for item in range(len(data_test)):
            print(f'{color.HEADER}First iteration: Sample {item}{color.ENDC}')
            lossT, (x1, x2), score = inference_siamese(item, model, data_test, threshold=0.008240703442447072,
                                                       optimizer=optimContrastive, scheduler=scheduler)


            # 4. Statistics
            print(f'{color.HEADER}Computing threshold ...{color.ENDC}')
            df1 = pd.DataFrame()
            for canal in range(score.shape[1]):
                lt = lossT.data.cpu().numpy()[:, canal]
                l, ls =np.zeros_like(lt), data_test[item][2][:,canal]

                dfDatos, th, pos = compute_threshold(lt, ls, levels=20)
                df1 = pd.concat([df1, pd.DataFrame([th])], ignore_index=True)

            if args.prod == 'TRUE':
                th = torch.from_numpy(df1.to_numpy())
                for mult in range(19):
                    print(f'{color.HEADER}Executing with the optimum threshold {th}...{color.ENDC}')
                    lossT, (x1, x2), score = inference_siamese(item, model, data_test, threshold=th,
                                                               optimizer=optimContrastive, scheduler=scheduler,
                                                               unique=False, prod=mult+1)

                    print(f'{color.HEADER}Grabbing data for sample {item}...{color.ENDC}')
                    for canal in range(score.shape[1]):
                        result, pred = pot_eval_siamese(score.data.cpu().numpy()[:, canal],
                                                        data_test[item][2][:, 0],
                                                        pot_th=th,
                                                        item=item,
                                                        code=data_test[item][4])
                        df = pd.concat([df, pd.DataFrame([result])], ignore_index=True)

                print(f'{color.BLUE}-----------------------------{color.ENDC}')
                df.to_csv('plots/TransformerSiamesCirce_CIRCE/statsE.csv')

            else:
                th = torch.from_numpy(df1.to_numpy())
                print(f'{color.HEADER}Executing with the optimum threshold {th}...{color.ENDC}')
                lossT, (x1, x2), score = inference_siamese(item, model, data_test, threshold=th,
                                                           optimizer=optimContrastive, scheduler=scheduler, unique=False)

                # 5. Plot curves
                print(f'{color.HEADER}Printing curves...{color.ENDC}')
                if args.test:
                    #plotEspectrogramas(x1[0], x2[0])
                    plotterSiamese(f'{args.model}_{args.dataset}_{item}', x1[0], x2[0], lossT,
                                   data_test[item][2], score, th, data_test[item][4])

                # 6. Grabbing data
                print(f'{color.HEADER}Grabbing data for sample {item}...{color.ENDC}')
                for canal in range(score.shape[1]):
                    result, pred = pot_eval_siamese(score.data.cpu().numpy()[:, canal],
                                                    data_test[item][2][:, 0],
                                                    pot_th=th,
                                                    item=item,
                                                    code=data_test[item][4])
                    df = pd.concat([df, pd.DataFrame([result])], ignore_index=True)

                print(f'{color.BLUE}-----------------------------{color.ENDC}')

        print(f'{color.GREEN}Grabbing ALL DATA...{color.ENDC}')
        df.to_csv('plots/TransformerSiamesCirce_CIRCE/stats.csv')


def TranAD_mode():
    a=1


if __name__ == '__main__':
    if args.modo == 'CIRCE':
        CIRCE_mode()
    if args.modo == 'TranAD':
        TranAD_mode()
