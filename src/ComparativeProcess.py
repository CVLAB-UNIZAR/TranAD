from src.data import StandardDataSet
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from src.utils import *

class ComparativeProcess(object):
    def __init__(self, faltas_file: str, simu_csv_file: str) -> None:
        super().__init__()
        self.faltas_file = faltas_file
        self.simu_csv_file = simu_csv_file

        self.data = StandardDataSet('processed/CIRCE/faltas_1', 'data/CIRCE/ResumenBloqueSimulaciones1-200.csv', './',
                              mode='train')
        self.data_test = StandardDataSet('processed/CIRCE/faltas_1', 'data/CIRCE/ResumenBloqueSimulaciones1-200.csv', './',
                               mode='_test_1')
        self.model, self.optimizer, self.scheduler, self.epoch, self.accuracy_list = self.load_model(args.model,
                                                                                                     self.data.dims)
        self.trainD = None
        self.trainO = None
        self.testD = None
        self.testO = None

    def prepare_data(self):
        self.train_loader = DataLoader(self.data.faltas, batch_size=self.data.faltas.shape[0])

        self.trainD = next(iter(self.train_loader))
        self.trainO = self.trainD
        if self.model.name in ['Attention', 'DAGMM', 'USAD', 'MSCRED', 'CAE_M', 'GDN', 'MTAD_GAT', 'MAD_GAN'] or 'TranAD' in self.model.name:
            self.trainD = self.convert_to_windows(self.trainD, self.model)

    def prepare_data_test(self, item):
        self.test_loader = DataLoader(self.data_test.faltas[item], batch_size=self.data_test.faltas.shape[1])
        self.testD = next(iter(self.test_loader))
        self.testO = self.testD
        if self.model.name in ['Attention', 'DAGMM', 'USAD', 'MSCRED', 'CAE_M', 'GDN', 'MTAD_GAT', 'MAD_GAN'] or 'TranAD' in self.model.name:
            self.testD = self.convert_to_windows(self.testD, self.model)

    def compute_score(self, loss, output, threshold, unique):
        if unique:
            score = (loss > threshold)*1.0
        else:
            score = torch.from_numpy(np.zeros_like(output))
            for i in range(threshold.shape[0]):
                    score[:, i] = torch.from_numpy(loss[:, i]) > threshold[i]

        return score

    def convert_to_windows(self, data, model):
        windows = []
        w_size = model.n_window
        for i, g in enumerate(data):
            if i >= w_size:
                w = data[i - w_size:i]
            else:
                w = torch.cat([data[0].repeat(w_size - i, 1), data[0:i]])
            windows.append(
                w if 'TranAD' or 'TranCIRCE' or 'OSContrastiveTransformer' in args.model or 'Attention' in args.model
                else w.view(
                    -1))
        return torch.stack(windows)

    def load_model(self, modelname, dims):
        import src.models
        model_class = getattr(src.models, modelname)
        model = model_class(dims).double()
        optimizer = torch.optim.AdamW(model.parameters(), lr=model.lr, weight_decay=1e-5)
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
        return model, optimizer, scheduler, epoch, accuracy_list
