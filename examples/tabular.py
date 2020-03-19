import sys
sys.path.append('.')
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from sklearn.preprocessing import StandardScaler

from ap_perf import PerformanceMetric, MetricLayer

np.random.seed(0)

def load_data(ds, standardize=True):
    df = pd.read_csv("examples/data-cv/" + ds + ".csv", sep=',', header=None)
    X_all = df.values[:,:-1].astype(np.float32)
    y_all = df.values[:,-1]
    y_all = (y_all >= 7).astype(np.float32)  # to binary

    n_all = len(y_all)
    id_perm = np.random.permutation(n_all)
    n_tr = n_all // 2
    
    # split
    X_tr = X_all[id_perm[:n_tr], :]
    X_ts = X_all[id_perm[n_tr:], :]
    y_tr = y_all[id_perm[:n_tr]]
    y_ts = y_all[id_perm[n_tr:]]

    # normalize
    if standardize:
        scaler = StandardScaler()
        scaler.fit(X_tr)

        X_tr = scaler.transform(X_tr)
        X_ts = scaler.transform(X_ts)

    return X_tr, X_ts, y_tr, y_ts


class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return [self.X[idx, :], self.y[idx]]


class Net(nn.Module):
    def __init__(self, nvar):
        super().__init__()
        self.fc1 = nn.Linear(nvar, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x.squeeze()


# performance metric
# F-Beta
class F_beta(PerformanceMetric):
    def __init__(self, beta):
        self.beta = beta

    def define(self, C):
        return ((1 + self.beta ** 2) * C.tp) / ( (self.beta ** 2) * C.ap + C.pp)  


f2 = F_beta(2)
f2.initialize()
f2.enforce_special_case_positive()

# performance metric
pm = f2


ds = "whitewine"
X_tr, X_ts, y_tr, y_ts = load_data(ds, standardize=True)

trainset = TabularDataset(X_tr, y_tr)
testset = TabularDataset(X_ts, y_ts)

batch_size = 25
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

method = "ap-perf"              # uncomment if we want to use ap-perf objective 
# method = "bce-loss"           # uncomment if we want to use bce-loss objective

torch.manual_seed(1)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

nvar = X_tr.shape[1]
model = Net(nvar).to(device)

if method == "ap-perf":
    criterion = MetricLayer(f2).to(device)
    lr = 3e-3
    weight_decay = 1e-3
else:
    criterion = nn.BCEWithLogitsLoss().to(device)
    lr = 1e-2
    weight_decay = 1e-3

optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

for epoch in range(100):

    for i, (inputs, labels) in enumerate(trainloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        output = model(inputs)

        loss = criterion(output, labels)

        loss.backward()
        optimizer.step()

        sys.stdout.write("\r#%d | progress : %d%%" % (epoch,  int(100 * (i+1) / len(trainloader))))
        sys.stdout.flush()

    print()

    # evaluate after each epoch
    model.eval()

    # train
    train_data = torch.tensor(X_tr).to(device)
    tr_output = model(train_data)
    tr_pred = (tr_output >= 0.0).float()
    tr_pred_np = tr_pred.cpu().numpy()

    train_acc = np.sum(y_tr == tr_pred_np) / len(y_tr)
    train_metric = pm.compute_metric(tr_pred_np, y_tr)
    
    # test
    test_data = torch.tensor(X_ts).to(device)
    ts_output = model(test_data)
    ts_pred = (ts_output >= 0.0).float()
    ts_pred_np = ts_pred.cpu().numpy()

    test_acc = np.sum(y_ts == ts_pred_np) / len(y_ts)
    test_metric = pm.compute_metric(ts_pred_np, y_ts)

    model.train()

    print('#{} | Acc tr: {:.5f} | Acc ts: {:.5f} | Metric tr: {:.5f} | Metric ts: {:.5f}'.format(
        epoch, train_acc, test_acc, train_metric, test_metric))






