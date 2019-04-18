import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from ipypb import track as tqdm

# constants / options
path_train = './data/train.csv'
path_test = './data/test.csv'
path_models = './models/'
path_plots = './plots/'
path_logs = './logs/'

# helpers
def load_train(path=path_train):
    data = pd.read_csv(path)
    y = data['class_label']
    data.drop('class_label', axis=1, inplace=True)
    return data, y

def load_test(path=path_test):
    data = pd.read_csv(path)
    return data

# cv still is not supported on GPU
def cv_catboost(X, y, params, n_splits, seed):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    res=[]
    for tr_idx, te_idx in kf.split(X):
        print('Processing fold...')
        X_tr, X_te = X.iloc[tr_idx, :], X.iloc[te_idx, :]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

        cat = CatBoostClassifier(**params)
        cat.fit(X_tr, y_tr, eval_set=(X_te, y_te))

        res.append(f1_score(y_te, cat.predict(X_te), average='weighted'))
    return np.mean(res), np.std(res)

# packing metrics and loss results into easily plottable format
def get_catboost_metrics(cat):
    res = cat.get_evals_result()
    df_eval = pd.DataFrame({
        'train_f1': res['learn']['TotalF1'],
        'test_f1': res['validation_0']['TotalF1']
        })
    df_loss = pd.DataFrame({
        'train_loss': res['learn']['MultiClass'],
        'test_loss': res['validation_0']['MultiClass']
        })
    return {'eval': df_eval, 'loss': df_loss}

##### nn helpers
def process_train_nn(X, y):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.05, random_state=42)
    tX_tr = torch.FloatTensor(X_tr.values)
    ty_tr = torch.LongTensor(y_tr.values)
    tX_te = torch.FloatTensor(X_te.values)
    ty_te = torch.LongTensor(y_te.values)

    train_dl = DataLoader(TensorDataset(tX_tr, ty_tr), batch_size=1024, shuffle=False)
    test_dl = DataLoader(TensorDataset(tX_te, ty_te), batch_size=1024, shuffle=False)
    return train_dl, test_dl

def test_nn(model, dataloader, criterion, device='cpu'):
    model.eval()
    class_total, class_correct = [0] * 14, [0] * 14
    y_pred, y_true = [], []
    loss_iter = []

    with torch.no_grad():
        for data, labels in tqdm(dataloader): # TODO
            data.to(device)
            outputs = model(data)
            if device == 'cuda':
                outputs.to('cpu')
            _, predicted = torch.max(outputs, 1)
            y_pred = np.concatenate((y_pred, predicted.numpy()))
            y_true = np.concatenate((y_true, labels.numpy()))
            loss = criterion(outputs, labels.type(torch.LongTensor))
            loss_iter.append(loss.detach().numpy())

    return f1_score(y_true, y_pred, average='weighted'), accuracy_score(y_true, y_pred), np.mean(loss_iter)

def train_nn(model, train_dataloader, test_dataloader, n_epochs, lr, device='cpu'):
    criterion = nn.NLLLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=1e-5, momentum=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    log_train = {'loss' : [], 'f1': [], 'acc': []}
    log_test = {'loss' : [], 'f1': [], 'acc': []}
    model.to('cpu')
    l1_crit = nn.L1Loss(size_average=False)

    for epoch in range(n_epochs):
        model.train()
        for data, target in tqdm(train_dataloader):
            optimizer.zero_grad()
            data.to(device)
            pred = model(data)
            if device == 'cuda':
                pred.to('cpu')
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
        scheduler.step()   

        f1, acc, l = test_nn(model, train_dataloader, criterion, device)
        log_train['loss'].append(l)
        log_train['f1'].append(f1)
        log_train['acc'].append(acc)

        f1, acc, l = test_nn(model, test_dataloader, criterion, device)
        log_test['loss'].append(l)
        log_test['f1'].append(f1)
        log_test['acc'].append(acc)

        print('Train loss: {}, acc: {}, f1: {}'.format(log_train['loss'][-1], log_train['acc'][-1], log_train['f1'][-1]))
        print('Test loss: {}, acc: {}, f1: {}'.format(log_test['loss'][-1], log_test['acc'][-1], log_test['f1'][-1]))
    return model, log_train, log_test