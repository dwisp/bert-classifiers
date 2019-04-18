import os
from os.path import join
import helpers as hlp
import hyppars as pr
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, cross_validate, KFold, train_test_split
# from sklearn.utils.class_weight import compute_class_weight

## pytorch
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

# data loading and manipulation
X, y = hlp.load_train()
test = hlp.load_test()
# w = compute_class_weight('balanced', np.arange(0, 14), y)

# train / test split
X_tr, X_te, y_tr, y_te = train_test_split(X, y, train_size=0.75, random_state=pr.rseed)

# evaluating models on the training data
# RANDOM FOREST
metrics_rf = GridSearchCV(
    RandomForestClassifier(
        n_estimators=200,
        random_state=pr.rseed,
        class_weight='balanced',
        min_samples_split=3,
        n_jobs=-1
    ),
    pr.grid_rf,
    scoring=make_scorer(f1_score, average='weighted'),
    n_jobs=-1,
    cv=4,
    verbose=1
)
metrics_rf.fit(X, y)
# writing best parameters
pd.DataFrame(metrics_rf.best_params_, index=[0]).\
    to_csv(join(hlp.path_models, 'pars_rf.csv'), index=False)

# CATBOOST
model_cat = CatBoostClassifier(
    depth=5,
    learning_rate=0.13,
    iterations=2000,
    bagging_temperature=0.35,
    l2_leaf_reg=1.5,
    loss_function='MultiClass',
    eval_metric='TotalF1',
    random_state=pr.rseed,
    thread_count=-1,
    task_type = 'GPU'
)

model_cat.fit(X_tr, y_tr, eval_set=(X_te, y_te))

# saving metrics and model
metric_and_loss = hlp.get_catboost_metrics(model_cat)
metric_and_loss['eval'].to_csv(join(hlp.path_logs, 'cat_eval.csv'))
metric_and_loss['loss'].to_csv(join(hlp.path_logs, 'cat_loss.csv'))

### PyTorch NN ###
torch.manual_seed(pr.rseed)
np.random.seed(pr.rseed)
# setting up a GPU device to train on
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# getting the tensors
module_list = [
    nn.BatchNorm1d(770, eps=1e-05, momentum=0.1, affine=True),
    nn.Linear(in_features=770, out_features=1024, bias=True),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(in_features=1024, out_features=256, bias=True),
    nn.ReLU(),
    nn.Linear(in_features=256, out_features=14, bias=True),
    nn.LogSoftmax()
]
model = nn.Sequential(*module_list)

train_dl, test_dl = hlp.process_train_nn(X, y)
model, nn_log_train, nn_log_test = hlp.train_nn(model, train_dl, test_dl, 20, 1e-3, 'cuda')

pd.DataFrame(nn_log_train).to_csv(join(hlp.path_logs, 'nn_train.csv'))
pd.DataFrame(nn_log_test).to_csv(join(hlp.path_logs, 'nn_test.csv'))

tX_subm = torch.FloatTensor(test.values)
model.eval()
y_subm = model(tX_subm)

pred_class = np.argmax(y_subm.detach().numpy(), axis=1)
pred_class = pred_class.tolist()

# saving model and its predictions
subm = pd.DataFrame(data=pred_class, columns=['class_label'])
subm.index.name = 'id'
subm.to_csv('nn_preds.csv')

torch.save(model, join(hlp.path_models, 'nn.pth'))