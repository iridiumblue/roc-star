# Note - this sample codes need cleaning up, and automatic loading of its test dataset.

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import traceback
import sys

epoches = 100
INITIAL_LR=0.001

import os
KAGGLE=False
if KAGGLE :
    ipath = '../input/imagetest/'
    img_files = os.listdir(ipath)
    def train_path(p): return f"../input/imagetest/{p}"
else:
    ipath = '../input/imagetest/train'
    img_files = os.listdir(ipath)
    def train_path(p): return f"../input/imagetest/train/{p}"

img_files = list(map(train_path, img_files))

from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn import preprocessing
import torch
import numpy
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from sklearn.metrics import mean_squared_error, roc_auc_score, accuracy_score
from keras.utils import Progbar

import time
import code
# Any results you write to the current directory are saved as output.

BATCH_SIZE=80
USE_ROC_STAR=True

def epoch_update_gamma(y_true,y_pred, epoch=-1):
"""
    Calculate gamma from last epoch's targets and predictions.
    Gamma is updated at the end of each epoch.

    y_true: `Tensor`. Targets (labels).  Float either 0.0 or 1.0 .
    y_pred: `Tensor` . Predictions.
"""
        DELTA = 2
        SUB_SAMPLE_SIZE = 10000.0
        pos = y_pred[y_true==1]
        neg = y_pred[y_true==0] # yo pytorch, no boolean tensors or operators?  Wassap?
        # subsample the training set for performance
        cap_pos = pos.shape[0]
        cap_neg = neg.shape[0]
        pos = pos[torch.rand_like(pos) < SUB_SAMPLE_SIZE/cap_pos]
        neg = neg[torch.rand_like(neg) < SUB_SAMPLE_SIZE/cap_neg]
        ln_pos = pos.shape[0]
        ln_neg = neg.shape[0]
        pos_expand = pos.view(-1,1).expand(-1,ln_neg).reshape(-1)
        neg_expand = neg.repeat(ln_pos)
        diff = neg_expand - pos_expand
        ln_All = diff.shape[0]
        Lp = diff[diff>0] # because we're taking positive diffs, we got pos and neg flipped.
        ln_Lp = Lp.shape[0]-1
        diff_neg = -1.0 * diff[diff<0]
        diff_neg = diff_neg.sort()[0]
        ln_neg = diff_neg.shape[0]-1
        ln_neg = max([ln_neg, 0])
        left_wing = int(ln_Lp*DELTA)
        left_wing = max([0,left_wing])
        left_wing = min([ln_neg,left_wing])
        if diff_neg.shape[0] > 0 :
           gamma = diff_neg[left_wing]
        else:
           gamma = 0.2
        L1 = diff[diff>-1.0*gamma]
        ln_L1 = L1.shape[0]
        if epoch > -1 :
            return gamma
        return 0.10


def roc_star_loss( _y_true, y_pred, gamma, _epoch_true, epoch_pred):
"""
   Nearly direct loss function for AUC.
   See article,
   C. Reiss, "Roc-star : An objective function for ROC-AUC that actually works."
   https://github.com/iridiumblue/articles/blob/master/roc_star.md

    _y_true: `Tensor`. Targets (labels).  Float either 0.0 or 1.0 .
     y_pred: `Tensor` . Predictions.
    gamma  : `Float` Gamma, as derived from last epoch.
    _epoch_true: `Tensor`.  Targets (labels) from last epoch.
    epoch_pred : `Tensor`.  Predicions from last epoch.
"""
        #convert labels to boolean
        y_true = (_y_true>=0.50)
        epoch_true = (_epoch_true>=0.50)

        # if batch is either all true or false return small random stub value.
        if torch.sum(y_true)==0 or torch.sum(y_true) == y_true.shape[0]: return torch.sum(y_pred)*1e-8

        pos = y_pred[y_true]
        neg = y_pred[~y_true]

        epoch_pos = epoch_pred[epoch_true]
        epoch_neg = epoch_pred[~epoch_true]

        # Take random subsamples of the training set, both positive and negative.
        max_pos = 1000 # Max number of positive training samples
        max_neg = 1000 # Max number of positive training samples
        cap_pos = epoch_pos.shape[0]
        cap_neg = epoch_neg.shape[0]
        epoch_pos = epoch_pos[torch.rand_like(epoch_pos) < max_pos/cap_pos]
        epoch_neg = epoch_neg[torch.rand_like(epoch_neg) < max_neg/cap_pos]

        ln_pos = pos.shape[0]
        ln_neg = neg.shape[0]

        # sum positive batch elements agaionst (subsampled) negative elements
        if ln_pos>0 :
            pos_expand = pos.view(-1,1).expand(-1,epoch_neg.shape[0]).reshape(-1)
            neg_expand = epoch_neg.repeat(ln_pos)

            diff2 = neg_expand - pos_expand + gamma
            l2 = diff2[diff2>0]
            m2 = l2 * l2
            len2 = l2.shape[0]
        else:
            m2 = torch.tensor([0], dtype=torch.float).cuda()
            len2 = 0

        # Similarly, compare negative batch elements against (subsampled) positive elements
        if ln_neg>0 :
            pos_expand = epoch_pos.view(-1,1).expand(-1, ln_neg).reshape(-1)
            neg_expand = neg.repeat(epoch_pos.shape[0])

            diff3 = neg_expand - pos_expand + gamma
            l3 = diff3[diff3>0]
            m3 = l3*l3
            len3 = l3.shape[0]
        else:
            m3 = torch.tensor([0], dtype=torch.float).cuda()
            len3=0

        if (torch.sum(m2)+torch.sum(m3))!=0 :
           res2 = (torch.sum(m2)+torch.sum(m3))/(len2+len3)
        else:
           res2 = torch.sum(m2)+torch.sum(m3)

        res2 = torch.where(torch.isnan(res2), torch.zeros_like(res2), res2)

        return res2


class CatDogDataset(Dataset):
    def __init__(self, image_paths, transform):
        super().__init__()
        self.paths = image_paths
        self.len = len(self.paths)
        self.transform = transform

    def __len__(self): return self.len

    def __getitem__(self, index):
        path = self.paths[index]
        image = Image.open(path).convert('RGB')
        image = self.transform(image)
        label = 0 if 'cat' in path else 1
        return (image, label)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
import random
random.shuffle(img_files)
train_files = img_files[:20000]
train_tt = [1*('cat' in o) for o in train_files]

valid = img_files[20000:]
train_ds = CatDogDataset(train_files, transform)
#code.interact(local=dict(globals(), **locals()))
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE)
num_batches=len(train_dl)
valid_ds = CatDogDataset(valid, transform)
valid_dl = DataLoader(valid_ds, batch_size=BATCH_SIZE)
#code.interact(local=dict(globals(), **locals()))
class CatAndDogNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size=(5, 5), stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size=(5, 5), stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size=(3, 3), padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        #MIDDLE=50
        MIDDLE=500
        self.fc1 = nn.Linear(in_features= 64 * 6 * 6, out_features=MIDDLE)
        self.fc2 = nn.Linear(in_features=MIDDLE, out_features=int(MIDDLE/10))
        self.fc3 = nn.Linear(in_features=int(MIDDLE/10), out_features=2)
        self.drop = nn.Dropout(0.20)
        self.drop2= nn.Dropout(0.20)


    def forward(self, X):
        X = F.relu(self.bn1(self.conv1(X)))

        X = F.max_pool2d(X, 2)

        X = F.relu(self.bn2(self.conv2(X)))
        X = F.max_pool2d(X, 2)

        X = F.relu(self.bn3(self.conv3(X)))
        X = F.max_pool2d(X, 2)

#         print(X.shape)
        X = X.view(X.shape[0], -1)
        X = F.relu(self.fc1(X))
        X = self.drop(X)
        X = F.relu(self.fc2(X))
        X = self.drop2(X)
        X = self.fc3(X)
        X = X[:,1]-X[:,0]


#         X = torch.sigmoid(X)
        return X

model = CatAndDogNet().cuda()
losses = []


start = time.time()
loss_fn = nn.CrossEntropyLoss()



lr_range_4_momentum = [INITIAL_LR,INITIAL_LR/40.0]

optimizer = torch.optim.AdamW(model.parameters(), lr = INITIAL_LR,weight_decay=1e-4)




last_epoch_y_pred = torch.tensor( 1.0-numpy.random.rand(len(train_ds))/2.0, dtype=torch.float).cuda()
#code.interact(local=dict(globals(), **locals()))
last_epoch_y_t    = torch.tensor([o for o in train_tt],dtype=torch.float).cuda()
epoch_gamma=0.20
print("\rTraining ....\r")
for epoch in range(epoches):
    progbar = Progbar(num_batches, stateful_metrics=["loss","accuracy","LR","momentum"])

    epoch_loss = 0
    epoch_accuracy = 0
    epoch_y_pred=[]
    epoch_y_t=[]
    i=0
    for X, y in train_dl:
      try:
        X = X.cuda()
        b_t = y
        y = y.cuda()
        preds = model(X)
        b_preds = preds.tolist()



        if USE_ROC_STAR and epoch>-1 :
          #code.interact(local=dict(globals(), **locals()))
          loss = roc_star_loss(y,preds,epoch_gamma, last_epoch_y_t, last_epoch_y_pred)
        else:
          #code.interact(local=dict(globals(), **locals()))
          loss = F.binary_cross_entropy(F.sigmoid(preds), 1.0*y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_loss = loss.item()

        c_lr,c_mom= 0,0 #scheduler.batch(batch_loss,i, num_batches)

        epoch_y_pred.extend(b_preds)
        epoch_y_t.extend(b_t)

        accuracy = accuracy_score(b_preds>=np.median(b_preds),b_t)
        #accuracy = ((preds.argmax(dim=1) == y).float().mean())
        epoch_accuracy += accuracy
        epoch_loss += loss
        if not KAGGLE :
            progbar.update(
              i,
              values=[
                ("loss", batch_loss),
                ("accuracy",accuracy),
                ("LR", c_lr),
                ("momentum ", c_mom),
              ]
            )
        #print('.', end='', flush=True)
        i+=1
      except:
          traceback.print_exc(file=sys.stdout)
          code.interact(local=dict(globals(), **locals()))


    #c_epoch_lr=scheduler.epoch()
    #print("\r*SideWinder recentered learning rate to ",c_epoch_lr)
    last_epoch_y_pred = torch.tensor(epoch_y_pred).cuda()
    last_epoch_y_t = torch.tensor(epoch_y_t).cuda()
    auc = roc_auc_score(epoch_y_t,epoch_y_pred)
    epoch_accuracy = accuracy_score(1*(np.array(epoch_y_pred)>=np.median(epoch_y_pred)), epoch_y_t)

    epoch_gamma = epoch_update_gamma(last_epoch_y_t, last_epoch_y_pred, epoch)

    epoch_loss = epoch_loss / len(train_dl)
    losses.append(epoch_loss)
    print("TRAIN Epoch: {}, auc {:.4f}, train loss: {:.4f}, train accuracy: {:.4f}, time: {}".format(epoch, auc, epoch_loss, epoch_accuracy, time.time() - start))
    epoch_y_pred=[]
    epoch_y_t=[]
    with torch.no_grad():
        val_epoch_loss = 0
        val_epoch_accuracy = 0
        for val_X, val_y in valid_dl:
            val_X = val_X.cuda()
            b_t = val_y
            val_y = val_y.cuda()
            val_preds = model(val_X)
            b_preds = val_preds.tolist()
            val_loss = 0 #loss_fn(val_preds, val_y)
            epoch_y_pred.extend(b_preds)
            epoch_y_t.extend(b_t)
            val_epoch_loss += val_loss
            val_accuracy = 0 #((val_preds.argmax(dim=1) == val_y).float().mean())
            val_epoch_accuracy += val_accuracy

        auc = roc_auc_score(epoch_y_t,epoch_y_pred)
        epoch_accuracy = accuracy_score(1*(np.array(epoch_y_pred)>=np.median(epoch_y_pred)), epoch_y_t)
        #val_epoch_accuracy = val_epoch_accuracy/len(valid_dl)
        val_epoch_loss = val_epoch_loss / len(valid_dl)
        print("VALID Epoch: {}, auc {:.4f},valid loss: {:.4f}, valid accuracy: {:.4f}, time: {}\n".format(epoch, auc, val_epoch_loss, epoch_accuracy, time.time() - start))
