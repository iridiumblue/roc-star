'''
A demonstration of a new and experimental loss function which directly targets AUC/ROC,
and is seen to outperform BxE in early testing. See paper up here -
https://github.com/iridiumblue/roc-star.

The test is a simple sentiment analysis binary classifier turned loose on tweets from Twitter.
Text embeddings have been precomputed and pickled for speed.
TRUNC truncates the training set, set it to -1 for all 1.6 M sample tweets.

Note that for the first epoch (only), the loss function is BxE. That is just to kickstart
the new loss function roc_star_loss. That's a good practice to stick to if you want
to give this a try for your model. Note also that roc_star_loss requires a call to
epoch_update_gamma at the end of each epoch.
'''


from warnings import simplefilter
import time
from copy import copy
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence
import numpy as np
import _pickle
import gc
from pathlib2 import Path
from sklearn.metrics import roc_auc_score
from trains import Task
from trains import StorageManager
from pkbar import Kbar as Progbar
import argparse
import os
from tempfile import gettempdir


# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)

x_train_torch,x_valid_torch,y_train_torch,y_valid_torch = None,None,None,None
embedding_matrix = None
task=None
logger=None
best_result={}
max_features = 200000
embed_size = 300


def init(h_params):
    global x_train_torch, x_valid_torch, y_train_torch, y_valid_torch
    global embedding_matrix
    global task, logger

    print("Recovering tokenized text from pickle ...")
    tokenized = StorageManager.get_local_copy(
        remote_url="https://allegro-datasets.s3.amazonaws.com/roc_star_data"
                   "/tokenized.pkl.zip",
        name="tokenized",
    )
    embedding = StorageManager.get_local_copy(
        remote_url="https://allegro-datasets.s3.amazonaws.com/roc_star_data"
                   "/embedding.pkl.zip",
        name="embedding",
    )

    x_train, x_valid, y_train, y_valid = _pickle.load(open(Path(tokenized, "tokenized.pkl"), "rb"))
    print("Reusing pickled embedding ...")
    embedding_matrix = _pickle.load(open(Path(embedding, "embedding.pkl"), "rb"))

    print("Moving data to GPU ...")
    if h_params.trunc>-1 :
        print(f"\r\r * * WARNING training set truncated to first {h_params.trunc} items.\r\r")

    x_train_torch = torch.tensor(x_train[:h_params.trunc], dtype=torch.long).cuda()
    x_valid_torch = torch.tensor(x_valid, dtype=torch.long).cuda()
    y_train_torch = torch.tensor(y_train[:h_params.trunc], dtype=torch.float32).cuda()
    y_valid_torch = torch.tensor(y_valid, dtype=torch.float32).cuda()
    del x_train, y_train, x_valid, y_valid; gc.collect(2)

    task_name = "ROC" if h_params.use_roc_star else "BxE"
    task = Task.init(project_name='Roc-star Loss', task_name='Roc star')
    logger = task.get_logger()


def epoch_update_gamma(y_true, y_pred, epoch=-1, delta=2):
        """
        Calculate gamma from last epoch's targets and predictions.
        Gamma is updated at the end of each epoch.
        y_true: `Tensor`. Targets (labels).  Float either 0.0 or 1.0 .
        y_pred: `Tensor` . Predictions.
        """
        sub_sample_size = 2000.0
        pos = y_pred[y_true==1]
        neg = y_pred[y_true==0] # yo pytorch, no boolean tensors or operators?  Wassap?
        # subsample the training set for performance
        cap_pos = pos.shape[0]
        cap_neg = neg.shape[0]
        pos = pos[torch.rand_like(pos) < sub_sample_size/cap_pos]
        neg = neg[torch.rand_like(neg) < sub_sample_size/cap_neg]
        ln_pos = pos.shape[0]
        ln_neg = neg.shape[0]
        pos_expand = pos.view(-1,1).expand(-1,ln_neg).reshape(-1)
        neg_expand = neg.repeat(ln_pos)
        diff = neg_expand - pos_expand
        Lp = diff[diff>0] # because we're taking positive diffs, we got pos and neg flipped.
        ln_Lp = Lp.shape[0]-1
        diff_neg = -1.0 * diff[diff<0]
        diff_neg = diff_neg.sort()[0]
        ln_neg = diff_neg.shape[0]-1
        ln_neg = max([ln_neg, 0])
        left_wing = int(ln_Lp*delta)
        left_wing = max([0,left_wing])
        left_wing = min([ln_neg,left_wing])
        default_gamma = torch.tensor(0.2, dtype=torch.float).cuda()
        if diff_neg.shape[0] > 0 :
            gamma = diff_neg[left_wing]
        else:
            gamma = default_gamma # default=torch.tensor(0.2, dtype=torch.float).cuda() #zoink
        L1 = diff[diff>-1.0*gamma]
        if epoch > -1 :
            return gamma
        else :
            return default_gamma


def roc_star_loss(_y_true, y_pred, gamma, _epoch_true, epoch_pred):
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
        else:
            m2 = torch.tensor([0], dtype=torch.float).cuda()

        # Similarly, compare negative batch elements against (subsampled) positive elements
        if ln_neg>0 :
            pos_expand = epoch_pos.view(-1,1).expand(-1, ln_neg).reshape(-1)
            neg_expand = neg.repeat(epoch_pos.shape[0])

            diff3 = neg_expand - pos_expand + gamma
            l3 = diff3[diff3>0]
            m3 = l3*l3
        else:
            m3 = torch.tensor([0], dtype=torch.float).cuda()

        if (torch.sum(m2)+torch.sum(m3))!=0 :
            res2 = torch.sum(m2)/max_pos+torch.sum(m3)/max_neg
        else:
            res2 = torch.sum(m2)+torch.sum(m3)

        res2 = torch.where(torch.isnan(res2), torch.zeros_like(res2), res2)

        return res2


#https://github.com/keitakurita/Better_LSTM_PyTorch/blob/master/better_lstm/model.py
class VariationalDropout(nn.Module):
    """
    Applies the same dropout mask across the temporal dimension
    See https://arxiv.org/abs/1512.05287 for more details.
    Note that this is not applied to the recurrent activations in the LSTM like the above paper.
    Instead, it is applied to the inputs and outputs of the recurrent layer.
    """
    def __init__(self, dropout, batch_first=False):
        super().__init__()
        self.dropout = dropout
        self.batch_first = batch_first

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.dropout <= 0.:
            return x

        is_packed = isinstance(x, PackedSequence)
        if is_packed:
            x, batch_sizes = x
            max_batch_size = int(batch_sizes[0])
        else:
            batch_sizes = None
            max_batch_size = x.size(0)

        # Drop same mask across entire sequence
        if self.batch_first:
            m = x.new_empty(max_batch_size, 1, x.size(2), requires_grad=False).bernoulli_(1 - self.dropout)
        else:
            m = x.new_empty(1, max_batch_size, x.size(2), requires_grad=False).bernoulli_(1 - self.dropout)
        x = x.masked_fill(m == 0, 0) / (1 - self.dropout)

        if is_packed:
            return PackedSequence(x, batch_sizes)
        else:
            return x


#https://github.com/keitakurita/Better_LSTM_PyTorch/blob/master/better_lstm/model.py
class LSTM(nn.LSTM):
    def __init__(self, *args, dropouti: float=0.,
                 dropoutw: float=0., dropouto: float=0.,
                 batch_first=True, unit_forget_bias=True, **kwargs):
        super().__init__(*args, **kwargs, batch_first=batch_first)
        self.unit_forget_bias = unit_forget_bias
        self.dropoutw = dropoutw
        self.input_drop = VariationalDropout(dropouti,
                                             batch_first=batch_first)
        self.output_drop = VariationalDropout(dropouto,
                                              batch_first=batch_first)
        self._init_weights()

    def _init_weights(self):
        """
        Use orthogonal init for recurrent layers, xavier uniform for input layers
        Bias is 0 except for forget gate
        """
        for name, param in self.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "bias" in name and self.unit_forget_bias:
                nn.init.zeros_(param.data)
                param.data[self.hidden_size:2 * self.hidden_size] = 1

    def _drop_weights(self):
        for name, param in self.named_parameters():
            if "weight_hh" in name:
                getattr(self, name).data = \
                    torch.nn.functional.dropout(param.data, p=self.dropoutw,
                                                training=self.training).contiguous()

    def forward(self, input, hx=None):
        self._drop_weights()
        input = self.input_drop(input)
        seq, state = super().forward(input, hx=hx)
        return self.output_drop(seq), state



class NeuralNet(nn.Module):
    def __init__(self, embedding_matrix,h_params):
        super(NeuralNet, self).__init__()
        embed_size = embedding_matrix.shape[1]

        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.h_params = copy(h_params)

        self.c1 = nn.Conv1d(300,kernel_size=2,out_channels=300,padding=1)
        LSTM_UNITS=h_params.lstm_units
        BIDIR = h_params.bidirectional

        LSTM_OUT = 2* LSTM_UNITS  if BIDIR else LSTM_UNITS

        self.lstm1 = LSTM(embed_size, LSTM_UNITS, dropouti=h_params.dropout_i,dropoutw=h_params.dropout_w, dropouto=h_params.dropout_o,bidirectional=BIDIR, batch_first=True)
        self.lstm2 = LSTM(LSTM_OUT, LSTM_UNITS,   dropouti=h_params.dropout_i,dropoutw=h_params.dropout_w, dropouto=h_params.dropout_o,bidirectional=BIDIR, batch_first=True)


        self.linear1 = nn.Linear(2*LSTM_OUT, 2*LSTM_OUT)
        self.linear2 = nn.Linear(2*LSTM_OUT, 2*LSTM_OUT)

        self.hey_norm = nn.LayerNorm(2*LSTM_OUT)

        self.linear_out = nn.Linear(2*LSTM_OUT, h_params.dense_hidden_units)
        self.linear_xtra = nn.Linear(h_params.dense_hidden_units,int(h_params.dense_hidden_units/2))
        self.linear_xtra2 = nn.Linear(int(h_params.dense_hidden_units/2),int(h_params.dense_hidden_units/4))
        self.linear_out2= nn.Linear(int(h_params.dense_hidden_units/4), 1)



    def forward(self, x):
        h_embedding = self.embedding(x)
        h1 = h_embedding.permute(0, 2, 1)

        q1 =  self.c1(h1)
        f1 =  q1.permute(0, 2, 1)
        h_lstm1, _ = self.lstm1(f1)
        h_lstm2, _ = self.lstm2(h_lstm1)

        # global average pooling
        avg_pool = torch.mean(h_lstm2, 1)
        # global max pooling
        max_pool, _ = torch.max(h_lstm2, 1)

        h_conc = torch.cat((max_pool, avg_pool), 1)

        h_conc_linear1  = self.linear1(h_conc)
        h_conc_linear2  = self.linear2(h_conc)

        hidden = h_conc + h_conc_linear1 + h_conc_linear2
        hidden = F.selu(self.linear_out(hidden))
        hidden =  F.selu(self.linear_xtra(hidden))
        hidden =  F.selu(self.linear_xtra2(hidden))
        hidden =  F.sigmoid(self.linear_out2(hidden))

        result=hidden.flatten()

        return result



def train_model(h_params, model, x_train, x_valid, y_train, y_valid,  lr,
                batch_size=1000, n_epochs=20, title='', graph=''):
    global best_result
    param_lrs = [{'params': param, 'lr': lr} for param in model.parameters()]
    optimizer = torch.optim.AdamW(param_lrs, lr=h_params.initial_lr,
                betas=(0.9, 0.999),
                eps=1e-6,
                amsgrad=False
                )

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_valid, y_valid), batch_size=batch_size, shuffle=False)

    num_batches = len(x_train)/batch_size
    print(len(x_train))
    results=[]

    for epoch in range(n_epochs):
        start_time = time.time()
        model.train()
        avg_loss = 0.


        progbar =Progbar(num_batches, stateful_metrics=['train-auc'])

        whole_y_pred=np.array([])
        whole_y_t=np.array([])

        for i,data in enumerate(train_loader):
            x_batch = data[:-1][0]
            y_batch = data[-1]

            y_pred = model(x_batch)

            if h_params.use_roc_star and epoch>0 :

               if i==0 : print('*Using Loss Roc-star')
               loss = roc_star_loss(y_batch,y_pred,epoch_gamma, last_whole_y_t, last_whole_y_pred)

            else:
               if i==0 : print('*Using Loss BxE')
               loss = F.binary_cross_entropy(y_pred, 1.0*y_batch)

            logger.report_scalar(title="Loss", series="trains loss",
                                 value=loss, iteration=epoch * len(x_train) + i)

            optimizer.zero_grad()
            loss.backward()
            # To prevent gradient explosions resulting in NaNs
            # https://discuss.pytorch.org/t/nan-loss-in-rnn-model/655/8
            # https://github.com/pytorch/examples/blob/master/word_language_model/main.py
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

            optimizer.step()

            whole_y_pred = np.append(whole_y_pred, y_pred.clone().detach().cpu().numpy())
            whole_y_t    = np.append(whole_y_t, y_batch.clone().detach().cpu().numpy())

            if i>0:
                if i%50==1 :
                   try:
                      train_roc_val = roc_auc_score(whole_y_t>=0.5, whole_y_pred)
                   except:

                      train_roc_val=-1

                   progbar.update(
                      i,
                      values=[
                          ("loss", np.mean(loss.detach().cpu().numpy())),
                          ("train-auc", train_roc_val)
                      ]
                   )

        model.eval()
        last_whole_y_t = torch.tensor(whole_y_t).cuda()
        last_whole_y_pred = torch.tensor(whole_y_pred).cuda()

        all_valid_preds = np.array([])
        all_valid_t = np.array([])
        for i, valid_data in enumerate(valid_loader):
            x_batch = valid_data[:-1]
            y_batch = valid_data[-1]

            y_pred = model(*x_batch).detach().cpu().numpy()
            y_t = y_batch.detach().cpu().numpy()

            all_valid_preds=np.concatenate([all_valid_preds,y_pred],axis=0)
            all_valid_t = np.concatenate([all_valid_t,y_t],axis=0)

        epoch_gamma = epoch_update_gamma(last_whole_y_t, last_whole_y_pred, epoch,h_params.delta)

        try:
          valid_auc = roc_auc_score(all_valid_t>=0.5, all_valid_preds)
        except:
          valid_auc=-1

        try:
           train_roc_val = roc_auc_score(whole_y_t>=0.5, whole_y_pred)
        except:
           train_roc_val=-1

        elapsed_time = time.time() - start_time
        if epoch==0 :
           print("\n\n* * * * * * * * * * * Params :", title," :: ", graph)
        print('\nEpoch {}/{} \t loss={:.4f} \t time={:.2f}s'.format(
              epoch + 1, n_epochs, avg_loss, elapsed_time))

        print("Gamma = ", epoch_gamma)
        print("Validation AUC = ", valid_auc)
        if not ('auc' in best_result) or best_result['auc']<valid_auc :
            best_result['auc']= valid_auc
            best_result['params'] = copy(h_params)
            best_result['epoch'] = epoch+1
            torch.save(model.state_dict(), os.path.join(gettempdir(), "roc-star.pt"))
            print('* * grabbing ', best_result)


        print("\r Training AUC = ", train_roc_val)
        if valid_auc>0 and train_roc_val>0 :
            logger.report_scalar(title="Accuracy", series="validation accuracy",
                 value=valid_auc, iteration=epoch)
            logger.report_scalar(title="Accuracy", series="train accuracy",
                 value=train_roc_val, iteration=epoch)

        results.append({
           'valid_auc': valid_auc,
           'train_auc': train_roc_val,
           'gamma': epoch_gamma.item()
        })

        print(f'TRAINS results page: {task._get_app_server()}/projects/{task.project}/experiments/{task.id}/output/log')

        print()

    return results


def run(h_params,embedding_matrix, title='',graph=''):
    model = NeuralNet(embedding_matrix,h_params)
    model.cuda()
    h_params.dropout_i,h_params.dropout_o,h_params.dropout_w = (h_params.dropout,h_params.dropout,h_params.dropout)
    run_result = train_model(h_params,model, x_train_torch, x_valid_torch, y_train_torch, y_valid_torch,  lr=h_params.initial_lr,
                batch_size=h_params.batch_size, n_epochs=h_params.epochs,title=title,graph=graph)
    return run_result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--use-roc-star', type=bool, default=True)
    parser.add_argument('--delta', type=float, default=2)
    parser.add_argument('--initial-lr', type=float, default=1e-3)
    parser.add_argument('--dense-hidden-units', type=int, default=1024)
    parser.add_argument('--lstm-units', type=int, default=64)
    parser.add_argument('--batch-size', type=int,default=128)
    parser.add_argument('--bidirectional', type=bool, default=True)
    parser.add_argument('--spacial-dropout', type=float, default=0.00)
    parser.add_argument('--dropout-w', type=float,default=0.18)
    parser.add_argument('--dropout-i', type=float,default=0.18)
    parser.add_argument('--dropout-o', type=float,default=0.18)
    parser.add_argument('--dropout', type=float,default=0.20)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--trunc', type=int, default = 200000,help='truncates data, -1 for no truncation')
    parser.add_argument('--maxlen', type=int, default=30)
    parser.add_argument('--seed', type=int,default=43)

    FLAGS, unparsed = parser.parse_known_args()

    torch.manual_seed(FLAGS.seed)
    torch.backends.cudnn.deterministic = True

    init(FLAGS)

    title = "ROC_STAR" if FLAGS.use_roc_star else "BXE"
    run(FLAGS, embedding_matrix, title=title, graph=title)

    print(f'TRAINS results page: {task._get_app_server()}/projects/{task.project}/experiments/{task.id}/output/log')
    print('\r\r\r * * Best validation score ',best_result)

