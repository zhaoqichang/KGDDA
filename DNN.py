
import argparse
import time
import pandas as pd
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import os
from sklearn import metrics
import copy
from pytorchtools import EarlyStopping
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve, auc



n_classes = 2
seed = 10
epochs = 200
weight_decay = 0.001
dropout = 0.1
cuda = torch.cuda.is_available()
if cuda:
    print('Using GPU')

def classification_accuracy(Q, data_loader,X_dim):
    Q.eval()
    labels = []
    scores = []

    loss = 0
    correct = 0
    for batch_idx, (X, target) in enumerate(data_loader):
        X = X.view(-1, X_dim)
        X, target = Variable(X), Variable(target)

        if cuda:
            X, target = X.cuda(), target.cuda()

        output = Q(X)
        output_probability = F.softmax(output, dim=1)

        labels.extend(target.data.tolist())
        if cuda:
            scores.extend(output_probability.cpu().data.numpy()[:, 1].tolist())
        else:
            scores.extend(output_probability.data.numpy()[:, 1].tolist())

        loss += F.cross_entropy(output, target, size_average=False).item()

        pred = output_probability.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()

    loss /= len(data_loader.dataset)
    acc = correct / len(data_loader.dataset)

    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    precision, recall, _ = metrics.precision_recall_curve(labels, scores)
    aupr = metrics.auc(recall, precision)

    return loss, acc, auc, aupr

#train(model, fc_solver, train_labeled_loader)
def train(fc_dnn, fc_solver, train_labeled_loader,X_dim):
    TINY = 1e-15

    fc_dnn.train()
    for X, target in train_labeled_loader:
        X, target = Variable(X), Variable(target)
        if cuda:
            X = X.cuda()
            target = target.cuda()

        X = X.view(-1, X_dim)
        out = fc_dnn(X)

        fc_solver.zero_grad()
        loss = F.cross_entropy(out + TINY, target)
        loss.backward()
        fc_solver.step()

    return loss.item()


def report_loss(epoch, loss):
    print()
    print('Epoch-{}; loss: {:.4}'.format(epoch, loss))


class FC_DNN(nn.Module):
    def __init__(self,X_dim):
        super(FC_DNN, self).__init__()
        self.lin1 = nn.Linear(X_dim, 1024)
        self.lin2 = nn.Linear(1024, 512)
        self.lin3 = nn.Linear(512, 256)
        self.lin4 = nn.Linear(256, 128)
        self.cat = nn.Linear(128, n_classes)

    def forward(self, x):
        x = x.to(torch.float32)
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=dropout, training=self.training)

        x = self.lin2(x)
        x = F.relu(x)
        x = F.dropout(x, p=dropout, training=self.training)

        x = self.lin3(x)
        x = F.relu(x)
        x = F.dropout(x, p=dropout, training=self.training)

        x = self.lin4(x)
        x = F.relu(x)
        x = F.dropout(x, p=dropout, training=self.training)

        xcat = self.cat(x)
        return xcat


class TE_DNN(nn.Module):
    def __init__(self, X_dim, nhead, num_layers,dim_feedforward,sequence_length=4):
        super(TE_DNN, self).__init__()
        self.sequence_length = sequence_length
        self.embedding_dim = X_dim // sequence_length

        norm = nn.LayerNorm(self.embedding_dim)

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.embedding_dim, nhead=nhead,dim_feedforward=dim_feedforward),
            num_layers=num_layers,
            norm=norm
        )
        self.lin1 = nn.Linear(X_dim, 1024)
        self.lin2 = nn.Linear(1024, 512)
        self.lin3 = nn.Linear(512, 256)
        self.lin4 = nn.Linear(256, 128)
        self.cat = nn.Linear(128, n_classes)

    def forward(self, x):
        # 增加 sequence length 维度
        x = x.float()
        batch_size = x.size(0)
        # (batch_size, embedding_size) -> (batch_size, sequence_length, embedding_dim)
        x = x.view(batch_size, self.sequence_length, self.embedding_dim)

        # Transpose to (sequence_length, batch_size, embedding_dim) for transformer
        x = x.transpose(0, 1)

        # Apply transformer encoder
        x = self.transformer_encoder(x)

        # Transpose back to (batch_size, sequence_length, embedding_dim)
        x = x.transpose(0, 1)

        # Flatten to (batch_size, embedding_size)
        x = x.contiguous().view(batch_size, -1)


        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=dropout, training=self.training)

        x = self.lin2(x)
        x = F.relu(x)
        x = F.dropout(x, p=dropout, training=self.training)

        x = self.lin3(x)
        x = F.relu(x)
        x = F.dropout(x, p=dropout, training=self.training)

        x = self.lin4(x)
        x = F.relu(x)
        x = F.dropout(x, p=dropout, training=self.training)

        xcat = self.cat(x)
        return xcat

def generate_model(i,args,train_labeled_loader, valid_loader,X_dim):
    print('generating new model')
    torch.manual_seed(10)

    if args.mode == 'TE' or args.mode =='TE_2' or args.mode =='TE_4':
        model = TE_DNN(X_dim,args.nhead,args.num_layers,args.dim_feedforward)
    elif args.mode =='TE_3':
        model = TE_DNN(X_dim, args.nhead, args.num_layers, args.dim_feedforward,6)
    else:
        model = FC_DNN(X_dim)

    if cuda:
        model = model.cuda()

    lr = 0.0001
    fc_solver = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    start = time.time()
    max_valid_acc, max_valid_auc,max_valid_aupr= 0,0,0
    result_model = None

    train_loss, train_acc, train_auc,train_aupr = classification_accuracy(model, train_labeled_loader,X_dim)   #过了一遍
    print('no train loss {}, acc {}, auc {} ,aupr {} '.format(train_loss, train_acc, train_auc,train_aupr))

    early_stopping = EarlyStopping(patience=20, verbose=True,path=f'checkpoint_KGEM_{args.mode}_{args.split}_{args.valid_auc}_{args.patient_epoch}_{args.drdi_feature}_{args.nhead}_{args.num_layers}_{args.dim_feedforward}_{i}.pt')
    flag = 0
    for epoch in range(epochs):
        loss = train(model, fc_solver, train_labeled_loader,X_dim)
        report_loss(epoch, loss)

        train_loss, train_acc, train_auc, train_aupr = classification_accuracy(
            model, train_labeled_loader,X_dim)
        print('Train loss {:.4}, acc {:.4}, auc {:.4},aupr {} '.format(
            train_loss, train_acc, train_auc, train_aupr))

        valid_loss, valid_acc, valid_auc, valid_aupr = classification_accuracy(
            model, valid_loader,X_dim)
        print('Valid loss {:.4}, acc {:.4}, auc {:.4},aupr {} '.format(
            valid_loss, valid_acc, valid_auc, valid_aupr))

        if valid_auc > args.valid_auc:
            fc_solver = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

        if epoch < args.patient_epoch:
            continue

        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            flag = epoch -20
            print("Early stopping")
            break

    model.load_state_dict(torch.load(f'checkpoint_KGEM_{args.mode}_{args.split}_{args.valid_auc}_{args.patient_epoch}_{args.drdi_feature}_{args.nhead}_{args.num_layers}_{args.dim_feedforward}_{i}.pt'))
    end = time.time()
    print('Training time: {:.4} seconds'.format(end - start))
    result_model = model
    return result_model,flag