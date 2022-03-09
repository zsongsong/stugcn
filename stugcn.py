import argparse
import math
import time
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import os
import random
from utils import geneTimeAdj, circshift, load_KB
from tqdm import tqdm

import EarlyStopping
import utils
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def tensor_from_numpy(x, device):
    return torch.from_numpy(x).to(device)


class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.rand(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.rand(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, adjacency, input_feature):
        batch_size, num_nodes, input_size = input_feature.shape
        input_feature = torch.reshape(input_feature, shape=[batch_size * num_nodes, input_size])
        support = torch.matmul(input_feature, self.weight)
        support = torch.reshape(support, shape=[batch_size, num_nodes, self.output_dim])
        support = support.permute(1, 0, 2)
        support = torch.reshape(support, shape=[num_nodes, batch_size * self.output_dim])
        output = torch.sparse.mm(adjacency, support)
        output = torch.reshape(output, shape=[num_nodes, batch_size, self.output_dim])
        output = output.permute(1, 0, 2)
        if self.use_bias:
            output += self.bias
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.input_dim) + ' -> ' \
               + str(self.output_dim) + ')'


class Satt(nn.Module):
    def __init__(self, adjacency, k, input_dim, hidden_dim):
        super(Satt, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.gcn1 = GraphConvolution(input_dim, hidden_dim)
        self.gcn11 = GraphConvolution(hidden_dim, hidden_dim)
        self.gcn2 = GraphConvolution(hidden_dim, hidden_dim)
        self.gcn22 = GraphConvolution(hidden_dim, hidden_dim)
        self.gcn3 = GraphConvolution(hidden_dim, hidden_dim)
        self.gcn33 = GraphConvolution(hidden_dim, hidden_dim)
        self.k = k

        self.adjk1 = (torch.ones(k, adjacency.shape[0]) + torch.randn(k, adjacency.shape[0])).to(device)
        self.adjkk = torch.eye(k, k).to(device)

        adjacency1 = torch.cat((adjacency, self.adjk1), dim=0)
        atemp = torch.cat((self.adjk1.T, self.adjkk), dim=0)
        adjacency1 = torch.cat((adjacency1, atemp), dim=1)

        degree = adjacency1.sum(1).reshape(-1)
        d_hat = torch.diag(torch.pow(degree, -0.5).flatten())
        self.adjacency1 = d_hat * adjacency1 * d_hat

    def forward(self, input_feature):
        batch_size, num_nodes, input_size = input_feature.shape
        glok = torch.zeros(batch_size, self.k, input_size).to(device)
        input = torch.cat((input_feature.float(), glok), dim=1)
        gcn1 = torch.tanh(self.gcn1(self.adjacency1, input))
        gcn11 = torch.sigmoid(self.gcn11(self.adjacency1, input))
        output = torch.mul(gcn1, gcn11)
        gcn2 = torch.tanh(self.gcn2(self.adjacency1, output))
        gcn22 = torch.sigmoid(self.gcn22(self.adjacency1, output))
        output = torch.mul(gcn2, gcn22)
        gcn3 = torch.tanh(self.gcn3(self.adjacency1, output))
        gcn33 = torch.sigmoid(self.gcn33(self.adjacency1, output))
        gcn3 = torch.mul(gcn3, gcn33)
        c = torch.cat((torch.eye(num_nodes, num_nodes),
                       torch.zeros(self.k, num_nodes)), dim=0).to(device)
        gcn3 = gcn3.permute(1, 0, 2)
        gcn3 = torch.reshape(gcn3, shape=[num_nodes + self.k, batch_size * self.hidden_dim])
        gcn3 = torch.mm(c.T, gcn3)
        output = torch.reshape(gcn3, shape=[num_nodes, batch_size, self.hidden_dim])
        output = output.permute(1, 0, 2)
        return output


class Tatt(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_nodes,  L, timelen=288 * 7):
        super(Tatt, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.timelen = timelen
        self.gcn1 = GraphConvolution(input_dim, hidden_dim)
        self.gcn11 = GraphConvolution(hidden_dim, hidden_dim)
        self.gcn2 = GraphConvolution(hidden_dim, hidden_dim)
        self.gcn22 = GraphConvolution(hidden_dim, hidden_dim)
        self.gcn3 = GraphConvolution(hidden_dim, hidden_dim)
        self.gcn33 = GraphConvolution(hidden_dim, hidden_dim)
        self.conv1 = nn.Conv1d(in_channels=num_nodes, out_channels=num_nodes, kernel_size=(577,))
        adjacency = geneTimeAdj(288 * 3, L)

        degree = adjacency.sum(1).reshape(-1)
        d_hat = torch.diag(torch.pow(degree, -0.5).flatten())
        adjacency = d_hat * adjacency * d_hat
        self.adjacency = adjacency.float().to(device)
        self.c = torch.cat((torch.eye(288, 288), torch.zeros(timelen - 288, 288)))
        self.c1 = torch.cat((torch.eye(288 * 3, 288 * 3), torch.zeros(timelen - 288 * 3, 288 * 3)))
        self.x = torch.rand(timelen, input_dim).to(device)
        x = load_KB("b").T.float()  # bay_kb metr_kb
        mean, std = torch.mean(x), torch.std(x)
        self.x = ((x - mean) / std).to(device)

    def forward(self, input_feature, week=0, hour=0):
        batch_size, num_nodes, input_size = input_feature.shape
        c = torch.zeros(batch_size, self.timelen, 288)
        c1 = torch.zeros(batch_size, self.timelen, 288 * 3)
        for i in range(batch_size):
            c[i] = circshift(self.c, week[i] * 288 + hour[i], 0)
            c1[i] = circshift(self.c1, week[i] * 288 + hour[i] - 288, 0)  
        c = c.to(device)
        c1 = c1.to(device)

        x = self.x.unsqueeze(0).expand(batch_size, self.timelen, self.input_dim)
        # 下采样
        cTx = torch.matmul(c.permute(0, 2, 1), x)
        # 上采样
        cin = torch.matmul(c, (input_feature.permute(0, 2, 1) - cTx))
        c1T = c1.permute(0, 2, 1)
        temp = torch.matmul(c1T, x + cin)

        gcn1 = torch.tanh(self.gcn1(self.adjacency, temp))
        gcn11 = torch.sigmoid(self.gcn11(self.adjacency, temp))
        output = torch.mul(gcn1, gcn11)
        gcn2 = torch.tanh(self.gcn2(self.adjacency, output))
        gcn22 = torch.sigmoid(self.gcn22(self.adjacency, output))
        output = torch.mul(gcn2, gcn22)
        gcn3 = torch.tanh(self.gcn3(self.adjacency, output))
        gcn33 = torch.sigmoid(self.gcn33(self.adjacency, output))
        output = torch.mul(gcn3, gcn33)
        #下采样
        output = output.permute(0, 2, 1)
        output = self.conv1(output)
        return output


class Model(nn.Module):
    def __init__(self, adjacency, input_dimS, input_dimT, hidden_dimS,
                 hidden_dimT, num_nodes, L=6, virtualnode=30):
        super(Model, self).__init__()
        self.daylen = 288
        self.Satt1 = Satt(adjacency, virtualnode, input_dimS, hidden_dimS)
        self.Tatt1 = Tatt(input_dimT, hidden_dimT, num_nodes, L)

        self.conv1d = nn.Conv1d(in_channels=64 * num_nodes,
                                out_channels=num_nodes, kernel_size=(1,))
        self.gru = nn.GRU(input_size=1, hidden_size=64, num_layers=1)

    def forward(self, input_feature, week=0, hour=0):
        gcn1 = self.Satt1(input_feature)
        gcn2 = self.Tatt1(input_feature, week=week, hour=hour)
        gcnx = torch.squeeze(torch.add(gcn1, gcn2))
        gcnx = gcnx.unsqueeze(-1)

        pred, _ = self.gru(gcnx)
        pred = pred.permute(1, 0, 2)
        output = pred.reshape(self.daylen, -1)
        output = output.unsqueeze(0)
        output = output.permute(0, 2, 1)
        pred = self.conv1d(output)


        return pred


def res(model, valX, valTE, valY, mean, std, args, log):
    start = time.time()
    model.eval()  # 评估模式, 这会关闭dropout
    num_val = valX.shape[0]
    pred = []
    label = []
    num_batch = math.ceil(num_val / args.batch_size)

    with torch.no_grad():
        for batch_idx in tqdm(range(num_batch)):
            start_idx = batch_idx * args.batch_size
            end_idx = min(num_val, (batch_idx + 1) * args.batch_size)

            X = torch.from_numpy(valX[start_idx: end_idx]).float().to(device)
            y = valY[start_idx: end_idx]
            te = torch.from_numpy(valTE[start_idx: end_idx]).to(device)
            X = X.permute(0, 2, 1)
            y_hat = model(X, week=te[:, 0, 0], hour=te[:, 0, 1])
            y_hat = y_hat.permute(0, 2, 1)
            pred.append(y_hat.cpu().numpy() * std + mean)
            label.append(y)

    pred = np.concatenate(pred, axis=0)
    label = np.concatenate(label, axis=0)

    mae, rmse, mape, wape = metric(pred, label)
    utils.log_string(log, 'average, mae: %.4f, rmse: %.4f, mape: %.4f, wape: %.4f' % (mae, rmse, mape, wape))
    utils.log_string(log, 'eval time: %.1f' % (time.time() - start))


    return mae, rmse, mape


def train(model, trainX, trainTE, trainY, valX, valTE, valY, testX, testTE, testY, mean, std, args, log):
    num_train = trainX.shape[0]
    min_loss = 10000000.0

    model.train()
    early_stopping = EarlyStopping.EarlyStop(patience=7, verbose=True, delta=0)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.learning_rate)

    it = 0
    for epoch in tqdm(range(1, args.max_epoch + 1)):
        model.train()
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        permutation = np.random.permutation(num_train)
        trainX = trainX[permutation]
        trainTE = trainTE[permutation]
        trainY = trainY[permutation]
        num_batch = math.ceil(num_train / args.batch_size)

        for batch_idx in tqdm(range(num_batch)):
            start_idx = batch_idx * args.batch_size
            end_idx = min(num_train, (batch_idx + 1) * args.batch_size)

            X = torch.from_numpy(trainX[start_idx: end_idx]).float().to(device)
            y = torch.from_numpy(trainY[start_idx: end_idx]).float().to(device)

            te = torch.from_numpy(trainTE[start_idx: end_idx]).to(device)
            X = X.permute(0, 2, 1)
            optimizer.zero_grad()
            y_hat = model(X, week=te[:, 0, 0], hour=te[:, 0, 1])
            y_hat = y_hat.permute(0, 2, 1)
            loss = _compute_loss(y, y_hat * std + mean)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            train_l_sum += loss.cpu().item()

            batch_count += 1

        utils.log_string(log, 'epoch %d, lr %.6f, loss %.4f, time %.1f sec'
                         % (epoch, optimizer.param_groups[0]['lr'], train_l_sum / batch_count, time.time() - start))

        mae, rmse, mape = res(model, valX, valTE, valY, mean, std, args, log)
        early_stopping(mae, model)
        # 若满足 early stopping 要求，每一个学习率至少使用2次
        it = it + 1
        if early_stopping.early_stop and (it >= 2):
            it = 0
            print("Early stopping")
            # 结束模型训练
            for p in optimizer.param_groups:
                p['lr'] *= 0.9
            # 获得 early stopping 时的模型参数
            model.load_state_dict(torch.load('checkpoint.pt'))
            early_stopping.early_stop = False
        if mae < min_loss:
            min_loss = mae
            torch.save(model, args.model_file)
            test(model, testX, testTE, testY, mean, std, args, log)


def test(model, valX, valTE, valY, mean, std, args, log):
    model = torch.load(args.model_file)
    mae, rmse, mape = res(model, valX, valTE, valY, mean, std, args, log)
    return mae, rmse, mape


def _compute_loss(y_true, y_predicted):
    return masked_mae(y_predicted, y_true, 0.0)


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def metric(pred, label):
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = np.not_equal(label, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(pred, label)).astype(np.float32)
        rmse = np.square(mae)
        mape = np.divide(mae, label)
        mae = np.nan_to_num(mae * mask)
        wape = np.divide(np.sum(mae), np.sum(label))
        mae = np.mean(mae)
        rmse = np.nan_to_num(rmse * mask)
        rmse = np.sqrt(np.mean(rmse))
        mape = np.nan_to_num(mape * mask)
        mape = np.mean(mape)
    return mae, rmse, mape, wape

