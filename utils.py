import numpy as np
import pandas as pd
import logging
import os
import pickle
import scipy.sparse as sp
import sys
import torch
from scipy.sparse import linalg
from rpca import inexact_augmented_lagrange_multiplier
from tqdm import tqdm
import scipy.io as scio

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def loadmetrP288():
    # Traffic
    df = pd.read_hdf('data/metr/metr-la.h5')
    Traffic = df.values
    # train/val/test
    num_step = int(df.shape[0] / 288)
    Traffic = Traffic[: num_step * 288]
    # X, Y
    P = 288
    Q = 288
    TrafficX, TrafficY = seq2instance(Traffic, P, Q)
    # normalization
    mean, std = np.mean(TrafficX), np.std(TrafficX)
    TrafficX = (TrafficX - mean) / std

    # temporal embedding
    Time = df.index
    dayofweek = np.reshape(Time.weekday, newshape=(-1, 1))
    timeofday = (Time.hour * 3600 + Time.minute * 60 + Time.second) \
                // (5 * 60)
    timeofday = np.reshape(timeofday, newshape=(-1, 1))
    Time = np.concatenate((dayofweek, timeofday), axis=-1)
    # train/val/test
    Time = Time[: num_step * 288]
    # shape = (num_sample, P + Q, 2)
    TimeTE = seq2instance(Time, P, Q)
    TimeTE = np.concatenate(TimeTE, axis=1).astype(np.int32)
    permutation = np.random.permutation(num_step - 1)
    TrafficX = TrafficX[permutation]
    TrafficY = TrafficY[permutation]
    TimeTE = TimeTE[permutation]
    train_steps = int(0.7 * num_step)
    test_steps = int(0.2 * num_step)
    val_steps = num_step - train_steps - test_steps
    trainX = TrafficX[: train_steps]
    trainTE = TimeTE[: train_steps]
    trainY = TrafficY[: train_steps]
    valX = TrafficX[round(train_steps): round(train_steps + val_steps)]
    valTE = TimeTE[round(train_steps): round(train_steps + val_steps)]
    valY = TrafficY[round(train_steps): round(train_steps + val_steps)]
    testX = TrafficX[round(train_steps + val_steps):round(train_steps + val_steps + test_steps)]
    testTE = TimeTE[round(train_steps + val_steps):round(train_steps + val_steps + test_steps)]
    testY = TrafficY[round(train_steps + val_steps):round(train_steps + val_steps + test_steps)]

    return (trainX, trainTE, trainY, valX, valTE, valY, testX, testTE, testY,
            mean, std)


def loadbayP288():
    # Traffic
    df = pd.read_hdf('data/bay/PeMS.h5')
    Traffic = df.values
    # train/val/test
    num_step = int(df.shape[0] / 288)
    Traffic = Traffic[: num_step * 288]
    # X, Y
    P = 288
    Q = 288
    TrafficX, TrafficY = seq2instance(Traffic, P, Q)
    # normalization
    mean, std = np.mean(TrafficX), np.std(TrafficX)
    TrafficX = (TrafficX - mean) / std

    # temporal embedding
    Time = df.index
    dayofweek = np.reshape(Time.weekday, newshape=(-1, 1))
    timeofday = (Time.hour * 3600 + Time.minute * 60 + Time.second) \
                // (5 * 60)
    timeofday = np.reshape(timeofday, newshape=(-1, 1))
    Time = np.concatenate((dayofweek, timeofday), axis=-1)
    # train/val/test
    Time = Time[: num_step * 288]
    # shape = (num_sample, P + Q, 2)
    TimeTE = seq2instance(Time, P, Q)
    TimeTE = np.concatenate(TimeTE, axis=1).astype(np.int32)
    permutation = np.random.permutation(num_step - 1)
    TrafficX = TrafficX[permutation]
    TrafficY = TrafficY[permutation]
    TimeTE = TimeTE[permutation]
    train_steps = int(0.7 * num_step)
    test_steps = int(0.2 * num_step)
    val_steps = num_step - train_steps - test_steps
    trainX = TrafficX[: train_steps]
    trainTE = TimeTE[: train_steps]
    trainY = TrafficY[: train_steps]
    valX = TrafficX[round(train_steps): round(train_steps + val_steps)]
    valTE = TimeTE[round(train_steps): round(train_steps + val_steps)]
    valY = TrafficY[round(train_steps): round(train_steps + val_steps)]
    testX = TrafficX[round(train_steps + val_steps):round(train_steps + val_steps + test_steps)]
    testTE = TimeTE[round(train_steps + val_steps):round(train_steps + val_steps + test_steps)]
    testY = TrafficY[round(train_steps + val_steps):round(train_steps + val_steps + test_steps)]

    return (trainX, trainTE, trainY, valX, valTE, valY, testX, testTE, testY,
            mean, std)


def load_KB(str):
    week = scio.loadmat('data/' + str + ".mat")
    return torch.tensor(week[str])


def gen_KB(trainX, trainTE, mean, std, str):
    daynum, daylen, num_nodes = trainX.shape  # 82 288 207
    trainXsort = np.zeros(trainX.shape)
    trainTEsort = np.zeros(trainTE.shape)
    ind = 0
    week0 = week1 = week2 = week3 = week4 = week5 = week6 = 0
    for i in range(0, daynum):
        if trainTE[i, 0, 0] == 0:
            trainTEsort[ind] = trainTE[i]
            trainXsort[ind] = trainX[i]
            ind = ind + 1
            week0 = week0 + 1
    for i in range(0, daynum):
        if trainTE[i, 0, 0] == 1:
            trainTEsort[ind] = trainTE[i]
            trainXsort[ind] = trainX[i]
            ind = ind + 1
            week1 = week1 + 1
    for i in range(0, daynum):
        if trainTE[i, 0, 0] == 2:
            trainTEsort[ind] = trainTE[i]
            trainXsort[ind] = trainX[i]
            ind = ind + 1
            week2 = week2 + 1
    for i in range(0, daynum):
        if trainTE[i, 0, 0] == 3:
            trainTEsort[ind] = trainTE[i]
            trainXsort[ind] = trainX[i]
            ind = ind + 1
            week3 = week3 + 1
    for i in range(0, daynum):
        if trainTE[i, 0, 0] == 4:
            trainTEsort[ind] = trainTE[i]
            trainXsort[ind] = trainX[i]
            ind = ind + 1
            week4 = week4 + 1
    for i in range(0, daynum):
        if trainTE[i, 0, 0] == 5:
            trainTEsort[ind] = trainTE[i]
            trainXsort[ind] = trainX[i]
            ind = ind + 1
            week5 = week5 + 1
    for i in range(0, daynum):
        if trainTE[i, 0, 0] == 6:
            trainTEsort[ind] = trainTE[i]
            trainXsort[ind] = trainX[i]
            ind = ind + 1
            week6 = week6 + 1
    trainXsort = trainXsort * std + mean
    trainXsort = np.swapaxes(trainXsort, 1, 2).reshape(daynum, -1)

    weekind = week0
    A, _ = inexact_augmented_lagrange_multiplier(
        np.squeeze(trainXsort[0:weekind, :]))
    week0A = np.mean(A.reshape(week0, num_nodes, 288), 0)
    A, _ = inexact_augmented_lagrange_multiplier(
        np.squeeze(trainXsort[weekind:weekind + week1, :]))
    week1A = np.mean(A.reshape(week1, num_nodes, 288), 0)
    weekind = weekind + week1
    A, _ = inexact_augmented_lagrange_multiplier(
        np.squeeze(trainXsort[weekind:weekind + week2, :]))
    week2A = np.mean(A.reshape(week2, num_nodes, 288), 0)
    weekind = weekind + week2
    A, _ = inexact_augmented_lagrange_multiplier(
        np.squeeze(trainXsort[weekind:weekind + week3, :]))
    week3A = np.mean(A.reshape(week3, num_nodes, 288), 0)
    weekind = weekind + week3
    A, _ = inexact_augmented_lagrange_multiplier(
        np.squeeze(trainXsort[weekind:weekind + week4, :]))
    week4A = np.mean(A.reshape(week4, num_nodes, 288), 0)
    weekind = weekind + week4
    A, _ = inexact_augmented_lagrange_multiplier(
        np.squeeze(trainXsort[weekind:weekind + week5, :]))
    week5A = np.mean(A.reshape(week5, num_nodes, 288), 0)
    weekind = weekind + week5
    A, _ = inexact_augmented_lagrange_multiplier(
        np.squeeze(trainXsort[weekind:weekind + week6, :]))
    week6A = np.mean(A.reshape(week6, num_nodes, 288), 0)
    week = np.hstack([week0A, week1A, week2A, week3A, week4A, week5A, week6A])
    scio.savemat("data/" + str + ".mat", {str: week})


def gen_KB0(trainX, trainTE, mean, std, str):
    daynum, daylen, num_nodes = trainX.shape  # 82 288 207 
    week = np.zeros((num_nodes, 2016)) + 0.000000001
    #     week = np.random.rand(num_nodes, 2016)
    scio.savemat("data/" + str + ".mat", {str: week})


def geneTimeAdj(timelen, L=12):
    L = L + 1
    T = round(timelen / 3)
    adjA = torch.zeros(timelen, timelen)
    adj1 = torch.zeros(T, T)
    adj2 = torch.zeros(T, T)
    adj3 = torch.zeros(T, T)
    # adj1.detach().numpy()
    for i in range(T):
        for j in range(T):
            distance1 = np.abs(j - i)
            distance2 = j - i
            if distance1 < L:
                adj1[i, j] = 1 - distance1 / L
            if distance2 > T - L:
                adj1[i, j] = 1 - (T - distance2) / L
    for i in range(T):
        for j in range(T):
            distance1 = np.abs(j - i)
            distance3 = i - j
            if distance1 < L:
                adj2[i, j] = 1 - distance1 / L
            if distance3 > T - L:
                adj2[i, j] = 1 - (T - distance3) / L
    for i in range(T):
        for j in range(T):
            distance1 = np.abs(j - i)
            distance2 = j - i
            distance3 = i - j
            if distance1 < L:
                adj3[i, j] = 1 - distance1 / L
            if distance2 > T - L:
                adj3[i, j] = 1 - (T - distance2) / L
            if distance3 > T - L:
                adj3[i, j] = 1 - (T - distance3) / L
    for i in range(3):
        adjA[T * i:T * i + T, T * 0:T * 0 + T] = adj1
        adjA[T * i:T * i + T, T * 1:T * 1 + T] = adj3
        adjA[T * i:T * i + T, T * 2:T * 2 + T] = adj2
    return adjA


def get_adjacency_matrix(distance_df_filename, num_of_vertices,
                         type_='connectivity', id_filename=None):
    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information

    num_of_vertices: int, the number of vertices

    type_: str, {connectivity, distance}

    Returns
    ----------
    A: np.ndarray, adjacency matrix

    '''
    import csv

    A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                 dtype=np.float32)

    if id_filename:
        with open(id_filename, 'r') as f:
            id_dict = {int(i): idx
                       for idx, i in enumerate(f.read().strip().split('\n'))}
        with open(distance_df_filename, 'r') as f:
            f.readline()
            reader = csv.reader(f)
            for row in reader:
                if len(row) != 3:
                    continue
                i, j, distance = int(row[0]), int(row[1]), float(row[2])
                A[id_dict[i], id_dict[j]] = 1
                A[id_dict[j], id_dict[i]] = 1
        return A

    # Fills cells in the matrix with distances.
    with open(distance_df_filename, 'r') as f:
        f.readline()
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 3:
                continue
            i, j, distance = int(row[0]), int(row[1]), float(row[2])
            if type_ == 'connectivity':
                A[i, j] = 1
                A[j, i] = 1
            elif type == 'distance':
                A[i, j] = 1 / distance
                A[j, i] = 1 / distance
            else:
                raise ValueError("type_ error, must be "
                                 "connectivity or distance!")
    return A


def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


def circshift(matrix, shiftnum1, shiftnum2):
    h, w = matrix.shape
    matrix = torch.vstack((matrix[(h - shiftnum1):, :], matrix[:(h - shiftnum1), :]))
    matrix = torch.hstack((matrix[:, (w - shiftnum2):], matrix[:, :(w - shiftnum2)]))
    return matrix


def load_graph_data(pkl_filename):
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    return sensor_ids, sensor_id_to_ind, adj_mx


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


# log string
def log_string(log, string):
    log.write(string + '\n')
    log.flush()
    print(string)


def seq2instance(data, P, Q):
    num_step, dims = data.shape
    daynum = round(num_step / 288)
    num_sample = daynum - 1
    x = np.zeros(shape=(num_sample, P, dims))
    y = np.zeros(shape=(num_sample, Q, dims))
    for i in range(num_sample):
        x[i] = data[i * P: round(i * P + P)]
        y[i] = data[round(i * P + P): round(i * P + P + Q)]
    return x, y
