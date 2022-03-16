import argparse

import torch
import numpy as np

import stugcn
import utils


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=1,
                    help='batch size')
parser.add_argument('--max_epoch', type=int, default=200,
                    help='epoch to run')
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='initial learning rate')
parser.add_argument('--model_file', default='model/METR1.pth',
                    help='save the model to disk')
# parser.add_argument('--log_myvn0', default='log/log(bay)myvn0',
#                     help='log file')
# parser.add_argument('--log_myvn10', default='log/log(bay)myvn10',
#                     help='log file')
# parser.add_argument('--log_myvn20', default='log/log(bay)myvn20',
#                     help='log file')
# parser.add_argument('--log_myvn30', default='log/log(bay)myvn30',
#                     help='log file')
# parser.add_argument('--log_myvn40', default='log/log(bay)myvn40',
#                     help='log file')
# parser.add_argument('--log_myvn50', default='log/log(bay)myvn50',
#                     help='log file')

parser.add_argument('--log_myvn0', default='log/log(metr)myvn0',
                    help='log file')
parser.add_argument('--log_myvn10', default='log/log(metr)myvn10',
                    help='log file')
parser.add_argument('--log_myvn20', default='log/log(metr)myvn20',
                    help='log file')
parser.add_argument('--log_myvn30', default='log/log(metr)myvn30',
                    help='log file')
parser.add_argument('--log_myvn40', default='log/log(metr)myvn40',
                    help='log file')
parser.add_argument('--log_myvn50', default='log/log(metr)myvn50',
                    help='log file')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args = parser.parse_args()

for i in range(5):
    trainX, trainTE, trainY, valX, valTE, valY, testX, testTE, testY, mean, std = utils.loadmetrP288()
    _, _, adj_mx = utils.load_graph_data('data/metr/adj_mx.pkl')
    utils.gen_KB(trainX, trainTE, mean, std, 'b') # metr_kb
    # sensor_ids, sensor_id_to_ind, adj_mx = utils.load_graph_data('data/bay/adj_mx_bay.pkl')
    # trainX, trainTE, trainY, valX, valTE, valY, testX, testTE, testY, mean, std = utils.loadbayP288()
    # utils.gen_KB(trainX,trainTE,mean, std, 'b') # bay_kb
    adj_mx = torch.from_numpy(adj_mx).float().to(device)
    daynum, daylen, num_nodes = trainX.shape  # 82 288 207
    '''myvn0'''
    log = open(args.log_myvn0, 'a')
    model = stugcn.Model(adj_mx, input_dimS=288, input_dimT=num_nodes, hidden_dimS=288, hidden_dimT=num_nodes,
                        num_nodes=num_nodes, virtualnode=0).to(device)
    stugcn.train(model, trainX, trainTE, trainY, valX, valTE, valY, testX, testTE, testY, mean, std, args, log)
    stugcn.test(model, testX, testTE, testY, mean, std, args, log)
    '''myvn10'''
    log = open(args.log_myvn10, 'a')
    model = stugcn.Model(adj_mx, input_dimS=288, input_dimT=num_nodes, hidden_dimS=288, hidden_dimT=num_nodes,
                        num_nodes=num_nodes, virtualnode=10).to(device)
    stugcn.train(model, trainX, trainTE, trainY, valX, valTE, valY, testX, testTE, testY, mean, std, args, log)
    stugcn.test(model, testX, testTE, testY, mean, std, args, log)
    '''myvn20'''
    log = open(args.log_myvn20, 'a')
    model = stugcn.Model(adj_mx, input_dimS=288, input_dimT=num_nodes, hidden_dimS=288, hidden_dimT=num_nodes,
                        num_nodes=num_nodes, virtualnode=20).to(device)
    stugcn.train(model, trainX, trainTE, trainY, valX, valTE, valY, testX, testTE, testY, mean, std, args, log)
    stugcn.test(model, testX, testTE, testY, mean, std, args, log)
    '''myvn30'''
    log = open(args.log_myvn30, 'a')
    model = stugcn.Model(adj_mx, input_dimS=288, input_dimT=num_nodes, hidden_dimS=288, hidden_dimT=num_nodes,
                        num_nodes=num_nodes, virtualnode=30).to(device)
    stugcn.train(model, trainX, trainTE, trainY, valX, valTE, valY, testX, testTE, testY, mean, std, args, log)
    stugcn.test(model, testX, testTE, testY, mean, std, args, log)
    '''myvn40'''
    log = open(args.log_myvn40, 'a')
    model = stugcn.Model(adj_mx, input_dimS=288, input_dimT=num_nodes, hidden_dimS=288, hidden_dimT=num_nodes,
                        num_nodes=num_nodes, virtualnode=40).to(device)
    stugcn.train(model, trainX, trainTE, trainY, valX, valTE, valY, testX, testTE, testY, mean, std, args, log)
    stugcn.test(model, testX, testTE, testY, mean, std, args, log)
    '''myvn50'''
    log = open(args.log_myvn50, 'a')
    model = stugcn.Model(adj_mx, input_dimS=288, input_dimT=num_nodes, hidden_dimS=288, hidden_dimT=num_nodes,
                        num_nodes=num_nodes, virtualnode=50).to(device)
    stugcn.train(model, trainX, trainTE, trainY, valX, valTE, valY, testX, testTE, testY, mean, std, args, log)
    stugcn.test(model, testX, testTE, testY, mean, std, args, log)
