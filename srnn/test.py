'''
Train script for the structural RNN model
introduced in https://arxiv.org/abs/1511.05298

Author : Anirudh Vemula
Date : 29th March 2017
'''

import argparse
import os
import pickle
import time
import numpy as np
import torch
import re
from torch.autograd import Variable

from utils import DataLoader
from st_graph import ST_GRAPH
from model import SRNN
from criterion import Gaussian2DLikelihood
from dataloader import NewDataLoader, TrajectoryDataset


def main():
    parser = argparse.ArgumentParser()

    # RNN size
    parser.add_argument('--human_node_rnn_size', type=int, default=128,
                        help='Size of Human Node RNN hidden state')
    parser.add_argument('--human_human_edge_rnn_size', type=int, default=256,
                        help='Size of Human Human Edge RNN hidden state')

    # Input and output size
    parser.add_argument('--human_node_input_size', type=int, default=2,
                        help='Dimension of the node features')
    parser.add_argument('--human_human_edge_input_size', type=int, default=2,
                        help='Dimension of the edge features')
    parser.add_argument('--human_node_output_size', type=int, default=2,
                        help='Dimension of the node output')

    # Embedding size
    parser.add_argument('--human_node_embedding_size', type=int, default=64,
                        help='Embedding size of node features')
    parser.add_argument('--human_human_edge_embedding_size', type=int, default=64,
                        help='Embedding size of edge features')

    # Attention vector dimension
    parser.add_argument('--attention_size', type=int, default=64,
                        help='Attention size')

    # Sequence length
    parser.add_argument('--seq_length', type=int, default=39,
                        help='Sequence length')
    parser.add_argument('--pred_length', type=int, default=10,
                        help='Predicted sequence length')

    # Batch size
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size')

    # Number of epochs
    parser.add_argument('--num_epochs', type=int, default=200,
                        help='number of epochs')

    # Gradient value at which it should be clipped
    parser.add_argument('--grad_clip', type=float, default=10.,
                        help='clip gradients at this value')
    # Lambda regularization parameter (L2)
    parser.add_argument('--lambda_param', type=float, default=0.00005,
                        help='L2 regularization parameter')

    # Learning rate parameter
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='learning rate')
    # Decay rate for the learning rate parameter
    parser.add_argument('--decay_rate', type=float, default=0.99,
                        help='decay rate for the optimizer')

    # Dropout rate
    parser.add_argument('--dropout', type=float, default=0,
                        help='Dropout probability')

    # The leave out dataset
    parser.add_argument('--leaveDataset', type=int, default=0,
                        help='The dataset index to be left out in training')

    # Train from checkpoint
    parser.add_argument('--model', type=str, default='',
                        help='The checkpoint model path')

    args = parser.parse_args()

    test(args)


def test(args):
    data = np.load('./data_load2.npz')
    keys = list(data.keys())
    keys_train = keys[0: -3]
    keys_eval = keys[-2:]
    dataset_eval = TrajectoryDataset(data, args.seq_length - args.pred_length + 1, args.pred_length, keys_eval)
    dataloader = NewDataLoader(dataset_eval, batch_size=args.batch_size)

    stgraph = ST_GRAPH(1, args.seq_length + 1)

    net = torch.load(args.model, map_location=torch.device('cpu'))
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    dataloader.reset_batch_pointer()

    # For each batch
    for batch in range(dataloader.num_batches):
        if batch >= 10:
            break
        start = time.time()

        # Get batch data
        # x, _, _, d = dataloader.next_batch(randomUpdate=True)
        x = dataloader.next_batch()

        # Loss for this batch
        # loss_batch = 0
        batch_loss = Variable(torch.zeros(1))
        # batch_loss = batch_loss.cuda()

        # For each sequence in the batch
        for sequence in range(dataloader.batch_size):
            # Construct the graph for the current sequence
            stgraph.readGraph([x[sequence]])

            nodes, edges, nodesPresent, edgesPresent = stgraph.getSequence()

            # Convert to cuda variables
            # nodes = Variable(torch.from_numpy(nodes).float()).cuda()
            # edges = Variable(torch.from_numpy(edges).float()).cuda()
            nodes = Variable(torch.from_numpy(nodes).float())
            edges = Variable(torch.from_numpy(edges).float())

            # Define hidden states
            numNodes = nodes.size()[1]
            # hidden_states_node_RNNs = Variable(torch.zeros(numNodes, args.human_node_rnn_size)).cuda()
            # hidden_states_edge_RNNs = Variable(torch.zeros(numNodes * numNodes, args.human_human_edge_rnn_size)).cuda()
            hidden_states_node_RNNs = Variable(torch.zeros(numNodes, args.human_node_rnn_size))
            hidden_states_edge_RNNs = Variable(torch.zeros(numNodes * numNodes, args.human_human_edge_rnn_size))

            # cell_states_node_RNNs = Variable(torch.zeros(numNodes, args.human_node_rnn_size)).cuda()
            # cell_states_edge_RNNs = Variable(torch.zeros(numNodes * numNodes, args.human_human_edge_rnn_size)).cuda()
            cell_states_node_RNNs = Variable(torch.zeros(numNodes, args.human_node_rnn_size))
            cell_states_edge_RNNs = Variable(torch.zeros(numNodes * numNodes, args.human_human_edge_rnn_size))

            # Zero out the gradients
            net.zero_grad()

            # Forward prop
            outputs, _, _, _, _, _ = net(nodes[:args.seq_length], edges[:args.seq_length], nodesPresent[:-1],
                                         edgesPresent[:-1], hidden_states_node_RNNs, hidden_states_edge_RNNs,
                                         cell_states_node_RNNs, cell_states_edge_RNNs)

            # print(outputs.shape)
            # print(nodes.shape)
            # print(nodesPresent)
            # print('----------------')
            # raise KeyError

            # Compute loss
            # loss = Gaussian2DLikelihood(outputs, nodes[1:], nodesPresent[1:], args.pred_length)
            loss = net.get_square_loss(outputs, nodes[1:])
            batch_loss = batch_loss + loss
            optimizer.step()
            # print(loss)
            stgraph.reset()
        end = time.time()
        batch_loss = batch_loss / dataloader.batch_size
        # loss_batch = loss_batch / dataloader.batch_size
        # loss_epoch += loss_batch
        loss_batch = batch_loss.item()

        print('{}/{} , test_loss = {:.3f}, time/batch = {:.3f}'.format(batch, dataloader.num_batches,loss_batch, end - start))


if __name__ == '__main__':
    main()
