import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from arguments import get_args
from transformers import Random
from models import GCN, GraphCNN
from utils import test

import ast
import copy
from sklearn.model_selection import KFold
from collections import Counter
from torch_geometric.transforms.one_hot_degree import OneHotDegree

torch.set_num_threads(20)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):
    dataset = TUDataset(root=f'./Datasets/{args.dataset}', name=args.dataset)

    print(dataset)

    if dataset.num_features == 0 or args.initialize_node_features:
        if args.randomize:
            print('Using random node features')
            dataset.transform = Random()
        else:
            print('Using degree node features')
            max_degree = -1
            for data in dataset:
                edge_index = data.edge_index
                degrees = Counter(list(map(int, edge_index[0])))
                if max_degree < max(degrees.values()):
                    max_degree = max(degrees.values())

            dataset.transform = OneHotDegree(max_degree=max_degree, cat=False)

    # if dataset.num_features == 0:
    #     dataset.transform = Random()
    # elif args.randomize:
    #     dataset.transform = Random(vector_size=args.randomize)



    print("Use clean dataset: {}".format(bool(args.provided_idx)))
    if args.provided_idx:
        graph_idx = get_clean_graph_indices(args.dataset)
        dataset_size = len(graph_idx)
        print(f"Dataset size: {len(dataset)} -> {dataset_size}")
        shuffled_idx = copy.deepcopy(graph_idx)

    else:
        dataset_size = len(dataset)
        shuffled_idx = list(range(dataset_size))
        print(f"Dataset size: {len(dataset)}")

    global_train_acc = []
    global_test_acc = []
    global_loss = []
    kf = KFold(10, shuffle=True)

    for train_index, test_index in kf.split(shuffled_idx):
        test_dataset = [dataset[int(idx)] for idx in test_index]
        train_dataset = [dataset[int(idx)] for idx in train_index]

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

        classes = set()
        for b in train_loader:
            classes = classes.union(list(map(int, b.y)))
        print('Possible train classes', classes)

        classes = set()
        for b in test_loader:
            classes = classes.union(list(map(int, b.y)))
        print('Possible test classes', classes)

        #model = GCN(dataset.num_features, dataset.num_classes, args.hidden, args.dropout).to(device)

        model = GraphCNN(5, 2, dataset.num_features, 64, 
                        dataset.num_classes, 0.5, False, 
                        "sum", "sum", device).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        for epoch in range(1, args.num_epochs):

            model.train()

            if epoch % 50 == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 0.5 * param_group['lr']

            train_loss = 0
            for data in train_loader:
                data = data.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = F.nll_loss(output, data.y)
                loss.backward()
                train_loss += loss.item() * data.num_graphs
                optimizer.step()

            train_loss = train_loss / len(train_dataset)

            train_acc = test(model, train_loader, device)
            test_acc = test(model, test_loader, device)

            print('Epoch: {:03d}, Train Loss: {:.7f}, '
                  'Train Acc: {:.7f}, Test Acc: {:.7f}'.format(epoch, train_loss,
                                                               train_acc, test_acc))
            global_loss.append(train_loss)
            global_train_acc.append(train_acc)
            global_test_acc.append(test_acc)

    lm, ls, trm, trs, tem, tes = np.mean(global_loss), np.std(global_loss), np.mean(global_train_acc), np.std(
        global_train_acc), np.mean(global_test_acc), np.std(global_test_acc)
    print('After 10-Fold XVal: Average Train Loss: {:.4f} +- {:.4f}\n'
          'Average Train Acc: {:.4f}+-{:.4f}\n Average Test Acc: {:.4f}+-{:.4f}'.format(lm, ls, trm, trs, tem, tes)
          )
    with open('../gnn_results/results.txt', 'a+') as f:
        print("gin {} {} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}".format(int(args.provided_idx), args.dataset, lm, ls, trm, trs, tem, tes), file=f)


def get_clean_graph_indices(dataset_name, path_to_orbits='../results_no_labels/orbits/',
                            path_to_dataset='../datasets/'):
    '''
    Return indices of the dataset that should be included for training.
    It gets graph orbits and keep one graph from each orbit if orbit contains the same labels,
    or else removes entirely the orbit.
    :param dataset_name:
    :param path_to_orbits:
    :return:
    '''
    dataset = TUDataset(root=f'./Datasets/{dataset_name}', name=dataset_name)

    # get a list of lists, with graphs that belong to orbits
    with open(path_to_orbits + f'{dataset_name}_orbits.txt') as f:
        true_orbits = [list(map(int, ast.literal_eval(''.join(line.split()[2:])))) for line in f]

    # get target labels for each graphs
    graph_labels = dict()
    with open(path_to_dataset + f'{dataset_name}/{dataset_name}_graph_labels.txt') as f:
        for i, line in enumerate(f):
            graph_labels[i + 1] = line.strip()

    # get labels in each orbit
    orbit_labels = [[graph_labels[graph] for graph in orbit] for orbit in true_orbits]

    # keep a representative of the orbit
    orbit_graphs = []
    for i, orbit in enumerate(true_orbits):
        assert len(orbit) == len(orbit_labels[i])
        if len(set(orbit_labels[i])) == 1:  # there is only one label in the orbit
            orbit_graphs.append(orbit[0])  # keep only first graph from the orbit

    # calculate all graphs needed to be removed
    iso_graphs = set()
    for orbit in true_orbits:
        iso_graphs = iso_graphs.union(orbit)

    print(len(iso_graphs), len(iso_graphs.difference(orbit_graphs)), len(true_orbits))
    iso_graphs = iso_graphs.difference(orbit_graphs)

    graph_idx = []
    for i in range(len(dataset)):
        if i + 1 not in iso_graphs:
            graph_idx.append(i)

    return graph_idx


if __name__ == "__main__":
    args = get_args()
    args.dataset = 'PROTEINS'
    # args.num_epochs = 2
    # args.provided_idx = True
    args.initialize_node_features = 1
    main(args)

    console = []
