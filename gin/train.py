import os
import shutil

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
from sklearn.model_selection import KFold, train_test_split
from collections import Counter
from torch_geometric.transforms.one_hot_degree import OneHotDegree

np.random.seed(42)
torch.manual_seed(42)

torch.set_num_threads(20)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_Y_iso_idx_and_labels(orbits, train_graph_idx, test_graph_idx, dataset, homogeneous=False):
    '''Function returns indices and labels of graphs in the test that have
    isomorphic graphs in the train such that the whole orbit for this graph
    has the same label.
    '''
    # getting index of iso graphs in the test
    iso_graphs = [x - 1 for orb in orbits for x in orb]
    iso_test = set()
    for orb in orbits:
        new_orb = [x - 1 for x in orb]
        # at least one isomorphic graph is present in train
        if len(set(train_graph_idx).intersection(new_orb)) > 0:
            # add intersection of isomorphic graphs with the test idx
            iso_test.update(set(test_graph_idx).intersection(new_orb))
    iso_train = set(train_graph_idx).intersection(iso_graphs)

    # get orbits for which train and test iso exists
    iso_train_orbits_graphs = []
    iso_test_keep = []
    for graph in iso_test:
        for orb in orbits:
            new_orb = [x - 1 for x in orb]
            if graph in new_orb:
                # check the number of labels in orbit
                orb_labels = set([dataset[graph].y.item() for graph in new_orb])
                if not homogeneous:
                    iso_train_orbits_graphs.append(new_orb)
                    iso_test_keep.append(graph)
                elif len(orb_labels) == 1:
                    # keep orbit
                    iso_train_orbits_graphs.append(new_orb)
                    iso_test_keep.append(graph)
                break
    iso_test_idx = [test_graph_idx.index(graph_idx) for graph_idx in iso_test_keep]
    print('Iso train {}, test {}, total {}'.format(len(iso_train), len(iso_test_keep), len(iso_graphs)))

    # select graphs orbits in the train
    iso_train_orbits_idx = [set(train_graph_idx).intersection(orb) for orb in iso_train_orbits_graphs]
    # select the most popular label from the train orbit
    iso_labels = [Counter([dataset[idx].y.item() for idx in orb_idx]).most_common(1)[0][0] for orb_idx in iso_train_orbits_idx]
    return iso_test_idx, iso_labels


def get_dataset_classes(loader):
    classes = []
    for b in loader:
        classes += list(map(int, b.y))
    return Counter(classes)


def test_model(model, loader, device, iso_test_idx, iso_labels=None):
    model.eval()

    correct = 0
    iso_correct = 0
    for data in loader:
        data = data.to(device)
        output = model(data)
        pred = output.max(dim=1)[1]
        if iso_labels is not None:  # replace some of the labels
            tmp = pred.numpy()
            tmp[iso_test_idx] = iso_labels
            pred = torch.tensor(tmp)
        correct += pred.eq(data.y).sum().item()
        iso_correct += pred[iso_test_idx].eq(data.y[iso_test_idx]).sum().item()
    return correct / len(loader.dataset), iso_correct / len(iso_test_idx) if len(iso_test_idx) else 1


def main(args):
    dataset = TUDataset(root=f'./Datasets/{args.dataset}', name=args.dataset)

    print(dataset)

    # if dataset.num_features == 0 or args.initialize_node_features:
    if args.initialize_node_features:
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

    print("Use clean dataset: {}".format(bool(args.clean_dataset)))
    graph_idx, orbits = get_clean_graph_indices(args.dataset, path_to_orbits=args.orbits_path)
    print('Found {} orbits from {}'.format(len(orbits), args.orbits_path))

    if args.clean_dataset:
        dataset_size = len(graph_idx)
        print(f"Dataset size: {len(dataset)} -> {dataset_size}")
        shuffled_idx = copy.deepcopy(graph_idx)

    else:
        dataset_size = len(dataset)
        shuffled_idx = list(range(dataset_size))
        print(f"Dataset size: {len(dataset)}")

    print('Class labels:', Counter([int(dataset[int(idx)].y) for idx in shuffled_idx]))

    global_test_acc_original_hom = [] # accuracy of original model
    global_test_acc_iso_original_hom = [] # accuracy of original model on homogeneous Y_iso
    global_test_acc_hom = [] # accuracy of peering model on homogeneous
    global_test_acc_iso_hom = [] # accuracy of peering model on homogeneous Y_iso
    global_test_acc_original_all = [] # accuracy of original model (same as first)
    global_test_acc_iso_original_all = [] # accuracy of original model on all Y_iso
    global_test_acc_all = [] # accuracy of peering model on all
    global_test_acc_iso_all = [] # accuracy of peering model on all Y_iso


    global_loss = []
    epoch_trains = []
    epoch_vals = []
    epoch_tests = []

    kf = KFold(args.num_kfold, shuffle=True)  # 20% for test size
    pos2idx = dict(enumerate(shuffled_idx))

    for xval, (train_index, test_index) in enumerate(kf.split(shuffled_idx)):

        test_dataset = [dataset[pos2idx[idx]] for idx in test_index]
        train_val_dataset = [dataset[pos2idx[idx]] for idx in train_index]

        test_graph_idx = [pos2idx[idx] for idx in test_index]
        train_graph_idx = [pos2idx[idx] for idx in train_index]

        # split on train and val
        train_dataset, val_dataset = train_test_split(train_val_dataset, test_size=0.2)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=len(val_dataset))
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

        print(len(train_dataset), len(val_dataset), len(test_dataset))

        iso_test_idx_hom, iso_test_labels_hom = get_Y_iso_idx_and_labels(orbits, train_graph_idx, test_graph_idx, dataset, homogeneous=True)
        iso_test_idx_all, iso_test_labels_all = get_Y_iso_idx_and_labels(orbits, train_graph_idx, test_graph_idx, dataset, homogeneous=False)

        print('Possible train classes', get_dataset_classes(train_loader))
        print('Possible val classes', get_dataset_classes(val_loader))
        print('Possible test classes', get_dataset_classes(test_loader))

        # model = GCN(dataset.num_features, dataset.num_classes, args.hidden, args.dropout).to(device)

        model = GraphCNN(5, 2, dataset.num_features, 64,
                         dataset.num_classes, 0.5, False,
                         "sum", "sum", device).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        best_score = -1
        best_model = None

        epoch_train = []
        epoch_val = []
        epoch_test = []

        print('Running {} epochs'.format(args.num_epochs))

        for epoch in range(1, args.num_epochs + 1):

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
            val_acc = test(model, val_loader, device)
            test_acc = test(model, test_loader, device)

            epoch_train.append(train_acc)
            epoch_val.append(val_acc)
            epoch_test.append(test_acc)

            if val_acc > best_score and epoch >= 50:
                best_score = val_acc
                best_test_score = test_acc
                best_epoch = epoch
                best_model = copy.deepcopy(model)

            print('Xval: {:03d}, Epoch: {:03d}, Train Loss: {:.4f}, '
                  'Train Acc: {:.4f}, Val Acc: {:.4f}, Test Acc: {:.4f}'.format(xval, epoch, train_loss,
                                                                                train_acc, val_acc, test_acc))

        test_acc_original_hom, test_acc_iso_original_hom = test_model(best_model, test_loader, device, iso_test_idx_hom)
        test_acc_hom, test_acc_iso_hom = test_model(best_model, test_loader, device, iso_test_idx_hom, iso_test_labels_hom)

        test_acc_original_all, test_acc_iso_original_all = test_model(best_model, test_loader, device, iso_test_idx_all)
        test_acc_all, test_acc_iso_all = test_model(best_model, test_loader, device, iso_test_idx_all, iso_test_labels_all)


        print('Xval {:03d} Best model accuracy on test {:.4f} vs {:.4f} ({:.4f} {})'.format(xval, test_acc,
                                                                                            best_test_score, best_score,
                                                                                            best_epoch))
        global_test_acc_original_hom.append(test_acc_original_hom)
        global_test_acc_iso_original_hom.append(test_acc_iso_original_hom)
        global_test_acc_hom.append(test_acc_hom)
        global_test_acc_iso_hom.append(test_acc_iso_hom)

        global_test_acc_original_all.append(test_acc_original_all)
        global_test_acc_iso_original_all.append(test_acc_iso_original_all)
        global_test_acc_all.append(test_acc_all)
        global_test_acc_iso_all.append(test_acc_iso_all)

        epoch_trains.append(epoch_train)
        epoch_vals.append(epoch_val)

    with open('../gnn_results/epochs_{}.txt'.format(args.dataset), 'w') as f:
        print(epoch_trains, file=f)
        print(epoch_vals, file=f)
        print(epoch_tests, file=f)

    test_mean_original_hom, test_std_original_hom = np.mean(global_test_acc_original_hom), np.std(global_test_acc_original_hom)
    test_iso_mean_original_hom, test_iso_std_original_hom = np.mean(global_test_acc_iso_original_hom), np.std(global_test_acc_iso_original_hom)
    test_mean_hom, test_std_hom = np.mean(global_test_acc_hom), np.std(global_test_acc_hom)
    test_iso_mean_hom, test_iso_std_hom = np.mean(global_test_acc_iso_hom), np.std(global_test_acc_iso_hom)

    test_mean_original_all, test_std_original_all = np.mean(global_test_acc_original_all), np.std(global_test_acc_original_all)
    test_iso_mean_original_all, test_iso_std_original_all = np.mean(global_test_acc_iso_original_all), np.std(global_test_acc_iso_original_all)
    test_mean_all, test_std_all = np.mean(global_test_acc_all), np.std(global_test_acc_all)
    test_iso_mean_all, test_iso_std_all = np.mean(global_test_acc_iso_all), np.std(global_test_acc_iso_all)

    print(
        'After 10-Fold XVal: Original-Hom Test Acc: {:.4f}+-{:.4f} Test Iso Acc: {:.4f}+-{:.4f}'.format(test_mean_original_hom, test_std_original_hom,
                                                                                                   test_iso_mean_original_hom,
                                                                                                   test_iso_std_original_hom))
    print('After 10-Fold XVal: Peering-Hom Test Acc: {:.4f}+-{:.4f} Test Iso Acc: {:.4f}+-{:.4f}'.format(test_mean_hom,
                                                                                                     test_std_hom,
                                                                                                     test_iso_mean_hom,
                                                                                                     test_iso_std_hom))
    print(
        'After 10-Fold XVal: Original-All Test Acc: {:.4f}+-{:.4f} Test Iso Acc: {:.4f}+-{:.4f}'.format(test_mean_original_all, test_std_original_all,
                                                                                                   test_iso_mean_original_all,
                                                                                                   test_iso_std_original_all))
    print('After 10-Fold XVal: Peering-All Test Acc: {:.4f}+-{:.4f} Test Iso Acc: {:.4f}+-{:.4f}'.format(test_mean_all,
                                                                                                     test_std_all,
                                                                                                     test_iso_mean_all,
                                                                                                     test_iso_std_all))
    with open('../gnn_results/results.txt', 'a+') as f:
        print("original-hom gin {} {} {} {:.3f} {:.3f} {:.3f} {:.3f}".format(args.orbits_path, int(args.clean_dataset), args.dataset,
                                                                             test_mean_original_hom,
                                                                             test_std_original_hom,
                                                                             test_iso_mean_original_hom,
                                                                             test_iso_std_original_hom),
              file=f)
        print("peering-hom gin {} {} {} {:.3f} {:.3f} {:.3f} {:.3f}".format(args.orbits_path, int(args.clean_dataset), args.dataset,
                                                                            test_mean_hom,
                                                                            test_std_hom,
                                                                            test_iso_mean_hom,
                                                                            test_iso_std_hom),
              file=f)

        print("original-all gin {} {} {} {:.3f} {:.3f} {:.3f} {:.3f}".format(args.orbits_path, int(args.clean_dataset), args.dataset,
                                                                             test_mean_original_all,
                                                                             test_std_original_all,
                                                                             test_iso_mean_original_all,
                                                                             test_iso_std_original_all),
              file=f)
        print("peering-all gin {} {} {} {:.3f} {:.3f} {:.3f} {:.3f}".format(args.orbits_path, int(args.clean_dataset), args.dataset,
                                                                            test_mean_all,
                                                                            test_std_all,
                                                                            test_iso_mean_all,
                                                                            test_iso_std_all),
              file=f)

    return best_model


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

    iso_graphs = iso_graphs.difference(orbit_graphs)

    clean_graph_idx = [idx for idx in range(len(dataset)) if i + 1 not in iso_graphs]

    return clean_graph_idx, true_orbits


if __name__ == "__main__":
    args = get_args()
    args.dataset = 'MUTAG'
    # args.num_epochs = 3
    # args.orbits_path = '../results_node_labels/orbits/'
    # args.clean_dataset = True
    args.initialize_node_features = True
    main(args)

    console = []
