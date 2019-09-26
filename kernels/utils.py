from igraph import Graph
import numpy as np
import ast
import random
from sklearn import svm
from sklearn.metrics import accuracy_score
import pathlib
import networkx as nx
from collections import Counter


def save_to_graphml(dataset, path):
    names = []
    dataset_name = dataset.name.lower()
    for i, graph in enumerate(dataset):

        g = nx.Graph()
        # adding (attributed) edges
        if graph.num_edge_features == 0:
            g.add_edges_from(zip(*graph.edge_index.numpy()))
        else:
            ex, ey = np.where(graph.edge_attr == 1)
            edge_attributes = np.vstack((graph.edge_index.numpy(), ey + 1))
            g.add_weighted_edges_from(zip(*edge_attributes), weight='e_label')

        # adding node attributes
        vx, vy = np.where(graph.x == 1)
        node_attributes = dict(zip(vx, vy + 1))
        nx.set_node_attributes(g, node_attributes, 'v_label')

        name = f'{dataset_name}_{i+1}.graphml'
        names.append(name)
        pathlib.Path(f'{path}/data').mkdir(parents=True, exist_ok=True)
        nx.write_graphml(g, f'{path}/data/{name}')
    with open(f'{path}/{dataset_name}.list', 'w') as f:
        for name in names:
            f.write(f'{name}\n')

def save_to_graphml2(dataset, path):
    names = []
    dataset_name = dataset.name.lower()
    for i, graph in enumerate(dataset):
        g = Graph()
        g.add_vertices(graph.num_nodes)
        g.add_edges([(i,j) for (i, j) in zip(graph.edge_index.data.numpy()[0], graph.edge_index.data.numpy()[1])])
        g.vs['label'] = [list(attr).index(1)+1 for attr in graph.x.data.numpy()]
        if not dataset.num_edge_features == 0:
            g.es['label'] = [list(attr).index(1)+1 for attr in graph.edge_attr.data.numpy()]
        name = f'{dataset_name}_{i+1}.graphml'
        names.append(name)
        pathlib.Path(f'{path}/data').mkdir(parents=True, exist_ok=True)
        g.write_graphml(f'{path}/data/{name}')
    with open(f'{path}/{dataset_name}.list', 'w') as f:
        for name in names:
            f.write(f'{name}\n')

def read_kernel_matrix(path):
    '''
    Reads computed kernel matrix and returns numpy array.
    '''
    data = []
    with open(path, 'r') as f:
        for line in f.readlines():
            data.append(np.fromstring(line, dtype=float, sep=',')[:-1])   
    return np.array(data)

class Evaluation(object):
    '''
    Evaluating a Kernel matrix on SVM classification accuracy.
    
    By providing a Kernel matrix M and labels y on initialization, 
    you can run self.evaluate(k=10) to get accuracy results on k=10
    cross validation test sets of your matrix.
    '''
    def __init__(self, matrix, labels, args, verbose=False):
        '''
        Initialize evaluation.
        :param matrix: feature matrix (either kernel or embeddings)
        :param labels: labels for each row
        '''
        self.K = matrix
        self.y = labels
        self.verbose = verbose
        self.args = args

        self.global_test_acc_original_hom = []  # accuracy of original model
        self.global_test_acc_iso_original_hom = []  # accuracy of original model on homogeneous Y_iso
        self.global_test_acc_hom = []  # accuracy of peering model on homogeneous
        self.global_test_acc_iso_hom = []  # accuracy of peering model on homogeneous Y_iso
        self.global_test_acc_original_all = []  # accuracy of original model (same as first)
        self.global_test_acc_iso_original_all = []  # accuracy of original model on all Y_iso
        self.global_test_acc_all = []  # accuracy of peering model on all
        self.global_test_acc_iso_all = []  # accuracy of peering model on all Y_iso

    def split(self, alpha=.8):
        y = np.copy(self.y)
        K = np.copy(self.K)
        N = K.shape[0]

        perm = np.random.permutation(N)
        for i in range(N):
            K[:, i] = K[perm, i]
        for i in range(N):
            K[i, :] = K[i, perm]

        y = y[perm]

        n1 = int(alpha * N)  # training number
        n2 = int((1 - alpha) / 2 * N)  # validation number

        K_train = K[:n1, :n1]
        y_train = y[:n1]
        K_val = K[n1:(n1 + n2), :n1]
        y_val = y[n1:(n1 + n2)]
        K_test = K[(n1 + n2):, :(n1 + n2)]
        y_test = y[(n1 + n2):]
        K_train_val = K[:(n1 + n2), :(n1 + n2)]
        y_train_val = y[:(n1 + n2)]

        return K_train, K_val, K_test, y_train, y_val, y_test, K_train_val, y_train_val

    def kfold(self, k=10):
        '''
        Generator yields a fold of train-val-test split for cross-validation.
        :param k: number of folds.
        :return:
        '''
        y = np.copy(self.y)
        K = np.copy(self.K)
        N = K.shape[0]

        # permute matrix first
        perm = np.random.permutation(N)
        for i in range(N):
            K[:, i] = K[perm, i]
        for i in range(N):
            K[i, :] = K[i, perm]

        y = y[perm]

        mapping = dict(enumerate(perm))

        test_idx = [(N // k) * ix for ix in range(k)] + [N]
        for ix in range(k):
            test_range = list(range(test_idx[ix], test_idx[ix + 1]))
            test_original_inds = [mapping[ind] for ind in test_range]

            train_val_range = [ix for ix in range(N) if ix not in test_range]
            K_train_val = K[np.ix_(train_val_range, train_val_range)]
            y_train_val = y[train_val_range]

            K_test = K[np.ix_(test_range, train_val_range)]
            y_test = y[test_range]

            val_range = random.sample(train_val_range, N // k)
            train_range = [ix for ix in train_val_range if ix not in val_range]
            train_original_inds = [mapping[ind] for ind in train_range]
            K_train = K[np.ix_(train_range, train_range)]
            y_train = y[train_range]

            K_val = K[np.ix_(val_range, train_range)]
            y_val = y[val_range]
            yield K_train, K_val, K_test, y_train, y_val, y_test, K_train_val, y_train_val, \
                train_original_inds, test_original_inds

    def split_embeddings(self, alpha=.8):
        '''Split embeddings matrix.'''
        K = np.copy(self.K)
        y = np.copy(self.y)
        K_train_val, K_test, y_train_val, y_test = train_test_split(K, y, test_size=1 - alpha)
        K_train, K_val, y_train, y_val = train_test_split(K_train_val, y_train_val, test_size=1 - alpha)
        return K_train, K_val, K_test, y_train, y_val, y_test, K_train_val, y_train_val

    def run_SVM(self,
                K_train, K_val, K_test, y_train, y_val, y_test, K_train_val, y_train_val,
                iso_test_idx_hom, iso_test_labels_hom,
                iso_test_idx_all, iso_test_labels_all):
        '''Run SVM on kernel matrix using train-val-test split.'''

        C_grid = [0.001, 0.01, 0.1, 1, 10]
        val_scores = []
        for i in range(len(C_grid)):
            # Train a model on Train data
            model = svm.SVC(kernel='precomputed', C=C_grid[i])
            model.fit(K_train, y_train)

            # Predict a model on Validation data
            y_val_pred = model.predict(K_val)
            val_scores.append(accuracy_score(y_val, y_val_pred))

        # re-train a model on Train + Validation data
        max_idx = np.argmax(val_scores)
        model = svm.SVC(kernel='precomputed', C=C_grid[max_idx])
        model.fit(K_train_val, y_train_val)

        # Predict the final model on Test data
        y_test_pred = model.predict(K_test)
        if self.verbose:
            print(y_test_pred)

        test_acc_original_hom, test_acc_iso_original_hom = test_model(y_test_pred, y_test, iso_test_idx_hom)
        test_acc_hom, test_acc_iso_hom = test_model(y_test_pred, y_test, iso_test_idx_hom, iso_test_labels_hom)

        test_acc_original_all, test_acc_iso_original_all = test_model(y_test_pred, y_test, iso_test_idx_all)
        test_acc_all, test_acc_iso_all = test_model(y_test_pred, y_test, iso_test_idx_all, iso_test_labels_all)

        self.global_test_acc_original_hom.append(test_acc_original_hom)
        self.global_test_acc_iso_original_hom.append(test_acc_iso_original_hom)
        self.global_test_acc_hom.append(test_acc_hom)
        self.global_test_acc_iso_hom.append(test_acc_iso_hom)

        self.global_test_acc_original_all.append(test_acc_original_all)
        self.global_test_acc_iso_original_all.append(test_acc_iso_original_all)
        self.global_test_acc_all.append(test_acc_all)
        self.global_test_acc_iso_all.append(test_acc_iso_all)

        return val_scores[max_idx], accuracy_score(y_test, y_test_pred), C_grid[max_idx]

    def evaluate(self, dataset, k=10):
        '''
        Performs k-fold cross-validation of kernel matrix using SVM model.
        :param k: number of folds
        :return: list of k accuracies on a test split.
        '''

        graph_idx, orbits = get_clean_graph_indices(dataset, path_to_orbits=self.args.orbits_path)
        print('Found {} orbits from {}'.format(len(orbits), self.args.orbits_path))
        gen = self.kfold(k=k)

        accs = []
        for ix, (K_train, K_val, K_test, y_train, y_val, y_test, K_train_val, y_train_val, train_original_inds, test_original_inds) in enumerate(gen):

            iso_test_idx_all, iso_test_labels_all = get_Y_iso_idx_and_labels(orbits, train_original_inds, test_original_inds, self.y, homogeneous=True)
            iso_test_idx_hom, iso_test_labels_hom = get_Y_iso_idx_and_labels(orbits, train_original_inds, test_original_inds, self.y, homogeneous=False)

            val, acc, c_max = self.run_SVM(K_train, K_val, K_test, y_train, y_val, y_test, K_train_val, y_train_val,
                                           iso_test_idx_hom, iso_test_labels_hom,
                                           iso_test_idx_all, iso_test_labels_all)
            accs.append(acc)
            if self.verbose:
                print("Scored {} on validation and {} on test with C = {}".format(val, acc, c_max))

        test_mean_original_hom, test_std_original_hom = np.mean(self.global_test_acc_original_hom), np.std(
            self.global_test_acc_original_hom)
        test_iso_mean_original_hom, test_iso_std_original_hom = np.mean(self.global_test_acc_iso_original_hom), np.std(
            self.global_test_acc_iso_original_hom)
        test_mean_hom, test_std_hom = np.mean(self.global_test_acc_hom), np.std(self.global_test_acc_hom)
        test_iso_mean_hom, test_iso_std_hom = np.mean(self.global_test_acc_iso_hom), np.std(self.global_test_acc_iso_hom)

        test_mean_original_all, test_std_original_all = np.mean(self.global_test_acc_original_all), np.std(
            self.global_test_acc_original_all)
        test_iso_mean_original_all, test_iso_std_original_all = np.mean(self.global_test_acc_iso_original_all), np.std(
            self.global_test_acc_iso_original_all)
        test_mean_all, test_std_all = np.mean(self.global_test_acc_all), np.std(self.global_test_acc_all)
        test_iso_mean_all, test_iso_std_all = np.mean(self.global_test_acc_iso_all), np.std(self.global_test_acc_iso_all)

        print(
            'After 10-Fold XVal: Original-Hom Test Acc: {:.4f}+-{:.4f} Test Iso Acc: {:.4f}+-{:.4f}'.format(
                test_mean_original_hom, test_std_original_hom,
                test_iso_mean_original_hom,
                test_iso_std_original_hom))
        print('After 10-Fold XVal: Peering-Hom Test Acc: {:.4f}+-{:.4f} Test Iso Acc: {:.4f}+-{:.4f}'.format(
            test_mean_hom,
            test_std_hom,
            test_iso_mean_hom,
            test_iso_std_hom))
        print(
            'After 10-Fold XVal: Original-All Test Acc: {:.4f}+-{:.4f} Test Iso Acc: {:.4f}+-{:.4f}'.format(
                test_mean_original_all, test_std_original_all,
                test_iso_mean_original_all,
                test_iso_std_original_all))
        print('After 10-Fold XVal: Peering-All Test Acc: {:.4f}+-{:.4f} Test Iso Acc: {:.4f}+-{:.4f}'.format(
            test_mean_all,
            test_std_all,
            test_iso_mean_all,
            test_iso_std_all))

        path = f'./kernel_results/{self.args.dataset}/{self.args.kernel}/'
        pathlib.Path(f'{path}').mkdir(parents=True, exist_ok=True)
        with open(f'{path}results.txt', 'a+') as f:
            print("original-hom gin {} {} {} {:.3f} {:.3f} {:.3f} {:.3f}".format(self.args.orbits_path,
                                                                                 int(self.args.clean_dataset), self.args.dataset,
                                                                                 test_mean_original_hom,
                                                                                 test_std_original_hom,
                                                                                 test_iso_mean_original_hom,
                                                                                 test_iso_std_original_hom),
                  file=f)
            print(
                "peering-hom gin {} {} {} {:.3f} {:.3f} {:.3f} {:.3f}".format(self.args.orbits_path, int(self.args.clean_dataset),
                                                                              self.args.dataset,
                                                                              test_mean_hom,
                                                                              test_std_hom,
                                                                              test_iso_mean_hom,
                                                                              test_iso_std_hom),
                file=f)

            print("original-all gin {} {} {} {:.3f} {:.3f} {:.3f} {:.3f}".format(self.args.orbits_path,
                                                                                 int(self.args.clean_dataset), self.args.dataset,
                                                                                 test_mean_original_all,
                                                                                 test_std_original_all,
                                                                                 test_iso_mean_original_all,
                                                                                 test_iso_std_original_all),
                  file=f)
            print(
                "peering-all gin {} {} {} {:.3f} {:.3f} {:.3f} {:.3f}".format(self.args.orbits_path, int(self.args.clean_dataset),
                                                                              self.args.dataset,
                                                                              test_mean_all,
                                                                              test_std_all,
                                                                              test_iso_mean_all,
                                                                              test_iso_std_all),
                file=f)
        return accs


def get_clean_graph_indices(dataset, path_to_orbits='../results_no_labels/orbits/',
                            path_to_dataset='./Datasets/Pytorch_geometric/'):
    '''
    Return indices of the dataset that should be included for training.
    It gets graph orbits and keep one graph from each orbit if orbit contains the same labels,
    or else removes entirely the orbit.
    :param dataset_name:
    :param path_to_orbits:
    :return:
    '''

    # get a list of lists, with graphs that belong to orbits
    with open(path_to_orbits + f'{dataset.name}_orbits.txt') as f:
        true_orbits = [list(map(int, ast.literal_eval(''.join(line.split()[2:])))) for line in f]

    # get target labels for each graphs
    graph_labels = dict()
    with open(path_to_dataset + f'{dataset.name}/raw/{dataset.name}_graph_labels.txt') as f:
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


def get_Y_iso_idx_and_labels(orbits, train_graph_idx, test_graph_idx, y, homogeneous=False):
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
                orb_labels = set([y[graph] for graph in new_orb])
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
    iso_labels = [Counter([y[idx] for idx in orb_idx]).most_common(1)[0][0] for orb_idx in iso_train_orbits_idx]
    return iso_test_idx, iso_labels


def test_model(preds, y,  iso_test_idx, iso_labels=None):
    if iso_labels is not None:
        preds[iso_test_idx] = iso_labels
    correct = (preds == y).sum()
    iso_correct = (preds[iso_test_idx] == y[iso_test_idx]).sum()
    return correct / len(preds), iso_correct / len(iso_test_idx) if len(iso_test_idx) else 1

