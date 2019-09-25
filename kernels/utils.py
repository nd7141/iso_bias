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

        self.global_train_acc = []
        self.global_test_acc = []
        self.global_test_acc_iso = []
        self.global_test_acc2 = []
        self.global_test_acc_iso2 = []
        self.global_test_acc3 = []
        self.global_test_acc_iso3 = []
        self.global_test_acc4 = []
        self.global_test_acc_iso4 = []

        if self.args.orbits_path2:
            self.global_test_acc5 = []
            self.global_test_acc_iso5 = []
            self.global_test_acc6 = []
            self.global_test_acc_iso6 = []
            self.global_test_acc7 = []
            self.global_test_acc_iso7 = []
            self.global_test_acc8 = []
            self.global_test_acc_iso8 = []

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
                iso_test_idx, iso_test_labels, iso_test_idx3, iso_test_labels3,
                iso_test_idx5 = None, iso_test_labels5 = None, iso_test_idx7 = None, iso_test_labels7 = None):
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

        test_acc, test_acc_iso = test_model(y_test_pred, y_test, iso_test_idx)
        test_acc2, test_acc_iso2 = test_model(y_test_pred, y_test, iso_test_idx, iso_test_labels)
        test_acc3, test_acc_iso3 = test_model(y_test_pred, y_test, iso_test_idx3, iso_test_labels3)
        test_acc4, test_acc_iso4 = test_model(y_test_pred, y_test, iso_test_idx3)

        if self.args.orbits_path2:
            test_acc5, test_acc_iso5 = test_model(y_test_pred, y_test, iso_test_idx5)
            test_acc6, test_acc_iso6 = test_model(y_test_pred, y_test, iso_test_idx5, iso_test_labels5)
            test_acc7, test_acc_iso7 = test_model(y_test_pred, y_test, iso_test_idx7, iso_test_labels7)
            test_acc8, test_acc_iso8 = test_model(y_test_pred, y_test, iso_test_idx7)

        self.global_test_acc.append(test_acc)
        self.global_test_acc_iso.append(test_acc_iso)
        self.global_test_acc2.append(test_acc2)
        self.global_test_acc_iso2.append(test_acc_iso2)
        self.global_test_acc3.append(test_acc3)
        self.global_test_acc_iso3.append(test_acc_iso3)
        self.global_test_acc4.append(test_acc4)
        self.global_test_acc_iso4.append(test_acc_iso4)

        if self.args.orbits_path2:
            self.global_test_acc5.append(test_acc5)
            self.global_test_acc_iso5.append(test_acc_iso5)
            self.global_test_acc6.append(test_acc6)
            self.global_test_acc_iso6.append(test_acc_iso6)
            self.global_test_acc7.append(test_acc7)
            self.global_test_acc_iso7.append(test_acc_iso7)
            self.global_test_acc8.append(test_acc8)
            self.global_test_acc_iso8.append(test_acc_iso8)

        return val_scores[max_idx], accuracy_score(y_test, y_test_pred), C_grid[max_idx]

    def evaluate(self, dataset, k=10):
        '''
        Performs k-fold cross-validation of kernel matrix using SVM model.
        :param k: number of folds
        :return: list of k accuracies on a test split.
        '''

        graph_idx, orbits = get_clean_graph_indices(dataset, path_to_orbits=self.args.orbits_path)
        print('Found {} orbits from {}'.format(len(orbits), self.args.orbits_path))
        if self.args.orbits_path2:
            graph_idx2, orbits2 = get_clean_graph_indices(dataset, path_to_orbits=self.args.orbits_path2)
            print('Found {} orbits from {}'.format(len(orbits2), self.args.orbits_path2))

        gen = self.kfold(k=k)

        accs = []
        for ix, (K_train, K_val, K_test, y_train, y_val, y_test, K_train_val, y_train_val, train_original_inds, test_original_inds) in enumerate(gen):

            iso_test_idx, iso_test_labels = get_Y_iso_idx_and_labels(orbits, train_original_inds, test_original_inds, self.y,
                                                                     homogeneous=True)
            iso_test_idx3, iso_test_labels3 = get_Y_iso_idx_and_labels(orbits, train_original_inds, test_original_inds, self.y,
                                                                       homogeneous=False)
            if self.args.orbits_path2:
                iso_test_idx5, iso_test_labels5 = get_Y_iso_idx_and_labels(orbits2, train_original_inds, test_original_inds,
                                                                           self.y,
                                                                           homogeneous=True)
                iso_test_idx7, iso_test_labels7 = get_Y_iso_idx_and_labels(orbits2, train_original_inds, test_original_inds,
                                                                           self.y,
                                                                           homogeneous=False)
            else:
                iso_test_idx5, iso_test_labels5 = None, None
                iso_test_idx7, iso_test_labels7 = None, None

            val, acc, c_max = self.run_SVM(K_train, K_val, K_test, y_train, y_val, y_test, K_train_val, y_train_val,
                                           iso_test_idx, iso_test_labels, iso_test_idx3, iso_test_labels3,
                                           iso_test_idx5, iso_test_labels5, iso_test_idx7, iso_test_labels7)
            accs.append(acc)
            if self.verbose:
                print("Scored {} on validation and {} on test with C = {}".format(val, acc, c_max))

        test_mean, test_std = np.mean(self.global_test_acc), np.std(self.global_test_acc)
        test_iso_mean, test_iso_std = np.mean(self.global_test_acc_iso), np.std(self.global_test_acc_iso)
        test_mean2, test_std2 = np.mean(self.global_test_acc2), np.std(self.global_test_acc2)
        test_iso_mean2, test_iso_std2 = np.mean(self.global_test_acc_iso2), np.std(self.global_test_acc_iso2)
        test_mean3, test_std3 = np.mean(self.global_test_acc3), np.std(self.global_test_acc3)
        test_iso_mean3, test_iso_std3 = np.mean(self.global_test_acc_iso3), np.std(self.global_test_acc_iso3)
        test_mean4, test_std4 = np.mean(self.global_test_acc4), np.std(self.global_test_acc4)
        test_iso_mean4, test_iso_std4 = np.mean(self.global_test_acc_iso4), np.std(self.global_test_acc_iso4)

        if self.args.orbits_path2:
            test_mean5, test_std5 = np.mean(self.global_test_acc5), np.std(self.global_test_acc5)
            test_iso_mean5, test_iso_std5 = np.mean(self.global_test_acc_iso5), np.std(self.global_test_acc_iso5)
            test_mean6, test_std6 = np.mean(self.global_test_acc6), np.std(self.global_test_acc6)
            test_iso_mean6, test_iso_std6 = np.mean(self.global_test_acc_iso6), np.std(self.global_test_acc_iso6)
            test_mean7, test_std7 = np.mean(self.global_test_acc7), np.std(self.global_test_acc7)
            test_iso_mean7, test_iso_std7 = np.mean(self.global_test_acc_iso7), np.std(self.global_test_acc_iso7)
            test_mean8, test_std8 = np.mean(self.global_test_acc8), np.std(self.global_test_acc8)
            test_iso_mean8, test_iso_std8 = np.mean(self.global_test_acc_iso8), np.std(self.global_test_acc_iso8)

        print(
            'After 10-Fold XVal: Model-1 Test Acc: {:.4f}+-{:.4f} Test Iso Acc: {:.4f}+-{:.4f}'.format(test_mean,
                                                                                                       test_std,
                                                                                                       test_iso_mean,
                                                                                                       test_iso_std))
        print('After 10-Fold XVal: Model-2 Test Acc: {:.4f}+-{:.4f} Test Iso Acc: {:.4f}+-{:.4f}'.format(test_mean2,
                                                                                                         test_std2,
                                                                                                         test_iso_mean2,
                                                                                                         test_iso_std2))
        print(
            'After 10-Fold XVal: Model-3 Test Acc: {:.4f}+-{:.4f} Test Iso Acc: {:.4f}+-{:.4f}'.format(test_mean3,
                                                                                                       test_std3,
                                                                                                       test_iso_mean3,
                                                                                                       test_iso_std3))
        print('After 10-Fold XVal: Model-4 Test Acc: {:.4f}+-{:.4f} Test Iso Acc: {:.4f}+-{:.4f}'.format(test_mean4,
                                                                                                         test_std4,
                                                                                                         test_iso_mean4,
                                                                                                         test_iso_std4))
        if self.args.orbits_path2:
            print('After 10-Fold XVal: Model-5 Test Acc: {:.4f}+-{:.4f} Test Iso Acc: {:.4f}+-{:.4f}'.format(test_mean5,
                                                                                                             test_std5,
                                                                                                             test_iso_mean5,
                                                                                                             test_iso_std5))
            print('After 10-Fold XVal: Model-6 Test Acc: {:.4f}+-{:.4f} Test Iso Acc: {:.4f}+-{:.4f}'.format(test_mean6,
                                                                                                             test_std6,
                                                                                                             test_iso_mean6,
                                                                                                             test_iso_std6))
            print('After 10-Fold XVal: Model-7 Test Acc: {:.4f}+-{:.4f} Test Iso Acc: {:.4f}+-{:.4f}'.format(test_mean7,
                                                                                                             test_std7,
                                                                                                             test_iso_mean7,
                                                                                                             test_iso_std7))
            print('After 10-Fold XVal: Model-8 Test Acc: {:.4f}+-{:.4f} Test Iso Acc: {:.4f}+-{:.4f}'.format(test_mean8,
                                                                                                             test_std8,
                                                                                                             test_iso_mean8,
                                                                                                             test_iso_std8))
        path = f'./kernel_results/{self.args.dataset}/{self.args.kernel}/'
        pathlib.Path(f'{path}').mkdir(parents=True, exist_ok=True)
        with open(f'{path}results.txt', 'a+') as f:
            print("model-1 kernel {} {} {} {:.3f} {:.3f} {:.3f} {:.3f}".format(self.args.orbits_path, int(self.args.clean_dataset),
                                                                            self.args.dataset, test_mean, test_std,
                                                                            test_iso_mean, test_iso_std),
                  file=f)
            print("model-2 kernel {} {} {} {:.3f} {:.3f} {:.3f} {:.3f}".format(self.args.orbits_path, int(self.args.clean_dataset),
                                                                            self.args.dataset, test_mean2, test_std2,
                                                                            test_iso_mean2, test_iso_std2),
                  file=f)
            print("model-3 kernel {} {} {} {:.3f} {:.3f} {:.3f} {:.3f}".format(self.args.orbits_path, int(self.args.clean_dataset),
                                                                            self.args.dataset, test_mean3, test_std3,
                                                                            test_iso_mean3, test_iso_std3),
                  file=f)
            print("model-4 kernel {} {} {} {:.3f} {:.3f} {:.3f} {:.3f}".format(self.args.orbits_path, int(self.args.clean_dataset),
                                                                            self.args.dataset, test_mean4, test_std4,
                                                                            test_iso_mean4, test_iso_std4),
                  file=f)
            if self.args.orbits_path2:
                print("model-5 kernel {} {} {} {:.3f} {:.3f} {:.3f} {:.3f}".format(self.args.orbits_path2,
                                                                                int(self.args.clean_dataset),
                                                                                self.args.dataset, test_mean5, test_std5,
                                                                                test_iso_mean5, test_iso_std5),
                      file=f)
                print("model-6 kernel {} {} {} {:.3f} {:.3f} {:.3f} {:.3f}".format(self.args.orbits_path2,
                                                                                int(self.args.clean_dataset),
                                                                                self.args.dataset, test_mean6, test_std6,
                                                                                test_iso_mean6, test_iso_std6),
                      file=f)
                print("model-7 kernel {} {} {} {:.3f} {:.3f} {:.3f} {:.3f}".format(self.args.orbits_path2,
                                                                                int(self.args.clean_dataset),
                                                                                self.args.dataset, test_mean7, test_std7,
                                                                                test_iso_mean7, test_iso_std7),
                      file=f)
                print("model-8 kernel {} {} {} {:.3f} {:.3f} {:.3f} {:.3f}".format(self.args.orbits_path2,
                                                                                int(self.args.clean_dataset),
                                                                                self.args.dataset, test_mean8, test_std8,
                                                                                test_iso_mean8, test_iso_std8),
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

