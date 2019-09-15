from igraph import Graph
import numpy as np
import os
import random
from sklearn import svm
from sklearn.metrics import accuracy_score
import pathlib

def save_to_graphml(dataset, path):
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
    def __init__(self, matrix, labels, verbose=False):
        '''
        Initialize evaluation.
        :param matrix: feature matrix (either kernel or embeddings)
        :param labels: labels for each row
        '''
        self.K = matrix
        self.y = labels
        self.verbose = verbose

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

        test_idx = [(N // k) * ix for ix in range(k)] + [N]
        for ix in range(k):
            test_range = list(range(test_idx[ix], test_idx[ix + 1]))

            train_val_range = [ix for ix in range(N) if ix not in test_range]
            K_train_val = K[np.ix_(train_val_range, train_val_range)]
            y_train_val = y[train_val_range]


            K_test = K[np.ix_(test_range, train_val_range)]
            y_test = y[test_range]

            val_range = random.sample(train_val_range, N // k)
            train_range = [ix for ix in train_val_range if ix not in val_range]
            K_train = K[np.ix_(train_range, train_range)]
            y_train = y[train_range]

            K_val = K[np.ix_(val_range, train_range)]
            y_val = y[val_range]
            yield K_train, K_val, K_test, y_train, y_val, y_test, K_train_val, y_train_val

    def split_embeddings(self, alpha=.8):
        '''Split embeddings matrix.'''
        K = np.copy(self.K)
        y = np.copy(self.y)
        K_train_val, K_test, y_train_val, y_test = train_test_split(K, y, test_size=1 - alpha)
        K_train, K_val, y_train, y_val = train_test_split(K_train_val, y_train_val, test_size=1 - alpha)
        return K_train, K_val, K_test, y_train, y_val, y_test, K_train_val, y_train_val

    def run_SVM(self,
                K_train, K_val, K_test, y_train, y_val, y_test, K_train_val, y_train_val):
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
        return val_scores[max_idx], accuracy_score(y_test, y_test_pred), C_grid[max_idx]

    def evaluate(self, k=10):
        '''
        Performs k-fold cross-validation of kernel matrix using SVM model.
        :param k: number of folds
        :return: list of k accuracies on a test split.
        '''
        gen = self.kfold(k=k)

        accs = []
        for ix, (K_train, K_val, K_test, y_train, y_val, y_test, K_train_val, y_train_val) in enumerate(gen):
            val, acc, c_max = self.run_SVM(K_train, K_val, K_test, y_train, y_val, y_test, K_train_val, y_train_val)
            accs.append(acc)
            if self.verbose:
                print("Scored {} on validation and {} on test with C = {}".format(val, acc, c_max))
        return accs
