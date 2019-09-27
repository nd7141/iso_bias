import networkx as nx
import matplotlib.pyplot as plt
import os

def get_degrees(G):
    degrees = G.degree()
    d2n = dict()
    for node, degree in degrees:
        d2n.setdefault(degree, []).append(node)
    return sorted(d2n.items())

def read_adj(fn):
    print(fn)
    G = nx.Graph()
    with open(fn) as f:
        header = next(f)
        n, m = int(header[0]), int(header[1])
        for line in f:
            if line:
                s = line.split()
                G.add_edge(int(s[0]), int(s[1]))
    assert len(G) == n, f"Got {len(G)} {n}"
    return G

if __name__ == '__main__':

    dataset = 'COLLAB'

    print("graph, label, nodes, edges, degrees, groups")

    ds = ['Synthie',
 'Tox21_AHR',
 'Tox21_AR',
 'Tox21_ARE',
 'Tox21_AR-LBD',
 'Letter-high',
 'Letter-low',
 'Letter-med',
 'Cuneiform',
 'DBLP_v1',
 'DHFR',
 'PROTEINS',
 'PTC_FM',
 'PTC_FR',
 'PTC_MM',
 'PTC_MR',
 'SYNTHETIC',
 'MSRC_21',
 'MSRC_21C',
 'MSRC_9',
 'Mutagenicity',
 'OHSU',
 'COX2',
 'COX2_MD',
 'DHFR_MD',
 'ER_MD',
 'FIRSTMM_DB',
 'KKI']
    results_dir = 'results_node_labels/'
    for dataset in ds:
        with open(f"{results_dir}/{dataset}_all_groups.txt", 'w') as g:
            for i in range(1, 10000):
                try:
                    G = read_adj(f"datasets/data_adj/{dataset}_adj/graph_{i}.adj")
                    G = nx.convert_node_labels_to_integers(G)
                    group_fn = f"{results_dir}/{dataset}_groups/graph_{i}.txt"
                    lines = ''
                    if os.path.exists(group_fn):
                        with open(f"{results_dir}/{dataset}_groups/graph_{i}.txt") as f:
                            lines = list(map(lambda x: x.strip(), f.readlines()))

                    print(i, len(G), len(G.edges()), get_degrees(G), lines, file=g)
                except FileNotFoundError:
                    print('Finished with',i, 'datasets')
                    break