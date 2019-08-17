import networkx as nx
import os


def writeg(graph, fn):
    n = len(graph)
    m = len(graph.edges())
    with open(fn, 'w+') as f:
        f.write(f"{n} {m}\n")
        for e in graph.edges():
            f.write(f"{e[0]} {e[1]}\n")


def formatg(graph):
    "Format graph for correct writing"
    nodes = graph.nodes()
    mapping = dict(zip(nodes, range(len(nodes))))
    return nx.relabel_nodes(graph, mapping)

def write_node_labels(fn, node_labels):
    with open(fn, 'w') as f:
        for i, label in enumerate(node_labels):
            f.write(f"{i} {label}\n")

def write_edge_labels(edges, fn, edge_labels):
    assert len(edges) == len(edge_labels)
    with open(fn, 'w') as f:
        for i in range(len(edges)):
            e = edges[i]
            lab = edge_labels[i]
            f.write(f"{e[0]} {e[1]} {lab}\n")

def convert_dortmund_to_graphml(folder):
    fns = os.listdir(folder)
    graphs_fn = indicator_fn = graph_labels_fn = \
        node_labels_fn = edge_labels_fn = None
    for fn in fns:
        if 'A.txt' in fn:
            graphs_fn = folder + fn
        elif '_graph_indicator.txt' in fn:
            indicator_fn = folder + fn
        elif '_graph_labels.txt' in fn:
            graph_labels_fn = folder + fn
        elif '_node_labels.txt' in fn:
            node_labels_fn = folder + fn
        elif '_edge_labels.txt' in fn:
            edge_labels_fn = folder + fn

    fn = list(filter(lambda x: x.endswith('_A.txt'), fns))[0]
    dataset = fn[:fn.find('_A.txt')]
    output_folder = f"datasets/data_adj/{dataset}_adj/"
    print(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    with open(indicator_fn) as f:
        nodes2graph = dict()
        for i, line in enumerate(f):
            nodes2graph[i + 1] = int(line.strip())

    if node_labels_fn:
        node_labels_f = open(node_labels_fn)
    if edge_labels_fn:
        edge_labels_f = open(edge_labels_fn)
    with open(graphs_fn) as f:
        current_graph = 1
        edges = []
        for i, line in enumerate(f):
            l = line.strip().split(',')
            u, v = int(l[0]), int(l[1])
            g1, g2 = nodes2graph[u], nodes2graph[v]
            assert g1 == g2, 'Nodes should be connected in the same graph. Line {}, graphs {} {}'. \
                format(i, g1, g2)

            if g1 != current_graph:  # assumes indicators are sorted
                # print(g1, current_graph, edges)
                G = nx.Graph()
                G.add_edges_from(edges)
                G = formatg(G)
                if node_labels_fn:
                    node_labels = [int(next(node_labels_f)) for _ in range(len(G))]
                if edge_labels_fn:
                    edge_labels = [int(next(edge_labels_f)) for _ in range(2 * len(G.edges()))]
                # print(len(G.edges()), len(G.nodes()), len(set(edges)), len(G)*(len(G)-1)/2)
                writeg(G, output_folder + 'graph_{}.adj'.format(current_graph))
                if node_labels_fn:
                    write_node_labels(output_folder + '{}.node_labels'.format(current_graph), node_labels)
                if edge_labels_fn:
                    write_edge_labels(edges, output_folder + '{}.edge_labels'.format(current_graph), edge_labels)
                edges = []
                current_graph += 1
                if current_graph % 1000 == 0:
                    print('Finished {} dataset'.format(current_graph - 1))

            edges.append((u, v))
    if node_labels_fn:
        node_labels_f.close()
    if edge_labels_fn:
        edge_labels_f.close()


if __name__ == "__main__":
    # dir = "datasets/data_graphml/data_graphml/NCI109/"
    # out = "datasets/NCI109_adj/"
    # os.makedirs(out, exist_ok=True)
    # fns = os.listdir(dir)
    # for fn in fns:
    #     if fn.endswith('.graphml'):
    #         G = formatg(nx.read_graphml(dir + fn))
    #         writeg(G, out + fn.split('.')[0] + '.adj')

    dataset = 'COLLAB'
    ds = ['FIRSTMM_DB',
          'OHSU',
          'KKI',
          'Peking_1',
          'MUTAG',
          'MSRC_21C',
          'MSRC_9',
          'Cuneiform',
          'SYNTHETIC',
          'COX2_MD',
          'BZR_MD',
          'PTC_MM',
          'PTC_MR',
          'PTC_FM',
          'PTC_FR',
          'DHFR_MD',
          'Synthie',
          'BZR',
          'ER_MD',
          'COX2',
          'MSRC_21',
          'ENZYMES',
          'DHFR',
          'IMDB-BINARY',
          'PROTEINS',
          'DD',
          'IMDB-MULTI',
          'AIDS',
          'REDDIT-BINARY',
          'Letter-high',
          'Letter-low',
          'Letter-med',
          'Fingerprint',
          'COIL-DEL',
          'COIL-RAG',
          'NCI1',
          'NCI109',
          'FRANKENSTEIN',
          'Mutagenicity',
          'REDDIT-MULTI-5K',
          'COLLAB',
          'Tox21_ARE',
          'Tox21_aromatase',
          'Tox21_MMP',
          'Tox21_ER',
          'Tox21_HSE',
          'Tox21_AHR',
          'Tox21_PPAR-gamma',
          'Tox21_AR-LBD',
          'Tox21_p53',
          'Tox21_ER_LBD',
          'Tox21_ATAD5',
          'Tox21_AR',
          'REDDIT-MULTI-12K',
          'DBLP_v1']
    ds = ['MUTAG']
    for dataset in ds:
        # try:
            print(dataset)
            convert_dortmund_to_graphml(f'datasets/{dataset}/')
        # except Exception as e:
        #     print('Failed with', dataset, e)

    from pprint import pprint

    l = ['FIRSTMM_DB',
         'OHSU',
         'KKI',
         'Peking_1',
         'MUTAG',
         'MSRC_21C',
         'MSRC_9',
         'Cuneiform',
         'SYNTHETIC',
         'COX2_MD',
         'BZR_MD',
         'PTC_MM',
         'PTC_MR',
         'PTC_FM',
         'PTC_FR',
         'DHFR_MD',
         'Synthie',
         'BZR',
         'ER_MD',
         'COX2',
         'MSRC_21',
         'ENZYMES',
         'DHFR',
         'IMDB-BINARY',
         'PROTEINS',
         'DD',
         'IMDB-MULTI',
         'AIDS',
         'REDDIT-BINARY',
         'Letter-high',
         'Letter-low',
         'Letter-med',
         'Fingerprint',
         'COIL-DEL',
         'COIL-RAG',
         'NCI1',
         'NCI109',
         'FRANKENSTEIN',
         'Mutagenicity',
         'REDDIT-MULTI-5K',
         'COLLAB',
         'Tox21_ARE',
         'Tox21_aromatase',
         'Tox21_MMP',
         'Tox21_ER',
         'Tox21_HSE',
         'Tox21_AHR',
         'Tox21_PPAR-gamma',
         'Tox21_AR-LBD',
         'Tox21_p53',
         'Tox21_ER_LBD',
         'Tox21_ATAD5',
         'Tox21_AR',
         'REDDIT-MULTI-12K',
         'DBLP_v1']
    import os

    os.listdir()

    ['AIDS',
     'BZR',
     'BZR_MD',
     'COIL-DEL',
     'COIL-RAG',
     'COLLAB',
     'COX2',
     'COX2_MD',
     'Cuneiform',
     'DBLP_v1',
     'DD',
     'DHFR',
     'DHFR_MD',
     'ENZYMES',
     'ER_MD',
     'Fingerprint',
     'FIRSTMM_DB',
     'FRANKENSTEIN',
     'IMDB-BINARY',
     'IMDB-MULTI',
     'KKI',
     'Letter-high',
     'Letter-low',
     'Letter-med',
     'MSRC_21',
     'MSRC_21C',
     'MSRC_9',
     'MUTAG',
     'Mutagenicity',
     'NCI1',
     'NCI109',
     'OHSU',
     'Peking_1',
     'PROTEINS',
     'PTC_FM',
     'PTC_FR',
     'PTC_MM',
     'PTC_MR',
     'REDDIT-BINARY',
     'REDDIT-MULTI-12K',
     'REDDIT-MULTI-5K',
     'SYNTHETIC',
     'Synthie',
     'Tox21_AHR',
     'Tox21_AR',
     'Tox21_ARE',
     'Tox21_AR-LBD',
     'Tox21_aromatase',
     'Tox21_ATAD5',
     'Tox21_ER',
     'Tox21_ER_LBD',
     'Tox21_HSE',
     'Tox21_MMP',
     'Tox21_p53',
     'Tox21_PPAR-gamma',
     'TRIANGLES',
     'TWITTER-Real-Graph-Partia']
