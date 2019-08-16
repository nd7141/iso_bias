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

def convert_dortmund_to_graphml(folder):
    fns = os.listdir(folder)
    graphs_fn = folder + list(filter(lambda x: x.endswith('A.txt'), fns))[0]
    indicator_fn = folder + list(filter(lambda x: x.endswith('_graph_indicator.txt'), fns))[0]
    labels_fn = folder + list(filter(lambda x: x.endswith('_graph_labels.txt'), fns))[0]

    fn = list(filter(lambda x: x.endswith('_A.txt'), fns))[0]
    dataset = fn[:fn.find('_A.txt')]
    output_folder = f"datasets/data_adj/{dataset}_adj/"
    print(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    with open(indicator_fn) as f:
        nodes2graph = dict()
        for i, line in enumerate(f):
            nodes2graph[i+1] = int(line.strip())

    with open(graphs_fn) as f:
        current_graph = 1
        edges = []
        for i, line in enumerate(f):
            l = line.strip().split(',')
            u, v = int(l[0]), int(l[1])
            g1, g2 = nodes2graph[u], nodes2graph[v]
            assert g1 == g2 , 'Nodes should be connected in the same graph. Line {}, graphs {} {}'.\
                format(i, g1, g2)

            if g1 != current_graph: # assumes indicators are sorted
                # print(g1, current_graph, edges)
                G = nx.Graph()
                G.add_edges_from(edges)
                G = formatg(G)
                # print(len(G.edges()), len(G.nodes()), len(set(edges)), len(G)*(len(G)-1)/2)
                writeg(G, output_folder + 'graph_{}.adj'.format(current_graph))
                edges = []
                current_graph += 1
                if current_graph % 1000 == 0:
                    print('Finished {} dataset'.format(current_graph-1))

            edges.append((u, v))

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
    for dataset in ds:
        try:
            print(dataset)
            convert_dortmund_to_graphml(f'datasets/{dataset}/')
        except Exception as e:
            print('Failed with', dataset, e)

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
