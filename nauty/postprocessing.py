import os
import re
from collections import defaultdict as ddict

def get_iso_graphs(fns):
    '''get a list of isomorphic instances for each graph'''
    iso_graphs = ddict(list)
    for fn in fns:
        gs = re.findall('\d+', fn)
        iso_graphs[gs[0]].append(gs[1])
        iso_graphs[gs[1]].append(gs[0])
    return iso_graphs

def get_orbits(iso_graphs):
    covered_graphs = set()
    orbits = []
    for graph, its_isomorphic in iso_graphs.items():
        if graph not in covered_graphs:
            orbits.append([graph] + its_isomorphic)
            covered_graphs.union([graph] + its_isomorphic)
    return orbits

if __name__ == '__main__':
    # verifying that found isomorphic graphs have the same order and size (double check)
    dataset = 'COLLAB'
    ds = [
        'FIRSTMM_DB',
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
        'REDDIT-MULTI-12K'
    ]

    use_node_labels = False
    compute_groups = False
    if use_node_labels:
        results_dir = 'nauty_results/results_node_labels/'
    else:
        results_dir = 'nauty_results/results_no_labels/'

    ds = [
        'MUTAG',
        'IMDB-BINARY',
    ]

    for dataset in ds:
        print(dataset)

        iso_dir = f'{results_dir}/{dataset}_iso/'
        out_fn = f"{results_dir}/orbits/"
        os.makedirs(out_fn, exist_ok=True)

        # get files with isomorphic graphs
        try:
            fns = list(filter(lambda x: x.endswith('.adj'), os.listdir(iso_dir)))
        except:
            continue

        # get orbits for a data set
        iso_graphs = get_iso_graphs(fns)
        orbits = get_orbits(iso_graphs)

        # write orbit number, size of orbit, and graphs of orbit into file
        with open(out_fn + f"{dataset}_orbits.txt", "w") as f:
            for i, l in enumerate(sorted(orbits, key=lambda x: len(x), reverse=True)):
                print(i, len(l), sorted(l, key=lambda x: int(x)), file=f)

