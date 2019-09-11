import os
import re
from collections import defaultdict as ddict

import pprint

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
    # ds = ['Cuneiform']

    use_node_labels = False
    compute_groups = False
    if use_node_labels:
        results_dir = 'nauty_results/results_node_labels/'
    else:
        results_dir = 'nauty_results/results_no_labels/'

    for dataset in ds:
        print(dataset)

        # res_fn = f'results2/{dataset}_all_groups.txt'
        # with open(res_fn) as f:
        #     d = dict()
        #     for line in f:
        #         s = line.split()
        #         d[s[0]] = s[1:]
        # try:
        #     fns = list(filter(lambda x: x.endswith('.adj'), os.listdir(iso_dir)))
        # except FileNotFoundError as e:
        #     print("Didn't find a file", dataset)
        # for fn in fns:
        #     try:
        #         gs = re.findall('\d+', fn)
        #         meta1 = d[gs[0]]
        #         meta2 = d[gs[1]]
        #         assert meta1[0] == meta2[0] and meta1[1] == meta2[1]
        #     except KeyError as e:
        #         continue

        # count the number of isomorphic groups
        # results_dir = 'results_node_labels/'

        iso_dir = f'{results_dir}/{dataset}_iso/'
        out_fn = f"{results_dir}/orbits/"
        os.makedirs(out_fn, exist_ok=True)

        try:
            fns = list(filter(lambda x: x.endswith('.adj'), os.listdir(iso_dir)))
        except:
            continue
        iso_graphs = ddict(list)
        for fn in fns:
            gs = re.findall('\d+', fn)
            iso_graphs[gs[0]].append(gs[1])
            iso_graphs[gs[1]].append(gs[0])

        covered = set()
        count = 0
        orbits = []
        for key, value in iso_graphs.items():
            if key not in covered:
                # print(count, key, *value)
                orbits.append([key] + value)
                count += 1
                covered.add(key)
                covered = covered.union(value)
        # s = set()
        # for orbit in orbits:
        #     print(len(s), orbit, len(s.intersection(orbit)))
        #     assert len(s.intersection(orbit)) == 0
        #     s = s.union(orbit)
        with open(out_fn + f"{dataset}_orbits.txt", "w") as f:
            for i, l in enumerate(sorted(orbits, key=lambda x: len(x), reverse=True)):
                print(i, len(l), sorted(l, key=lambda x: int(x)), file=f)

