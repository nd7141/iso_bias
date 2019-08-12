import os
import re
from collections import defaultdict as ddict

if __name__ == '__main__':
    # verifying that found isomorphic graphs have the same order and size (double check)
    dataset = 'COLLAB'
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
    for dataset in ds:
        print(dataset)
        res_fn = f'results/{dataset}_all_groups.txt'
        iso_dir = f'results/{dataset}_iso2/'

        with open(res_fn) as f:
            d = dict()
            for line in f:
                s = line.split()
                d[s[0]] = s[1:]

        try:
            fns = list(filter(lambda x: x.endswith('.adj'), os.listdir(iso_dir)))
        except FileNotFoundError as e:
            print("Didn't find a file", dataset)
            continue
        for fn in fns:
            try:
                gs = re.findall('\d+', fn)
                meta1 = d[gs[0]]
                meta2 = d[gs[1]]
                assert meta1[0] == meta2[0] and meta1[1] == meta2[1]
            except KeyError as e:
                continue

        # count the number of isomorphic groups
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
        with open(f"results/{dataset}_orbits.txt", "w") as f:
            for i, l in enumerate(sorted(orbits, key=lambda x: len(x), reverse=True)):
                print(i, len(l), sorted(l, key=lambda x: int(x)), file=f)

