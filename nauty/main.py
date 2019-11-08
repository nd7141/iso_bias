from subprocess import check_call
import os
import re
import time
import multiprocessing as mp
from functools import partial
from itertools import combinations, chain
from collections import defaultdict as ddict
from pprint import pprint


def run_nauty(f1, f2, use_node_labels=False):
    '''it will check for isomorphism and write a file outf if isomorphism exists'''
    outf = outdir + f1.split('/')[-1] + '-' + f2.split('/')[-1]
    i = re.findall("\d+", f1)[-1]
    j = re.findall("\d+", f2)[-1]
    f1 = indir + f1
    f2 = indir + f2
    if use_node_labels:
        label_fn1 = indir + f"{i}.node_labels"
        label_fn2 = indir + f"{j}.node_labels"
        if os.path.exists(label_fn1) and os.path.exists(label_fn1):
            check_call([f"./iso {f1} {f2} {outf} {label_fn1} {label_fn2}"], cwd='.', shell=True)
    else:
        check_call([f"./iso {f1} {f2} {outf}"], cwd='.', shell=True)


def get_graph_stats(fn):
    '''output number of nodes and edges'''
    with open(indir + fn) as f:
        line = next(f)
        n, m = line.split()
        return int(n), int(m)


def get_equivalent_graphs(fns):
    '''group graphs together according to equivalent stats'''
    stats2graphs = ddict(list)
    for fn in fns:
        stats2graphs[get_graph_stats(fn)].append(fn)
    return stats2graphs


def get_dataset_pairs(fns):
    '''get pairs for each group of graphs'''
    stats2graphs = get_equivalent_graphs(fns)
    generators = []
    for graphs in stats2graphs.values():
        if len(graphs) > 1:
            generators.append(combinations(graphs, 2))
    return chain(*generators)

def get_graph_groups(results_dir, dataset, indir):
    outdir = f'{results_dir}/{dataset}_groups/'
    exec = 'gr'
    compile_exec = f"gcc -o {exec} nauty_gr.c nauty26r11/nauty.a"
    check_call(compile_exec, cwd=".", shell=True)

    os.makedirs(outdir, exist_ok=True)
    fns = os.listdir(indir)
    for fn in fns:
        if fn.endswith(".adj"):
            # print(fn)
            inf = indir + fn
            outf = outdir + fn.split('.')[0] + '.txt'
            check_call(f"./{exec} {inf} {outf}", cwd=".", shell=True)

if __name__ == '__main__':
    # experiment: write genrators for each graph
    # https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets#contact
    dataset = 'REDDIT-BINARY'
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

    ds = [
        'MUTAG',
        'IMDB-BINARY',
    ]

    for dataset in ds:
        print(dataset)
        os.makedirs(results_dir, exist_ok=True)
        indir = f'datasets_nauty/{dataset}_adj/'

        # compile program for getting groups for each graph
        if compute_groups:
            get_graph_groups(results_dir, dataset, indir)

        # this will be used to generate files that are isomorphic to each other
        outdir = f'{results_dir}/{dataset}_iso/'
        if (not os.path.exists(indir + f"1.node_labels") and use_node_labels):
            print(f"Files for node labels is not present but use_node_labels=True. Skip the dataset {dataset}")
            continue
        os.makedirs(outdir, exist_ok=True)

        # compile program for running graph isomorphism
        if use_node_labels:
            compile_exec = "gcc -o iso nauty_iso_labels.c nauty26r11/nauty.a"
        else:
            compile_exec = "gcc -o iso nauty_iso.c nauty26r11/nauty.a"
        check_call(compile_exec, cwd=".", shell=True)

        # run pairwise graph isomorphism test for a data set
        fns = list(filter(lambda x: x.endswith('.adj'), os.listdir(indir)))
        f = partial(run_nauty, use_node_labels=use_node_labels)  # function to call executable

        # this will generate only pairs that have the same number of nodes/edges
        pairs = get_dataset_pairs(fns)

        pool = mp.Pool(processes=64)
        pool.starmap(f, pairs)
        pool.close()
