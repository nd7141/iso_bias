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
    with open(indir + fn) as f:
        line = next(f)
        n, m = line.split()
        return int(n), int(m)


def get_equivalent_graphs(fns):
    stats2graphs = ddict(list)
    for fn in fns:
        stats2graphs[get_graph_stats(fn)].append(fn)
    return stats2graphs


def get_dataset_pairs(fns):
    stats2graphs = get_equivalent_graphs(fns)
    generators = []
    for graphs in stats2graphs.values():
        if len(graphs) > 1:
            generators.append(combinations(graphs, 2))
    return chain(*generators)


if __name__ == '__main__':

    use_node_labels = True
    compute_groups = True
    results_dir = 'results_no_labels/'

    # experiment: write genrators for each graph
    # https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets#contact
    dataset = 'REDDIT-BINARY'
    ds = [
        # 'FIRSTMM_DB',
        # 'OHSU',
        # 'KKI',
        # 'Peking_1',
        # 'MUTAG',
        # 'MSRC_21C',
        # 'MSRC_9',
        # 'Cuneiform',
        # 'SYNTHETIC',
        # 'COX2_MD',
        # 'BZR_MD',
        # 'PTC_MM',
        # 'PTC_MR',
        # 'PTC_FM',
        # 'PTC_FR',
        # 'DHFR_MD',
        # 'Synthie',
        # 'BZR',
        # 'ER_MD',
        # 'COX2',
        # 'MSRC_21',
        # 'ENZYMES',
        # 'DHFR',
        # 'IMDB-BINARY',
        # 'PROTEINS',
        # 'DD',
        # 'IMDB-MULTI',
        # 'AIDS',
        # 'REDDIT-BINARY',
        # 'Letter-high',
        # 'Letter-low',
        # 'Letter-med',
        # 'Fingerprint',
        # 'COIL-DEL',
        # 'COIL-RAG',
        # 'NCI1',
        # 'NCI109',
        # 'FRANKENSTEIN',
        # 'Mutagenicity',
        # 'REDDIT-MULTI-5K',
        # 'COLLAB',
        # 'Tox21_ARE',
        # 'Tox21_aromatase',
        # 'Tox21_MMP',
        # 'Tox21_ER',
        # 'Tox21_HSE',
        # 'Tox21_AHR',
        # 'Tox21_PPAR-gamma',
        # 'Tox21_AR-LBD',
        # 'Tox21_p53',
        'Tox21_ER_LBD',
        'Tox21_ATAD5',
        'Tox21_AR',
        'REDDIT-MULTI-12K'
    ]
    ds = ['Cuneiform']

    for dataset in ds:
        print(dataset)
        os.makedirs(results_dir, exist_ok=True)
        indir = f'datasets/data_adj/{dataset}_adj/'
        outdir = f'{results_dir}/{dataset}_groups/'

        # compile program for running graph isomorphism
        if compute_groups:
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
                    # try:
                    check_call(f"./{exec} {inf} {outf}", cwd=".", shell=True)
                    # except Exception as e:
                    #     print(f"File: {fn}. Exception: {e}")

        #  experiment: run graph isomorphism for all graphs in a folder, write results to outf
        # ds = ['DD', 'enzymes', 'NCI1', 'NCI109']
        # ds = ['enzymes']
        # ds = ['REDDIT-MULTI-5K', 'REDDIT-MULTI-12K']
        # for dataset in ds:
        indir = f'datasets/data_adj/{dataset}_adj/'

        outdir = f'{results_dir}/{dataset}_iso/'
        if (not os.path.exists(indir + f"1.node_labels") and use_node_labels):
            print(f"Files for node labels is not present but use_node_labels=True. Skip the dataset {dataset}")
            continue
        os.makedirs(outdir, exist_ok=True)

        outf = f'{results_dir}/{dataset}_iso.txt'
        if os.path.exists(outf):
            check_call(f"rm {outf}", cwd=".", shell=True)

        # compile program for running graph isomorphism
        if use_node_labels:
            compile_exec = "gcc -o iso nauty_iso_labels.c nauty26r11/nauty.a"
        else:
            compile_exec = "gcc -o iso nauty_iso.c nauty26r11/nauty.a"
        check_call(compile_exec, cwd=".", shell=True)

        # run graph isomorphism
        fns = list(filter(lambda x: x.endswith('.adj'), os.listdir(indir)))
        N = len(fns)
        ids = []
        start = time.time()

        f = partial(run_nauty, use_node_labels=use_node_labels)  # function to call executable

        # pprint(get_equivalent_graphs(fns))
        pairs = get_dataset_pairs(fns)
        # pairs = combinations(fns, 2)

        pool = mp.Pool(processes=64)
        pool.starmap(f, pairs)
        pool.close()

        # for i in range(N-1):
        #     start2i = time.time()
        #     for j in range(i+1, N):
        #         f1 = indir + fns[i]
        #         f2 = indir + fns[j]
        #         print(f1, f2)
        #         check_call(f"./iso {f1} {f2} {outf} &" , cwd = '.', shell= True)
        #         ids.append([f1, f2])
        #     finish2i = time.time()
        #     itsec = (finish2i-start2i)/(N-i-1)
        #     print('Total time spent on {} iteration'.format(i), finish2i - start2i)
        #     print('Average time per pair', itsec)
        #     print('Estimated time remaining', (N-i)*(N-i-1)*itsec/2)
        # end = time.time()
        # print('Total time spent:', end - start)

        # write labels of graphs
        # with open(f'datasets/data_graphml/{dataset}.label') as f:
        #     labels = next(f).strip().split()
        # fn2label = dict(zip(sorted(fns, key=lambda x: int(re.findall('\d+', x)[0])), labels))
        #
        # print(fn2label)

        # lines = []
        #
        # with open(outf) as f:
        #     for c, line in enumerate(f):
        #         f1 = ids[c][0].split('/')[-1]
        #         f2 = ids[c][1].split('/')[-1]
        #         lines.append(line.strip() + f" {fn2label[f1]} {fn2label[f2]}\n")
        #
        # with open(outf, 'w') as f:
        #     f.write(''.join(lines))
