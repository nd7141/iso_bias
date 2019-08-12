from subprocess import check_call
import os
import re
import time
import multiprocessing as mp
from functools import partial
from itertools import combinations

def run_nauty(f1, f2):
    outf = outdir + f1.split('/')[-1] + '-' + f2.split('/')[-1]
    f1 = indir + f1
    f2 = indir + f2
    check_call(f"./iso {f1} {f2} {outf} &", cwd='.', shell=True)

if __name__ == '__main__':

    # experiment: write genrators for each graph
    # https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets#contact
    dataset = 'REDDIT-BINARY'
    # ds = [
 # 'Tox21_AHR',
 # 'Tox21_AR',
 # 'Tox21_ARE',
 # 'Tox21_AR-LBD',
 # 'Tox21_aromatase',
 # 'Tox21_ATAD5',
 # 'Tox21_ER',
 # 'Tox21_ER_LBD',
 # 'Tox21_HSE',
 # 'Tox21_MMP',
 # 'Tox21_p53',
 # 'Tox21_PPAR-gamma',
 # 'TWITTER-Real-Graph-Partial',
 # 'Letter-high',
 # 'Letter-low',
 # 'Letter-med',
 # 'Cuneiform',
 # 'DBLP_v1',
 # 'DHFR',
 # 'PROTEINS',
 # 'PTC_FM',
 # 'PTC_FR',
 # 'PTC_MM',
 # 'PTC_MR',
 # 'SYNTHETIC',
 # 'MSRC_21',
 # 'MSRC_21C',
 # 'MSRC_9',
 # 'OHSU',
 # 'COX2',
 # 'COX2_MD',
 # 'DHFR_MD',
 # 'ER_MD',
 # 'FIRSTMM_DB',
 # 'KKI',
 #        'Mutagenicity']
    ds = ['MUTAG']
    for dataset in ds:
        os.makedirs('results', exist_ok=True)
        indir = f'datasets/data_adj/{dataset}_adj/'
        outdir = f'results/{dataset}_groups/'

        # compile program for running graph isomorphism
        exec = 'gr'
        compile_exec = f"gcc -o {exec} nauty_gr.c nauty26r11/nauty.a"
        check_call(compile_exec, cwd = ".", shell=True)

        os.makedirs(outdir, exist_ok = True)
        fns = os.listdir(indir)
        for fn in fns:
            if fn.endswith(".adj"):
                print(fn)
                inf = indir + fn
                outf = outdir + fn.split('.')[0] + '.txt'
                # try:
                check_call(f"./{exec} {inf} {outf}", cwd = ".", shell=True)
                # except Exception as e:
                #     print(f"File: {fn}. Exception: {e}")

    #  experiment: run graph isomorphism for all graphs in a folder, write results to outf
    # ds = ['DD', 'enzymes', 'NCI1', 'NCI109']
    # ds = ['enzymes']
    # ds = ['REDDIT-MULTI-5K', 'REDDIT-MULTI-12K']
    # for dataset in ds:
    #     indir = f'datasets/data_adj/{dataset}_adj/'
    #
    #     outdir = f'results/{dataset}_iso2/'
    #     os.makedirs(outdir, exist_ok=True)
    #
    #     outf = f'results/{dataset}_iso.txt'
    #     if os.path.exists(outf):
    #         check_call(f"rm {outf}", cwd =".", shell = True)
    #
    #     # compile program for running graph isomorphism
    #     compile_exec = "gcc -o iso nauty_iso.c nauty26r11/nauty.a"
    #     check_call(compile_exec, cwd = ".", shell=True)
    #
    #     # run graph isomorphism
    #     fns = list(filter(lambda x: x.endswith('.adj'), os.listdir(indir)))
    #     N = len(fns)
    #     ids = []
    #     start = time.time()
    #
    #     f = partial(run_nauty,
    #                 outf=outf)
    #     pairs = combinations(fns, 2)
    #
    #     pool = mp.Pool(processes=64)
    #     pool.starmap(run_nauty, pairs)
    #     pool.close()

        # for i in range(N-1):
        #     start2i = time.time()
        #     for j in range(i+1, N):
        #         f1 = indir + fns[i]
        #         f2 = indir + fns[j]
        #         check_call(f"./iso {f1} {f2} {outf}", cwd = '.', shell= True)
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