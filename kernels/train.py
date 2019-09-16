import pathlib
import subprocess
from utils import save_to_graphml, read_kernel_matrix, Evaluation
from arguments import get_args
from torch_geometric.datasets import TUDataset
from collections import Counter
from torch_geometric.transforms.one_hot_degree import OneHotDegree


def main(args):
    path = pathlib.Path('./src/gkernel')
    if not path.is_file():
        subprocess.call(["make"], cwd="./src", shell=True)
    dataset = TUDataset(root=f'{args.dir}/Pytorch_geometric/{args.dataset}', name=args.dataset)

    if dataset.num_features == 0:
        max_degree = -1
        for data in dataset:
            edge_index = data.edge_index
            degrees = Counter(list(map(int, edge_index[0])))
            if max_degree < max(degrees.values()):
                max_degree = max(degrees.values())

        dataset.transform = OneHotDegree(max_degree=max_degree, cat=False)

    path = pathlib.Path(f'{args.dir}/GraphML/{args.dataset}/{args.dataset.lower()}_{args.kernel}.kernel')
    if not path.is_file():
        save_to_graphml(dataset, f'{args.dir}/GraphML/{args.dataset}')
        cmd = ['./src/gkernel']
        cmd.append('-k')
        cmd.append(args.kernel)
        if args.parameter:
            cmd.append('-p')
            cmd.append(args.parameter)
        cmd.append('-i')
        cmd.append(f'{args.dir}/GraphML/{args.dataset}/{args.dataset.lower()}.list')
        cmd.append('-g')
        cmd.append(f'{args.dir}/GraphML/{args.dataset}/data/')
        cmd.append('-o')
        cmd.append(f'{args.dir}/GraphML/{args.dataset}/{args.dataset.lower()}_{args.kernel}.kernel')
        subprocess.call(cmd)

    K = read_kernel_matrix(f'{args.dir}/GraphML/{args.dataset}/{args.dataset.lower()}_{args.kernel}.kernel')

    y = dataset.data.y.data.numpy()

    ev = Evaluation(K, y, verbose=True)

    accs = ev.evaluate()

if __name__ == "__main__":
    args = get_args()
    main(args)