import numpy as np
import torch
import torch.nn.functional as F
from gin.arguments import get_args
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from gin.transformers import Random
from gin.models import GCN
from gin.utils import test

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():

    args = get_args()

    dataset = TUDataset(root=args.dir, name=args.dataset).shuffle()

    if dataset.num_features == 0:
        dataset.transform = Random()
    elif args.randomize:
        dataset.transform = Random(vector_size=args.randomize)

    dataset_size = len(dataset)

    split = int(np.floor(dataset_size * args.split))

    test_dataset = dataset[split:]
    train_dataset = dataset[:split]

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    model = GCN(dataset.num_features, dataset.num_classes, args.hidden, args.dropout).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(1, args.num_epochs):

        model.train()

        if epoch % 50 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.5 * param_group['lr']

        train_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data.x, data.edge_index, data.batch)
            loss = F.nll_loss(output, data.y)
            loss.backward()
            train_loss += loss.item() * data.num_graphs
            optimizer.step()

        train_loss = train_loss / len(train_dataset)

        train_acc = test(model, train_loader, device)
        test_acc = test(model, test_loader, device)

        print('Epoch: {:03d}, Train Loss: {:.7f}, '
              'Train Acc: {:.7f}, Test Acc: {:.7f}'.format(epoch, train_loss,
                                                           train_acc, test_acc))


if __name__ == "__main__":
    main()




