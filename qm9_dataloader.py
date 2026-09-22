import torch
import torch_geometric
import numpy as np
import os
from torch_geometric.loader import DataLoader
flag = [True]
def remove_self_loop(data):
    if flag[0]: 
        print("transform")
        flag[0]=False
    edge_index = data.edge_index
    edge_attr = data.edge_attr
    new_edge_index, new_edge_attr = torch_geometric.utils.remove_self_loops(edge_index, edge_attr)
    data.edge_index = new_edge_index
    data.edge_attr = new_edge_attr
    return data

def get_train_val_test():
    qm9_dataset = torch_geometric.datasets.QM9(root='data/QM9', pre_transform=remove_self_loop)
    np.random.seed(0)
    print("Warning seed is constant")
    perm = torch.from_numpy(np.random.permutation(len(qm9_dataset)))

    train_idx = perm[:100000]
    index_test = int(0.1 * len(qm9_dataset)) + 1
    test_idx   = perm[100000:100000 + index_test]
    val_idx  = perm[100000 + index_test:]

    train_dataset = qm9_dataset[train_idx]
    val_dataset   = qm9_dataset[val_idx]
    test_dataset  = qm9_dataset[test_idx]
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=os.getenv("BATCH_SIZE"), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=os.getenv("BATCH_SIZE"))
    test_loader = DataLoader(test_dataset, batch_size=os.getenv("BATCH_SIZE"))

    return train_loader, val_loader, test_loader

