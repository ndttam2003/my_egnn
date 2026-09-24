import torch
import torch_geometric
import numpy as np
import os
from torch_geometric.loader import DataLoader
import dotenv
dotenv.load_dotenv(override=True)
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

def z_to_15_features(z, charge_power=2, charge_scale=9.0):
    """
    Chuyển đổi tensor atomic numbers z thành vector đặc trưng 15 chiều giống hệt code gốc EGNN.
    
    Tham số:
        z: Tensor 1D chứa số proton (data.z hoặc batch.z), shape [N]
        charge_power: Bậc đa thức lớn nhất (mặc định = 2) -> sinh ra bậc 0, 1, 2 (3 bậc)
        charge_scale: Hệ số chuẩn hóa (mặc định = 9.0 là Z max của Flo)
    
    Trả về:
        atom_scalars: Tensor kích thước [N, 15]
    """
    device = z.device
    
    # 1. Danh sách 5 nguyên tố trong QM9: H (1), C (6), N (7), O (8), F (9)
    species = torch.tensor([1, 6, 7, 8, 9], device=device)
    
    # 2. Tạo One-hot 5 chiều: so sánh z với từng nguyên tố -> shape [N, 5]
    one_hot = (z.unsqueeze(-1) == species.unsqueeze(0)).float()
    
    # 3. Tính các bậc lũy thừa của Z: (Z/9)^0, (Z/9)^1, (Z/9)^2 -> shape [N, 3]
    powers = torch.arange(charge_power + 1.0, device=device, dtype=torch.float32)
    charge_tensor = (z.unsqueeze(-1).float() / charge_scale).pow(powers)
    
    # 4. Tích Tensor (Outer Product) giữa One-hot [N, 5, 1] và Lũy thừa [N, 1, 3]
    # -> Kết quả: [N, 5, 3] rồi làm phẳng thành [N, 15]
    atom_scalars = (one_hot.unsqueeze(-1) * charge_tensor.unsqueeze(1)).view(-1, 15)
    
    return atom_scalars

def cal_relative_matrix(coord, edge_index):
    # neighbor_coord = []
    # print(vmap_coor(edge_index[0], edge_index[1]))
    # for src, target in edge_index.T:
    #     dist = (coord[target] - coord[src])
    #     print(dist.shape)
    #     neighbor_coord.append(dist)
    # print(neighbor_coord)
    # return torch.vstack(neighbor_coord)
    vmap_coor = torch.func.vmap(lambda s, t : coord[t] - coord[s])
    return vmap_coor(edge_index[0], edge_index[1])

def transfrom_data(data):
    z = data.z
    data.feature = z_to_15_features(z)
    relatix_matrix = cal_relative_matrix(data.pos, data.edge_index)
    data.relatix_matrix = relatix_matrix
    return data

def get_train_val_test():
    qm9_dataset = torch_geometric.datasets.QM9(root='data/QM9', pre_transform=remove_self_loop)
    qm9_dataset_transformed = [transfrom_data(x) for x in qm9_dataset ]
    np.random.seed(0)
    print("Warning seed is constant")
    perm = torch.from_numpy(np.random.permutation(len(qm9_dataset_transformed)))

    train_idx = perm[:100000]
    index_test = int(0.1 * len(qm9_dataset_transformed)) + 1
    test_idx   = perm[100000:100000 + index_test]
    val_idx  = perm[100000 + index_test:]

    # train_dataset = qm9_dataset_transformed[train_idx]
    # val_dataset   = qm9_dataset_transformed[val_idx]
    # test_dataset  = qm9_dataset_transformed[test_idx]
    train_dataset = [qm9_dataset_transformed[i] for i in train_idx]
    val_dataset   = [qm9_dataset_transformed[i] for i in val_idx]
    test_dataset  = [qm9_dataset_transformed[i] for i in test_idx]
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=int(os.getenv("BATCH_SIZE")), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=int(os.getenv("BATCH_SIZE")))
    test_loader = DataLoader(test_dataset, batch_size=int(os.getenv("BATCH_SIZE")))

    return train_loader, val_loader, test_loader

