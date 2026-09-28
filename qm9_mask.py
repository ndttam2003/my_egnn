import os
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv

import jax
import jax.numpy as jnp
from flax import nnx
import optax
import orbax.checkpoint as ocp
from tensorboardX import SummaryWriter
import torch_geometric

from qm9_dataloader import get_train_val_test
from models.gcl_jax import E_GCL_JAX

load_dotenv(override=True)

# =========================================================================
# 0. CẤU HÌNH BIẾN MÔI TRƯỜNG & STATIC SHAPES
# =========================================================================
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 96))
MAX_NODE_PER_GRAPH = int(os.getenv("max_node_per_graph", 29))

# Mỗi phân tử QM9 (tối đa 29 nút) có tối đa 29 * 28 = 812 cạnh (Complete Graph không self-loop)
DEFAULT_MAX_EDGE_PER_GRAPH = MAX_NODE_PER_GRAPH * (MAX_NODE_PER_GRAPH - 1)
MAX_EDGE_PER_GRAPH = int(os.getenv("max_edge_per_graph", DEFAULT_MAX_EDGE_PER_GRAPH))

MAX_TOTAL_NODES = BATCH_SIZE * MAX_NODE_PER_GRAPH
MAX_TOTAL_EDGES = BATCH_SIZE * MAX_EDGE_PER_GRAPH

# =========================================================================
# 1. ĐỊNH NGHĨA MODEL EGNN CÓ HỖ TRỢ MASKING
# =========================================================================
class EGNN(nnx.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        rngs: nnx.Rngs,
        edge_attr_dim: int = 0,
        act_fn=nnx.silu,
        n_layers: int = 4,
        attention: bool = False,
        recurrent: bool = True,
        use_bias: bool = True,
    ):
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.layers = nnx.List()
        for _ in range(n_layers):
            self.layers.append(
                E_GCL_JAX(
                    input_dim=hidden_dim,
                    hidden_dim=hidden_dim,
                    output_dim=hidden_dim,
                    rngs=rngs,
                    edge_attr_dim=edge_attr_dim,
                    act_fn=act_fn,
                    attention=attention,
                    recurrent=recurrent,
                    use_bias=use_bias,
                )
            )

        self.embedding = nnx.Linear(
            in_features=input_dim,
            out_features=hidden_dim,
            rngs=rngs,
        )

        self.node_dec = nnx.Sequential(
            nnx.Linear(
                in_features=hidden_dim,
                out_features=hidden_dim,
                kernel_init=jax.nn.initializers.glorot_normal(),
                rngs=rngs,
            ),
            act_fn,
            nnx.Linear(
                in_features=hidden_dim,
                out_features=hidden_dim,
                kernel_init=jax.nn.initializers.glorot_normal(),
                rngs=rngs,
            ),
        )

        self.graph_dec = nnx.Sequential(
            nnx.Linear(
                in_features=hidden_dim,
                out_features=hidden_dim,
                kernel_init=jax.nn.initializers.glorot_normal(),
                rngs=rngs,
            ),
            act_fn,
            nnx.Linear(
                in_features=hidden_dim,
                out_features=1,
                kernel_init=jax.nn.initializers.glorot_normal(),
                rngs=rngs,
            ),
        )

    def __call__(
        self,
        hidden_state: jax.Array,        # [MAX_TOTAL_NODES, input_dim]
        coordinate: jax.Array,          # [MAX_TOTAL_NODES, 3]
        relative_coordinate: jax.Array, # [MAX_TOTAL_EDGES, 3] hoặc tương tự
        edge_index: jax.Array,          # [2, MAX_TOTAL_EDGES]
        node_mask: jax.Array,           # [MAX_TOTAL_NODES, 1]
        edge_mask: jax.Array,           # [MAX_TOTAL_EDGES, 1]
        batch: jax.Array,               # [MAX_TOTAL_NODES]
        num_graphs: int = BATCH_SIZE,
        edge_attr: jax.Array = None,
        rngs: nnx.Rngs = None,
    ):
        h = self.embedding(hidden_state)

        for i in range(self.n_layers):
            h, coordinate, _ = self.layers[i](
                hidden_state=h,
                coordinate=coordinate,
                relative_coordinate=relative_coordinate,
                edge_index=edge_index,
                edge_attr=edge_attr,
                batch=batch,
                num_graphs=num_graphs,
            )

        h = self.node_dec(h)

        # 1. Gom tụ về NUM_GRAPHS (gồm cả đồ thị ảo)
        graph_embedding = jnp.zeros((num_graphs, self.hidden_dim)).at[batch].add(h)

        # 2. CẮT BỎ ĐỒ THỊ ẢO: Chỉ lấy đúng batch_size đồ thị thật [0 : batch_size]!
        graph_embedding = graph_embedding[:BATCH_SIZE]

        # 3. Dự đoán nhãn cho batch_size phân tử thật
        pred = self.graph_dec(graph_embedding).squeeze(-1)  # shape: [batch_size]
        return pred, graph_embedding


# =========================================================================
# 2. HÀM CHUẨN BỊ PADDED BATCH (ĐẢM BẢO STATIC SHAPE CHO JIT)
# =========================================================================
def prepare_padded_batch(batch, max_nodes: int, max_edges: int, batch_size: int) -> dict:
    # 1. Trích xuất dữ liệu bằng .detach().cpu().numpy() để tương thích NumPy 2.0+
    feat_tensor = batch.feature if hasattr(batch, 'feature') else batch.x
    feat_real = feat_tensor.detach().cpu().numpy()
    pos_real = batch.pos.detach().cpu().numpy()
    rel_real = batch.relatix_matrix.detach().cpu().numpy()
    edge_index_real = batch.edge_index.detach().cpu().numpy()
    batch_idx_real = batch.batch.detach().cpu().numpy()

    num_real_nodes = pos_real.shape[0]
    num_real_edges = edge_index_real.shape[1]

    pad_nodes = max_nodes - num_real_nodes
    pad_edges = max_edges - num_real_edges

    # 2. Pad Node Features & Coordinates
    hidden_state = np.pad(feat_real, ((0, pad_nodes), (0, 0)), mode='constant', constant_values=0)
    coordinate = np.pad(pos_real, ((0, pad_nodes), (0, 0)), mode='constant', constant_values=0)

    # 3. Node Mask
    node_mask = np.zeros((max_nodes, 1), dtype=np.float32)
    node_mask[:num_real_nodes] = 1.0

    # 4. Thùng rác cho Edge Index
    dummy_node_idx = max_nodes - 1
    edge_index = np.full((2, max_edges), fill_value=dummy_node_idx, dtype=np.int32)
    edge_index[:, :num_real_edges] = edge_index_real

    # 5. Pad Relative Coordinate
    if rel_real.ndim == 2:
        rel_len = rel_real.shape[0]
        pad_rel = max_edges - rel_len
        relative_coordinate = np.pad(rel_real, ((0, pad_rel), (0, 0)), mode='constant', constant_values=0)
    else:
        relative_coordinate = rel_real

    # 6. Edge Mask
    edge_mask = np.zeros((max_edges, 1), dtype=np.float32)
    edge_mask[:num_real_edges] = 1.0

    # 7. Batch Index
    batch_index = np.full(max_nodes, fill_value=batch_size, dtype=np.int32)
    batch_index[:num_real_nodes] = batch_idx_real

    # 8. Labels
    labels = batch.y[:, 1].detach().cpu().numpy()

    return {
        "hidden_state": jnp.array(hidden_state, dtype=jnp.float32),
        "coordinate": jnp.array(coordinate, dtype=jnp.float32),
        "relative_coordinate": jnp.array(relative_coordinate, dtype=jnp.float32),
        "edge_index": jnp.array(edge_index, dtype=jnp.int32),
        "node_mask": jnp.array(node_mask, dtype=jnp.float32),
        "edge_mask": jnp.array(edge_mask, dtype=jnp.float32),
        "batch": jnp.array(batch_index, dtype=jnp.int32),
        "edge_attr": None,
        "labels": jnp.array(labels, dtype=jnp.float32),
    }


# =========================================================================
# 3. CÁC BƯỚC TÍNH TOÁN ĐƯỢC JIT BẰNG @nnx.jit
# =========================================================================
def loss_fn(model: nnx.Module, batch_data: dict,):
    # Lấy labels ra tính loss
    labels = batch_data["labels"]
    inputs = {k: v for k, v in batch_data.items() if k != "labels"}
    pred, _ = model(**inputs)
    loss = jnp.mean(jnp.abs(pred - labels))
    return loss

@nnx.jit
def train_step(model: nnx.Module, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch: dict):
    """Bước huấn luyện được JIT biên dịch 1 lần duy nhất."""
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=False)
    loss, grads = grad_fn(model, batch)
    metrics.update(loss=loss)
    optimizer.update(model, grads)

@nnx.jit
def eval_step(model: nnx.Module, metrics: nnx.MultiMetric, batch: dict):
    """Bước đánh giá được JIT biên dịch 1 lần duy nhất."""
    loss = loss_fn(model, batch)
    metrics.update(loss=loss)


# =========================================================================
# 4. CÁC HÀM RUN EPOCH
# =========================================================================
def train_one_epoch(model, optimizer, metrics, train_loader, epoch: int):
    for batch in tqdm(train_loader, desc=f"Training epoch {epoch}"):
        # Bỏ qua nếu batch cuối không đủ BATCH_SIZE (để tránh recompilation do đổi shape)
        # if batch.num_graphs < BATCH_SIZE:
        #     continue
        padded_batch = prepare_padded_batch(batch, MAX_TOTAL_NODES, MAX_TOTAL_EDGES, BATCH_SIZE)
        train_step(model, optimizer, metrics,  batch=padded_batch)

    computed = metrics.compute()
    metrics.reset()
    return computed

def evaluate(model, metrics, loader, desc: str = "Evaluating"):
    for batch in tqdm(loader, desc=desc):
        if batch.num_graphs < BATCH_SIZE:
            continue
        padded_batch = prepare_padded_batch(batch, MAX_TOTAL_NODES, MAX_TOTAL_EDGES, BATCH_SIZE)
        eval_step(model, metrics, padded_batch)

    computed = metrics.compute()
    metrics.reset()
    return computed


# =========================================================================
# 5. QUẢN LÝ CHECKPOINT VỚI MAX_CHECKPOINTS
# =========================================================================
def init_checkpoint_manager(save_dir: str, max_checkpoints: int) -> ocp.CheckpointManager:
    """Tự động giữ tối đa max_checkpoints gần nhất bằng Orbax."""
    options = ocp.CheckpointManagerOptions(
        max_to_keep=max_checkpoints,
        create=True
    )
    return ocp.CheckpointManager(
        os.path.abspath(save_dir),
        options=options
    )

def save_model_checkpoint(manager: ocp.CheckpointManager, step: int, model: nnx.Module):
    _, state = nnx.split(model)
    manager.save(step, args=ocp.args.StandardSave(state))
    manager.wait_until_finished()


# =========================================================================
# 6. HÀM MAIN CHÍNH
# =========================================================================
def main():
    print(f"--- Cấu hình JAX Static Shape ---")
    print(f"BATCH_SIZE: {BATCH_SIZE}")
    print(f"Max nodes per graph: {MAX_NODE_PER_GRAPH} -> Total padded nodes: {MAX_TOTAL_NODES}")
    print(f"Max edges per graph: {MAX_EDGE_PER_GRAPH} -> Total padded edges: {MAX_TOTAL_EDGES}")

    train_loader, val_loader, test_loader = get_train_val_test()

    rngs = nnx.Rngs(0)
    model = EGNN(
        input_dim=15,
        hidden_dim=128,
        rngs=rngs,
        edge_attr_dim=0,
    )

    learning_rate = float(os.getenv("LR", 1e-3))
    optimizer = nnx.Optimizer(
        model, optax.adamw(learning_rate=learning_rate), wrt=nnx.Param
    )
    metrics = nnx.MultiMetric(loss=nnx.metrics.Average('loss'))

    train_model = nnx.view(model)
    eval_model = nnx.view(model)

    writer = SummaryWriter(logdir=os.getenv("TENSOR_BOARD_DIR", "runs/egnn_qm9"))

    # Thiết lập Checkpoint Manager với max_checkpoints từ env
    save_dir = os.path.join("./", os.getenv("save_dir", "checkpoints"))
    max_checkpoints = int(os.getenv("max_checkpoints", 3))
    ckpt_manager = init_checkpoint_manager(save_dir, max_checkpoints)

    eval_every = int(os.getenv("eval_every", 1))
    num_epoch = int(os.getenv("num_epoch", 100))

    # Vòng lặp huấn luyện chính
    for epoch in range(1, num_epoch + 1):
        # 1. Huấn luyện (JIT được biên dịch ở batch đầu tiên, các batch sau chạy cực nhanh)
        train_metrics = train_one_epoch(train_model, optimizer, metrics, train_loader, epoch)
        for metric, val in train_metrics.items():
            writer.add_scalar(f'train_{metric}', val, epoch)

        # 2. Đánh giá Validation định kỳ
        if epoch % eval_every == 0 or epoch == num_epoch:
            val_metrics = evaluate(eval_model, metrics, val_loader, desc=f"Validation epoch {epoch}")
            for metric, val in val_metrics.items():
                writer.add_scalar(f'val_{metric}', val, epoch)

            # Lưu checkpoint có kiểm soát max_checkpoints
            save_model_checkpoint(ckpt_manager, step=epoch, model=model)

    # 3. Đánh giá Test cuối cùng
    test_metrics = evaluate(eval_model, metrics, test_loader, desc="Test evaluation")
    for metric, val in test_metrics.items():
        writer.add_scalar(f'test_{metric}', val, num_epoch)

    print(f"\nHuấn luyện hoàn tất! Checkpoint được lưu tại: {save_dir}")

if __name__ == "__main__":
    main()