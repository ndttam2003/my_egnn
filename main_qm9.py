import torch_geometric
import jax
import jax.numpy as jnp
from flax import nnx
import optax
from tensorboardX import SummaryWriter
import orbax.checkpoint as ocp
import tensorboard
from qm9_dataloader import get_train_val_test
from models.gcl_jax import E_GCL_JAX
from dotenv import load_dotenv
import os
from tqdm import tqdm
load_dotenv(override=True)

class EGNN(nnx.Module):
    def __init__(self,
                input_dim, 
                hidden_dim, 
                rngs: nnx.Rngs,
                edge_attr_dim = 0,
                act_fn = nnx.silu,
                n_layers=4,
                attention = False,
                recurrent = True,
                use_bias=True,

                 ):

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.layers = nnx.List()
        for i in range(n_layers):
            self.layers.append(
                E_GCL_JAX(
                    input_dim = hidden_dim, 
                    hidden_dim = hidden_dim, 
                    output_dim = hidden_dim,
                    rngs = rngs,
                    edge_attr_dim = edge_attr_dim,
                    act_fn = act_fn,
                    attention = attention,
                    recurrent = recurrent,
                    use_bias=use_bias,
                )
            )

        self.embedding = nnx.Linear(
            in_features=input_dim,
            out_features=hidden_dim,
            rngs=rngs,
        )

        self.node_dec = nnx.Sequential(
            nnx.Linear(in_features= hidden_dim, 
                        out_features=hidden_dim,
                        kernel_init=jax.nn.initializers.glorot_normal(),
                        rngs=rngs, 
                        ),
            act_fn,
            nnx.Linear(in_features=hidden_dim,
                        out_features=hidden_dim,
                        kernel_init=jax.nn.initializers.glorot_normal(),
                        rngs=rngs, 
                        )
        )

        self.graph_dec = nnx.Sequential(
            nnx.Linear(in_features= hidden_dim, 
                        out_features=hidden_dim,
                        kernel_init=jax.nn.initializers.glorot_normal(),
                        rngs=rngs, 
                        ),
            act_fn,
            nnx.Linear(in_features=hidden_dim,
                        out_features=1,
                        kernel_init=jax.nn.initializers.glorot_normal(),
                        rngs=rngs, 
                        )
        )

    def __call__(self, 
                    hidden_state: jax.Array, 
                    coordinate: jax.Array,
                    relative_coordinate: jax.Array,
                    edge_index: jax.Array, 
                    edge_attr: jax.Array = None, 
                    batch =None,
                    rngs = None,
                    ):
        h = self.embedding(hidden_state)
        # print(h.shape)
        for i in range(self.n_layers):
            # print(i)
            h, _, _ = self.layers[i](
                h, 
                coordinate,
                relative_coordinate,
                edge_index, 
                edge_attr = edge_attr, 
                batch =batch,
                rngs = rngs,
            )

        h = self.node_dec(h)

        if batch is None:
            graph_embedding = h.sum(axis = 0)
        else:
            graph_embedding = jnp.zeros((batch[-1] + 1, h.shape[1])).at[batch].add(h)

        pred = self.graph_dec(graph_embedding)

        return pred, graph_embedding

class MeanAbsoluteError(nnx.metrics.Average):
    def __init__(self, argname: str = 'values'):
        super().__init__(argname=argname)

    def update(self, *, predictions: jax.Array, targets: jax.Array, **_) -> None:  # type: ignore[override]
        if predictions.shape != targets.shape:
            raise ValueError(
                f'Expected predictions.shape==labels.shape, '
                f'got {predictions.shape} and {targets.shape}'
            )
        super().update(values=jnp.abs(predictions - targets).mean())


def loss_fn(model, batch:dict, rngs: nnx.Rngs | None = None):
  labels = batch.pop("labels")
  pred, graph_embedding = model(**batch, rngs = rngs)
  loss = jnp.mean(jnp.abs(pred - labels))
  return loss

def train_step(model, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, rngs: nnx.Rngs, batch):
  """Train for a single step."""
  grad_fn = nnx.value_and_grad(loss_fn, has_aux=False)
  loss,  grads = grad_fn(model, batch, rngs)
  metrics.update(loss=loss)  # In-place updates.
  optimizer.update(model, grads)  # In-place updates.

def pred_step(model, batch:None):
  logits = model(**batch, rngs = None)
  return logits

def eval_step(model, metrics: nnx.MultiMetric, batch):
  loss = loss_fn(model, batch)
  metrics.update(loss=loss)  # In-place updates.


# =========================================================================
# 1. HÀM CHUẨN BỊ BATCH VÀ CÁC BƯỚC RUN EPOCH
# =========================================================================

def prepare_batch(batch) -> dict:
    """Chuyển đổi PyG batch sang dictionary dữ liệu JAX."""
    return {
        "hidden_state": jnp.array(batch.feature),
        "coordinate": jnp.array(batch.pos),
        "relative_coordinate": jnp.array(batch.relatix_matrix),
        "edge_index": jnp.array(batch.edge_index),
        "edge_attr": None,
        "batch": jnp.array(batch.batch),
        "labels": jnp.array(batch.y[:, 1])
    }

def train_one_epoch(model, optimizer, metrics, rngs, train_loader, epoch: int):
    """Huấn luyện 1 epoch."""
    for batch in tqdm(train_loader, desc=f"Training, epoch = {epoch}: "):
        input_training = prepare_batch(batch)
        train_step(model, optimizer, metrics, rngs, batch=input_training)
    
    # Tính toán và reset metrics cho epoch
    computed_metrics = metrics.compute()
    metrics.reset()
    return computed_metrics

def evaluate(model, metrics, loader, desc: str = "Evaluating: "):
    """Đánh giá trên tập validation hoặc test."""
    for batch in tqdm(loader, desc=desc):
        input_data = prepare_batch(batch)
        eval_step(model, metrics, input_data)
        
    computed_metrics = metrics.compute()
    metrics.reset()
    return computed_metrics

# =========================================================================
# 2. HÀM QUẢN LÝ VÀ LƯU CHECKPOINT VỚI MAX_CHECKPOINTS
# =========================================================================

def init_checkpoint_manager(save_dir: str, max_checkpoints: int) -> ocp.CheckpointManager:
    """Khởi tạo CheckpointManager kiểm soát số lượng checkpoint tối đa."""
    options = ocp.CheckpointManagerOptions(
        max_to_keep=max_checkpoints,
        create=True
    )
    # Orbax CheckpointManager tự động xóa các checkpoint cũ khi vượt quá max_to_keep
    return ocp.CheckpointManager(
        os.path.abspath(save_dir),
        options=options
    )

def save_model_checkpoint(manager: ocp.CheckpointManager, step: int, model: nnx.Module):
    """Hàm lưu trạng thái model (state) vào checkpoint."""
    _, state = nnx.split(model)
    manager.save(step, args=ocp.args.StandardSave(state))
    manager.wait_until_finished()

# =========================================================================
# 3. HÀM MAIN ĐÃ ĐƯỢC THU GỌN
# =========================================================================

def main():
    train_loader, val_loader, test_loader = get_train_val_test()

    model = EGNN(
        input_dim=15,
        hidden_dim=128,
        rngs=nnx.Rngs(0),
        edge_attr_dim=0,
    )

    learning_rate = float(os.getenv("LR", 1e-3))
    optimizer = nnx.Optimizer(
        model, optax.adamw(learning_rate=learning_rate), wrt=nnx.Param
    )
    metrics = nnx.MultiMetric(loss=nnx.metrics.Average('loss'))

    train_model = nnx.view(model)
    eval_model = nnx.view(model)

    writer = SummaryWriter(logdir=os.getenv("TENSOR_BOARD_DIR"))
    rngs = nnx.Rngs(0)

    # Khởi tạo checkpoint manager với max_checkpoints từ biến môi trường
    save_dir = os.path.join("./", os.getenv("save_dir", "checkpoints"))
    max_checkpoints = int(os.getenv("max_checkpoints", 3))
    ckpt_manager = init_checkpoint_manager(save_dir, max_checkpoints)

    eval_every = int(os.getenv("eval_every", 1))
    num_epoch = int(os.getenv("num_epoch", 100))

    # Vòng lặp Training
    for epoch in range(1, num_epoch + 1):
        # 1. Train
        train_metrics = train_one_epoch(train_model, optimizer, metrics, rngs, train_loader, epoch)
        for metric, val in train_metrics.items():
            writer.add_scalar(f'train_{metric}', val, epoch)

        # 2. Validation định kỳ & Lưu checkpoint
        if epoch % eval_every == 0 or epoch == num_epoch:
            val_metrics = evaluate(eval_model, metrics, val_loader, desc=f"Validation, epoch = {epoch}: ")
            for metric, val in val_metrics.items():
                writer.add_scalar(f'val_{metric}', val, epoch)

            # Lưu checkpoint tại mỗi lần validate (tự động xóa checkpoint cũ nếu > max_checkpoints)
            save_model_checkpoint(ckpt_manager, step=epoch, model=model)

    # 3. Đánh giá cuối cùng trên Test set (đã sửa dùng test_loader)
    test_metrics = evaluate(eval_model, metrics, test_loader, desc="Test: ")
    for metric, val in test_metrics.items():
        writer.add_scalar(f'test_{metric}', val, num_epoch)

    print("Training finished. Checkpoints saved at:", save_dir)

if __name__ == "__main__":
    main()

