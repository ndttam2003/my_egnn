import jax
import torch_geometric
from flax import nnx
import optax
import jax.numpy as jnp
from qm9_dataloader import get_train_val_test
from models.gcl_jax import E_GCL_JAX
from dotenv import load_dotenv
import os
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


# if __name__ == "__main__":
#     from torch_geometric.utils.smiles import from_smiles
#     from qm9_dataloader import remove_self_loop, transfrom_data
#     sample = torch_geometric.datasets.QM9(root='data/QM9', pre_transform=remove_self_loop)
#     sample = [transfrom_data(i) for i in sample[:10]]
#     from torch_geometric.loader import DataLoader
#     sample = next(iter(DataLoader(sample, batch_size = 2)))

#     print(sample)
#     print(sample.name)

#     hidden_state: jax.Array = jnp.array(sample.x)
#     coordinate: jax.Array = jnp.array(sample.pos)
#     relative_coordinate: jax.Array = jnp.array(sample.relatix_matrix)
#     edge_index: jax.Array= jnp.array(sample.edge_index) 
#     edge_attr: jax.Array = jnp.array(sample.edge_attr) 
#     batch = jnp.array(sample.batch)

#     edge_attr = None
#     model = EGNN(
#         input_dim=11,
#         hidden_dim=16,
#         rngs=nnx.Rngs(0),
#         edge_attr_dim=edge_attr.shape[1] if edge_attr is not None else 0,
#     )

#     # nnx.display(model)

#     pred, graph_embedding = model(hidden_state, 
#                      coordinate,
#                      relative_coordinate,
#                      edge_index, 
#                      edge_attr =edge_attr, 
#                      batch =batch)

#     print("pred = ", pred)
#     print(pred.shape)
#     print(graph_embedding)
#     print(graph_embedding.shape)

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

def eval_step(model, metrics: nnx.MultiMetric, batch):
  loss = loss_fn(model, batch)
  metrics.update(loss=loss)  # In-place updates.


def main():

    model = EGNN(
            input_dim=11,
            hidden_dim=16,
            rngs=nnx.Rngs(0),
            edge_attr_dim=4,
        )

    learning_rate = int(os.getenv("LR"))

    optimizer = nnx.Optimizer(
        model, optax.adamw(learning_rate = learning_rate), wrt=nnx.Param
    )
    metrics = nnx.MultiMetric(
        mae=MeanAbsoluteError(),
        loss=nnx.metrics.Average('loss'),
    )

    # nnx.display(optimizer)

    


        


