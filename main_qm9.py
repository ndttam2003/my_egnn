import jax
import torch
import torch_geometric
from flax import nnx
import jax.numpy as jnp
from qm9_dataloader import get_train_val_test
from .models.gcl_jax import E_GCL_JAX

class EGNN(nnx.Module):
    def __init__(self,
                input_dim, 
                hidden_dim, 
                output_dim,
                rngs: nnx.Rngs,
                edge_attr_dim = 0,
                act_fn = nnx.silu,
                n_layers=4,
                attention = False,
                recurrent = True,
                use_bias=True,

                 ):

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers

        self.embedding = nnx.Linear(
            in_features=input_dim,
            out_features=output_dim,
            rngs=rngs,
        )

        


