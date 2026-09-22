from flax import nnx
import jax
import jax.numpy as jnp
from gnn_utiles import gcn_norm
import torch
import torch_geometric

        

class E_GCL_JAX(nnx.Module):
    def __init__(self, input_dim, 
                 hidden_dim, 
                 rngs: nnx.Rngs,
                 edge_attr_dim = 0,
                 act_fn = nnx.relu,
                 use_bias=True,
                 ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.rngs = rngs

        distance_dim = 1
        

        self.phi_edge = nnx.Sequential(
            nnx.Linear(in_features=input_dim *2 + # hi, hj
                                                distance_dim + # ||xi-xj||^2
                                                edge_attr_dim, # aji
                        out_features=hidden_dim,
                        kernel_init=jax.nn.initializers.glorot_normal(),
                        rngs=rngs, 
                        use_bias=use_bias,
                        ),
            act_fn,
            nnx.Linear(in_features=hidden_dim,
                        out_features=hidden_dim,
                        kernel_init=jax.nn.initializers.glorot_normal(),
                        rngs=rngs, 
                        use_bias=use_bias,
                        )

        )

        self.phi_x = nnx.Sequential(
            nnx.Linear(in_features=input_dim,
                              out_features=hidden_dim,
                              kernel_init=jax.nn.initializers.glorot_normal(),
                              rngs=rngs, 
                              use_bias=use_bias,
                              ),
            act_fn,
            nnx.Linear(in_features=hidden_dim,
                        out_features=hidden_dim,
                        kernel_init=jax.nn.initializers.glorot_normal(),
                        rngs=rngs, 
                        use_bias=use_bias,
                        )
        )

        self.phi_h_mlp =nnx.Sequential(
            nnx.Linear(in_features=input_dim,
                              out_features=hidden_dim,
                              kernel_init=jax.nn.initializers.glorot_normal(),
                              rngs=rngs, 
                              use_bias=use_bias,
                              ),
            act_fn,
            nnx.Linear(in_features=hidden_dim,
                        out_features=hidden_dim,
                        kernel_init=jax.nn.initializers.glorot_normal(),
                        rngs=rngs, 
                        use_bias=use_bias,
                        )
        )


        # self.lin = nnx.Linear(in_features=input_dim,
        #                       out_features=hidden_dim,
        #                       kernel_init=jax.nn.initializers.glorot_normal(),
        #                       rngs=rngs, 
        #                       use_bias=use_bias,
        #                       )

        
    def __call__(self, 
                 x: jax.Array, 
                 edge_index: jax.Array, 
                 edge_weight: jax.Array = None, 
                 rngs = None,
                 ):
        if self.normalize:
            edge_index, edge_weight = gcn_norm(edge_index, 
                                                edge_weight, 
                                                num_nodes=x.shape[0]
                                                )

        x = self.lin(x)

        out = self.propergate(x, edge_index, edge_weight)
        if self.bias:
            out = out + self.bias_custom[None]

        return out

    def propergate(self, 
                    x: jax.Array, 
                    edge_index: jax.Array, 
                    edge_weight: jax.Array = None, #shape [nume edge, 1]
                    ):
        src , target = edge_index
        x_j = x[src] # [num edge, num feature]
        m_ij = x_j if edge_weight is None else x_j * edge_weight.reshape(-1,1)
        sum_aggreate = jnp.zeros(x.shape, dtype=x.dtype).at[target].add(m_ij)

        return sum_aggreate