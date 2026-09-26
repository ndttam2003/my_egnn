from flax import nnx
import jax
import jax.numpy as jnp
import torch
import torch_geometric

        

class E_GCL_JAX(nnx.Module):
    def __init__(self, input_dim, 
                 hidden_dim, 
                 output_dim,
                 rngs: nnx.Rngs,
                 edge_attr_dim = 0,
                 act_fn = nnx.relu,
                 attention = False,
                 recurrent = True,
                 use_bias=True,
                 ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.rngs = rngs
        self.recurrent = recurrent

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
            nnx.Linear(in_features=hidden_dim,
                              out_features=hidden_dim,
                              kernel_init=jax.nn.initializers.glorot_uniform(),
                              rngs=rngs, 
                              use_bias=use_bias,
                              ),
            act_fn,
            nnx.Linear(in_features=hidden_dim,
                        out_features=1,
                        kernel_init=jax.nn.initializers.glorot_normal(),
                        rngs=rngs, 
                        use_bias=False,
                        )
        )

        self.phi_h_mlp =nnx.Sequential(
            nnx.Linear(in_features=input_dim + # h_i
                                    hidden_dim, # m_i
                              out_features=hidden_dim,
                              kernel_init=jax.nn.initializers.glorot_normal(),
                              rngs=rngs, 
                              use_bias=use_bias,
                              ),
            act_fn,
            nnx.Linear(in_features=hidden_dim,
                        out_features=output_dim,
                        kernel_init=jax.nn.initializers.glorot_normal(),
                        rngs=rngs, 
                        use_bias=use_bias,
                        )
        )


        self.attention = attention
        if self.attention:
            self.att_mlp = nnx.Sequential(
                nnx.Linear(in_features=hidden_dim, 
                                out_features=1,
                                kernel_init=jax.nn.initializers.glorot_normal(),
                                rngs=rngs, 
                                use_bias=use_bias,
                                ),
                nnx.sigmoid,

            )
        # self.lin = nnx.Linear(in_features=input_dim,
        #                       out_features=hidden_dim,
        #                       kernel_init=jax.nn.initializers.glorot_normal(),
        #                       rngs=rngs, 
        #                       use_bias=use_bias,
        #                       )

        
    def __call__(self, 
                 hidden_state: jax.Array, 
                 coordinate: jax.Array,
                 relative_coordinate: jax.Array,
                 edge_index: jax.Array, 
                 edge_attr: jax.Array = None, 
                 batch =None,
                 rngs = None,
                 ):

        distances = jnp.linalg.vector_norm(relative_coordinate, keepdims=True, axis=-1)
        
        if batch is None:
            num_nodes = hidden_state.shape[0]
            C = 1 / (num_nodes - 1)
        else:
            num_nodes = jnp.zeros((batch[-1] + 1,), dtype=int).at[batch].add(jnp.ones_like(batch)).reshape(-1,1)
            C = 1 / (num_nodes[batch] - 1)

        messages = self.propergate_edge(hidden_state, distances, edge_index, edge_attr)
        new_coord = self.update_coord(coordinate, relative_coordinate, messages, edge_index, C)

        aggreate = jnp.zeros((hidden_state.shape[0], messages.shape[1]), 
                             dtype=hidden_state.dtype
                             ).at[edge_index[1]].add(messages)

        new_hidden_state = self.phi_h_mlp(
            jnp.concat([hidden_state, aggreate], axis=1)
        )

        if self.recurrent:
            new_hidden_state = new_hidden_state + hidden_state
        return new_hidden_state, new_coord, aggreate

    def propergate_edge(self, 
                    hidden_state: jax.Array, 
                    distances: jax.Array,
                    edge_index: jax.Array, 
                    edge_attr: jax.Array = None, #shape [nume edge, 1]
                    ):
        src , target = edge_index

        h_i = hidden_state[target]
        h_j = hidden_state[src]

        if edge_attr is None:
            out = jnp.concat([h_i, h_j, distances], axis=1)
        else:
            out = jnp.concat([h_i, h_j, distances, edge_attr], axis=1)

        out = self.phi_edge(out)

        if self.attention:
            out = self.att_mlp(out) * out

        return out


        # x_j = x[src] # [num edge, num feature]
        # m_ij = x_j if edge_weight is None else x_j * edge_weight.reshape(-1,1)
        # sum_aggreate = jnp.zeros(x.shape, dtype=x.dtype).at[target].add(m_ij)

        # return sum_aggreate

    def update_coord(self,coordinate, relative_coordinate, messages, edge_index, C):
        src, target = edge_index
        move = jnp.zeros(coordinate.shape, dtype=coordinate.dtype)
        scale_relative = relative_coordinate * self.phi_x(messages)
        move = C * move.at[target].add(scale_relative)

        return coordinate + move

if __name__ == "__main__":
    from qm9_dataloader import get_train_val_test
    train, _, _ = get_train_val_test()
    sample = next(iter(train))

    print(sample)

    hidden_state: jax.Array = jnp.array(sample.x)
    coordinate: jax.Array = jnp.array(sample.pos)
    relative_coordinate: jax.Array = jnp.array(sample.relatix_matrix)
    edge_index: jax.Array= jnp.array(sample.edge_index) 
    edge_attr: jax.Array = jnp.array(sample.edge_attr) 
    batch = jnp.array(sample.batch)

    edge_attr = None
    model = E_GCL_JAX(
        input_dim=11,
        hidden_dim=16,
        output_dim=11,
        rngs=nnx.Rngs(0),
        edge_attr_dim=edge_attr.shape[1] if edge_attr is not None else 0,
    )

    new_hidden_state, aggreate = model(hidden_state, 
                     coordinate,
                     relative_coordinate,
                     edge_index, 
                     edge_attr =None, 
                     batch =batch)

    print(new_hidden_state)
    print(new_hidden_state.shape)
    print(aggreate)
    print(aggreate.shape)



