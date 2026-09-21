from flax import nnx
import jax
import jax.numpy as jnp
from gnn_utiles import gcn_norm
import torch
import torch_geometric

        

class GCNLayer(nnx.Module):
    def __init__(self, input_num_feature, 
                 output_num_feature, 
                 rngs: nnx.Rngs,
                 normalize: bool = True,
                 bias: bool = True,
                 ):
        self.input_num_feature = input_num_feature
        self.output_num_feature = output_num_feature
        self.rngs = rngs
        self.normalize = normalize

        self.lin = nnx.Linear(in_features=input_num_feature,
                              out_features=output_num_feature,
                              kernel_init=jax.nn.initializers.glorot_normal(),
                              rngs=rngs, 
                              use_bias=False,
                              )

        self.bias = bias
        if bias:
            self.bias_custom = nnx.Param(jnp.zeros((output_num_feature,), dtype=self.lin.kernel.dtype))

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

if __name__ == "__main__":
    from torch_geometric.utils import from_smiles

    data = from_smiles("CCO", with_hydrogen=True)
    print(data)

    jax_gcn = GCNLayer(9, 3, nnx.Rngs(0))

    print(jax_gcn(jnp.array(data.x), jnp.array(data.edge_index)))
    print(jax_gcn.lin.kernel)

    print("=========== torch =======")
    torch_gcn  = torch_geometric.nn.GCNConv(
        9,
        3,
    )
    print(torch_gcn.lin.weight)

    with torch.no_grad():
        torch_gcn.lin.weight.copy_(torch.tensor(jax_gcn.lin.kernel).T)

    print("After copy")
    
    print(torch_gcn.lin.weight)

    print(torch_gcn(data.x.float(), data.edge_index))

    print("===== Compare =======")

    print(jax_gcn(jnp.array(data.x), jnp.array(data.edge_index)) == jnp.array(torch_gcn(data.x.float(), data.edge_index).detach()))
