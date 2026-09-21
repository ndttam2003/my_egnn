import jax
import jax.numpy as jnp

def maybe_num_nodes(edge_index: jax.Array, num_nodes = None):
    if num_nodes is not None:
        return num_nodes
    else:
        return edge_index.max() + 1

def add_remaining_self_loops_jax(
        edge_index : jax.Array, 
        edge_attr: jax.Array = None, 
        fill_value =1.0, 
        num_nodes = None
    ):
    """
    Example:
        >>> edge_index = torch.tensor([[0, 1],
        ...                            [1, 0]])
        >>> edge_weight = torch.tensor([0.5, 0.5])
        >>> add_remaining_self_loops(edge_index, edge_weight)
        (tensor([[0, 1, 0, 1],
                [1, 0, 0, 1]]),
        tensor([0.5000, 0.5000, 1.0000, 1.0000]))
    """

    N = maybe_num_nodes(edge_index, num_nodes)

    src = edge_index[0]
    tar = edge_index[1]
    # print("src = ", src)
    # print(tar)

    mask = src != tar

    # print("mask = ", mask)
    # print("src[(mask)] = ", src[(mask)])

    loop_index = jnp.arange(0, N).reshape((1, -1))
    loop_index = jnp.repeat(loop_index, repeats=2, axis=0)
    if edge_attr is not None:
        loop_attr = jnp.full((N,) + edge_attr.shape[1:], fill_value)
        # print(loop_attr)
        inv_mask = (~mask)
        loop_attr = loop_attr.at[src[inv_mask]].set(edge_attr[inv_mask])

        edge_attr = jnp.concat([edge_attr[mask], loop_attr], axis=0)

    edge_index = jnp.concat([edge_index[:, mask], loop_index], axis=1)

    return edge_index, edge_attr

if __name__ == "__main__":
    edge_index = jnp.array([[0,0,1],
                            [0,1,0]])
    edge_weight = jnp.array([0.5, 0.5, 0.5])
    print(add_remaining_self_loops_jax(edge_index, edge_weight))
    edge_weight = jnp.array([[0.5, 0.5],[0.5, 0.5], [0.5, 0.5]])
    print(add_remaining_self_loops_jax(edge_index, edge_weight))




def gcn_norm(
    edge_index : jax.Array,
    edge_weight: jax.Array = None,
    num_nodes: int = None,
    improved: bool = False,
    add_self_loops: bool = True,
    ):
    fill_value = 2.0 if improved else 1.0
    num_nodes = maybe_num_nodes(edge_index=edge_index, num_nodes=num_nodes)

    if add_self_loops:
        edge_index, edge_weight = add_remaining_self_loops_jax(
            edge_index, edge_weight, fill_value, num_nodes
        )

    if edge_weight is None:
        edge_weight = jax.numpy.ones((edge_index.shape[1],))

    row, col = edge_index[0], edge_index[1]
    deg = jnp.zeros(num_nodes)
    idx = col
    # print("edge_weight = ", edge_weight)
    deg = deg.at[idx].add(edge_weight)
    # print("deg = ", deg)
    # inv_deg = jnp.power(deg, -0.5)
    
    # inv_deg = jnp.where(inv_deg != float("inf"), inv_deg, 0)
    inv_deg = jnp.where(deg > 0, jax.lax.rsqrt(deg), 0)
    # print("inv_deg = ", inv_deg)
    d_i = inv_deg[row]
    d_j = inv_deg[col]

    edge_weight = d_i * edge_weight * d_j

    return edge_index, edge_weight


if __name__ == "__main__":
    print(f"=============== gcn_norm ===========")
    edge_index = jnp.array([[0,1],
                            [1,0]])
    edge_weight = jnp.array([0.5, 0.5])

    print(gcn_norm(edge_index=edge_index, edge_weight=edge_weight))

    import torch

    import torch_geometric
    from torch_geometric.nn.conv import gcn_conv
    print(print(gcn_conv.gcn_norm(torch.tensor(edge_index),
                                  edge_weight=torch.tensor(edge_weight)

                                  )))




    