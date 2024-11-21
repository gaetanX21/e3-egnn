from torch_geometric.data import Data
from torch_geometric.data import Batch
import torch
import numpy as np
from scipy.stats import ortho_group
from torch_geometric.nn import MessagePassing


def permute_graph(data, perm):
    """Helper function for permuting PyG Data object attributes consistently.
    /!\ j'ai passé une heure dessus mais je suis sûr qu'elle fonctionne!
    """
    # Permute the node attribute ordering
    data = data.clone()

    invperm = torch.argsort(perm)

    data.x = data.x[perm]
    data.pos = data.pos[perm]
    # # Permute the edge index
    data.edge_index = torch.stack([invperm[data.edge_index[0]], invperm[data.edge_index[1]]])
    # no need to permute edge_attr!

    return data


def permutation_invariance_unit_test(module, dataloader):
    """Unit test for checking whether a module (GNN model) is 
    permutation invariant.
    """
    data = next(iter(dataloader))
    out_1 = module(data)
    perm = torch.randperm(data.x.shape[0])
    data = permute_graph(data, perm) # permuted copy
    out_2 = module(data)
    # Check whether output varies after applying transformations
    return torch.allclose(out_1, out_2, atol=1e-04)


def permutation_equivariance_unit_test(module, dataloader):
    """Unit test for checking whether a module (GNN layer) is 
    permutation equivariant.
    """
    data = next(iter(dataloader))
    out_1 = module(data.x, data.edge_index, data.edge_attr)
    perm = torch.randperm(data.x.shape[0])
    data = permute_graph(data, perm) # permuted copy
    out_2 = module(data.x, data.edge_index, data.edge_attr)
    # Check whether output varies after applying transformations
    return torch.allclose(out_1[perm], out_2, atol=1e-4)


def random_orthogonal_matrix(dim=3):
  """
  Helper function to sample a random orthogonal matrix of shape (dim, dim)
  """
  Q = torch.tensor(ortho_group.rvs(dim=dim)).float()
  return Q


def rot_trans_invariance_unit_test(module, dataloader):
    """Unit test for checking whether a module (GNN model/layer) is 
    rotation and translation invariant.
    """
    data = next(iter(dataloader))

    # Forward pass on original example
    # Note: We have written a conditional forward pass so that the same unit
    #       test can be used for both the GNN model as well as the layer.
    #       The functionality for layers will be useful subsequently. 
    if isinstance(module, MessagePassing): # layer
       out_1 = module(data.x, data.pos, data.edge_index, data.edge_attr)
    else: # model
        out_1 = module(data)
        

    Q = random_orthogonal_matrix(dim=3)
    t = torch.rand(3)
    data.pos = torch.matmul(data.pos, Q) + t

    # Forward pass on rotated + translated example
    if isinstance(module, MessagePassing):
        out_2 = module(data.x, data.pos, data.edge_index, data.edge_attr)
    else:
        out_2 = module(data)

    # Check whether output varies after applying transformations
    return torch.allclose(out_1, out_2, atol=1e-4)


def rot_trans_equivariance_unit_test(module, dataloader):
    """Unit test for checking whether a module (GNN layer) is 
    rotation and translation equivariant.
    """
    data = next(iter(dataloader))

    out_1, pos_1 = module(data.x, data.pos, data.edge_index, data.edge_attr)

    Q = random_orthogonal_matrix(dim=3)
    t = torch.rand(3)
    data.pos = torch.matmul(data.pos, Q) + t


    # Forward pass on rotated + translated example
    out_2, pos_2 = module(data.x, data.pos, data.edge_index, data.edge_attr)
    
    # Check whether output varies after applying transformations
    pos_1 = torch.matmul(pos_1, Q) + t
    return torch.allclose(out_1, out_2, atol=1e-4) and torch.allclose(pos_1, pos_2, atol=1e-4)