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
    data = next(iter(dataloader))[0]
    out_1 = module(data)
    perm = torch.randperm(data.x.shape[0])
    data = permute_graph(data, perm) # permuted copy
    out_2 = module(data)
    # Check whether output varies after applying transformations
    return torch.allclose(out_1, out_2, atol=1e-04)


def permutation_equivariance_unit_test(module, dataloader, use_edge_attr):
    """Unit test for checking whether a module (GNN layer) is 
    permutation equivariant.
    """
    data = next(iter(dataloader))[0]

    if use_edge_attr:
        out_h_1, out_pos_1 = module(data.x, data.pos, data.edge_index, data.edge_attr)
    else:
        out_h_1, out_pos_1 = module(data.x, data.pos, data.edge_index)
    
    # Permute the node attribute ordering
    perm = torch.randperm(data.x.shape[0])
    data = permute_graph(data, perm) # permuted copy

    if use_edge_attr:
        out_h_2, out_pos_2 = module(data.x, data.pos, data.edge_index, data.edge_attr)
    else:
        out_h_2, out_pos_2 = module(data.x, data.pos, data.edge_index)

    # Check whether output varies after applying transformations
    return torch.allclose(out_h_1[perm], out_h_2, atol=1e-4) and torch.allclose(out_pos_1[perm], out_pos_2, atol=1e-4)


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
    data = next(iter(dataloader))[0]

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


def rot_trans_equivariance_unit_test(module, dataloader, use_edge_attr):
    """Unit test for checking whether a module (GNN layer) is 
    rotation and translation equivariant.
    """
    data = next(iter(dataloader))[0]

    if use_edge_attr:
        out_1, pos_1 = module(data.x, data.pos, data.edge_index, data.edge_attr)
    else:
        out_1, pos_1 = module(data.x, data.pos, data.edge_index)

    Q = random_orthogonal_matrix(dim=3)
    t = torch.rand(3)
    data.pos = torch.matmul(data.pos, Q) + t


    # Forward pass on rotated + translated example
    if use_edge_attr:
        out_2, pos_2 = module(data.x, data.pos, data.edge_index, data.edge_attr)
    else:
        out_2, pos_2 = module(data.x, data.pos, data.edge_index)
    
    # Check whether output varies after applying transformations
    pos_1 = torch.matmul(pos_1, Q) + t
    return torch.allclose(out_1, out_2, atol=1e-4) and torch.allclose(pos_1, pos_2, atol=1e-4)


def full_test(model, dataloader, use_edge_attr):
    """Run all tests on a model (GNN model) and looks at its layer properties too.
    """
    # Create a dummy dataset
    data = next(iter(dataloader))[0]

    layer = model.convs[0]

    # test the layers
    layer_perm_equiv = permutation_equivariance_unit_test(layer, dataloader, use_edge_attr)
    layer_rot_trans_equiv = rot_trans_equivariance_unit_test(layer, dataloader, use_edge_attr)
    print(f"Layer permutation equivariance: {layer_perm_equiv}")
    print(f"Layer rotation and translation equivariance: {layer_rot_trans_equiv}")

    # test the model
    model_perm_invar = permutation_invariance_unit_test(model, dataloader)
    model_rot_trans_invar = rot_trans_invariance_unit_test(model, dataloader)
    print(f"Model permutation invariance: {model_perm_invar}")
    print(f"Model rotation and translation invariance: {model_rot_trans_invar}")