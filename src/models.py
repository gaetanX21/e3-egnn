import torch
from torch_geometric.nn import radius_graph



class E3EquivariantGNN(torch.nn.Module):
    pass
    # def __init__(self, input_irreps, hidden_irreps, output_irreps):
    #     super().__init__()
    #     self.layer1 = FullyConnectedTensorProduct(input_irreps, hidden_irreps, hidden_irreps)
    #     self.layer2 = FullyConnectedTensorProduct(hidden_irreps, hidden_irreps, output_irreps)
    #     self.fc = torch.nn.Linear(output_irreps.dim, 1)  # Binary classification

    # def forward(self, data):
    #     # Node features and positions
    #     x = data.x  # Node features
    #     pos = data.pos  # Node positions

    #     # Graph connectivity
    #     edge_index = radius_graph(pos, r=10.0)  # Use a cutoff radius (10 is arbitrary)

    #     # Equivariant message passing
    #     x = self.layer1(x, edge_index, pos)
    #     x = torch.nn.functional.relu(x)
    #     x = self.layer2(x, edge_index, pos)

    #     # Classification output
    #     out = self.fc(x.mean(dim=0))  # Global pooling for classification
    #     return torch.sigmoid(out)
    


class SE3EquivariantGNN(torch.nn.Module):
    pass
    # def __init__(self, input_irreps, hidden_irreps, output_irreps):
    #     super().__init__()
    #     self.layer1 = FullyConnectedTensorProduct(input_irreps, hidden_irreps, hidden_irreps, irreps_out="1e")
    #     self.layer2 = FullyConnectedTensorProduct(hidden_irreps, hidden_irreps, output_irreps, irreps_out="1e")
    #     self.fc = torch.nn.Linear(output_irreps.dim, 1)  # Binary classification

    # def forward(self, data):
    #     # Node features and positions
    #     x = data.x  # Node features
    #     pos = data.pos  # Node positions

    #     # Graph connectivity
    #     edge_index = radius_graph(pos, r=10.0)  # Use a cutoff radius

    #     # Equivariant message passing
    #     x = self.layer1(x, edge_index, pos)
    #     x = torch.nn.functional.relu(x)
    #     x = self.layer2(x, edge_index, pos)

    #     # Classification output
    #     out = self.fc(x.mean(dim=0))  # Global pooling for classification
    #     return torch.sigmoid(out)