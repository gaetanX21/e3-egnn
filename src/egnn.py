import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter

class EGCL(MessagePassing):
    """
    Equivariant Graph Convolution Layer (EGCL) for predicting noise in molecules.
    Inherited from the MessagePassing class in PyTorch Geometric.
    """
    def __init__(self, node_dim=NODE_FEATURES+1, aggr='add'):
        super().__init__(aggr=aggr)
        
        self.mlp_msg_h = nn.Sequential(
            nn.Linear(node_dim*2+1, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, node_dim),
            nn.SiLU()
        )

        self.mlp_attn_msg_h = nn.Sequential(
            nn.Linear(node_dim, 1),
            nn.Sigmoid()
        )

        self.mlp_upd_h = nn.Sequential(
            nn.Linear(node_dim*2, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, node_dim)
            # tempting to put a SiLU here, but it's not in the paper
        )

        self.mlp_msg_x = nn.Sequential(
            nn.Linear(node_dim*2+1, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, 1)
            # again, tempting to put a SiLU here, but it's not in the paper
        )

    def forward(self, h, x, edge_index):
        out = self.propagate(edge_index, h=h, x=x)
        return out
    
    def message(self, h_i, h_j, x_i, x_j):
        d_ij = torch.linalg.norm(x_j-x_i, dim=-1, keepdim=True)
        m_ij = self.mlp_msg_h(torch.cat([h_i, h_j, d_ij**2], dim=-1))
        attn = self.mlp_attn_msg_h(m_ij)
        msg_h = attn * m_ij
        msg_x = (x_i-x_j)/(d_ij+1) * self.mlp_msg_x(torch.cat([h_i, h_j, d_ij**2], dim=-1))
        return msg_h, msg_x
    
    def aggregate(self, inputs, index):
        inputs_h, inputs_x = inputs
        reduce = 'sum' if self.aggr=='add' else self.aggr # 'add' aggregation is called 'sum' in scatter
        agg_h = scatter(inputs_h, index, dim=self.node_dim, reduce=reduce)
        agg_x = scatter(inputs_x, index, dim=self.node_dim, reduce=reduce)
        return agg_h, agg_x
    
    def update(self, aggr_out, h, x):
        agg_h, agg_x = aggr_out
        out_h = self.mlp_upd_h(torch.cat([h, agg_h], dim=-1))
        out_x = x + agg_x
        return out_h, out_x


class EGNN(nn.Module):
    """
    Equivariant Graph Neural Network (EGNN) for predicting noise in molecules.
    eps_x, eps_h = EGNN(z_t^(x), z_t^(h), t/T) - [z_t^(x), 0, 0]
    Notice that we append time t to each node as a feature. (the 12th feature)
    """
    def __init__(self, node_dim=NODE_FEATURES+1, num_layers=4, aggr='add'):
        super().__init__()
        self.convs = nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(EGCL(node_dim=node_dim, aggr=aggr))

    def forward(self, data, t):
        num_nodes = data.h.shape[0]
        h = torch.cat([data.h, t*torch.ones(num_nodes, 1)], dim=-1)
        x = data.x
        for conv in self.convs:
            h_upd, x_upd = conv(h, x, data.edge_index)
            h = h + h_upd
            x = x_upd
        x_cog = x.mean(dim=0, keepdim=True)
        x = x - x_cog # remove center of gravity cf. Section 3.2 in paper
        return torch.cat([x, h], dim=-1)[:,:-1] # remove time feature from output
