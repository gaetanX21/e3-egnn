import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU
from torch_geometric.utils import scatter


# Linear regression model (baseline)
class LinReg(nn.Module):
    """Baseline linear regression model for graph property prediction"""
    def __init__(self, atom_features=11, out_dim=1):
        super().__init__()
        self.lin = Linear(atom_features+3, out_dim)
        self.pool = global_mean_pool

    def forward(self, data):       
        inputs = torch.cat([data.x, data.pos], dim=-1)
        inputs = self.lin(inputs)
        out = self.pool(inputs, data.batch) # (n, d) -> (batch_size, d) [because PyG concatenates all the graphs into one big graph]
        return out.view(-1)


# Message Passing Neural Network (MPNN)
class MPNNLayer(MessagePassing):
    def __init__(self, emb_dim, edge_dim, aggr='add'):
        """Message Passing Neural Network Layer.

        Args:
            emb_dim: (int) - hidden dimension `d` to embed node features
            edge_dim: (int) - edge feature dimension `d_e`
            aggr: (str) - aggregation function `\oplus` (sum/mean/max)
        """
        super().__init__(aggr=aggr)

        self.emb_dim = emb_dim
        self.edge_dim = edge_dim

        self.mlp_msg = Sequential(
            Linear(2*emb_dim + edge_dim, emb_dim), ReLU(),
            BatchNorm1d(emb_dim), Linear(emb_dim, emb_dim), ReLU(),
          )
        
        self.mlp_upd = Sequential(
            Linear(2*emb_dim, emb_dim), ReLU(),
            BatchNorm1d(emb_dim), Linear(emb_dim, emb_dim), ReLU(),
          )

    def forward(self, h, edge_index, edge_attr):
        out = self.propagate(edge_index, h=h, edge_attr=edge_attr)
        return out

    def message(self, h_i, h_j, edge_attr):
        msg = torch.cat([h_i, h_j, edge_attr], dim=-1)
        return self.mlp_msg(msg)
    
    def aggregate(self, inputs, index):
        return scatter(inputs, index, dim=self.node_dim, reduce=self.aggr)
    
    def update(self, aggr_out, h):
        upd_out = torch.cat([h, aggr_out], dim=-1)
        out_h = self.mlp_upd(upd_out)
        return out_h
    
    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(emb_dim={self.emb_dim}, aggr={self.aggr})')


class MPNN(nn.Module):
    def __init__(self, num_layers=4, emb_dim=64, atom_features=11, edge_dim=4, out_dim=1):
        super().__init__()
        self.lin_in = Linear(atom_features+3, emb_dim)   
        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(MPNNLayer(emb_dim, edge_dim, aggr='add'))
        self.pool = global_mean_pool
        self.lin_pred = Linear(emb_dim, out_dim)
        
    def forward(self, data):
        # we directly append pos to the node features
        h = torch.cat([data.x, data.pos], dim=-1)
        h = self.lin_in(h)
        for conv in self.convs:
            h = h + conv(h, data.edge_index, data.edge_attr) # (n, d) -> (n, d)
        h_graph = self.pool(h, data.batch) # (n, d) -> (batch_size, d)
        out = self.lin_pred(h_graph) # (batch_size, d) -> (batch_size, 1)
        return out.view(-1)


# E(3) Equivariant Graph Neural Network (EGNN) (without edge features)
class E3EGCL(MessagePassing):
    """
    E(3) Equivariant Graph Convolution Layer (EGCL)
    """
    def __init__(self, atom_features, aggr='add'):
        super().__init__(aggr=aggr)
        
        self.mlp_msg_h = nn.Sequential(
            nn.Linear(atom_features*2+1, atom_features),
            nn.SiLU(),
            nn.Linear(atom_features, atom_features),
            nn.SiLU()
        )

        self.mlp_attn_msg_h = nn.Sequential(
            nn.Linear(atom_features, 1),
            nn.Sigmoid()
        )

        self.mlp_upd_h = nn.Sequential(
            nn.Linear(atom_features*2, atom_features),
            nn.SiLU(),
            nn.Linear(atom_features, atom_features)
        )

        self.mlp_msg_x = nn.Sequential(
            nn.Linear(atom_features*2+1, atom_features),
            nn.SiLU(),
            nn.Linear(atom_features, atom_features),
            nn.SiLU(),
            nn.Linear(atom_features, 1)
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


class E3EGNN(nn.Module):
    """
    E(3) Equivariant Graph Neural Network (EGNN).
    """
    def __init__(self, atom_features=11, num_layers=4, aggr='add'):
        super().__init__()
        self.convs = nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(E3EGCL(atom_features=atom_features, aggr=aggr))
        self.pool = global_mean_pool
        self.lin_pred = nn.Linear(atom_features, 1)

    def forward(self, data):
        h = data.x
        x = data.pos
        for conv in self.convs:
            h_upd, x_upd = conv(h, x, data.edge_index)
            h = h + h_upd
            x = x_upd
        x_cog = x.mean(dim=0, keepdim=True)
        x = x - x_cog # remove center of gravity cf. Section 3.2 in paper
        h_graph = self.pool(h, data.batch) # (n, d) -> (batch_size, d)
        out = self.lin_pred(h_graph)
        return out.view(-1)
    

# E(3) Equivariant Graph Neural Network (EGNN) (with edge features)
class E3EGCL_edge(MessagePassing):
    """
    E(3) Equivariant Graph Convolution Layer (EGCL) + WITH EDGE FEATURES
    """
    def __init__(self, atom_features, edge_dim, aggr='add'):
        super().__init__(aggr=aggr)
        
        self.mlp_msg_h = nn.Sequential(
            nn.Linear(atom_features*2+1+edge_dim, atom_features),
            nn.SiLU(),
            nn.Linear(atom_features, atom_features),
            nn.SiLU()
        )

        self.mlp_attn_msg_h = nn.Sequential(
            nn.Linear(atom_features, 1),
            nn.Sigmoid()
        )

        self.mlp_upd_h = nn.Sequential(
            nn.Linear(atom_features*2, atom_features),
            nn.SiLU(),
            nn.Linear(atom_features, atom_features)
        )

        self.mlp_msg_x = nn.Sequential(
            nn.Linear(atom_features*2+1+edge_dim, atom_features),
            nn.SiLU(),
            nn.Linear(atom_features, atom_features),
            nn.SiLU(),
            nn.Linear(atom_features, 1)
        )

    def forward(self, h, x, edge_index, edge_attr):
        out = self.propagate(edge_index, h=h, x=x, edge_attr=edge_attr)
        return out
    
    def message(self, h_i, h_j, x_i, x_j, edge_attr):
        d_ij = torch.linalg.norm(x_j-x_i, dim=-1, keepdim=True)
        m_ij = self.mlp_msg_h(torch.cat([h_i, h_j, d_ij**2, edge_attr], dim=-1))
        attn = self.mlp_attn_msg_h(m_ij)
        msg_h = attn * m_ij
        msg_x = (x_i-x_j)/(d_ij+1) * self.mlp_msg_x(torch.cat([h_i, h_j, d_ij**2, edge_attr], dim=-1))
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


class E3EGNN_edge(nn.Module):
    """
    E(3) Equivariant Graph Neural Network (EGNN) + WITH EDGE FEATURES
    """
    def __init__(self, atom_features=11, edge_dim=4, num_layers=4, aggr='add'):
        super().__init__()
        self.convs = nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(E3EGCL_edge(atom_features=atom_features, aggr=aggr, edge_dim=edge_dim))
        self.pool = global_mean_pool
        self.lin_pred = nn.Linear(atom_features, 1)

    def forward(self, data):
        h = data.x
        x = data.pos
        for conv in self.convs:
            h_upd, x_upd = conv(h, x, data.edge_index, data.edge_attr)
            h = h + h_upd
            x = x_upd
        x_cog = x.mean(dim=0, keepdim=True)
        x = x - x_cog # remove center of gravity cf. Section 3.2 in paper
        h_graph = self.pool(h, data.batch) # (n, d) -> (batch_size, d)
        out = self.lin_pred(h_graph)
        return out.view(-1)