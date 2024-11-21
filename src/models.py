import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU
from torch_geometric.utils import scatter


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
    

class MLP(nn.Module):
    """Baseline MLP model for graph property prediction"""
    def __init__(self, atom_features=11, hidden_dim=64, out_dim=1):
        super().__init__()
        self.lin_in = Linear(atom_features+3, hidden_dim)
        self.mlp = Sequential(
            Linear(hidden_dim, hidden_dim), ReLU(),
            Linear(hidden_dim, hidden_dim), ReLU(),
            Linear(hidden_dim, hidden_dim), ReLU(),
        )
        self.lin_out = Linear(hidden_dim, out_dim)
        self.pool = global_mean_pool

    def forward(self, data):
        inputs = torch.cat([data.x, data.pos], dim=-1)
        inputs = self.lin_in(inputs)
        inputs = self.mlp(inputs)
        inputs_graph = self.pool(inputs, data.batch)
        out = self.lin_out(inputs_graph)
        return out.view(-1)


class MPNNLayer(MessagePassing):
    def __init__(self, emb_dim=64, edge_dim=4, aggr='add'):
        """Message Passing Neural Network Layer.
        Doesn't use the atom positions yet.

        Args:
            emb_dim: (int) - hidden dimension `d` to embed node features
            edge_dim: (int) - edge feature dimension `d_e`
            aggr: (str) - aggregation function `\oplus` (sum/mean/max)
        """
        # Set the aggregation function
        super().__init__(aggr=aggr)

        self.emb_dim = emb_dim
        self.edge_dim = edge_dim

        # MLP `\psi` for computing messages `m_ij`
        # Implemented as a stack of Linear->BN->ReLU->Linear->BN->ReLU
        # dims: (2d + d_e) -> d
        self.mlp_msg = Sequential(
            Linear(2*emb_dim + edge_dim, emb_dim), ReLU(),
            BatchNorm1d(emb_dim), Linear(emb_dim, emb_dim), ReLU(),
          )
        
        # MLP `\phi` for computing updated node features `h_i^{l+1}`
        # Implemented as a stack of Linear->BN->ReLU->Linear->BN->ReLU
        # dims: 2d -> d
        self.mlp_upd = Sequential(
            Linear(2*emb_dim, emb_dim), ReLU(),
            BatchNorm1d(emb_dim), Linear(emb_dim, emb_dim), ReLU(),
          )

    def forward(self, h, edge_index, edge_attr):
        """
        The forward pass updates node features `h` via one round of message passing.

        As our MPNNLayer class inherits from the PyG MessagePassing parent class,
        we simply need to call the `propagate()` function which starts the 
        message passing procedure: `message()` -> `aggregate()` -> `update()`.
        
        The MessagePassing class handles most of the logic for the implementation.
        To build custom GNNs, we only need to define our own `message()`, 
        `aggregate()`, and `update()` functions (defined subsequently).

        Args:
            h: (n, d) - initial node features
            edge_index: (e, 2) - pairs of edges (i, j)
            edge_attr: (e, d_e) - edge features

        Returns:
            out: (n, d) - updated node features
        """
        out = self.propagate(edge_index, h=h, edge_attr=edge_attr)
        return out

    def message(self, h_i, h_j, edge_attr):
        """Step (1) Message

        The `message()` function constructs messages from source nodes j 
        to destination nodes i for each edge (i, j) in `edge_index`.

        The arguments can be a bit tricky to understand: `message()` can take 
        any arguments that were initially passed to `propagate`. Additionally, 
        we can differentiate destination nodes and source nodes by appending 
        `_i` or `_j` to the variable name, e.g. for the node features `h`, we
        can use `h_i` and `h_j`. 
        
        This part is critical to understand as the `message()` function
        constructs messages for each edge in the graph. The indexing of the
        original node features `h` (or other node variables) is handled under
        the hood by PyG.

        Args:
            h_i: (e, d) - destination node features
            h_j: (e, d) - source node features
            edge_attr: (e, d_e) - edge features
        
        Returns:
            msg: (e, d) - messages `m_ij` passed through MLP `\psi`
        """
        msg = torch.cat([h_i, h_j, edge_attr], dim=-1)
        return self.mlp_msg(msg)
    
    def aggregate(self, inputs, index):
        """Step (2) Aggregate

        The `aggregate` function aggregates the messages from neighboring nodes,
        according to the chosen aggregation function ('sum' by default).

        Args:
            inputs: (e, d) - messages `m_ij` from destination to source nodes
            index: (e, 1) - list of source nodes for each edge/message in `input`

        Returns:
            aggr_out: (n, d) - aggregated messages `m_i`
        """
        return scatter(inputs, index, dim=self.node_dim, reduce=self.aggr)
    
    def update(self, aggr_out, h):
        """
        Step (3) Update

        The `update()` function computes the final node features by combining the 
        aggregated messages with the initial node features.

        `update()` takes the first argument `aggr_out`, the result of `aggregate()`, 
        as well as any optional arguments that were initially passed to 
        `propagate()`. E.g. in this case, we additionally pass `h`.

        Args:
            aggr_out: (n, d) - aggregated messages `m_i`
            h: (n, d) - initial node features

        Returns:
            upd_out: (n, d) - updated node features passed through MLP `\phi`
        """
        upd_out = torch.cat([h, aggr_out], dim=-1)
        return self.mlp_upd(upd_out)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(emb_dim={self.emb_dim}, aggr={self.aggr})')


class MPNN(nn.Module):
    def __init__(self, num_layers=4, emb_dim=64, in_dim=11, edge_dim=4, out_dim=1):
        """Message Passing Neural Network model for graph property prediction

        Args:
            num_layers: (int) - number of message passing layers `L`
            emb_dim: (int) - hidden dimension `d`
            in_dim: (int) - initial node feature dimension `d_n`
            edge_dim: (int) - edge feature dimension `d_e`
            out_dim: (int) - output dimension (fixed to 1)
        """
        super().__init__()
        
        # Linear projection for initial node features
        # dim: d_n -> d
        self.lin_in = Linear(in_dim, emb_dim)
        
        # Stack of MPNN layers
        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(MPNNLayer(emb_dim, edge_dim, aggr='add'))
        
        # Global pooling/readout function `R` (mean pooling)
        # PyG handles the underlying logic via `global_mean_pool()`
        self.pool = global_mean_pool

        # Linear prediction head
        # dim: d -> out_dim
        self.lin_pred = Linear(emb_dim, out_dim)
        
    def forward(self, data):
        """
        Args:
            data: (PyG.Data) - batch of PyG graphs

        Returns: 
            out: (batch_size, out_dim) - prediction for each graph
        """
        h = self.lin_in(data.x) # (n, d_n) -> (n, d)
        
        for conv in self.convs:
            h = h + conv(h, data.edge_index, data.edge_attr) # (n, d) -> (n, d)
            # Note that we add a residual connection after each MPNN layer

        h_graph = self.pool(h, data.batch) # (n, d) -> (batch_size, d)

        out = self.lin_pred(h_graph) # (batch_size, d) -> (batch_size, 1)

        return out.view(-1)
    

class E3EGCL(MessagePassing):
    """
    E(3) Equivariant Graph Convolution Layer (EGCL)
    """
    def __init__(self, node_dim, aggr='add'):
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
        )

        self.mlp_msg_x = nn.Sequential(
            nn.Linear(node_dim*2+1, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, 1)
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
    def __init__(self, node_dim, num_layers=4, aggr='add'):
        super().__init__()
        self.convs = nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(E3EGCL(node_dim=node_dim, aggr=aggr))

    def forward(self, data):
        h = data.h
        x = data.x
        for conv in self.convs:
            h_upd, x_upd = conv(h, x, data.edge_index)
            h = h + h_upd
            x = x_upd
        x_cog = x.mean(dim=0, keepdim=True)
        x = x - x_cog # remove center of gravity cf. Section 3.2 in paper
        return torch.cat([x, h], dim=-1)
