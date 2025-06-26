import numpy as np
import torch
from torch import nn
from torch.functional import F
from torch.optim import Adam
from torch_geometric.nn import MessagePassing
from torch.nn import Sequential as Seq, Linear as Lin, ReLU

from torch.autograd import Variable, grad


def get_edge_index(n, sim):
    """
    Constructs the edge index for a graph with n nodes.
    
    Parameters:
      - n: number of nodes.
      - sim: simulation type. If sim is 'string' or 'string_ball', the nodes are
             assumed to be connected in a chain (only adjacent nodes are connected).
             Otherwise, a complete graph (without self-loops) is created.
    
    Returns:
      - edge_index: a torch.tensor of shape [2, num_edges] representing the connectivity.
    """
    if sim in ['string', 'string_ball']:
        # For a chain structure, only connect adjacent nodes.
        row = torch.arange(0, n - 1)
        col = torch.arange(1, n)
        # Create bidirectional edges: (i, i+1) and (i+1, i)
        edge_index = torch.cat([
            torch.stack([row, col], dim=0),
            torch.stack([col, row], dim=0)
        ], dim=1)
    else:
        # For a complete graph, create edges between every pair of nodes (excluding self-loops).
        adj = np.ones((n, n)) - np.eye(n)
        row, col = np.where(adj)
        edge_index = torch.tensor([row, col], dtype=torch.long)
    return edge_index



class GeneralGN(MessagePassing):
    """
    Graph Network (single message passing step).

    Performs one round of message passing: computes edge messages, aggregates them,
    and updates node features.

    Parameters:
      n_f (int):        Input node feature dimension.
      msg_dim (int):    Edge message dimension (output of edge_model).
      ndim (int):       Output node feature dimension after update.
      hidden (int):     Hidden layer dimension for MLPs. Default is 300.
      aggr (str):       Aggregation method: 'add' in this project.

    Attributes:
      edge_model (Sequential): Neural network mapping [2*n_f] -> msg_dim for each edge.
      node_model (Sequential): Neural network mapping [n_f + msg_dim] -> ndim for each node.
    """
    def __init__(self, n_f, msg_dim, ndim, hidden=300, aggr='add'):
        super(GeneralGN, self).__init__(aggr=aggr) 
        self.edge_model = Seq(
            Lin(2 * n_f, hidden),
            ReLU(),
            Lin(hidden, hidden),
            ReLU(),
            Lin(hidden, hidden),
            ReLU(),
            Lin(hidden, msg_dim)
        )
        
        self.node_model = Seq(
            Lin(msg_dim + n_f, hidden),
            ReLU(),
            Lin(hidden, hidden),
            ReLU(),
            Lin(hidden, hidden),
            ReLU(),
            Lin(hidden, ndim)
        )



    def forward(self, x, edge_index):
        """
        Computes a single message passing step.
    
        Parameters:
          x (Tensor):          Node feature matrix of shape [N, n_f].
          edge_index (LongTensor): Edge index tensor of shape [2, E].

        Returns:
          out (Tensor):        Updated node features of shape [N, ndim].
        
        """
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)
      
    def message(self, x_i, x_j):
        """
        Compute messages for each directed edge.

        Parameters:
          x_i (Tensor): Node features for target nodes, shape [E, n_f].
          x_j (Tensor): Node features for source nodes, shape [E, n_f].

        Returns:
          msgs (Tensor): Message vectors for each edge, shape [E, msg_dim].
        """
        edge_inputs = torch.cat([x_i, x_j], dim=1)  
        return self.edge_model(edge_inputs)
    
    def update(self, aggr_out, x):
        """
        Update node features after aggregation.

        Parameters:
          aggr_out (Tensor): Aggregated messages for each node, shape [N, msg_dim].
          x (Tensor):        Original node features, shape [N, n_f].

        Returns:
          new_feats (Tensor): Updated node features, shape [N, ndim].
        """
        node_inputs = torch.cat([x, aggr_out], dim=1)
        return self.node_model(node_inputs) 


class NbodyGNN(GeneralGN):
    """
    A specialized GNN for N-body system acceleration prediction.
    Inherits from GeneralGNN and integrates edge_index and loss computation.

    Parameters:
      n_f (int):            Input node feature dimension (e.g., position, velocity, mass).
      msg_dim (int):        Edge message dimension.
      ndim (int):           Output node feature dimension (acceleration dimension).
      dt (float):           Time step size (not used directly, reserved for multi-step integration).
      edge_index (LongTensor): Edge index tensor of shape [2, E].
      aggr (str):           Aggregation method: 'add', 'mean', or 'max'. Default is 'add'.
      hidden (int):         Hidden layer dimension for MLPs. Default is 300.
      nt (int):             Number of message passing iterations. Typically 1 for fully connected N-body.
      l1_reg_weight (float): L1 regularization weight for edge messages. Default is 0.0 (no L1).
   
    """
    def __init__(
		self, n_f, msg_dim, ndim, 
		edge_index, aggr='add', hidden=300, nt=1, l1_reg_weight=0.0):

        super(NbodyGNN, self).__init__(n_f, msg_dim, ndim, hidden=hidden, aggr=aggr,)
       
        self.nt = nt
        self.edge_index = edge_index
        self.ndim = ndim
        self.l1_reg_weight = l1_reg_weight
    
    def predict_acceleration(self, data):
        """
        Predict node accelerations via message passing.

        Parameters:
          data (Data): PyG Data/Batch object containing:
                         - data.x: [N, n_f], node features.
                         - data.edge_index: [2, E], edge index.

        Returns:
          Predicted accelerations, shape [N, ndim].
        """
        return self.propagate(data.edge_index, x=data.x)
    
    def loss(self, data):
        """
        Compute loss: MAE on acceleration + optional L1 regularization on edge messages.

        Parameters:
          data (Data): PyG Data/Batch object containing:
                         - data.x: [N, n_f], node features.
                         - data.edge_index: [2, E], edge index.
                         - data.y: [N, ndim], true accelerations.
                         - data.num_graphs: Number of graphs in the batch.

        Returns:
          loss_value (Tensor): Scalar loss, equal to (MAE per graph + L1 regularization).
        """

        pred = self.predict_acceleration(data)
        batch_size = data.num_graphs  

  
        data_loss = F.l1_loss(pred, data.y, reduction='sum')

        total_loss = data_loss

        # ---- 2) optional L1 regularization on raw messages ----
        if self.l1_reg_weight > 0:
            # extract source/target features for every edge
            s_i = data.x[data.edge_index[0]]        # [E, n_f]
            s_j = data.x[data.edge_index[1]]        # [E, n_f]
            inp = torch.cat([s_i, s_j], dim=1)  # [E, 2*n_f]
            msgs = self.edge_model(inp)            # [E, msg_dim]
            # sum of abs over all edges & channels
            l1_sum = torch.sum(msgs.abs())

            # batch size (# graphs) and node count
            batch_size = data.num_graphs      
            n = data.x.size(0)
            n_per_graph = n // batch_size    
            reg_term = self.l1_reg_weight * batch_size * l1_sum / (n_per_graph**2) * n_per_graph                 
            total_loss = total_loss + reg_term

        return total_loss/batch_size
    

class VariationGN(MessagePassing):
    """
    A variational graph network layer with KL divergence regularization on the edge messages.
    """
    def __init__(self, n_f, msg_dim, ndim, hidden=300, aggr='add'):
        super().__init__(aggr=aggr)
        # edge encoder: outputs 2 * msg_dim for [mu, logvar]
        self.edge_encoder = Seq(
            Lin(2 * n_f, hidden),
            ReLU(),
            Lin(hidden, hidden),
            ReLU(),
            Lin(hidden, hidden),
            ReLU(),
            Lin(hidden, 2 * msg_dim),
        )
        # node decoder: maps aggregated messages + node features back to ndim
        self.node_decoder = Seq(
            Lin(msg_dim + n_f, hidden),
            ReLU(),
            Lin(hidden, hidden),
            ReLU(),
            Lin(hidden, hidden),
            ReLU(),
            Lin(hidden, ndim),
        )

    def forward(self, x, edge_index):
        """
        Perform one message-passing step.
        
        Args:
            x:      [N, n_f] node feature matrix
            edge_index: [2, E] edge indices
        Returns:
            [N, ndim] updated node features
        """
        # reset KL terms for this forward pass
        self.kl_terms = []
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        """
        Compute variational messages for each directed edge.
        
        Args:
            x_i: [E, n_f] features of target nodes
            x_j: [E, n_f] features of source nodes
        Returns:
            m:   [E, msg_dim] sampled message vectors
        """
        h = torch.cat([x_i, x_j], dim=-1)                # [E, 2*n_f]
        stats = self.edge_encoder(h)                     # [E, 2*msg_dim]
        mu, logvar = stats.chunk(2, dim=-1)              # each [E, msg_dim]

        std = torch.exp(0.5 * logvar)                    # [E, msg_dim]
        eps = torch.randn_like(std)                      # [E, msg_dim]
        m = mu + eps * std                               # reparameterized sample

        # KL divergence per edge: sum over msg_dim
        kl_edge = 0.5 * ((mu**2 + std**2 - logvar - 1).sum(dim=-1))  
        self.kl_terms.append(kl_edge)                    # list of [E] tensors

        return m

    def update(self, aggr_out, x):
        """
        Update node features by combining aggregated messages and input features.
        
        Args:
            aggr_out: [N, msg_dim] aggregated messages per node
            x:        [N, n_f] original node features
        Returns:
            [N, ndim] updated node features
        """
        h = torch.cat([x, aggr_out], dim=-1)             # [N, msg_dim + n_f]
        return self.node_decoder(h)


class VariationNbody(VariationGN):
    """
    A full GNN model for N-body acceleration prediction with variational edge messages.
    """
    def __init__(self, n_f, msg_dim, ndim, edge_index,
                 hidden=300, beta=1.0):
        super().__init__(n_f, msg_dim, ndim, hidden=hidden, aggr='add')

        self.edge_index = edge_index
        self.beta = beta

    def predict_acceleration(self, data):
        """
        Run message passing to predict accelerations.
        
        Args:
            data.x:          [N, n_f] node features
            data.edge_index: [2, E] edge indices
        Returns:
            [N, ndim] predicted accelerations
        """
        return self.forward(data.x, data.edge_index)

    def loss(self, data):
        """
        Compute VAE-style loss = data term + beta * KL term.
        
        Args:
            data.y:          [N, ndim] ground-truth accelerations
        Returns:
            scalar tensor of total loss (mean reduction)
        """
        # reset KL terms
        self.kl_terms = []

        # predict
        pred = self.predict_acceleration(data)

    
        data_loss = F.l1_loss(pred, data.y, reduction='mean')

        # KL loss (mean over all edges)
        all_kl = torch.cat(self.kl_terms)    # [total_edges_in_batch]
        kl_loss = all_kl.mean()

        # combined loss
        total = data_loss + self.beta * kl_loss

        # record for logging
        self.last_data_loss = data_loss.detach()
        self.last_kl_loss   = kl_loss.detach()
        self.last_total     = total.detach()

        return total
