import torch
import torch.nn as nn
from torch import scatter
from torch_geometric.utils import scatter
from torch_geometric.nn import global_mean_pool


class Normalizer(nn.Module):

    def __init__(self, num_feature, alpha=0.9):
        super(Normalizer, self).__init__()
        self.alpha = alpha
        self.num_feature = num_feature
        self.register_buffer('mean', torch.zeros(num_feature))
        self.register_buffer('var', torch.zeros(num_feature))
        self.register_buffer('std', torch.zeros(num_feature))
        self.w = nn.Parameter(torch.ones(num_feature))
        self.b = nn.Parameter(torch.zeros(num_feature))
        self.reset_stats()

    def forward(self, xs):
        xs = xs.view(-1, self.num_feature)  # Handling 1-batch case

        if self.training:
            mean_update = torch.mean(xs, dim=0)  # Get mean-values along the batch dimension
            self.mean = self.alpha * self.mean + (1 - self.alpha) * mean_update.data
            var_update = (1 - self.alpha) * torch.mean(torch.pow((xs - self.mean), 2), dim=0)
            self.var = self.alpha * self.var + var_update.data
            self.std = torch.sqrt(self.var + 1e-10)

        standardized = xs / self.std
        affined = standardized * torch.nn.functional.relu(self.w)

        return affined

    def reset_stats(self):
        self.mean.zero_()
        self.var.fill_(1)
        self.std.fill_(1)


class PhysicsInducedAttention(nn.Module):

    def __init__(self,
                 input_dim=3,
                 use_approx=True,
                 degree=5):
        """
        :param input_dim: (int) input_dim for PhysicsInducedBias layer
        :param use_approx: (bool) If True, exp() will be approximated with power series
        :param degree: (int) degree of power series approximation
        """
        super(PhysicsInducedAttention, self).__init__()
        self.input_dim = input_dim
        self.use_approx = use_approx
        self.degree = degree
        self.alpha = nn.Parameter(torch.zeros(1))
        self.r0 = nn.Parameter(torch.zeros(1))
        self.k = nn.Parameter(torch.zeros(1))

        self.alpha.data.fill_(1.0)
        self.r0.data.fill_(1.0)
        self.k.data.fill_(1.0)

        self.norm = Normalizer(self.input_dim)

    def forward(self, xs, degree=None):
        if degree is None:
            degree = self.degree
        interacting_coeiff = self.get_scaled_bias(xs, degree)
        interacting_coeiff = nn.functional.relu(interacting_coeiff)
        return interacting_coeiff

    @staticmethod
    def power_approx(fx, degree=5):
        ret = torch.ones_like(fx)
        fact = 1
        for i in range(1, degree + 1):
            fact = fact * i
            ret += torch.pow(fx, i) / fact
        return ret

    def get_scaled_bias(self, xs, degree=5):
        xs = self.norm(xs)
        eps = 1e-10
        inp = torch.split(xs, 1, dim=1)
        x, r, ws = inp[0], inp[1], inp[2]

        r0 = nn.functional.relu(self.r0 + eps)
        alpha = nn.functional.relu(self.alpha + eps)
        k = nn.functional.relu(self.k + eps)

        denom = r0 + k * x
        down_stream_effect = alpha * torch.pow((r0 / denom), 2)
        radial_input = -torch.pow((r / denom), 2)

        if self.use_approx:
            radial_effect = self.power_approx(radial_input, degree=degree)
        else:
            radial_effect = torch.exp(-torch.pow((r / denom), 2))

        interacting_coeiff = down_stream_effect * radial_effect

        return interacting_coeiff


def global_mean_pool_edge(data, updated_ef, device):
    # Map edges to graphs based on the source node
    edge_batch_mapping = data.batch[data.edge_index[0]]
    aggregated_ef = torch.zeros((data.num_graphs, updated_ef.size(1)), device=device)
    for graph_id in range(data.num_graphs):
        mask = (edge_batch_mapping == graph_id)
        if mask.sum() > 0:
            # Mean aggregation per graph
            aggregated_ef[graph_id] = updated_ef[mask].mean(dim=0)
    return aggregated_ef


class PIGN(nn.Module):
    """
    Pytorch-DGL implementation of the physics-induced Graph Network Layer
    "https://www.sciencedirect.com/science/article/pii/S0360544219315555"
    """

    def __init__(self,
                 edge_model: nn.Module,
                 node_model: nn.Module,
                 global_model: nn.Module,
                 residual: bool,
                 use_attention: bool):
        super(PIGN, self).__init__()

        self.glob_feat = None
        self.edge_model = edge_model
        self.node_model = node_model
        self.global_model = global_model
        if use_attention:
            self.attention_model = PhysicsInducedAttention(use_approx=False)

        self.residual = residual
        self.use_attention = use_attention

    def forward(self, data):
        """
        :param data: Data object containing graph, node features, edge features, and global features.
        """
        device = data.x.device
        edge_index = data.edge_index.to(device)  # Edge indices
        nf = torch.cat((data.x.to(device), data.pos.to(device)), dim=-1)  # Node features
        ef = data.edge_attr.to(device)  # Edge features
        gf = data.global_feats.to(device)  # Global features
        batch_mapping = data.batch  # Which graph each node belongs to

        # Edge update
        edge_repeated_u = gf[batch_mapping[edge_index[0]]]
        updated_ef = self.edge_update(nf, ef, edge_repeated_u, edge_index)

        # Node update
        node_repeated_u = gf[batch_mapping]
        updated_nf = self.node_update(updated_ef, nf, node_repeated_u, edge_index)

        aggregated_nf = global_mean_pool(updated_nf, batch=batch_mapping)
        aggregated_ef = global_mean_pool_edge(data, updated_ef, device)

        # Global update
        updated_u = self.global_update(aggregated_nf, aggregated_ef, gf)

        return updated_nf, updated_ef, updated_u

    def edge_update(self, nf, ef, rep_gf, g):
        src, dst = g
        model_input = torch.cat([nf[src], nf[dst], ef, rep_gf], dim=-1)
        updated_ef = self.edge_model(model_input)

        if self.use_attention:
            radial_dist = ef[:, 0]
            down_stream_dist = ef[:, 1]
            attn_input = torch.cat([down_stream_dist, radial_dist], dim=-1)
            weights = self.attention_model(attn_input)
            updated_ef = weights.unsqueeze(-1) * updated_ef

        return updated_ef

    def node_update(self, updated_ef, nf, repeated_us, edge_index):
        target_nodes = edge_index[1]
        aggregated_incoming_ef = scatter(updated_ef, target_nodes, dim=0, dim_size=nf.size(0), reduce='mean')
        nm_input = torch.cat([aggregated_incoming_ef, nf, repeated_us], dim=-1)
        updated_nf = self.node_model(nm_input)

        # Apply residual connection
        if self.residual:
            updated_nf += nf

        return updated_nf

    def global_update(self, aggregated_nf, aggregated_ef, gf):
        gm_input = torch.cat([aggregated_nf, aggregated_ef, gf], dim=-1)
        updated_gf = self.global_model(gm_input)

        # Apply residual connection
        if self.residual:
            updated_gf += gf

        return updated_gf