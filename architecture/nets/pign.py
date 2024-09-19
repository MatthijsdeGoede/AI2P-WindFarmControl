
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_mean_pool


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


class PIGN(MessagePassing):
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

        self.global_features = None
        self.edge_model = edge_model
        self.node_model = node_model
        self.global_model = global_model
        if use_attention:
            self.attention_model = PhysicsInducedAttention(use_approx=False)

        self.residual = residual
        self.use_attention = use_attention

    def forward(self, node_feat, edge_feat, glob_feat, edge_idx):
        self.global_features = glob_feat.unsqueeze(-1)
        
        # Use the learned functions to update the node and edge features
        # TODO: check how to repeat the global features correctly.
        repeated_gf = self.global_features.repeat_interleave(edge_feat.shape[0], dim=0).reshape(edge_feat.shape[0], -1)
        updated_ef = self.edge_update(node_feat, edge_feat, repeated_gf, edge_idx)
        updated_nf = self.propagate(edge_idx, x=node_feat, edge_attr=edge_feat)

        # Average the node and edge features across the neighbourhoods
        node_readout = global_mean_pool(updated_nf, batch=torch.zeros(updated_nf.size(0), dtype=torch.long,
                                                                      device=updated_nf.device))
        edge_readout = global_mean_pool(updated_ef, batch=torch.zeros(updated_ef.size(0), dtype=torch.long,
                                                                      device=updated_ef.device))

        # Update the global features using the aggregated node and edge features and the learned function
        global_input = torch.cat([node_readout, edge_readout, glob_feat], dim=-1)
        updated_gf = self.global_model(global_input)

        #  Add the output to its input to preserve information improving gradient flow during training
        if self.residual:
            updated_nf = updated_nf + node_feat
            updated_ef = updated_ef + edge_feat
            updated_gf = updated_gf + glob_feat

        return updated_nf, updated_ef, updated_gf


    def edge_update(self, node_feat, edge_feat, glob_feat, edge_idx):
        src, dst = edge_idx
        model_input = torch.cat([node_feat[src], node_feat[dst], edge_feat, glob_feat], dim=-1)
        updated_edge_attr = self.edge_model(model_input)

        if self.use_attention:
            radial_dist = edge_feat[:, 0]
            down_stream_dist = edge_feat[:, 1]
            attn_input = torch.cat([down_stream_dist, radial_dist], dim=-1)
            weights = self.attention_model(attn_input)
            updated_edge_attr = weights.unsqueeze(-1) * updated_edge_attr

        return updated_edge_attr

    def message(self, x_j, edge_feat):
        # TODO: check how to repeat the global features correctly.
        repeated_gf = self.global_features.repeat_interleave(x_j.shape[0], dim=0)
        return torch.cat([x_j, edge_feat, repeated_gf], dim=-1)

    def update(self, aggr_out):
        return self.node_model(aggr_out)
