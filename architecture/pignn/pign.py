import torch
import torch.nn as nn
from torch_geometric.utils import scatter
from torch_geometric.nn import global_mean_pool


class Normalizer(nn.Module):
    """
    A normalization layer that calculates and maintains a running mean and variance for input features
    and normalizes input tensors accordingly. Can be used to stabilize training.

    Attributes:
        num_feature (int): Number of features in the input.
        alpha (float): Smoothing factor for running mean and variance updates.
        mean (torch.Tensor): Running mean of the features.
        var (torch.Tensor): Running variance of the features.
        std (torch.Tensor): Standard deviation derived from variance.
        w (torch.Parameter): Scale parameter applied after normalization.
        b (torch.Parameter): Bias parameter applied after normalization.
    """

    def __init__(self, num_feature, alpha=0.9):
        """
        Initializes the Normalizer with feature dimensions and an optional smoothing factor for mean updates.

        Args:
            num_feature (int): Number of features in the input.
            alpha (float): Smoothing factor for running mean and variance updates (default: 0.9).
        """
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
        """
        Forward pass that normalizes the input using running mean and variance.

        Args:
            xs (torch.Tensor): Input tensor with features of shape (batch_size, num_feature).

        Returns:
            torch.Tensor: Normalized and affine-transformed input tensor.
        """
        xs = xs.view(-1, self.num_feature)  # Reshape to handle 1-batch cases

        if self.training:
            # Update running mean and variance
            mean_update = torch.mean(xs, dim=0)
            self.mean = self.alpha * self.mean + (1 - self.alpha) * mean_update.detach()
            var_update = (1 - self.alpha) * torch.mean(torch.pow((xs - self.mean), 2), dim=0)
            self.var = self.alpha * self.var + var_update.detach()
            self.std = torch.sqrt(self.var + 1e-10)

        standardized = xs / self.std
        affined = standardized * torch.nn.functional.relu(self.w)

        return affined

    def reset_stats(self):
        """Resets the mean, variance, and standard deviation statistics."""
        self.mean.zero_()
        self.var.fill_(1)
        self.std.fill_(1)


class PhysicsInducedAttention(nn.Module):
    """
    A physics-induced attention mechanism that calculates interaction coefficients
    using an engineering wake model: https://ideas.repec.org/a/eee/appene/v151y2015icp320-334.html.

    Attributes:
        input_dim (int): Number of input features.
        use_approx (bool): If True, approximates the exponential with a power series.
        degree (int): Degree of the power series approximation.
        alpha, r0, k (torch.Parameter): Learnable parameters for attention calculation.
        norm (Normalizer): Normalizer for feature inputs.
    """

    def __init__(self, input_dim=2, use_approx=True, degree=5):
        """
        Initializes PhysicsInducedAttention with the specified input dimensions,
        approximation settings, and normalization.

        Args:
            input_dim (int): Number of input features (default: 2).
            use_approx (bool): Use power series approximation for exponential function if True (default: True).
            degree (int): Degree of power series approximation for exponential (default: 5).
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
        """
        Forward pass that calculates interaction coefficients using scaled bias and optional power approximation.

        Args:
            xs (torch.Tensor): Input tensor with features of shape (batch_size, input_dim).
            degree (int, optional): Degree for power series approximation.

        Returns:
            torch.Tensor: Calculated interaction coefficients.
        """
        if degree is None:
            degree = self.degree
        interacting_coeiff = self.get_scaled_bias(xs, degree)
        interacting_coeiff = nn.functional.relu(interacting_coeiff)
        return interacting_coeiff

    @staticmethod
    def power_approx(fx, degree=5):
        """
        Approximates the exponential function with a Taylor series expansion.

        Args:
            fx (torch.Tensor): Input tensor for approximation.
            degree (int): Degree of Taylor series.

        Returns:
            torch.Tensor: Approximated exponential tensor.
        """
        ret = torch.ones_like(fx)
        fact = 1
        for i in range(1, degree + 1):
            fact = fact * i
            ret += torch.pow(fx, i) / fact
        return ret

    def get_scaled_bias(self, xs, degree=5):
        """
        Computes scaled bias based on normalized input features.

        Args:
            xs (torch.Tensor): Input tensor with features of shape (batch_size, input_dim).
            degree (int): Degree for power series approximation.

        Returns:
            torch.Tensor: Scaled bias for interaction coefficients.
        """
        xs = self.norm(xs)
        eps = 1e-10
        x, r = xs[:, 0], xs[:, 1]

        r0 = nn.functional.relu(self.r0 + eps)
        alpha = nn.functional.relu(self.alpha + eps)
        k = nn.functional.relu(self.k + eps)

        denom = r0 + k * x
        down_stream_effect = alpha * torch.pow((r0 / denom), 2)
        radial_input = -torch.pow((r / denom), 2)

        radial_effect = self.power_approx(radial_input, degree) if self.use_approx else torch.exp(-torch.pow((r / denom), 2))
        interacting_coeiff = down_stream_effect * radial_effect

        return interacting_coeiff


def global_mean_pool_edge(data, updated_ef, device):
    """
    Aggregates edge features by taking the mean across edges within each graph.

    Args:
        data (torch_geometric.data.Data): Graph data containing batch and edge_index.
        updated_ef (torch.Tensor): Updated edge features.
        device (torch.device): Device for computation.

    Returns:
        torch.Tensor: Aggregated edge features for each graph.
    """
    edge_graph_mapping = data.batch[data.edge_index[0]]
    aggregated_ef = torch.zeros((data.num_graphs, updated_ef.size(1)), device=device)
    for graph_id in range(data.num_graphs):
        mask = (edge_graph_mapping == graph_id)
        if mask.sum() > 0:
            aggregated_ef[graph_id] = updated_ef[mask].mean(dim=0)
    return aggregated_ef


class PIGN(nn.Module):
    """
    Physics-Induced Graph Network (PIGN) layer with edge, node, and global updates,
    designed to integrate physics-informed attention.

    Attributes:
        edge_model (nn.Module): Model to update edge features.
        node_model (nn.Module): Model to update node features.
        global_model (nn.Module): Model to update global features.
        attention_model (PhysicsInducedAttention, optional): Attention model for edge updates.
        residual (bool): If True, applies residual connections.
        use_attention (bool): If True, applies attention mechanism.
    """

    def __init__(self, edge_model: nn.Module, node_model: nn.Module, global_model: nn.Module, residual: bool, use_attention: bool):
        """
        Initializes PIGN layer with edge, node, and global update models.

        Args:
            edge_model (nn.Module): Model to update edge features.
            node_model (nn.Module): Model to update node features.
            global_model (nn.Module): Model to update global features.
            residual (bool): If True, applies residual connections.
            use_attention (bool): If True, applies attention mechanism.
        """
        super(PIGN, self).__init__()
        self.edge_model = edge_model
        self.node_model = node_model
        self.global_model = global_model
        self.residual = residual
        self.use_attention = use_attention
        if use_attention:
            self.attention_model = PhysicsInducedAttention(use_approx=False)

    def forward(self, data, nf, ef, gf):
        """
        Forward pass to update edge, node, and global features using the defined models and optional attention.

        Args:
            data (torch_geometric.data.Data): Graph data containing edge_index and batch.
            nf (torch.Tensor): Node features.
            ef (torch.Tensor): Edge features.
            gf (torch.Tensor): Global features.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Updated node, edge, and global features.
        """
        device = nf.device
        edge_index = data.edge_index.to(device)
        batch_mapping = data.batch

        edge_repeated_u = gf[batch_mapping[edge_index[0]]]
        updated_ef = self.edge_update(nf, ef, edge_repeated_u, edge_index)

        node_repeated_u = gf[batch_mapping]
        updated_nf = self.node_update(updated_ef, nf, node_repeated_u, edge_index[1])

        aggregated_nf = global_mean_pool(updated_nf, batch=batch_mapping)
        aggregated_ef = global_mean_pool_edge(data, updated_ef, device)
        updated_u = self.global_update(aggregated_nf, aggregated_ef, gf)

        return updated_nf, updated_ef, updated_u

    def edge_update(self, nf, ef, rep_gf, edge_index):
        """
        Updates edge features based on node and global features, with optional attention.

        Args:
            nf (torch.Tensor): Node features.
            ef (torch.Tensor): Edge features.
            rep_gf (torch.Tensor): Repeated global features for each edge.
            edge_index (torch.Tensor): Edge index tensor indicating connections.

        Returns:
            torch.Tensor: Updated edge features.
        """
        src, dst = edge_index
        model_input = torch.cat([nf[src], nf[dst], ef, rep_gf], dim=-1)
        updated_ef = self.edge_model(model_input)

        if self.use_attention:
            radial_dist, down_stream_dist = ef[:, 0], ef[:, 1]
            attn_input = torch.cat([down_stream_dist, radial_dist], dim=-1)
            weights = self.attention_model(attn_input)
            updated_ef = updated_ef * weights

        return updated_ef

    def node_update(self, updated_ef, nf, repeated_us, target_nodes):
        """
        Updates node features by aggregating incoming edge features and combining them with node and global features.

        Args:
            updated_ef (torch.Tensor): Updated edge features.
            nf (torch.Tensor): Node features.
            repeated_us (torch.Tensor): Repeated global features for each node.
            target_nodes (torch.Tensor): Target nodes for each edge.

        Returns:
            torch.Tensor: Updated node features.
        """
        aggregated_incoming_ef = scatter(updated_ef, target_nodes, dim=0, dim_size=nf.size(0), reduce='mean')
        nm_input = torch.cat([aggregated_incoming_ef, nf, repeated_us], dim=-1)
        updated_nf = self.node_model(nm_input)

        if self.residual:
            updated_nf = updated_nf + nf

        return updated_nf

    def global_update(self, aggregated_nf, aggregated_ef, gf):
        """
        Updates global features based on aggregated node and edge features.

        Args:
            aggregated_nf (torch.Tensor): Aggregated node features.
            aggregated_ef (torch.Tensor): Aggregated edge features.
            gf (torch.Tensor): Global features.

        Returns:
            torch.Tensor: Updated global features.
        """
        gm_input = torch.cat([aggregated_nf, aggregated_ef, gf], dim=-1)
        updated_gf = self.global_model(gm_input)

        if self.residual:
            updated_gf = updated_gf + gf

        return updated_gf
