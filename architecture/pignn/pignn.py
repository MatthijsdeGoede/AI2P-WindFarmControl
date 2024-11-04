import torch.nn as nn
from architecture.pignn.mlp import MLP
from architecture.pignn.pign import PIGN


class PowerPIGNN(nn.Module):
    """
    A Power Physics-Induced Graph Neural Network (PowerPIGNN) based on https://doi.org/10.1016/j.energy.2019.115883
    for predicting power output or related node and edge-level quantities.
    It utilizes multiple stacked PIGN layers with optional attention and normalization.

    Attributes:
        gn_layers (nn.ModuleList): A list of PIGN layers, each updating node, edge, and global features.
    """

    def __init__(self,
                 edge_in_dim: int,
                 node_in_dim: int,
                 global_in_dim: int,
                 edge_hidden_dim: int = 32,
                 node_hidden_dim: int = 32,
                 global_hidden_dim: int = 32,
                 output_dim: int = 1,
                 n_pign_layers: int = 3,
                 residual: bool = True,
                 input_norm: bool = True,
                 pign_mlp_params: dict = None):
        """
        Initializes a PowerPIGNN model with the specified dimensions for input, hidden, and output layers,
        along with options for residual connections, input normalization, and MLP parameters for PIGN layers.

        Args:
            edge_in_dim (int): Input dimension for edge features.
            node_in_dim (int): Input dimension for node features.
            global_in_dim (int): Input dimension for global features.
            edge_hidden_dim (int): Hidden dimension for edge features (default: 32).
            node_hidden_dim (int): Hidden dimension for node features (default: 32).
            global_hidden_dim (int): Hidden dimension for global features (default: 32).
            output_dim (int): Dimension of the output prediction (default: 1).
            n_pign_layers (int): Number of stacked PIGN layers (default: 3).
            residual (bool): If True, adds residual connections (default: True).
            input_norm (bool): If True, applies batch normalization to layer inputs (default: True).
            pign_mlp_params (dict): Parameters for MLPs within PIGN layers.
        """
        super(PowerPIGNN, self).__init__()

        if pign_mlp_params is None:
            pign_mlp_params = {'num_neurons': [256, 128], 'hidden_act': 'ReLU', 'out_act': 'ReLU'}

        # Define dimensions for input and output of each PIGN layer
        edge_in_dims = [edge_in_dim] + n_pign_layers * [edge_hidden_dim]
        edge_out_dims = n_pign_layers * [edge_hidden_dim] + [edge_hidden_dim]
        node_in_dims = [node_in_dim] + n_pign_layers * [node_hidden_dim]
        node_out_dims = n_pign_layers * [node_hidden_dim] + [node_hidden_dim]
        global_in_dims = [global_in_dim] + n_pign_layers * [global_hidden_dim]
        global_out_dims = n_pign_layers * [global_hidden_dim] + [global_hidden_dim]

        # Instantiate PIGN layers with input/output dimensions and other settings
        self.gn_layers = nn.ModuleList()
        dims = zip(edge_in_dims, edge_out_dims, node_in_dims, node_out_dims, global_in_dims, global_out_dims)
        for i, (ei, eo, ni, no, gi, go) in enumerate(dims):
            _residual = i >= 1 and residual
            _input_norm = 'batch' if input_norm else None
            use_attention = True if i == n_pign_layers else False  # Apply attention in the final layer

            em = MLP(ei + 2 * ni + gi, eo, input_norm=_input_norm, **pign_mlp_params)
            nm = MLP(ni + eo + gi, no, input_norm=_input_norm, **pign_mlp_params)
            gm = MLP(gi + eo + no, go, input_norm=_input_norm, **pign_mlp_params)
            layer = PIGN(em, nm, gm, residual=_residual, use_attention=use_attention)
            self.gn_layers.append(layer)

    def _forward_graph(self, data, nf, ef, gf):
        """
        Processes the graph through all PIGN layers sequentially, updating node, edge, and global features.

        Args:
            data (torch_geometric.data.Data): Graph data containing edge_index and batch.
            nf (torch.Tensor): Initial node features.
            ef (torch.Tensor): Initial edge features.
            gf (torch.Tensor): Initial global features.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Updated node, edge, and global features.
        """
        unf, uef, ug = nf, ef, gf
        for layer in self.gn_layers:
            unf, uef, ug = layer(data, unf, uef, ug)
        return unf, uef, ug

    def forward(self, data, nf, ef, gf):
        """
        Forward pass of the model, updating features and predicting power output.

        Args:
            data (torch_geometric.data.Data): Graph data containing edge_index and batch.
            nf (torch.Tensor): Node features.
            ef (torch.Tensor): Edge features.
            gf (torch.Tensor): Global features.

        Returns:
            torch.Tensor: Predicted power output, clipped between 0 and 1.
        """
        unf, uef, ug = self._forward_graph(data, nf, ef, gf)
        power_pred = self.reg(unf)
        return power_pred.clip(min=0.0, max=1.0)


class FlowPIGNN(PowerPIGNN):
    """
    A Flow Physics-Induced Graph Neural Network (FlowPIGNN) that extends PowerPIGNN for flow-based prediction tasks.
    It includes MLP and optional deconvolution for processing node embeddings to farm scale wind maps.

    Attributes:
        num_nodes (int): Number of nodes in the graph.
        mlp (MLP): MLP to reduce node embedding size.
        deconv_model (nn.Module): Optional deconvolution model for further feature processing.
    """

    def __init__(self,
                 edge_in_dim: int,
                 node_in_dim: int,
                 global_in_dim: int,
                 edge_hidden_dim: int = 32,
                 node_hidden_dim: int = 32,
                 global_hidden_dim: int = 32,
                 output_dim: int = 16384,
                 n_pign_layers: int = 3,
                 num_nodes: int = 10,
                 residual: bool = True,
                 input_norm: bool = True,
                 pign_mlp_params: dict = None,
                 reg_mlp_params: dict = None,
                 deconv_model: nn.Module = None):
        """
        Initializes a FlowPIGNN model with additional parameters for flow prediction, node processing,
        and optional deconvolution layers.

        Args:
            edge_in_dim (int): Input dimension for edge features.
            node_in_dim (int): Input dimension for node features.
            global_in_dim (int): Input dimension for global features.
            edge_hidden_dim (int): Hidden dimension for edge features.
            node_hidden_dim (int): Hidden dimension for node features.
            global_hidden_dim (int): Hidden dimension for global features.
            output_dim (int): Dimension of the output prediction.
            n_pign_layers (int): Number of stacked PIGN layers.
            num_nodes (int): Number of nodes in the graph.
            residual (bool): If True, enables residual connections.
            input_norm (bool): If True, applies batch normalization.
            pign_mlp_params (dict): Parameters for MLPs in PIGN layers.
            reg_mlp_params (dict): Parameters for MLP in regression layers.
            deconv_model (nn.Module): Deconvolution model for processing output embeddings.
        """
        super(FlowPIGNN, self).__init__(edge_in_dim, node_in_dim, global_in_dim, edge_hidden_dim, node_hidden_dim,
                                        global_hidden_dim, output_dim, n_pign_layers, residual, input_norm,
                                        pign_mlp_params, reg_mlp_params)
        self.num_nodes = num_nodes
        # MLP model to decrease node embedding size
        self.mlp = MLP(input_dim=num_nodes * node_hidden_dim, output_dim=64, num_neurons=[128, 128, 64], hidden_act='ReLU')
        # Optional deconvolution model
        self.deconv_model = deconv_model

    def forward(self, data, nf, ef, gf):
        """
        Forward pass that reshapes node features for flow prediction and optionally applies deconvolution.

        Args:
            data (torch_geometric.data.Data): Graph data containing edge_index and batch.
            nf (torch.Tensor): Node features.
            ef (torch.Tensor): Edge features.
            gf (torch.Tensor): Global features.

        Returns:
            torch.Tensor: Final output tensor with flow predictions, optionally processed by deconvolution.
        """
        unf, uef, ug = self._forward_graph(data, nf, ef, gf)
        output = unf.reshape(-1, 1, self.num_nodes, unf.size(1))

        # Apply deconvolution model if defined
        if self.deconv_model is not None:
            output_pignn = output.reshape(-1, 1, self.num_nodes * unf.size(1))
            output_mlp = self.mlp(output_pignn)
            output_mlp = output_mlp.reshape(-1, 1, 8, 8)
            output = self.deconv_model(output_mlp)

        return output
