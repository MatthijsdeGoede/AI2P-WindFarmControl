import torch.nn as nn

from WGN.deconv import DeConvNet
from WGN.mlp import MLP
from WGN.pign import PIGN


class PowerPIGNN(nn.Module):

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
                 pign_mlp_params: dict = None,
                 reg_mlp_params: dict = None):
        super(PowerPIGNN, self).__init__()

        if pign_mlp_params is None:
            pign_mlp_params = {'num_neurons': [256, 128], 'hidden_act': 'ReLU', 'out_act': 'ReLU'}

        if reg_mlp_params is None:
            reg_mlp_params = {'num_neurons': [64, 32, 16], 'hidden_act': 'ReLU', 'out_act': 'ReLU'}

        edge_in_dims = [edge_in_dim] + n_pign_layers * [edge_hidden_dim]
        edge_out_dims = n_pign_layers * [edge_hidden_dim] + [edge_hidden_dim]
        node_in_dims = [node_in_dim] + n_pign_layers * [node_hidden_dim]
        node_out_dims = n_pign_layers * [node_hidden_dim] + [node_hidden_dim]
        global_in_dims = [global_in_dim] + n_pign_layers * [global_hidden_dim]
        global_out_dims = n_pign_layers * [global_hidden_dim] + [global_hidden_dim]

        # instantiate PIGN layers
        self.gn_layers = nn.ModuleList()
        dims = zip(edge_in_dims, edge_out_dims, node_in_dims, node_out_dims, global_in_dims, global_out_dims)
        for i, (ei, eo, ni, no, gi, go) in enumerate(dims):
            _residual = i >= 1 and residual
            _input_norm = 'batch' if input_norm else None

            # apply `PhysicsInducedAttention` only to the last PGN Layer
            # maybe not optimal choice for the best-performing models
            # but it could provide straightforward way for analyzing attention scores.
            use_attention = True if i == n_pign_layers else False

            em = MLP(ei + 2 * ni + gi, eo, input_norm=_input_norm, **pign_mlp_params)
            nm = MLP(ni + eo + gi, no, input_norm=_input_norm, **pign_mlp_params)
            gm = MLP(gi + eo + no, go, input_norm=_input_norm, **pign_mlp_params)
            layer = PIGN(em, nm, gm, residual=_residual, use_attention=use_attention)
            self.gn_layers.append(layer)

        # regression layer : convert the node embedding to power predictions
        # self.reg = MLP(node_hidden_dim, output_dim, **reg_mlp_params)

    def _forward_graph(self, data, nf, ef, gf):
        unf, uef, ug = nf, ef, gf
        for layer in self.gn_layers:
            unf, uef, ug = layer(data, unf, uef, ug)
        return unf, uef, ug

    def forward(self, data, nf, ef, gf):
        unf, uef, ug = self._forward_graph(data, nf, ef, gf)
        power_pred = self.reg(unf)
        return power_pred.clip(min=0.0, max=1.0)


class FlowPIGNN(PowerPIGNN):
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
                 output_size: tuple = (128, 128)):
        super(FlowPIGNN, self).__init__(edge_in_dim, node_in_dim, global_in_dim, edge_hidden_dim, node_hidden_dim,
                                        global_hidden_dim, output_dim, n_pign_layers, residual, input_norm,
                                        pign_mlp_params, reg_mlp_params)
        self.num_nodes = num_nodes

        self.mlp = MLP(input_dim=num_nodes*node_hidden_dim, output_dim=64,num_neurons=[128, 128, 64], hidden_act='ReLU')
        # Actor model on the node embeddings
        self.deconv = DeConvNet(1, [64, 128, 256, 1], output_size=output_size)

    # Override
    def forward(self, data, nf, ef, gf):
        unf, uef, ug = self._forward_graph(data, nf, ef, gf)
        output_pignn = unf.reshape(-1, 1, self.num_nodes*unf.size(1))
        output_mlp = self.mlp(output_pignn)
        output_mlp = output_mlp.reshape(-1, 1, 8, 8)
        if self.deconv is not None:
            output = self.deconv(output_mlp)
        return output
