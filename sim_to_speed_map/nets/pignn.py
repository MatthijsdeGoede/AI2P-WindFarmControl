import torch
import torch.nn as nn

from architecture.nets.mlp import MLP
from architecture.nets.pign import PIGN


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
                 pign_params: dict = None,
                 pign_mlp_params: dict = None,
                 reg_mlp_params: dict = None):
        super(PowerPIGNN, self).__init__()

        if pign_params is None:
            pign_params = {'edge_aggregator': 'mean', 'global_node_aggr': 'mean', 'global_edge_aggr': 'mean'}

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
            layer = PIGN(em, nm, gm, residual=_residual, use_attention=use_attention, **pign_params)
            self.gn_layers.append(layer)

        # regression layer : convert the node embedding to power predictions
        self.reg = MLP(node_hidden_dim, output_dim, **reg_mlp_params)

    def _forward_graph(self, g, nf, ef, u):
        unf, uef, uu = nf, ef, u
        for layer in self.gn_layers:
            unf, uef, uu = layer(g, unf, uef, uu)
        return unf, uef, uu

    def forward(self, g, nf, ef, u):
        unf, uef, uu = self._forward_graph(g, nf, ef, u)
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
                 output_dim: int = 90000,
                 n_pign_layers: int = 3,
                 residual: bool = True,
                 input_norm: bool = True,
                 pign_params: dict = None,
                 pign_mlp_params: dict = None,
                 reg_mlp_params: dict = None):
        super(FlowPIGNN, self).__init__(edge_in_dim, node_in_dim, global_in_dim, edge_hidden_dim, node_hidden_dim,
                                        global_hidden_dim, output_dim, n_pign_layers, residual, input_norm, pign_params,
                                        pign_mlp_params, reg_mlp_params)

        if reg_mlp_params is None:
            reg_mlp_params = {'num_neurons': [64, 32, 16], 'hidden_act': 'ReLU', 'out_act': 'ReLU'}

        self.reg = MLP(edge_hidden_dim + node_hidden_dim + global_hidden_dim, output_dim, **reg_mlp_params)

    # Override
    def forward(self, g, nf, ef, u):
        unf, uef, uu = self._forward_base(g, nf, ef, u)
        flow_pred = self.reg(torch.cat([uef, unf, uu], dim=-1))
        return flow_pred
