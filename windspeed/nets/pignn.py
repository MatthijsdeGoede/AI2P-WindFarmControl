import torch.nn as nn

from windspeed.nets.mlp import MLP
from windspeed.nets.pign import PIGN


class PIGNN(nn.Module):

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
        super(PIGNN, self).__init__()

        if pign_params is None:
            pign_params = {'edge_aggregator': 'mean',
                          'global_node_aggr': 'mean',
                          'global_edge_aggr': 'mean'}

        if pign_mlp_params is None:
            pign_mlp_params = {'num_neurons': [256, 128],
                              'hidden_act': 'ReLU',
                              'out_act': 'ReLU'}

        if reg_mlp_params is None:
            reg_mlp_params = {'edge_aggregator': 'mean',
                              'global_node_aggr': 'mean',
                              'global_edge_aggr': 'mean'}

        edge_in_dims = [edge_in_dim] + n_pign_layers * [edge_hidden_dim]
        edge_out_dims = n_pign_layers * [edge_hidden_dim] + [edge_hidden_dim]
        node_in_dims = [node_in_dim] + n_pign_layers * [node_hidden_dim]
        node_out_dims = n_pign_layers * [node_hidden_dim] + [node_hidden_dim]
        global_in_dims = [global_in_dim] + n_pign_layers * [global_hidden_dim]
        global_out_dims = n_pign_layers * [global_hidden_dim] + [global_hidden_dim]

        # instantiate PIGN layers
        self.gn_layers = nn.ModuleList()
        dims = zip(edge_in_dims, edge_out_dims,
                   node_in_dims, node_out_dims,
                   global_in_dims, global_out_dims)
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
            l = PIGN(em, nm, gm, residual=_residual, use_attention=use_attention, **pign_params)
            self.gn_layers.append(l)

        # regression layer : convert the node embedding to power predictions or wind flow estimations
        self.reg = MLP(node_hidden_dim, output_dim,
                       **reg_mlp_params)

    def forward(self, g, nf, ef, u):
        unf, uef, uu = nf, ef, u
        for l in self.gn_layers:
            unf, uef, uu = l(g, unf, uef, uu)
        power_pred = self.reg(unf)
        return power_pred.clip(min=0.0, max=1.0)