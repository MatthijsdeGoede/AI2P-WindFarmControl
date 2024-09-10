from functools import partial

import dgl
import dgl.function as dgl_fn
import torch
import torch.nn as nn


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
                 use_attention: bool,
                 edge_aggregator: str = 'mean',
                 global_node_aggr: str = 'mean',
                 global_edge_aggr: str = 'mean'):
        super(PIGN, self).__init__()

        # trainable models
        self.edge_model = edge_model
        self.node_model = node_model
        self.global_model = global_model
        if use_attention:
            self.attention_model = PhysicsInducedAttention(use_approx=False)

        # residual hook
        self.residual = residual
        self.use_attention = use_attention

        # aggregators
        self.edge_aggr = getattr(dgl_fn, edge_aggregator)('m', 'agg_m')
        self.global_node_aggr = global_node_aggr
        self.global_edge_aggr = global_edge_aggr

    def forward(self, g, nf, ef, u):
        """
        :param g: graphs
        :param nf: node features
        :param ef: edge features
        :param u: global features
        :return:

        """
        with g.local_scope():
            g.ndata['_h'] = nf
            g.edata['_h'] = ef

            # update edges
            repeated_us = u.repeat_interleave(g.batch_num_edges(), dim=0)
            edge_update = partial(self.edge_update, repeated_us=repeated_us)
            g.apply_edges(func=edge_update)

            # update nodes
            repeated_us = u.repeat_interleave(g.batch_num_nodes(), dim=0)
            node_update = partial(self.node_update, repeated_us=repeated_us)
            g.pull(g.nodes(),
                   message_func=dgl_fn.copy_e('m', 'm'),
                   reduce_func=self.edge_aggr,
                   apply_node_func=node_update)

            # update global features
            node_readout = dgl.readout_nodes(g, 'uh', op=self.global_node_aggr)
            edge_readout = dgl.readout_edges(g, 'uh', op=self.global_edge_aggr)
            gm_input = torch.cat([node_readout, edge_readout, u], dim=-1)
            updated_u = self.global_model(gm_input)

            updated_nf = g.ndata['uh']
            updated_ef = g.edata['uh']

            if self.residual:
                updated_nf = updated_nf + nf
                updated_ef = updated_ef + ef
                updated_u = updated_u + u

            return updated_nf, updated_ef, updated_u

    def edge_update(self, edges, repeated_us):
        sender_nf = edges.src['_h']
        receiver_nf = edges.dst['_h']
        ef = edges.data['_h']

        # update edge features
        em_input = torch.cat([sender_nf, receiver_nf, ef, repeated_us], dim=-1)
        e_updated = self.edge_model(em_input)

        if self.use_attention:
            # compute edge weights
            dd = edges.data['down_stream_dist']
            rd = edges.data['radial_dist']
            attn_input = torch.cat([dd, rd], dim=-1)
            weights = self.attention_model(attn_input)
            updated_ef = weights * e_updated
        else:
            updated_ef = e_updated
        return {'m': updated_ef, 'uh': updated_ef}

    def node_update(self, nodes, repeated_us):
        agg_m = nodes.data['agg_m']
        nf = nodes.data['_h']
        nm_input = torch.cat([agg_m, nf, repeated_us], dim=-1)
        updated_nf = self.node_model(nm_input)
        return {'uh': updated_nf}
