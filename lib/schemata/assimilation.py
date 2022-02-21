import torch

from torch import nn
from lib.schemata.match import Match
from lib.schemata.graph_transformer import GraphTransformer
from lib.schemata.misc import xavier_init


class Assimilation(nn.Module):
    def __init__(self, in_edge_dim, hidden_edge_dim, out_edge_dim,
                 in_node_dim, hidden_node_dim, num_heads,
                 n_edge_class, n_node_class, mode, asm_num=2, freeze_base=False,
                 yesFuse=False, hard_att=False, sigmoid_uncertainty=False, accom=False, num_gt_layers=4):
        super(Assimilation, self).__init__()
        self.asm_num = asm_num
        self.mode = mode
        self.freeze_base = freeze_base
        self.fuse = yesFuse
        self.accom = accom
        self.num_gt_layers = num_gt_layers

        # First creating basic layers
        self.GT_layer = nn.ModuleList(
            [GraphTransformer(in_edge_dim,
                              hidden_edge_dim,
                              in_node_dim,
                              hidden_node_dim,
                              num_heads)])
        self.GT_layer.extend(
            [GraphTransformer(hidden_edge_dim,
                              out_edge_dim,
                              hidden_node_dim,
                              out_edge_dim,
                              num_heads) for i in range(num_gt_layers - 1)])

        self.match = Match(in_edge_feats=in_edge_dim, n_edge_classes=n_edge_class, in_node_feats=in_node_dim,
                           n_node_classes=n_node_class, sigmoid_uncertainty=sigmoid_uncertainty)

        if self.fuse:
            gain = nn.init.calculate_gain('leaky_relu', 0.2)
            self.edge_fc = nn.Sequential(*[
                xavier_init(nn.Linear(in_edge_dim, in_edge_dim * 4, bias=False), gain=gain),
                nn.LeakyReLU(inplace=True, negative_slope=0.2),
                xavier_init(nn.Linear(in_edge_dim * 4, in_edge_dim, bias=False), gain=gain),
                nn.LeakyReLU(inplace=True, negative_slope=0.2)])

            self.node_fc = nn.Sequential(*[
                xavier_init(nn.Linear(in_node_dim, in_node_dim * 4, bias=False), gain=gain),
                nn.LeakyReLU(inplace=True, negative_slope=0.2),
                xavier_init(nn.Linear(in_node_dim * 4, in_node_dim, bias=False), gain=gain),
                nn.LeakyReLU(inplace=True, negative_slope=0.2)])

            self.e_ln1 = nn.LayerNorm(in_edge_dim)
            self.e_ln2 = nn.LayerNorm(in_edge_dim)
            self.n_ln1 = nn.LayerNorm(in_node_dim)
            self.n_ln2 = nn.LayerNorm(in_node_dim)

        if self.freeze_base:
            for i in range(num_gt_layers):
                self.freeze_module(self.GT_layer[i])
            # self.freeze_module(self.match)
            self.freeze_module(self.edge_fc)
            self.freeze_module(self.node_fc)
            self.freeze_module(self.e_ln1)
            self.freeze_module(self.e_ln2)
            self.freeze_module(self.n_ln1)
            self.freeze_module(self.n_ln2)

    @staticmethod
    def freeze_module(module, is_param=False):
        if is_param:
            module.requires_grad = False
        else:
            for param in module.parameters():
                param.requires_grad = False

    def forward(self, init_node_emb, init_edge_emb, head_ind, tail_ind, is_training,
                gt_node_dists, gt_edge_dists, destroy_visual_input=False, keep_inds=None,
                boxes=None):
        edge_class = []
        node_class = []
        blocked_nodes_ind_v = None
        blocked_edges_ind_v = None
        blocked_nodes_ind = None
        blocked_edges_ind = None

        # To compute PKG results we used the following (commented out) snippet. This is not tested with current version.
        # init_node_emb[int(init_node_emb.shape[0]/2):] =
        # torch.zeros_like(init_node_emb[int(init_node_emb.shape[0]/2):], device=init_node_emb.device)
        # init_edge_emb[int(init_edge_emb.shape[0]/2):] =
        # torch.zeros_like(init_edge_emb[int(init_edge_emb.shape[0]/2):], device=init_edge_emb.device)

        if destroy_visual_input and is_training and init_node_emb.shape[0] > 0:
            # Note: Here we spot the nodes/edges that we should drop (exp 4 in paper).
            #       For the selected nodes/edges, 1. we destroy their visual input and 2. by passing the destroy index
            #       to self.match, we request for their direct schema. It will be
            #       returned in the vectors noisy_prior_edge_emb or noisy_prior_node_emb.
            node_mask_v = keep_inds
            blocked_nodes_ind_v = torch.where(node_mask_v == 0)[0]
            # Note: It is just sufficient to make the backward loss zero for non-visual entries at first prop by
            #       passing the indices to blocked inputs. (ref. train_rels.py)
            blocked_nodes_ind = blocked_nodes_ind_v.clone()

            head_blocked_ind = torch.nonzero(head_ind[:, None] == blocked_nodes_ind_v)[:, 0]
            tail_blocked_ind = torch.nonzero(tail_ind[:, None] == blocked_nodes_ind_v)[:, 0]
            blocked_edges_ind_v = torch.cat((head_blocked_ind, tail_blocked_ind))
            edge_mask_v = torch.ones(init_edge_emb.shape[0], device=init_edge_emb.device)
            edge_mask_v[blocked_edges_ind_v] = 0
            blocked_edges_ind_v = torch.where(edge_mask_v == 0)[0]
            blocked_edges_ind = blocked_edges_ind_v.clone()

        # Note: This is to make it more memory efficient for higher asms.
        #       Also it has to refresh at each assimilation.
        visual_node_emb = init_node_emb.clone()  # if (is_training and
        # (i == 0 or self.mode == 'predcls' or self.mode == 'sgcls') and not self.freeze_base)\
        # else init_node_emb.detach()
        visual_edge_emb = init_edge_emb.clone()  # if (is_training
        # and (i == 0 or self.mode == 'predcls' or self.mode == 'sgcls') and not self.freeze_base)\
        # else init_edge_emb.detach()

        if is_training and destroy_visual_input:
            # Remove Visual Embedding of Non-Visual Data! # --- Removed Detaching here (for i==0 shouldn't be)
            visual_node_emb = visual_node_emb * node_mask_v[:, None]
            visual_edge_emb = visual_edge_emb * edge_mask_v[:, None]

        z_edge, z_node = visual_edge_emb, visual_node_emb

        # Propagation (The Reasoning Module)
        for lyr in range(self.num_gt_layers):
            z_node, z_edge = self.GT_layer[lyr](node_emb=z_node,
                                                edge_emb=z_edge,
                                                head_ind=head_ind,
                                                tail_ind=tail_ind)
        # Note: Delta's are the attention values and contain the "schema messages",
        #       and alphas are the attention coefficients and contain the classification results.
        alpha_edge, delta_edge, alpha_node, delta_node = \
            self.match(node_emb=z_node,
                       edge_emb=z_edge,
                       is_training=is_training,
                       gt_node_dists=gt_node_dists,
                       gt_edge_dists=gt_edge_dists,
                       mode=self.mode,
                       node_destroy_index=blocked_nodes_ind_v,
                       edge_destroy_index=blocked_edges_ind_v,
                       boxes=boxes,
                       gt=True)
        edge_class.append(alpha_edge)
        node_class.append(alpha_node)

        if self.asm_num > 1:
            for i in range(self.asm_num - 1):
                # Note: deltas are updated in each loop
                if self.fuse:
                    edge_z_hat = self.e_ln1(delta_edge + visual_edge_emb)
                    f_edge_z_hat = self.edge_fc(edge_z_hat)
                    z_edge = self.e_ln2(f_edge_z_hat + edge_z_hat)

                    node_z_hat = self.n_ln1(delta_node + visual_node_emb)
                    f_node_z_hat = self.node_fc(node_z_hat)
                    z_node = self.n_ln2(f_node_z_hat + node_z_hat)
                else:
                    z_node, z_edge = delta_node, delta_edge
                for lyr in range(self.num_gt_layers):
                    z_node, z_edge = self.GT_layer[lyr](node_emb=z_node,
                                                        edge_emb=z_edge,
                                                        head_ind=head_ind,
                                                        tail_ind=tail_ind)

                alpha_edge, delta_edge, alpha_node, delta_node = \
                    self.match(node_emb=z_node,
                               edge_emb=z_edge,
                               is_training=is_training,
                               gt_node_dists=gt_node_dists,
                               gt_edge_dists=gt_edge_dists,
                               mode=self.mode,
                               node_destroy_index=blocked_nodes_ind_v,
                               edge_destroy_index=blocked_edges_ind_v,
                               boxes=boxes,
                               gt=False)

                edge_class.append(alpha_edge)
                node_class.append(alpha_node)

        return edge_class, node_class, blocked_edges_ind, blocked_nodes_ind
