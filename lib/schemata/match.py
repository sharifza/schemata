import torch

from torch import nn
import torch.nn.functional as F
from lib.nms import apply_nms


class Match(nn.Module):
    """
    Apply Attention Between Contextualized Scene Graph Representations and Schemata to match them.
    """

    def __init__(self,
                 in_edge_feats,
                 n_edge_classes,
                 in_node_feats,
                 n_node_classes,
                 hard_att=False,
                 sigmoid_uncertainty=False):
        """
        :param in_edge_feats: edge dim
        :param n_edge_classes: number of predicate classes
        :param in_node_feats: node dim
        :param n_node_classes: number of object classes
        """
        super(Match, self).__init__()
        self._in_edge_feats = in_edge_feats
        self.n_edge_classes = n_edge_classes
        self._in_node_feats = in_node_feats
        self.n_node_classes = n_node_classes
        self.hard_att = hard_att
        self.sigmoid_uncertainty = sigmoid_uncertainty

        self.edges_schema = nn.Parameter(torch.Tensor(in_edge_feats, n_edge_classes))
        self.nodes_schema = nn.Parameter(torch.Tensor(in_node_feats, n_node_classes))

        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('leaky_relu', param=0.2)
        nn.init.xavier_normal_(self.edges_schema, gain=gain)
        nn.init.xavier_normal_(self.nodes_schema, gain=gain)

    def forward(self, node_emb, edge_emb, is_training,
                gt_node_dists, gt_edge_dists,
                gt_node_labels, gt_edge_labels, epoch_num, last_asm, match0, mode, PKG,
                node_destroy_index=None, edge_destroy_index=None, boxes=None, accom=False, gt=False):
        # TODO: Check if these are still required (for predcls?)
        node_ground_schema = False
        edge_ground_schema = False
        # Returns schema of ground truth labels
        if mode == 'predcls':
            node_ground_schema = True
        if mode == 'sgcls' and PKG:
            node_ground_schema = True
            edge_ground_schema = True
        raw_edge_class, h_edge_emb = \
            self.send_message_kg2sg(feat=edge_emb,
                                    schema=self.edges_schema,
                                    is_training=is_training,
                                    gt_dist=gt_edge_dists,
                                    gt_label=gt_edge_labels,
                                    epoch_num=epoch_num,
                                    last_asm=last_asm,
                                    gt_schema=gt,
                                    destroy_index=edge_destroy_index,
                                    mode=mode,
                                    hard_att=self.hard_att,
                                    sigmoid_uncertainty=self.sigmoid_uncertainty,
                                    accom=accom,
                                    KG_emb=None
                                    )
        raw_node_class, h_node_emb = \
            self.send_message_kg2sg(feat=node_emb,
                                    schema=self.nodes_schema,
                                    is_training=is_training,
                                    gt_dist=gt_node_dists,
                                    gt_label=gt_node_labels,
                                    epoch_num=epoch_num,
                                    last_asm=last_asm,
                                    gt_schema=gt,
                                    destroy_index=node_destroy_index,
                                    boxes=boxes,
                                    mode=mode,
                                    hard_att=self.hard_att,
                                    sigmoid_uncertainty=self.sigmoid_uncertainty,
                                    accom=accom,
                                    KG_emb=None
                                    )

        return raw_edge_class, h_edge_emb, raw_node_class, h_node_emb

    @staticmethod
    def send_message_kg2sg(feat: torch.Tensor, schema: torch.Tensor, is_training: bool,
                           gt_dist, gt_label, mode, teacherForce=False,
                           curriculum=True,
                           epoch_num=0, inverse_temp=False,
                           last_asm=False, att_type='dot', gt_schema=False, destroy_index=None, boxes=None,
                           hard_att=False, sigmoid_uncertainty=False, accom=False, KG_emb=None):

        raw_att = feat @ schema
        if mode == 'predcls' and gt_schema:
            att = (torch.clone(gt_dist))
            # ignore background (=class 0) mask (schema of class 0 does not contain any information)
            if att.shape[1] != 151:
                edge_mask_s = torch.zeros(att.shape[0], device=att.device)
                att = att * edge_mask_s[:, None]
        else:
            if is_training:
                # if epoch_num < 3:
                #     att = (torch.clone(gt_dist))
                # else:
                #     att = F.softmax(raw_att, dim=1)
                att = (torch.clone(gt_dist))
            else:
                att = F.softmax(raw_att, dim=1)
            if mode == 'sgdet':
                # Remove Schema Message of Non-Max Schema Winner
                nms_mask = torch.zeros_like(att)
                for c_i in range(1, att.size(1)):
                    scores_ci = att[:, c_i]
                    boxes_ci = boxes[:, c_i]
                    keep = apply_nms(scores_ci, boxes_ci,
                                     pre_nms_topn=scores_ci.size(0), post_nms_topn=scores_ci.size(0),
                                     nms_thresh=0.3)
                    nms_mask[:, c_i][keep] = 1

                with torch.no_grad():
                    att *= nms_mask

        if destroy_index is not None:
            assert is_training
            if att.shape[1] == 151:
                att = att.index_copy(0, destroy_index, (torch.clone(gt_dist)[destroy_index]))
                if mode != 'predcls':
                    nodes_to_drop = torch.randint(0, destroy_index.shape[0], (int(destroy_index.shape[0] * 0.2),),
                                                  device=att.device)
                    att[destroy_index[nodes_to_drop]] *= 0
            elif gt_schema:
                edge_mask = torch.zeros(destroy_index.shape[0], device=att.device)
                att[destroy_index] = att[destroy_index] * edge_mask[:, None]
        else:  # FIXME: Not tested with PredCls
            if att.shape[1] != 151:
                edge_mask = torch.zeros(att.shape[0], device=att.device)
                att = att * edge_mask[:, None]

        schema_msg = att.detach() @ (torch.transpose(schema, 0, 1))

        return raw_att, schema_msg.detach()
