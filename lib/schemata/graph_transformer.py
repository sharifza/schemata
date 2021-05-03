import torch

from torch import nn
import torch.nn.functional as F
from lib.schemata.misc import sparse_softmax


class GraphTransformer(nn.Module):
    """
    An implementation of the graph transformer from [sharifzadeh2021]
    The objects from this class apply graph transformer over a graph with embeddings for each node and edge
    """

    def __init__(self,
                 in_edge_feats,
                 out_edge_feats,
                 in_node_feats,
                 out_node_feats,
                 num_heads
                 ):
        """
        :param in_edge_feats: input dimension for edge features
        :param out_edge_feats: output dimension for edge features
        :param in_node_feats: input dimension for node features
        :param out_node_feats: output dimension for node features
        :param num_heads: number of attention heads
        """
        super(GraphTransformer, self).__init__()
        self._num_heads = num_heads
        self._in_edge_feats = in_edge_feats
        self._out_edge_feats = out_edge_feats
        self._in_node_feats = in_node_feats
        self._out_node_feats = out_node_feats

        # Linear layers inside Graph Attention
        self.W_sr = nn.Linear(in_edge_feats, out_edge_feats)
        self.W_or = nn.Linear(in_edge_feats, out_edge_feats)

        self.W_rs = nn.Linear(in_node_feats, out_node_feats)
        self.W_ro = nn.Linear(in_node_feats, out_node_feats)

        # Attention linear layers
        self.node2edge_att_param = nn.Linear(in_node_feats + in_edge_feats, num_heads)
        self.edge2node_att_param = nn.Linear(in_node_feats + in_edge_feats, 1)

        # LayerNorm layers
        self.e_ln1 = nn.LayerNorm(in_edge_feats)
        self.e_ln2 = nn.LayerNorm(in_edge_feats)
        self.n_ln1 = nn.LayerNorm(in_node_feats)
        self.n_ln2 = nn.LayerNorm(in_node_feats)

        # Linear layers
        self.e_l1 = nn.Linear(in_edge_feats, in_edge_feats * 4)
        self.e_l2 = nn.Linear(in_edge_feats * 4, in_edge_feats)
        self.n_l1 = nn.Linear(in_node_feats, in_node_feats * 4)
        self.n_l2 = nn.Linear(in_node_feats * 4, in_node_feats)

        # Dropout layers (currently disabled)
        self.edge_drop = nn.Dropout(0.0)
        self.edge_drop2 = nn.Dropout(0.0)
        self.head_drop = nn.Dropout(0.0)
        self.tail_drop = nn.Dropout(0.0)
        self.node_drop = nn.Dropout(0.0)
        self.attn_drop = nn.Dropout(0.0)

        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('leaky_relu', 0.2)
        nn.init.xavier_normal_(self.W_or.weight, gain=gain)
        nn.init.xavier_normal_(self.W_sr.weight, gain=gain)
        nn.init.xavier_normal_(self.W_rs.weight, gain=gain)
        nn.init.xavier_normal_(self.W_ro.weight, gain=gain)
        nn.init.xavier_normal_(self.node2edge_att_param.weight, gain=gain)
        nn.init.xavier_normal_(self.edge2node_att_param.weight, gain=gain)
        nn.init.xavier_normal_(self.e_l1.weight, gain=gain)
        nn.init.xavier_normal_(self.e_l2.weight, gain=gain)
        nn.init.xavier_normal_(self.n_l1.weight, gain=gain)
        nn.init.xavier_normal_(self.n_l2.weight, gain=gain)

    def forward(self, node_emb, edge_emb, head_ind, tail_ind):
        """
        :param node_emb: shape: (n_nodes, d). The embeddings for nodes of a graph.
        :param edge_emb: shape: (n_edges, d). The embeddings for edges of a graph.
        :param head_ind: (n_edges,). The list of heads' node indices for each relation.
        :param tail_ind: (n_edges,). The list of tails' node indices for each relation.
        :return: (n_nodes, d), (n_edges, d). the updated node and edge embeddings
        """
        #: (n_nodes, d)
        feat_n = node_emb
        #: (n_edges, d)
        feat_e = edge_emb
        # Multi-head Attention for Edge Update
        new_feat_e = self.send_message_node_to_edge(feat_e=feat_e, feat_n=feat_n, head_indices=head_ind,
                                                    tail_indices=tail_ind, head_transformer=self.W_rs,
                                                    tail_transformer=self.W_ro,
                                                    att_transformer=self.edge2node_att_param)
        # Dropout
        new_feat_e = self.edge_drop(new_feat_e)
        # Residual Connection
        edge_z_hat = self.e_ln1(new_feat_e + feat_e)
        f_edge_z_hat = F.leaky_relu(self.e_l1(edge_z_hat), negative_slope=0.2)
        f_edge_z_hat = F.leaky_relu(self.edge_drop2(self.e_l2(f_edge_z_hat)), negative_slope=0.2)
        new_edge_emb = self.e_ln2(f_edge_z_hat + edge_z_hat)

        # Multihead Attention for Head Update
        new_head_emb = self.send_message_edge_to_node(feat_e=feat_e, feat_n=feat_n, node_indices=head_ind,
                                                      edge_transformer=self.W_sr,
                                                      att_transformer=self.node2edge_att_param)
        # Dropout
        new_head_emb = self.head_drop(new_head_emb)
        # Multi-head Attention for Tail Update
        new_tail_emb = self.send_message_edge_to_node(feat_e=feat_e, feat_n=feat_n, node_indices=tail_ind,
                                                      edge_transformer=self.W_or,
                                                      att_transformer=self.node2edge_att_param)
        # Dropout
        new_tail_emb = self.tail_drop(new_tail_emb)
        # Residual Connection and full node update
        node_z_hat = self.n_ln1(new_head_emb + new_tail_emb + feat_n)
        f_node_z_hat = F.leaky_relu(self.node_drop(self.n_l2(
            F.leaky_relu(self.n_l1(node_z_hat), negative_slope=0.2))), negative_slope=0.2)
        new_node_emb = self.n_ln2(f_node_z_hat + node_z_hat)

        return new_node_emb, new_edge_emb

    @staticmethod
    def send_message_edge_to_node(feat_e: torch.Tensor, feat_n: torch.Tensor,
                                  edge_transformer: nn.Module, att_transformer: nn.Module,
                                  node_indices: torch.LongTensor):
        """
        An efficient implementation of attentional message propagation from edge to node.
        """
        # attention: node to outgoing edge
        # select head nodes: (n_edges, d)
        head_node = feat_n.index_select(dim=0, index=node_indices)
        edge_feat_transform = edge_transformer(feat_e)
        # shape: (n_edges, num_heads)
        att = F.leaky_relu(att_transformer(torch.cat([head_node, edge_feat_transform], dim=-1)), negative_slope=0.2)
        att = sparse_softmax(att, index=node_indices)
        #: shape: (n_edges, num_heads, 1)
        att = att.unsqueeze(dim=-1)
        #: shape: (n_edges, 1, d)
        head_trans_edge_feat = edge_feat_transform.unsqueeze(dim=-2)
        # shape: (n_edges, num_heads, d)
        edge_messages = att * head_trans_edge_feat
        num_heads = edge_messages.shape[1]
        # shape: (n_edges, d)
        edge_messages = edge_messages.sum(dim=1) / num_heads
        # aggregate by node
        assert edge_messages.shape[0] == node_indices.shape[0]
        # Allocate all-zero output accumulator
        accumulator = torch.zeros_like(feat_n)
        # sum messages per node
        new_node_emb = accumulator.index_add_(dim=0, index=node_indices, source=edge_messages)
        return new_node_emb

    @staticmethod
    def send_message_node_to_edge(feat_e: torch.Tensor, feat_n: torch.Tensor,
                                  head_transformer: nn.Module, tail_transformer: nn.Module, att_transformer: nn.Module,
                                  head_indices: torch.LongTensor, tail_indices: torch.LongTensor):
        """
        An efficient implementation of attentional message propagation from node to edge.
        """
        # attention: node to outgoing edge
        # select head nodes: (n_edges, d)
        head_node = feat_n.index_select(dim=0, index=head_indices)
        trans_head_feat = head_transformer(head_node)
        # select tail nodes: (n_edges, d)
        tail_node = feat_n.index_select(dim=0, index=tail_indices)
        trans_tail_feat = tail_transformer(tail_node)
        # shape: (n_edges, num_heads)
        head_att = F.leaky_relu(att_transformer(torch.cat([feat_e, trans_head_feat], dim=-1)), negative_slope=0.2)
        # shape: (n_edges, num_tails)
        tail_att = F.leaky_relu(att_transformer(torch.cat([feat_e, trans_tail_feat], dim=-1)), negative_slope=0.2)
        # shape: (2, n_edges, 1)
        att = F.softmax(torch.stack((head_att, tail_att)), dim=0)
        # shape: (2, n_edges, d)
        out = att * torch.stack((trans_head_feat, trans_tail_feat))
        # shape: (n_edges, d)
        new_edge_emb = out.sum(dim=0)
        return new_edge_emb
