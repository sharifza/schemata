"""
Let's get the relationships yo
"""

import torch
import torch.nn as nn
import torch.nn.parallel
from torch.autograd import Variable
from torch.nn import functional as F
from lib.nms import apply_nms


from lib.schemata.assimilation import Assimilation
from lib.fpn.box_utils import bbox_overlaps, center_size
from lib.fpn.proposal_assignments.rel_assignments import rel_assignments
from torchvision.ops import RoIAlign
from lib.get_union_boxes import UnionBoxesAndFeats
from lib.object_detector import ObjectDetector, gather_res, load_vgg
from lib.pytorch_misc import to_onehot, arange, diagonal_inds, \
    Flattener, xavier_init
from lib.resnet import resnet_l4
from lib.surgery import filter_dets


class ScaleLayer(nn.Module):
    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


MODES = ('sgdet', 'sgcls', 'predcls')


class RelModel(nn.Module):
    """
    RELATIONSHIPS
    """

    # -- (CHANGED) changed the function's signature to support data fusion modes and depth models
    def __init__(self, classes,
                 rel_classes,
                 mode='sgdet',
                 num_gpus=1,
                 use_union=False,
                 require_overlap_det=True,
                 embed_dim=200,
                 hidden_dim=256,
                 pooling_dim=2048,
                 use_resnet=False,
                 thresh=0.01,
                 use_proposals=False,
                 use_bias=True,
                 asm_num=2,
                 freeze_base=False,
                 PKG=False,
                 vis_num=None,
                 destroy_vis=False,
                 allasm=False,
                 izs_split=None,
                 izs_vis_discard=False,
                 yesFuse=False,
                 hard_att=False,
                 sigmoid_uncertainty=False,
                 n_drop=False):

        """
        # TODO: update the param list description!
        :param classes: Object classes
        :param rel_classes: Relationship classes. None if were not using rel mode
        :param mode: (sgcls, predcls, or sgdet)
        :param num_gpus: how many GPUS 2 use
        :param use_union: Whether to use vision in the final product
        :param require_overlap_det: Whether two objects must intersect
        :param embed_dim: Dimension for all embeddings
        :param hidden_dim: LSTM hidden size
        :param obj_dim:
        """
        super(RelModel, self).__init__()
        self.classes = classes
        self.rel_classes = rel_classes
        self.num_gpus = num_gpus
        assert mode in MODES
        self.mode = mode
        self.epoch_num = None

        self.pooling_size = 7
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.obj_dim = 2048 if use_resnet else 4096
        self.pooling_dim = pooling_dim
        self.vis_num = vis_num
        self.destroy_vis = destroy_vis
        self.asm_num = asm_num
        self.PKG = PKG

        self.allasm=allasm
        self.use_bias = use_bias
        self.NM_bias = False
        self.PKG_bias = True
        self.use_union = use_union
        self.require_overlap = require_overlap_det and self.mode == 'sgdet'
        self.get_gt_pred = False
        if self.PKG:
            self.get_gt_pred = True
        self.izs_split = izs_split
        self.izs_vis_discard = izs_vis_discard
        self.n_drop = n_drop

        gain = nn.init.calculate_gain('leaky_relu', 0.2)

        # -- Define faster R-CNN network and it's related feature extractors
        self.detector = ObjectDetector(
            classes=classes,
            mode=('proposals' if use_proposals else 'refinerels') if mode == 'sgdet' else 'gtbox',
            use_resnet=use_resnet,
            thresh=thresh,
            max_per_img=64,
            get_gt_pred=self.get_gt_pred
        )
        self.rePN = False

        if self.use_union:
            self.union_boxes = UnionBoxesAndFeats(pooling_size=self.pooling_size, stride=16,
                                                  dim=1024 if use_resnet else 512)

        self.roi_fmap_obj = load_vgg(pretrained=False).classifier

        # -- Define different components' feature size
        features_dim = {'visual': self.hidden_dim,
                        'location': 20}
        self.location_size = 8
        self.node_emb_dim = features_dim['visual']
        self.node_hidden_dim = self.node_emb_dim
        self.edge_hidden_dim = self.node_emb_dim

        if self.asm_num > 0:
            self.asm = Assimilation(in_edge_dim=self.node_emb_dim,
                       hidden_edge_dim=self.edge_hidden_dim,
                       out_edge_dim=self.node_emb_dim,
                       in_node_dim=self.node_emb_dim,
                       hidden_node_dim=self.node_hidden_dim,
                       num_heads=5,
                       n_edge_class=self.num_rels,
                       n_node_class=self.num_classes,
                       asm_num=asm_num, mode=self.mode,
                       freeze_base=freeze_base,
                       yesFuse=yesFuse, hard_att=hard_att, sigmoid_uncertainty=sigmoid_uncertainty).cuda()

        # -- RGB visual features
        self.visual_hlayer = nn.Sequential(*[
            xavier_init(nn.Linear(self.obj_dim, self.node_emb_dim),gain=gain),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            nn.Dropout(0.8)
        ])

        # -- Location features
        self.location_hlayer = nn.Sequential(*[
            xavier_init(nn.Linear(self.location_size, features_dim['location']),gain=gain),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            nn.Dropout(0.1)
        ])
        self.location_scale = ScaleLayer(1.0)

        if self.use_union:
            edge_emb_dim = self.pooling_dim
            if use_resnet:
                self.roi_fmap = nn.Sequential(
                    resnet_l4(relu_end=False),
                    nn.AvgPool2d(self.pooling_size),
                    Flattener(),
                )
            else:
                roi_fmap = [
                    Flattener(),
                    load_vgg(use_dropout=False, use_relu=False, use_linear=pooling_dim == 4096, pretrained=False).classifier,
                ]
                if pooling_dim != 4096:
                    roi_fmap.append(nn.Linear(4096, pooling_dim))
                self.roi_fmap = nn.Sequential(*roi_fmap)
        else:
            edge_emb_dim = features_dim['location']

        self.edge_transform = nn.Sequential(*[
            xavier_init(nn.Linear(edge_emb_dim, self.node_emb_dim), gain=gain),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            nn.Dropout(0.1)])

        if freeze_base:
            self.freeze_module(self.edge_transform)
            self.freeze_module(self.location_scale)
            self.freeze_module(self.location_hlayer)
            self.freeze_module(self.visual_hlayer)
            self.freeze_module(self.roi_fmap_obj)
            self.freeze_module(self.detector)
            if self.use_union:
                self.freeze_module(self.union_boxes)
                self.freeze_module(self.roi_fmap)

    # -- (ADDED) freeze the given module
    @staticmethod
    def freeze_module(module):
        for param in module.parameters():
            param.requires_grad = False

    @staticmethod
    def unfreeze_module(module):
        for param in module.parameters():
            param.requires_grad = True

    @property
    def num_classes(self):
        return len(self.classes)

    @property
    def num_rels(self):
        return len(self.rel_classes)

    def get_rel_inds(self, rel_labels, im_inds, box_priors):
        # Get the relationship candidates
        prunned_inds = None
        if self.training or self.get_gt_pred:
            rel_inds = rel_labels[:, :3].data.clone()
        else:
            rel_cands = im_inds.data[:, None] == im_inds.data[None]
            rel_cands.view(-1)[diagonal_inds(rel_cands)] = 0

            # Require overlap for detection
            if self.require_overlap:
                rel_cands = rel_cands & (bbox_overlaps(box_priors.data,
                            box_priors.data) > 0)

            # # if there are fewer then 100 things then we might as well add some?
            # amt_to_add = 100 - rel_cands.long().sum()

            rel_cands = rel_cands.nonzero()
            if rel_cands.dim() == 0:
                rel_cands = im_inds.data.new(1, 2).fill_(0)

            rel_inds = torch.cat((im_inds.data[rel_cands[:, 0]][:, None], rel_cands), 1)
            # (im_inds.shape[0]-1) is because the self rels are removed (by doing rel_cands = rel_cands.nonzero())
            # prunned_inds = prunned_cands[:, 0] * (im_inds.shape[0]-1) + prunned_cands[:, 1]
        return rel_inds, prunned_inds

    def get_roi_features(self, features, rois):
        """
        Gets the ROI features
        :param features: [batch_size, dim, IM_SIZE/4, IM_SIZE/4] (features at level p2)
        :param rois: [num_rois, 5] array of [img_num, x0, y0, x1, y1].
        :return: [num_rois, #dim] array
        """
        feature_pool = RoIAlign((self.pooling_size, self.pooling_size), spatial_scale=1 / 16, sampling_ratio=-1)(
            features, rois)
        return self.roi_fmap_obj(feature_pool.view(rois.size(0), -1))

    def get_union_features(self, features, rois, pair_inds):
        """
        Classify the features
        :param features: [batch_size, dim, IM_SIZE/4, IM_SIZE/4]
        :param rois: [num_rois, 5] array of [img_num, x0, y0, x1, y1].
        :param pair_inds inds to use when predicting
        :return: score_pred, a [num_rois, num_classes] array
                 box_pred, a [num_rois, num_classes, 4] array
        """
        assert pair_inds.size(1) == 2
        uboxes = self.union_boxes(features, rois, pair_inds)
        return self.roi_fmap(uboxes)

    @staticmethod
    def get_loc_features(boxes, subj_inds, obj_inds):
        """
        Calculate the scale-invariant location feature
        :param boxes:
        :param subj_inds:
        :param obj_inds:
        :return:
        """
        boxes_centered = center_size(boxes.data)

        # -- Determine box's center and size (Subject's box)
        center_subj = boxes_centered[subj_inds][:, 0:2]
        size_subj = boxes_centered[subj_inds][:, 2:4]

        # -- Determine box's center and size (Object's box)
        center_obj = boxes_centered[obj_inds][:, 0:2]
        size_obj = boxes_centered[obj_inds][:, 2:4]

        # -- Calculate the scale-invariant location features of the subjects
        t_coord_subj = (center_subj - center_obj) / size_obj
        t_size_subj = torch.log(size_subj / size_obj)

        # -- Calculate the scale-invariant location features of the objects
        t_coord_obj = (center_obj - center_subj) / size_subj
        t_size_obj = torch.log(size_obj / size_subj)

        # -- Put everything together
        location_feature = Variable(torch.cat((t_coord_subj, t_size_subj,
                                               t_coord_obj, t_size_obj), 1))

        return location_feature

    def forward(self, x, im_sizes, image_offset,
                gt_boxes=None, gt_classes=None, gt_rels=None, proposals=None, train_anchor_inds=None,
                return_fmap=False, depth_imgs=None):
        """
        Forward pass for detection
        :param x: Images@[batch_size, 3, IM_SIZE, IM_SIZE]
        :param im_sizes: A numpy array of (h, w, scale) for each image.
        :param image_offset: Offset onto what image we're on for MGPU training (if single GPU this is 0)
        :param gt_boxes:

        Training parameters:
        :param gt_boxes: [num_gt, 4] GT boxes over the batch.
        :param gt_classes: [num_gt, 2] gt boxes where each one is (img_id, class)
        :param train_anchor_inds: a [num_train, 2] array of indices for the anchors that will
                                  be used to compute the training loss. Each (img_ind, fpn_idx)
        :return: If train:
            scores, boxdeltas, labels, boxes, boxtargets, rpnscores, rpnboxes, rellabels

            if test:
            prob dists, boxes, img inds, maxscores, classes

        """
        # # -- Discard the visual input of IZS images
        # if self.izs_vis_discard and self.destroy_vis \
        #         and self.izs_split and self.training:
        #     global_idx = im_sizes[:, 3]
        #     discard_idx = np.where(global_idx[:, None] == self.izs_split)[0]
        #     x[discard_idx] = 0

        result = self.detector(x, im_sizes, image_offset, gt_boxes, gt_classes, gt_rels, proposals,
                               train_anchor_inds, return_fmap=True)

        if result.is_none():
            return ValueError("heck")

        im_inds = result.im_inds - image_offset
        boxes = result.rm_box_priors

        if self.training and result.rel_labels is None:
            assert self.mode == 'sgdet'
            result.rel_labels = rel_assignments(im_inds.data, boxes.data, result.rm_obj_labels.data,
                                                gt_boxes.data, gt_classes.data, gt_rels.data,
                                                image_offset, filter_non_overlap=True,
                                                num_sample_per_gt=1)

        rel_inds, prunned_inds = self.get_rel_inds(result.rel_labels, im_inds, boxes)
        rois = torch.cat((im_inds[:, None].float(), boxes), 1)
        # TODO:
        #  1. Replicate sgcls and predcls for "IZS"
        #  2. Write a cleaner match function because now with the teacher forcing it is easier to write instead of
        #  using the destroy_ind
        #  3. Speed up predcls by skipping the forward passes somehow
        # Neural motifs backbone does NOT connect all entities to each other. They have a prunning step that removes
        # some connections (based on non-overlap and other sampling strategies they have which have shown promising).
        # But in our graph transformer we want a message propagation between ALL nodes. Therefore, here we
        # first create a fully connected adjacency matrix so everyone can attend to everyone else.
        # Then we find a mapping to the prunned entities by neural motifs.
        is_cand = (im_inds[:, None] == im_inds[None])
        # Self connections are handled by the residual connections in Assimilation class itself.
        is_cand.view(-1)[diagonal_inds(is_cand)] = 0
        all_rels = is_cand.nonzero()
        head_inds = all_rels[:, 0]
        tail_inds = all_rels[:, 1]
        # Map every prediction to the prunned graph that Neural Motifs backbone provides.
        ind_mapping = \
            torch.where((rel_inds[:, 1][:, None] == all_rels[:, 0]) & (rel_inds[:, 2][:, None] == all_rels[:, 1]))[1]

        # Some other gt label initializations
        if result.rel_labels is None:
            gt_edge_labels = None
            gt_edge_dists = None
        else:
            # gt_edge_labels = result.rel_labels[:, -1]
            gt_edge_labels = torch.zeros_like(all_rels[:, 0], device=all_rels.device)
            gt_edge_labels[ind_mapping] = result.rel_labels[:, -1]
            gt_edge_dists = F.one_hot(gt_edge_labels, num_classes=self.num_rels).float()
        if result.rm_obj_labels is None:
            gt_node_dists = None
        else:
            gt_node_dists = F.one_hot(result.rm_obj_labels.data, self.num_classes).float()

        # Feed the extracted features from the conv layers to the last VGG/ResNet layers. Here, only
        # the last 3 layers of VGG/ResNet are being trained. Everything else (in self.detector) is frozen
        result.obj_fmap = self.get_roi_features(result.fmap.detach(), rois)
        if self.use_union:
            # WARNING: Use_union (by Zellers et al) was not tested/used in Schemata.
            # edge_init_features = self.get_union_features(result.fmap.detach(), rois, rel_inds[:, 1:])
            edge_init_features = self.get_union_features(result.fmap.detach(), rois, all_rels)  # CHECK
        else:
            # Initiate edge features using location-based vectors
            location = self.get_loc_features(boxes, head_inds, tail_inds)
            location_fc1 = self.location_hlayer(location)
            edge_init_features = self.location_scale(location_fc1)

        # Assign features to nodes and edges after a few more transformations
        # Because of our GPU memory limit we definitely have to do this so that nodes/edges have smaller dims
        node_features = self.visual_hlayer(result.obj_fmap)
        edge_features = self.edge_transform(edge_init_features)

        # Contextualize, Assimilate and predict
        result.rel_dists, result.gat_obj_dist, result.blocked_edge_ind, result.blocked_node_ind \
            = self.asm(init_node_emb=node_features,
                       init_edge_emb=edge_features,
                       head_ind=head_inds,
                       tail_ind=tail_inds,
                       is_training=self.training,
                       gt_node_dists=gt_node_dists,
                       gt_edge_dists=gt_edge_dists,
                       destroy_visual_input=self.destroy_vis,
                       keep_inds=gt_classes[:, 2],
                       boxes=result.boxes_all
                       )
        if self.mode == 'predcls':
            # replace with one-hot distribution of ground truth labels with the predictions
            result.rm_obj_dists = F.one_hot(result.rm_obj_labels.data, self.num_classes).float()
            result.rm_obj_dists = result.rm_obj_dists * 1000 + (1 - result.rm_obj_dists) * (-1000)
            # if not self.training: #TODO: Allowing the schemas to adapt. Remember asm1 loss should be zeroed out.
            result.gat_obj_dist = [result.rm_obj_dists]
            for i in range(self.asm_num):  # Why did it have -1?
                result.gat_obj_dist.append(result.rm_obj_dists)

        for asm in range(self.asm_num):
            result.rel_dists[asm] = result.rel_dists[asm][ind_mapping]
        if result.blocked_edge_ind is not None:
            result.blocked_edge_ind = torch.where(result.blocked_edge_ind[:, None] == ind_mapping)[1]

        if self.training:
            return result
        # During evaluation ...:
        # The last obj_pred is the first asm so we need to reverse.
        rel_rep = []
        result.obj_scores = []
        result.obj_preds = []
        # Map the prediction to the relevant ones on the sparse graph that NM framework provides.
        # Need to handle the assimilating order of results.
        for asm in range(1, self.asm_num + 1):
            result.obj_scores.append(F.softmax(result.gat_obj_dist[-asm], dim=1))  # one2one mapping
            rel_rep.append(F.softmax(result.rel_dists[-asm], dim=1))
            # In SGDet, lets apply NMS again like NM does.
            nms_mask = torch.ones_like(result.gat_obj_dist[-asm].data)
            if not self.training and self.mode == 'sgdet':
                nms_mask = torch.zeros_like(result.gat_obj_dist[-asm].data)
                for c_i in range(1, result.gat_obj_dist[-asm].size(1)):
                    scores_ci = result.obj_scores[-1][:, c_i]
                    boxes_ci = result.boxes_all[:, c_i]
                    keep = apply_nms(scores_ci, boxes_ci,
                                     pre_nms_topn=scores_ci.size(0), post_nms_topn=scores_ci.size(0),
                                     nms_thresh=0.3)
                    nms_mask[:, c_i][keep] = 1

            # In other settings NMS mask is just ones here. In predcls result.gat_obj_dist is GT.
            result.obj_scores[-1] = nms_mask * result.obj_scores[-1]
            if self.mode == 'predcls':
                result.obj_preds.append(result.rm_obj_labels)
            else:
                result.obj_preds.append(result.obj_scores[-1][:, 1:].max(1)[1] + 1)
            twod_inds = arange(result.obj_preds[-1].data) * self.num_classes + result.obj_preds[-1].data
            result.obj_scores[-1] = result.obj_scores[-1].view(-1)[twod_inds]
            # During training we need only the latest asm output. So stop the loop.
            if not self.allasm or self.training:
                break
        # # This is the only list with the reverse order. Now the last rel_rel is the last asm.
        rel_rep.reverse()
        result.obj_scores.reverse()
        result.obj_preds.reverse()

        # Bbox regression
        if self.mode == 'sgdet':
            bboxes=result.boxes_all.view(-1, 4)[twod_inds].view(result.boxes_all.size(0), 4)
        else:
            # Boxes will get fixed by filter_dets function.
            bboxes=result.rm_box_priors

        if not self.allasm:
            asm_num = 1
        else:
            asm_num = self.asm_num
        # Filtering: Subject_Score * Pred_score * Obj_score, sorted and ranked
        return filter_dets(bboxes, result.obj_scores,
                           result.obj_preds, rel_inds[:, 1:], rel_rep, asm_num)

    def __getitem__(self, batch):
        """ Hack to do multi-GPU training"""
        batch.scatter()
        if self.num_gpus == 1:
            return self(*batch[0])

        replicas = nn.parallel.replicate(self, devices=list(range(self.num_gpus)))
        outputs = nn.parallel.parallel_apply(replicas, [batch[i] for i in range(self.num_gpus)])

        if self.training:
            return gather_res(outputs, 0, dim=0)
        return outputs

    def enable_dropout(self):
        for mm in self.modules():
            if mm.__class__.__name__.startswith('Dropout'):
                mm.train()
