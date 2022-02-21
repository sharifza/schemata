"""
Training script for scene graph detection. Integrated with my faster rcnn setup
"""

import os
import time

import numpy as np
import pandas as pd
import torch
from tensorboardX import SummaryWriter
from torch import optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import ModelConfig, BOX_SCALE, IM_SCALE
from dataloaders.visual_genome import VGDataLoader, VG
from lib.evaluation.sg_eval import BasicSceneGraphEvaluator
from lib.pytorch_misc import optimistic_restore, clip_grad_norm
from lib.pytorch_misc import print_para
from lib.pytorch_misc import set_random_seed
from misc.inverse_zeroshot import get_triples
from lib.rel_model_sharifza import RelModel

# -- Set random seed
rnd_seed = 13
set_random_seed(rnd_seed)

conf = ModelConfig()


# -- Load Inverse-Zeroshot split if provided
izs_split = None
izs_split_train = None
if conf.izs_file is not None:
    print(f"Loading izs file {conf.izs_file}...")
    izs_file = np.load(conf.izs_file, allow_pickle=True)
    izs_split = izs_file['val_rel'].tolist()
    izs_split_train = izs_file['train_rel'].tolist()

writer = SummaryWriter(comment='_run#' + conf.save_dir.split('/')[-1])
train, val, _ = VG.splits(num_val_im=conf.val_size, filter_duplicate_rels=True,
                          use_proposals=conf.use_proposals,
                          filter_non_overlap=conf.mode == 'sgdet',
                          izs_split=izs_split_train if not conf.destroy_vis else None)

train_loader, val_loader = VGDataLoader.splits(train, val, mode='rel',
                                               batch_size=conf.batch_size,
                                               num_workers=conf.num_workers,
                                               num_gpus=conf.num_gpus)
if conf.izs_file:
    if not conf.destroy_vis:
        izs_split_train = None
        # -- (Debug) Check if the inverse-zeroshot split is working correctly
        print("Checking the inverse-zeroshot split...")
        print("Training set size: ", len(train))
        discarded_triples = izs_file['triples'].tolist()
        kept_filenames = izs_file['filenames_rel'].tolist()
        _, triples_set_train = get_triples(train.relationships, train.gt_classes)
        # -- Double-check remaining triples
        assert (len(discarded_triples.intersection(triples_set_train)) == 0)
        # -- Double-check filenames
        for index, filename in enumerate(train.filenames):
            basename = os.path.basename(filename)
            assert (basename == kept_filenames[index])
    else:
        print("Checking the inverse-zeroshot split...")
        # -- (Debug) Double check if the filtered filenames and izs filenames match
        kept_filenames = izs_file['filenames_rel'].tolist()
        filtered_filenames = [os.path.basename(filename)
                              for index, filename in enumerate(train.filenames)
                              if index not in izs_split_train]
        assert (filtered_filenames == kept_filenames)

detector = RelModel(classes=train.ind_to_classes, rel_classes=train.ind_to_predicates, mode=conf.mode,
                    num_gpus=conf.num_gpus, use_union=conf.use_union, require_overlap_det=True,
                    hidden_dim=conf.hidden_dim, pooling_dim=conf.pooling_dim, use_resnet=conf.use_resnet,
                    use_proposals=conf.use_proposals, use_bias=conf.use_bias, asm_num=conf.asm_num,
                    freeze_base=conf.freeze_base, PKG=conf.PKG, destroy_vis=conf.destroy_vis, izs_split=izs_split_train,
                    izs_vis_discard=conf.izs_vis_discard, yesFuse=conf.yesFuse, hard_att=conf.hard_att,
                    sigmoid_uncertainty=conf.sigmoid_uncertainty, n_drop=conf.n_drop)

# Freeze the detector
for n, param in detector.detector.named_parameters():
    param.requires_grad = False

print(print_para(detector), flush=True)


def get_optimizer(lr):
    """
    Lower the learning rate on the VGG fully connected layers by 1/10th. It's a hack, but it helps
    stabilize the models.
    :param lr: learning rate
    :return: (optimizer, scheduler)
    """
    # Determine fully-connected layers and non fc layers
    fc_params = [p for n, p in detector.named_parameters() if n.startswith('roi_fmap') and p.requires_grad]
    non_fc_params = [p for n, p in detector.named_parameters() if not n.startswith('roi_fmap')]

    # -- Show the number of FC/non-FC parameters
    print("#FC params:{}, #non-FC params:{}".format(len(fc_params),
                                                    len(non_fc_params)))

    params = [{'params': fc_params, 'lr': lr / 10.0}, {'params': non_fc_params}]
    # params = [p for n,p in detector.named_parameters() if p.requires_grad]

    if conf.adam:
        _optimizer = optim.AdamW(params, lr=lr)  # , weight_decay=conf.l2) #weight_decay=conf.l2, lr=lr, eps=1e-3)
    else:
        _optimizer = optim.SGD(params, weight_decay=conf.l2, lr=lr, momentum=0.9)

    _scheduler = ReduceLROnPlateau(_optimizer, 'max', patience=3, factor=0.1,
                                   verbose=True, threshold=0.0001, threshold_mode='abs', cooldown=1)
    return _optimizer, _scheduler


# -- Load the checkpoint if it's provided
start_epoch = -1
if conf.ckpt is not None:
    ckpt = torch.load(conf.ckpt)
    # ckpt = load_model(conf.ckpt)

    # -- If the provided checkpoint is `vg-faster-rcnn`
    if conf.ckpt.endswith("vg-faster-rcnn.tar") or conf.ckpt.endswith("backbone.tar") \
            or conf.ckpt.endswith("vg-24.tar"):
        print("Loading The Backbone Checkpoint...")
        start_epoch = -1
        optimistic_restore(detector.detector, ckpt['state_dict'])
        if conf.use_union:
            detector.roi_fmap[1][0].weight.data.copy_(ckpt['state_dict']['roi_fmap.0.weight'])
            detector.roi_fmap[1][3].weight.data.copy_(ckpt['state_dict']['roi_fmap.3.weight'])
            detector.roi_fmap[1][0].bias.data.copy_(ckpt['state_dict']['roi_fmap.0.bias'])
            detector.roi_fmap[1][3].bias.data.copy_(ckpt['state_dict']['roi_fmap.3.bias'])
        detector.roi_fmap_obj[0].weight.data.copy_(ckpt['state_dict']['roi_fmap.0.weight'])
        detector.roi_fmap_obj[3].weight.data.copy_(ckpt['state_dict']['roi_fmap.3.weight'])
        detector.roi_fmap_obj[0].bias.data.copy_(ckpt['state_dict']['roi_fmap.0.bias'])
        detector.roi_fmap_obj[3].bias.data.copy_(ckpt['state_dict']['roi_fmap.3.bias'])
    # -- Otherwise
    else:
        print("Loading everything...")
        if conf.mode != 'sgdet':  # FIXME: a bit hacky. For fine-tuning sgdet we want to start from zero epoch for Currc
            start_epoch = ckpt['epoch']

        # -- Attach the extra checkpoint if provided
        if conf.extra_ckpt is not None:
            print("Attaching the extra checkpoint to the main one!")
            extra_ckpt_state_dict = torch.load(conf.extra_ckpt)
            ckpt['state_dict'].update(extra_ckpt_state_dict['state_dict'])

        # -- Remove unwanted weights from state_dict
        # if conf.asm_num > 1:
        #     remove_params(ckpt['state_dict'], rm_params)

        # -- Load the checkpoint
        if not optimistic_restore(detector, ckpt['state_dict']):
            start_epoch = -1

detector.cuda()


def train_epoch(epoch_num):
    detector.train()
    detector.epoch_num = epoch_num
    tr = []
    start = time.time()

    for b, batch in enumerate(train_loader):
        tr.append(train_batch(batch, verbose=b % (conf.print_interval * 10) == 0))  # b == 0))

        if b % conf.print_interval == 0 and b >= conf.print_interval:
            mn = pd.concat(tr[-conf.print_interval:], axis=1).mean(1)
            time_per_batch = (time.time() - start) / conf.print_interval
            print("\ne{:2d}b{:5d}/{:5d} {:.3f}s/batch, {:.1f}m/epoch".format(
                epoch_num, b, len(train_loader), time_per_batch, len(train_loader) * time_per_batch / 60))
            print(mn)

            if conf.mode is not 'predcls':
                writer.add_scalar('data/class_loss1', mn.class_loss0, (epoch_num * len(train_loader) + b))
            if conf.asm_num > 1 and not conf.PKG:  # TODO: fix PKG
                writer.add_scalar('data/class_loss2', mn.class_loss1, (epoch_num * len(train_loader) + b))
                writer.add_scalar('data/rel_loss2', mn.rel_loss1, (epoch_num * len(train_loader) + b))
            writer.add_scalar('data/rel_loss1', mn.rel_loss0, (epoch_num * len(train_loader) + b))
            writer.add_scalar('data/total_loss', mn.total, (epoch_num * len(train_loader) + b))

            print('-----------', flush=True)
            start = time.time()
    return pd.concat(tr, axis=1)


def train_batch(b, verbose=False):
    """
    :param b: contains:
          :param imgs: the image, [batch_size, 3, IM_SIZE, IM_SIZE]
          :param all_anchors: [num_anchors, 4] the boxes of all anchors that we'll be using
          :param all_anchor_inds: [num_anchors, 2] array of the indices into the concatenated
                                  RPN feature vector that give us all_anchors,
                                  each one (img_ind, fpn_idx)
          :param im_sizes: a [batch_size, 4] numpy array of (h, w, scale, num_good_anchors) for each image.

          :param num_anchors_per_img: int, number of anchors in total over the feature pyramid per img

          Training parameters:
          :param train_anchor_inds: a [num_train, 5] array of indices for the anchors that will
                                    be used to compute the training loss (img_ind, fpn_idx)
          :param gt_boxes: [num_gt, 4] GT boxes over the batch.
          :param gt_classes: [num_gt, 2] gt boxes where each one is (img_id, class)
    :return:
    """
    result = detector[b]

    losses = {}
    if conf.asm_num == 0:
        losses['class_loss%d' % 0] = F.cross_entropy(result.gat_obj_dist[0], result.rm_obj_labels)
        losses['rel_loss%d' % 0] = F.cross_entropy(result.rel_dists[0], result.rel_labels[:, -1])
    else:  # FIXME Make a better PKG training?
        if conf.PKG:
            range_top = conf.asm_num - 1
        else:
            range_top = conf.asm_num
        for i in range(range_top):
            gt_node_labels = result.rm_obj_labels.clone()
            gt_edge_labels = result.rel_labels[:, -1].clone()
            if i == 0 or conf.PKG:
                if result.blocked_node_ind != None:
                    gt_node_labels[result.blocked_node_ind] = -100
                if result.blocked_edge_ind != None:
                    gt_edge_labels[result.blocked_edge_ind] = -100
            # In SGDet because of the prunning this case might happen. NM used to put a self-connection to avoid
            # this. We just ignore it by putting a zero-loss.
            losses['class_loss%d' % i] = F.cross_entropy(result.gat_obj_dist[i], gt_node_labels)
            if gt_edge_labels.shape[0] == 0:
                losses['rel_loss%d' % i] = torch.zeros_like(losses['class_loss%d' % i],
                                                            device=losses['class_loss%d' % i].device)
            else:
                losses['rel_loss%d' % i] = F.cross_entropy(result.rel_dists[i], gt_edge_labels)
            if conf.mode == 'predcls' and i == 0:
                losses['class_loss%d' % i] *= 0
                losses['rel_loss%d' % i] *= 0
        # torch.nonzero(result.rel_labels[:, -1] != 0) list of non zeros

    loss = sum(losses.values())
    if conf.freeze_base:
        assert (conf.asm_num > 1)
        loss = losses['class_loss1'] + losses['rel_loss1']  # 1 is 2, in writer 2 is 2 :D

    optimizer.zero_grad()
    loss.backward()
    clip_grad_norm(
        [(n, p) for n, p in detector.named_parameters() if p.grad is not None],
        max_norm=conf.clip, verbose=verbose, clip=True)
    losses['total'] = loss
    optimizer.step()
    res = pd.Series({x: y.item() for x, y in losses.items()})
    return res


def val_epoch():
    detector.eval()
    evaluator = BasicSceneGraphEvaluator.all_modes()
    detection_results = []
    # detector.asm_num = 4
    # detector.GAT.asm_num = 4
    for val_b, batch in enumerate(val_loader):
        detection_results.append(detector[batch])
    if conf.allasm:
        asm_num = detector.asm_num
    else:
        asm_num = 1
    for asm in range(1, asm_num + 1):
        print("====ASM%d===" % asm)
        evaluator = BasicSceneGraphEvaluator.all_modes()
        for batch_num, results in enumerate(detection_results):
            val_batch(conf.num_gpus * batch_num, results, evaluator, -asm)
        evaluator[conf.mode].print_stats(epoch, writer, asm if asm_num > 1 else 2)
    # detector.asm_num = conf.asm_num
    # detector.GAT.asm_num = conf.asm_num
    return np.mean(evaluator[conf.mode].result_dict[conf.mode + '_recall'][100])


def val_batch(batch_num, det_res, evaluator, asm):
    # det_res = detector[b]
    if conf.num_gpus == 1:
        det_res = [det_res]

    for i, (boxes_i, objs_i, obj_scores_i, rels_i, pred_scores_i) in enumerate(det_res):
        gt_entry = {
            'gt_classes': val.gt_classes[batch_num + i].copy(),
            'gt_relations': val.relationships[batch_num + i].copy(),
            'gt_boxes': val.gt_boxes[batch_num + i].copy(),
            'izs_idx': izs_split.get(batch_num + i, None) if izs_split is not None else None
        }
        assert np.all(objs_i[asm][rels_i[asm][:, 0]] > 0) and np.all(objs_i[asm][rels_i[asm][:, 1]] > 0)
        # assert np.all(rels_i[:,2] > 0)

        pred_entry = {
            'pred_boxes': boxes_i * BOX_SCALE / IM_SCALE,
            'pred_classes': objs_i[asm],
            'pred_rel_inds': rels_i[asm],
            'obj_scores': obj_scores_i[asm],
            'rel_scores': pred_scores_i[asm],
        }

        evaluator[conf.mode].evaluate_scene_graph_entry(
            gt_entry,
            pred_entry,
        )


print("Training starts now!")
optimizer, scheduler = get_optimizer(conf.lr * conf.num_gpus * conf.batch_size)
for epoch in range(start_epoch + 1, start_epoch + 1 + conf.num_epochs):
    rez = train_epoch(epoch)
    print("overall{:2d}: ({:.3f})\n{}".format(epoch, rez.mean(1)['total'], rez.mean(1)), flush=True)
    if conf.save_dir is not None:
        torch.save({
            'epoch': epoch,
            'state_dict': detector.state_dict(),
            # {k:v for k,v in detector.state_dict().items() if not k.startswith('detector.')},
            # 'optimizer': optimizer.state_dict(),
        }, os.path.join(conf.save_dir, '{}-{}.tar'.format('vgrel', epoch)))

    if epoch % conf.val_iteration == 0:
        mAp = val_epoch()
        scheduler.step(mAp)

    # Stopping early stopping!
    # if any([pg['lr'] <= (conf.lr * conf.num_gpus * conf.batch_size)/99.0 for pg in optimizer.param_groups]):
    #     print("exiting training early", flush=True)
    #     break
