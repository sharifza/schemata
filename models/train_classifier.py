"""
Training script for Classification
"""
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from pycocotools.cocoeval import COCOeval
from torch import optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import ModelConfig, IM_SCALE, BOX_SCALE
from dataloaders.mscoco import CocoDetection, CocoDataLoader
from dataloaders.visual_genome import VGDataLoader, VG
from lib.object_detector import ObjectDetector
from lib.pytorch_misc import optimistic_restore, clip_grad_norm

ROOT_PATH = os.path.dirname(os.path.realpath(__file__))
cudnn.benchmark = True
conf = ModelConfig()

izs_split_train = None
if conf.izs_file is not None:
    print(f"Loading izs file {conf.izs_file}...")
    izs_file = np.load(conf.izs_file, allow_pickle=True)
    izs_split_train = izs_file['train_det'].tolist()

if conf.coco:
    train, val = CocoDetection.splits()
    val.ids = val.ids[:conf.val_size]
    train.ids = train.ids
    train_loader, val_loader = CocoDataLoader.splits(train, val, batch_size=conf.batch_size,
                                                     num_workers=conf.num_workers,
                                                     num_gpus=conf.num_gpus)
else:
    train, val, _ = VG.splits(num_val_im=conf.val_size, filter_non_overlap=False,
                              filter_empty_rels=False, use_proposals=conf.use_proposals,
                              izs_split=izs_split_train if not conf.destroy_vis else None)
    train_loader, val_loader = VGDataLoader.splits(train, val, batch_size=conf.batch_size,
                                                   num_workers=conf.num_workers,
                                                   num_gpus=conf.num_gpus)

# -- (Debug) Double check if the filtered filenames and izs filenames match
if conf.izs_file:
    print("Checking the inverse-zeroshot split...")
    print("Training set size: ", len(train))
    kept_filenames = izs_file['filenames_det'].tolist()
    filenames = [os.path.basename(filename)
                 for filename in train.filenames]
    assert (filenames == kept_filenames)

detector = ObjectDetector(classes=train.ind_to_classes, mode='gtbox',
                          num_gpus=conf.num_gpus, use_resnet=conf.use_resnet)
detector.cuda()

# Note: if you're doing the stanford setup, you'll need to change this to freeze the lower layers
if conf.use_proposals:
    for n, param in detector.named_parameters():
        if n.startswith('features'):
            param.requires_grad = False
#
# for n, param in detector.named_parameters():
#     if n.startswith('features'):
#         param.requires_grad = False

if conf.adam:
    optimizer = optim.Adam([p for p in detector.parameters() if p.requires_grad], lr=conf.lr* conf.num_gpus * conf.batch_size, weight_decay=conf.l2) #weight_decay=conf.l2, lr=lr, eps=1e-3)
else:
    optimizer = optim.SGD([p for p in detector.parameters() if p.requires_grad],
                      weight_decay=conf.l2, lr=conf.lr * conf.num_gpus * conf.batch_size, momentum=0.9)
scheduler = ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.1,
                              verbose=True, threshold=0.001, threshold_mode='abs', cooldown=1)

start_epoch = -1
if conf.ckpt is not None:
    # ckpt = load_model('checkpoints/deepcluster.pth.tar')
    ckpt = torch.load(conf.ckpt)
    if optimistic_restore(detector, ckpt['state_dict']):
        start_epoch = ckpt['epoch']


def train_epoch(epoch_num):
    detector.train()
    # Iterate through the convolutional layers
    p_count = 0
    if epoch_num >= 20:
        for p in detector.features[34:].parameters():
            p.requires_grad = True
    tr = []
    start = time.time()
    for b, batch in enumerate(train_loader):
        tr.append(train_batch(batch))

        if b % conf.print_interval == 0 and b >= conf.print_interval:
            mn = pd.concat(tr[-conf.print_interval:], axis=1).mean(1)
            time_per_batch = (time.time() - start) / conf.print_interval
            print("\ne{:2d}b{:5d}/{:5d} {:.3f}s/batch, {:.1f}m/epoch".format(
                epoch_num, b, len(train_loader), time_per_batch, len(train_loader) * time_per_batch / 60))
            print(mn)
            print('-----------', flush=True)
            start = time.time()
    return pd.concat(tr, axis=1)


def train_batch(b):
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
    scores = result.od_obj_dists
    labels = result.od_obj_labels
    class_loss = F.cross_entropy(scores, labels)

    loss = class_loss  # + box_loss
    res = pd.Series([class_loss.data, loss.data], ['class_loss', 'total'])

    optimizer.zero_grad()
    loss.backward()
    clip_grad_norm(
        [(n, p) for n, p in detector.named_parameters() if p.grad is not None],
        max_norm=conf.clip, clip=True)
    optimizer.step()

    return res


cls_accuracy = []
def val_epoch():
    detector.eval()
    # all_boxes is a list of length number-of-classes.
    # Each list element is a list of length number-of-images.
    # Each of those list elements is either an empty list []
    # or a numpy array of detection.
    vr = []
    cls_accuracy.clear()
    for val_b, batch in enumerate(val_loader):
        vr.append(val_batch(val_b, batch))
    vr = np.concatenate(vr, 0)
    if vr.shape[0] == 0:
        print("No detections anywhere")
        return 0.0

    # -- mAP accuracy (Since we are using GT bounding boxes, this metric doesn't really show valuable stuff)
    val_coco = val.coco
    coco_dt = val_coco.loadRes(vr)
    coco_eval = COCOeval(val_coco, coco_dt, 'bbox')
    coco_eval.params.imgIds = val.ids if conf.coco else [x for x in range(len(val))]

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    mAp = coco_eval.stats[1]

    # -- Classification accuracy
    print("\nClassification accuracy: %0.3f%%" % (np.mean(cls_accuracy) * 100.0))

    return mAp


def val_batch(batch_num, b):
    result = detector[b]
    if result is None:
        return np.zeros((0, 7))

    im_inds_np = result.im_inds.data.cpu().numpy()
    im_scales = b.im_sizes.reshape((-1, 3))[:, 2]

    # -- Determine predictions, scores, and boxes
    scores, cls_preds = result.rm_obj_dists[:, 1:].data.max(1)
    scores_np = scores.cpu().numpy()
    cls_preds_np = (cls_preds + 1).cpu().numpy()
    boxes_np = b.gt_boxes_primary.data.cpu().numpy()

    if conf.coco:
        boxes_np /= im_scales[im_inds_np][:, None]
        boxes_np[:, 2:4] = boxes_np[:, 2:4] - boxes_np[:, 0:2] + 1
        cls_preds_np[:] = [val.ind_to_id[c_ind] for c_ind in cls_preds_np]
        im_inds_np[:] = [val.ids[im_ind + batch_num * conf.batch_size * conf.num_gpus]
                         for im_ind in im_inds_np]
    else:
        boxes_np *= BOX_SCALE / IM_SCALE
        boxes_np[:, 2:4] = boxes_np[:, 2:4] - boxes_np[:, 0:2] + 1
        im_inds_np += batch_num * conf.batch_size * conf.num_gpus

    # -- Calculate pure object classification accuracy
    gt_classes = b.gt_classes_primary.data.cpu().numpy()[:, 1]
    det_array = np.equal(gt_classes, cls_preds_np).astype(np.float)
    # -- Split det_array w.r.t image indices (can be implemented using torch's scatter_add)
    det_split = np.split(det_array, np.cumsum(np.unique(im_inds_np, return_counts=True)[1])[:-1])
    cls_accuracy.extend([split.mean() for split in det_split])

    return np.column_stack((im_inds_np, boxes_np, scores_np, cls_preds_np))


print("Training starts now!")
for epoch in range(start_epoch + 1, start_epoch + 1 + conf.num_epochs):
    rez = train_epoch(epoch)
    print("overall{:2d}: ({:.3f})\n{}".format(epoch, rez.mean(1)['total'], rez.mean(1)), flush=True)
    mAp = val_epoch()
    scheduler.step(mAp)

    torch.save({
        'epoch': epoch,
        'state_dict': detector.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, os.path.join(conf.save_dir, '{}-{}.tar'.format('coco' if conf.coco else 'vg', epoch)))
