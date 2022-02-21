
from dataloaders.visual_genome import VGDataLoader, VG
import numpy as np
import torch
import pickle
from lib.pytorch_misc import set_random_seed
from config import ModelConfig
from lib.pytorch_misc import optimistic_restore
from lib.evaluation.sg_eval import BasicSceneGraphEvaluator
from tqdm import tqdm
from config import BOX_SCALE, IM_SCALE
import dill as pkl
import os
from lib.rel_model_sharifza import RelModel
conf = ModelConfig()

# -- Set random seed
rnd_seed = 13
set_random_seed(rnd_seed)

# -- Load Inverse-Zeroshot split if provided
izs_split = None
if conf.izs_file is not None:
    print(f"Loading izs file {conf.izs_file}...")
    izs_file = np.load(conf.izs_file, allow_pickle=True)
    izs_split = izs_file['test_rel' if conf.test else 'val_rel'].tolist()

train, val, test = VG.splits(num_val_im=conf.val_size, filter_duplicate_rels=True,
                             use_proposals=conf.use_proposals,
                             filter_non_overlap=conf.mode == 'sgdet',
                             # -- (ADDED) add depth related parameters
                             # use_depth=conf.fusion_mode != 'rgb_only',
                             # three_channels_depth=conf.pretrained_depth,
                             # enable_po=conf.enable_po
                             )
if conf.test:
    val = test
train_loader, val_loader = VGDataLoader.splits(train, val, mode='rel',
                                               batch_size=conf.batch_size,
                                               num_workers=conf.num_workers,
                                               num_gpus=conf.num_gpus,
                                               # -- (ADDED) add depth related parameters
                                               # use_depth=conf.fusion_mode != 'rgb_only'
                                               )

detector = RelModel(classes=train.ind_to_classes, rel_classes=train.ind_to_predicates, mode=conf.mode,
                    num_gpus=conf.num_gpus, use_union=conf.use_union, require_overlap_det=True,
                    hidden_dim=conf.hidden_dim, pooling_dim=conf.pooling_dim, use_resnet=conf.use_resnet,
                    use_proposals=conf.use_proposals, use_bias=conf.use_bias, asm_num=conf.asm_num,
                    freeze_base=conf.freeze_base, PKG=conf.PKG, allasm=conf.allasm, yesFuse=conf.yesFuse,
                    hard_att=conf.hard_att, sigmoid_uncertainty=conf.sigmoid_uncertainty, n_drop=conf.n_drop)


detector.cuda()
ckpt = torch.load(conf.ckpt)

optimistic_restore(detector, ckpt['state_dict'])
# if conf.mode == 'sgdet':
#     # det_ckpt = torch.load('checkpoints/new_vgdet/vg-19.tar')['state_dict']
#     det_ckpt = torch.load('checkpoints/vg-faster-rcnn.tar')['state_dict']
#     detector.detector.bbox_fc.weight.data.copy_(det_ckpt['bbox_fc.weight'])
#     detector.detector.bbox_fc.bias.data.copy_(det_ckpt['bbox_fc.bias'])
#     detector.detector.score_fc.weight.data.copy_(det_ckpt['score_fc.weight'])
#     detector.detector.score_fc.bias.data.copy_(det_ckpt['score_fc.bias'])

all_pred_entries = []

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

        assert np.all(objs_i[asm][rels_i[asm][:,0]] > 0) and np.all(objs_i[asm][rels_i[asm][:,1]] > 0)
        # assert np.all(rels_i[:,2] > 0)

        pred_entry = {
            'pred_boxes': boxes_i * BOX_SCALE/IM_SCALE,
            'pred_classes': objs_i[asm],
            'pred_rel_inds': rels_i[asm],
            'obj_scores': obj_scores_i[asm],
            'rel_scores': pred_scores_i[asm],
        }
        all_pred_entries.append(pred_entry)

        evaluator[conf.mode].evaluate_scene_graph_entry(
            gt_entry,
            pred_entry,
        )

evaluator = BasicSceneGraphEvaluator.all_modes(multiple_preds=conf.multi_pred)
if conf.cache is not None and os.path.exists(conf.cache): #Deprecated
    print("Found {}! Loading from it".format(conf.cache))
    with open(conf.cache,'rb') as f:
        all_pred_entries = pkl.load(f)
    for i, pred_entry in enumerate(tqdm(all_pred_entries)):
        gt_entry = {
            'gt_classes': val.gt_classes[i].copy(),
            'gt_relations': val.relationships[i].copy(),
            'gt_boxes': val.gt_boxes[i].copy(),
        }
        evaluator[conf.mode].evaluate_scene_graph_entry(
            gt_entry,
            pred_entry,
        )
    evaluator[conf.mode].print_stats()
else:
    detector.eval()
    evaluator = BasicSceneGraphEvaluator.all_modes(multiple_preds=conf.multi_pred)
    detection_results = []
    for val_b, batch in enumerate(tqdm(val_loader)):
            detection_results.append(detector[batch])
    if conf.allasm:
        asm_num = conf.asm_num
    else:
        asm_num=1
    for asm in range(1, asm_num + 1):
        print("====ASM%d===" % asm)
        evaluator = BasicSceneGraphEvaluator.all_modes(multiple_preds=conf.multi_pred)
        for batch_num, results in enumerate(detection_results):
            val_batch(conf.num_gpus * batch_num, results, evaluator, -asm)
        evaluator[conf.mode].print_stats()

    if conf.cache is not None:
        with open(conf.cache,'wb') as f:
            pkl.dump(all_pred_entries, f)
