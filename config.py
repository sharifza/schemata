"""
Configuration file!
"""
import os
from argparse import ArgumentParser
import numpy as np
import subprocess

ROOT_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(ROOT_PATH, 'data')

def path(fn):
    return os.path.join(DATA_PATH, fn)

def stanford_path(fn):
    return os.path.join(DATA_PATH, 'stanford_filtered', fn)

# =============================================================================
# Update these with where your data is stored ~~~~~~~~~~~~~~~~~~~~~~~~~
VG_IMAGES = os.path.join(DATA_PATH, 'VG_100K') #  '/home/rowan/datasets2/VG_100K_2/VG_100K'
VG_DEPTH_IMAGES = os.path.join(DATA_PATH, 'vg_depth_1024/VG_100K') #visual_genome/depth_images_512/' # -- (ADDED) Depth images path

# for required_data_folder in [VG_IMAGES, VG_DEPTH_IMAGES]:
#     if not os.path.exists(required_data_folder):
#         print("Directory ", required_data_folder, " should be created!")
#         if required_data_folder == VG_DEPTH_IMAGES:
#             os.mkdir(os.path.join(DATA_PATH, 'vg_depth_1024'))
#             os.mkdir(os.path.join(DATA_PATH, 'vg_depth_1024/VG_100K'))
#             sourcedir = "/nfs/data/sharifza/data/depth/VG_100K/vg_depth_1024/VG_100K/"
#         elif required_data_folder == VG_IMAGES:
#             os.mkdir(VG_IMAGES)
#             sourcedir = "/nfs/data/koner/data/VG_100K/"
#
#         print("Now copying images from NFS to local...")
#         # Handle the source folders for copying
#
#         # Call rsync to copy the data while showing a progress bar.
#         subprocess.call(["rsync", "-r", "--info=progress2", sourcedir, required_data_folder])
#         print("Copying finished!")
#     else:
#         print("Directory ", required_data_folder,  " already exists")

RCNN_CHECKPOINT_FN = path('faster_rcnn_500k.h5')

IM_DATA_FN = stanford_path('image_data.json')
VG_SGG_FN = stanford_path('VG-SGG.h5')
VG_SGG_DICT_FN = stanford_path('VG-SGG-dicts.json')
PROPOSAL_FN = stanford_path('proposals.h5')

COCO_PATH = '/home/rowan/datasets/mscoco'
# =============================================================================
# =============================================================================

MODES = ('sgdet', 'sgcls', 'predcls')

# -- (ADDED) Data fusion mode
FUSION_MODES = ('rgb_only', 'depth_only', 'fusion')

# -- (ADDED) Depth models
DEPTH_MODELS = ('alexnet', 'resnet18', 'resnet50', 'vgg', 'sqznet')

BOX_SCALE = 1024  # Scale at which we have the boxes
IM_SCALE = 592      # Our images will be resized to this res without padding

# Proposal assignments
BG_THRESH_HI = 0.5
BG_THRESH_LO = 0.0

RPN_POSITIVE_OVERLAP = 0.7
# IOU < thresh: negative example
RPN_NEGATIVE_OVERLAP = 0.3

# Max number of foreground examples
RPN_FG_FRACTION = 0.5
FG_FRACTION = 0.25
# Total number of examples
RPN_BATCHSIZE = 256
ROIS_PER_IMG = 256
REL_FG_FRACTION = 0.25
RELS_PER_IMG = 256

RELS_PER_IMG_REFINE = 64

BATCHNORM_MOMENTUM = 0.01
ANCHOR_SIZE = 16

ANCHOR_RATIOS = (0.23232838, 0.63365731, 1.28478321, 3.15089189) #(0.5, 1, 2)
ANCHOR_SCALES = (2.22152954, 4.12315647, 7.21692515, 12.60263013, 22.7102731) #(4, 8, 16, 32)

class ModelConfig(object):
    """Wrapper class for model hyperparameters."""
    def __init__(self):
        """
        Defaults
        """
        self.asm_num = None
        self.coco = None
        self.ckpt = None
        self.extra_ckpt = None
        self.save_dir = None
        self.lr = None
        self.batch_size = None
        self.val_size = None
        self.l2 = None
        self.clip = None
        self.num_gpus = None
        self.num_workers = None
        self.print_interval = None
        self.gt_box = None
        self.mode = None
        self.refine = None
        self.ad3 = False
        self.test = False
        self.adam = False
        self.multi_pred=False
        self.cache = None
        self.model = None
        self.use_proposals=False
        self.use_resnet=False
        self.use_tanh=False
        self.use_bias = False
        self.limit_vision=True
        self.num_epochs=None
        self.old_feats=False
        self.order=None
        self.det_ckpt=None
        self.nl_edge=None
        self.nl_obj=None
        self.hidden_dim=None
        self.pass_in_obj_feats_to_decoder = None
        self.pass_in_obj_feats_to_edge = None
        self.pooling_dim = None
        self.rec_dropout = None
        self.freeze_base = False
        self.PKG = False
        self.visual_share = None
        self.all_share = None
        self.visual_share_num = None
        self.destroy_vis = None
        self.use_union = None
        self.val_iteration = None
        self.allasm = False
        self.sigmoid_gate = False
        self.yesFuse = False
        self.hard_att = False
        self.sigmoid_uncertainty = False
        self.n_drop = False

        # -- (ADDED) add depth related parameters
        self.fusion_mode = None
        self.depth_model = None
        self.pretrained_depth = False
        self.tensorboard_ex = False
        self.enable_po = False
        self.izs_file = None
        self.izs_vis_discard = False
        self.enable_gates = False

        self.parser = self.setup_parser()
        self.args = vars(self.parser.parse_args())

        print("~~~~~~~~ Hyperparameters used: ~~~~~~~")
        for x, y in self.args.items():
            print("{} : {}".format(x, y))

        # -- (ADDED) print dataset path
        print("RGB dataset path: ", VG_IMAGES)
        # print("Depth dataset path: ", VG_DEPTH_IMAGES)

        self.__dict__.update(self.args)

        if len(self.ckpt) != 0:
            self.ckpt = os.path.join(ROOT_PATH, self.ckpt)
        else:
            self.ckpt = None

        if len(self.extra_ckpt) != 0:
            self.extra_ckpt = os.path.join(ROOT_PATH, self.extra_ckpt)
        else:
            self.extra_ckpt = None

        if len(self.cache) != 0:
            self.cache = os.path.join(ROOT_PATH, self.cache)
        else:
            self.cache = None

        if len(self.save_dir) == 0:
            self.save_dir = None
        else:
            self.save_dir = os.path.join(ROOT_PATH, self.save_dir)
            if not os.path.exists(self.save_dir):
                os.mkdir(self.save_dir)

        assert self.val_size >= 0

        if self.mode not in MODES:
            raise ValueError("Invalid mode: mode must be in {}".format(MODES))

        # -- (ADDED) add depth_fc model
        if self.model not in ('motifnet', 'stanford', 'sharifza', 'store_features',
                              # -- Depth models
                              'class',
                              'depth_fc',
                              'depth_fc_vision',
                              'depth_vision'):
            raise ValueError("Invalid model {}".format(self.model))

        if self.ckpt is not None and not os.path.exists(self.ckpt):
            raise ValueError("Ckpt file ({}) doesnt exist".format(self.ckpt))

        if self.extra_ckpt is not None and not os.path.exists(self.extra_ckpt):
            raise ValueError("Extra Ckpt file ({}) doesnt exist".format(self.extra_ckpt))

        # -- (ADDED) check if the provided fusion mode is correct
        if self.fusion_mode not in FUSION_MODES:
            raise ValueError("Invalid data fusion mode: mode must be in {}".format(FUSION_MODES))

        # -- (ADDED) check if the provided depth model is valid
        if self.depth_model not in DEPTH_MODELS:
            raise ValueError("Invalid depth model: model must be in {}".format(DEPTH_MODELS))

        # -- (ADDED) check if the provided inverse-zershot file exists
        if self.izs_file is not None and not os.path.exists(self.izs_file):
            raise ValueError("Inverse-Zeroshot split ({}) doesnt exist".format(self.izs_file))


    def setup_parser(self):
        """
        Sets up an argument parser
        :return:
        """
        parser = ArgumentParser(description='training code')


        # Options to deprecate
        parser.add_argument('-asm', dest='asm_num', help='asm num', type=int, default=2)
        parser.add_argument('-coco', dest='coco', help='Use COCO (default to VG)', action='store_true')
        parser.add_argument('-ckpt', dest='ckpt', help='Filename to load from', type=str, default='')
        parser.add_argument('-det_ckpt', dest='det_ckpt', help='Filename to load detection parameters from', type=str, default='')
        parser.add_argument('-extra_ckpt', dest='extra_ckpt', help='Filename to load extra checkpoint from', type=str,
                            default='')

        parser.add_argument('-save_dir', dest='save_dir',
                            help='Directory to save things to, such as checkpoints/save', default='', type=str)

        parser.add_argument('-ngpu', dest='num_gpus', help='cuantos GPUs tienes', type=int, default=3)
        parser.add_argument('-nwork', dest='num_workers', help='num processes to use as workers', type=int, default=1)

        parser.add_argument('-lr', dest='lr', help='learning rate', type=float, default=1e-3)

        parser.add_argument('-b', dest='batch_size', help='batch size per GPU',type=int, default=2)
        parser.add_argument('-val_size', dest='val_size', help='val size to use (if 0 we wont use val)', type=int, default=5000)

        parser.add_argument('-l2', dest='l2', help='weight decay', type=float, default=1e-4)
        parser.add_argument('-clip', dest='clip', help='gradients will be clipped to have norm less than this', type=float, default=5.0)
        parser.add_argument('-p', dest='print_interval', help='print during training', type=int,
                            default=100)
        parser.add_argument('-m', dest='mode', help='mode \in {sgdet, sgcls, predcls}', type=str,
                            default='sgdet')
        parser.add_argument('-model', dest='model', help='which model to use? (motifnet, stanford). If you want to use the baseline (NoContext) model, then pass in motifnet here, and nl_obj, nl_edge=0', type=str,
                            default='motifnet')
        parser.add_argument('-old_feats', dest='old_feats', help='Use the original image features for the edges', action='store_true')
        parser.add_argument('-order', dest='order', help='Linearization order for Rois (confidence -default, size, random)',
                            type=str, default='confidence')
        parser.add_argument('-cache', dest='cache', help='where should we cache predictions', type=str,
                            default='')
        parser.add_argument('-gt_box', dest='gt_box', help='use gt boxes during training', action='store_true')
        parser.add_argument('-limit_vision', dest='limit_vision', help='limit vision', action='store_false')
        parser.add_argument('-adam', dest='adam', help='use adam. Not recommended', action='store_true')
        parser.add_argument('-test', dest='test', help='test set', action='store_true')
        parser.add_argument('-multipred', dest='multi_pred', help='Allow multiple predicates per pair of box0, box1.', action='store_true')
        parser.add_argument('-nepoch', dest='num_epochs', help='Number of epochs to train the model for',type=int, default=25)
        parser.add_argument('-resnet', dest='use_resnet', help='use resnet instead of VGG', action='store_true')
        parser.add_argument('-proposals', dest='use_proposals', help='Use Xu et als proposals', action='store_true')
        parser.add_argument('-nl_obj', dest='nl_obj', help='Num object layers', type=int, default=1)
        parser.add_argument('-nl_edge', dest='nl_edge', help='Num edge layers', type=int, default=2)
        parser.add_argument('-hidden_dim', dest='hidden_dim', help='Num edge layers', type=int, default=256)
        parser.add_argument('-pooling_dim', dest='pooling_dim', help='Dimension of pooling', type=int, default=4096)
        parser.add_argument('-pass_in_obj_feats_to_decoder', dest='pass_in_obj_feats_to_decoder', action='store_true')
        parser.add_argument('-pass_in_obj_feats_to_edge', dest='pass_in_obj_feats_to_edge', action='store_true')
        parser.add_argument('-rec_dropout', dest='rec_dropout', help='recurrent dropout to add', type=float, default=0.1)
        parser.add_argument('-use_bias', dest='use_bias',  action='store_true')
        parser.add_argument('-use_tanh', dest='use_tanh',  action='store_true')
        parser.add_argument('-use_union', dest='use_union',  action='store_true')
        parser.add_argument('-freeze_base', dest='freeze_base', action='store_true')
        parser.add_argument('-destroy_vis', dest='destroy_vis', action='store_true')
        parser.add_argument('-PKG', dest='PKG', action='store_true')
        parser.add_argument('-visual_share', dest='visual_share', help='share of visual data to use', type=float,
                            default=1.0)
        parser.add_argument('-all_share', dest='all_share', help='share of all data to use', type=float,
                            default=1.0)
        parser.add_argument('-val_it', dest='val_iteration', help='how often to validate during training', type=float,
                            default=1)
        parser.add_argument('-allasm', dest='allasm', help='Should we print all assimilation results?', action='store_true')
        # -- (ADDED) Add `fusion_mode` and `depth_model` arguments
        parser.add_argument('-fusion_mode', dest='fusion_mode', help='data fusion mode \in {rgb_only, depth_only, fusion}', type=str,
                            default='rgb_only')
        parser.add_argument('-depth_model', dest='depth_model', help='depth model \in {alexnet, resnet, vgg}', type=str,
                            default='alexnet')
        # -- (ADDED) Add pre-trained flag for depth network
        # -- will convert the depth images to 3-channels
        parser.add_argument('-pretrained_depth', dest='pretrained_depth',  action='store_true')

        parser.add_argument('-sigmoid_gate', dest='sigmoid_gate', action='store_true')
        parser.add_argument('-yesFuse', dest='yesFuse', action='store_true')
        parser.add_argument('-hard_att', dest='hard_att', action='store_true')
        parser.add_argument('-sigmoid_uncertainty', dest='sigmoid_uncertainty', action='store_true')

        # -- (ADDED) add tensorboard_extra flag which is used to log extra information about depth process in tensorboard
        parser.add_argument('-tensorboard_ex', dest='tensorboard_ex',  action='store_true')
        parser.add_argument('-po', dest='enable_po',  action='store_true')
        parser.add_argument('-izs_file', dest='izs_file', help='Inverse-Zeroshot split.', type=str, default=None)
        parser.add_argument('-izs_vis_discard', dest='izs_vis_discard',  action='store_true')
        parser.add_argument('-gates', dest='enable_gates',  action='store_true')
        parser.add_argument('-n_drop', dest='n_drop', action='store_true')
        return parser
