#!/usr/bin/env bash
# Train the model without COCO pretraining
#python models/train_detector.py -b 8 -lr 1e-5 -save_dir checkpoints/backbone-5_noPre_adam5_freezeAllConv/ -nepoch 50 \
# -ngpu 1 -nwork 3 -p 500 -clip 5 -izs_file ./misc/izs_splits/5/izs_split_0.npz -gt_box -adam
python models/train_classifier.py -b 8 -lr 1e-5 -save_dir checkpoints/backbone-5_Pre_adam5/ -nepoch 50 \
 -ngpu 1 -nwork 3 -p 500 -clip 5 -adam -izs_file ./misc/izs_splits/5/izs_split_0.npz -adam
# If you want to evaluate on the frequency baseline now, run this command (replace the checkpoint with the
# best checkpoint you found).
#export CUDA_VISIBLE_DEVICES=0
# python models/eval_rel_count.py -ngpu 1 -b 6 -ckpt checkpoints/vgdet/vg-24.tar -nwork 1 -p 100 -test
#export CUDA_VISIBLE_DEVICES=1
#python models/eval_rel_count.py -ngpu 1 -b 6 -ckpt checkpoints/vgdet/vg-28.tar -nwork 1 -p 100 -test
#export CUDA_VISIBLE_DEVICES=2
#python models/eval_rel_count.py -ngpu 1 -b 6 -ckpt checkpoints/vgdet/vg-28.tar -nwork 1 -p 100 -test
#
#
