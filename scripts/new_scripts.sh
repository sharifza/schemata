#!/usr/bin/env bash

echo $1
if [[ $1 == "sgcls" ]]; then
    echo "SGCLS"
    python models/train_rels.py -m sgcls -model sharifza -b 12 -clip 5 \
        -p 500 -hidden_dim 512 -pooling_dim 4096 -lr 1e-5 -ngpu 1 -ckpt checkpoints/vg-faster-rcnn.tar  \
        -save_dir checkpoints/sgcls_2022 -nepoch 31 -asm 2 -adam -yesFuse
elif [[ $1 == "predcls" ]]; then
    echo "PREDCLS"
    python models/train_rels.py -m predcls -model sharifza -b 24 -clip 5 \
        -p 500 -hidden_dim 512 -pooling_dim 4096 -lr 1e-5 -ngpu 1 -ckpt checkpoints/vg-faster-rcnn.tar  \
        -save_dir checkpoints/predcls_2022 -nepoch 31 -asm 2 -adam -yesFuse
else
  echo "Script not found!"
fi