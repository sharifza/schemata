#!/usr/bin/env bash

if [ $1 == "eval_sgcls" ]; then
    python models/eval_rels.py -m sgcls -model sharifza -b 1 \
        -p 500 -hidden_dim 512 -pooling_dim 4096 -ngpu 1 -ckpt checkpoints/sgcls_2022/vgrel-14.tar\
        -asm 4 -allasm -test -yesFuse
fi
