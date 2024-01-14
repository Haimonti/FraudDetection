#!/bin/bash

### Feature Table ###
# Raw+Fina 42
# Raw 28
# FinRatio 14

dataset=aaer

BASEDIR=$(dirname "$0")
OUTDIR="${BASEDIR}/ckpt/"

if [ ! -d "${OUTDIR}" ]
then   
    echo "Output dir ${OUTDIR} does not exist, creating..."
    mkdir -p ${OUTDIR}
fi    

CUDA_VISIBLE_DEVICES=0 python -u main_cls_cv.py \
    --feat_d 28 \
    --hidden_d 10 \
    --boost_rate 1 \
    --lr 0.005 \
    --L2 .0e-3 \
    --num_nets 25 \
    --data ${dataset} \
    --tr ${BASEDIR}/../data/${dataset}.train \
    --te ${BASEDIR}/../data/${dataset}.test \
    --batch_size 16 \
    --epochs_per_stage 1 \
    --correct_epoch 1 \
    --model_order second \
    --normalization True \
    --cv False \
    --sparse False \
    --out_f ${OUTDIR}/${dataset}_cls.pth \
    --cuda
