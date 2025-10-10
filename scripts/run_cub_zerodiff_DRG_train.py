#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 17:52:45 2019
@author: akshita
"""
import os
os.system('''CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=3 python zerodiff_DRG_train.py \
--dataset CUB --gzsl --manualSeed 3483 --image_embedding res101 --class_embedding sent --eval_interval 1 \
 --encoded_noise --preprocessing --cuda \
--nepoch 300 --ngh 4096 --ndh 4096 --lr 0.0001 --classifier_lr 0.001 --lambda1 10 --critic_iter 5 \
--dataroot /Data_PHD_Backup/phd22_zihan_ye/datasets/xlsa17/data \
--nclass_all 200 --noiseSize 1024 --attSize 1024 --resSize 2048 \
--gamma_ADV 1 --gamma_VAE 0.0 --embed_type VA --gamma_recons 1.0 \
--n_T 4 --dim_t 1024 --gamma_x0 1.0 --gamma_xt 1.0 --gamma_dist 1.0 \
--batch_size 64 --syn_num 300 --split_percent 100 \
''')

# split_percent:100
# --batch_size 64 --syn_num 300 --split_percent 100 \

# split_percent:30
# --batch_size 64 --syn_num 100 --split_percent 30 \

# split_percent:10
# --batch_size 64 --syn_num 40 --split_percent 10 \