#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 17:52:45 2019
@author: ZihanYe
"""
import os
os.system('''CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=4 python zerodiff_DRG_train.py \
--gzsl --encoded_noise --manualSeed 9182 --preprocessing --cuda --image_embedding res101 \
--class_embedding att --class_embedding_norm --nepoch 300 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 \
--nclass_all 50 --dataroot datasets/xlsa17/data --dataset AWA2 \
--noiseSize 85 --attSize 85 --resSize 2048 \
--lr 0.0005 --classifier_lr 0.001 --gamma_recons 1.0 --freeze_dec \
--gamma_ADV 10 --gamma_VAE 1.0 --embed_type VA \
--n_T 4 --dim_t 85 --gamma_x0 1.0 --gamma_xt 1.0 --gamma_dist 0.0 \
--batch_size 64 --syn_num 1800 --split_percent 100 \
''')

# split_percent 100:
# --batch_size 64 --syn_num 1800 --split_percent 100 \

# split_percent 30:
# --batch_size 64 --syn_num 600 --split_percent 30 \

# split_percent 10:
# --batch_size 64 --syn_num 200 --split_percent 10 \


