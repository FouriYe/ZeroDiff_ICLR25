#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 17:52:45 2019
@author: akshita
"""
import os
os.system('''CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=3  python zerodiff_DFG_train.py \
--dataroot /Data_PHD_Backup/phd22_zihan_ye/datasets/xlsa17/data \
--dataset SUN --image_embedding res101 --class_embedding att --class_embedding_norm --eval_interval 5 \
--gzsl --manualSeed 4115 --encoded_noise --preprocessing --cuda \
--nepoch 400 --ngh 4096 --ndh 4096 --lr 0.0001 --classifier_lr 0.0005 --lambda1 10 --critic_iter 5 \
--nclass_all 717 --batch_size 64 --noiseSize 102 --attSize 102 --resSize 2048 \
--gamma_recons 0.01 --dec_lr 0.0001 \
--gamma_ADV 1 --gamma_VAE 1.0 --embed_type VA \
--n_T 4 --dim_t 102 --gamma_x0 1.0 --gamma_xt 1.0 --gamma_dist 0.0 --factor_dist 0.0 \
--split_percent 100 --syn_num 400 \
''')

# split_percent:100
# --split_percent 100 --syn_num 400 \

# split_percent:30
# --split_percent 30 --syn_num 120 \

# split_percent:10
# --split_percent 10 --syn_num 40 \