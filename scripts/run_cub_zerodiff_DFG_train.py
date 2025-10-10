#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 17:52:45 2019
@author: akshita
"""
import os
os.system('''CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=3 python zerodiff_DFG_train.py \
--gzsl --encoded_noise --manualSeed 3483 --preprocessing --cuda --image_embedding res101 \
--class_embedding sent --nepoch 300 --ngh 4096 --ndh 4096  --lambda1 10 --critic_iter 5 \
--nclass_all 200 --dataroot /Data_PHD_Backup/phd22_zihan_ye/datasets/xlsa17/data --dataset CUB --eval_interval 5 \
--batch_size 64 --noiseSize 1024 --attSize 1024 --resSize 2048 \
--lr 0.0001 --classifier_lr 0.001 --gamma_recons 0.01 --dec_lr 0.0001 \
--gamma_ADV 10 --gamma_VAE 1.0 --embed_type VA \
--n_T 4 --dim_t 1024 --gamma_x0 1.0 --gamma_xt 1.0 \
--gamma_dist 2.0 --factor_dist 1.5 \
--split_percent 100 --syn_num 1440 \
--netR_model_path ./out/CUB/zerodiff_DRG_100percent_att:sent_b:64_lr:0.0001_n_T:4_betas:0.1,20_gamma:ADV:10.0_VAE:1.0_x0:1.0_xt:1.0_dist:0.0_num:300_gzsl.tar \
''')

# split_percent 100:
# --split_percent 100 --syn_num 1440
# --netR_model_path ./out/CUB/zerodiff_DRG_100percent_att:sent_b:64_lr:0.0001_n_T:4_betas:0.1,20_gamma:ADV:10.0_VAE:1.0_x0:1.0_xt:1.0_dist:0.0_num:300_gzsl.tar \

# split_percent 30:
# --split_percent 30 --syn_num 440
# --netR_model_path ./out/CUB/zerodiff_DRG_30percent_att:sent_b:64_lr:0.0001_n_T:4_betas:0.1,20_gamma:ADV:10.0_VAE:1.0_x0:1.0_xt:1.0_dist:0.0_num:100_gzsl.tar \

# split_percent 10:
# --split_percent 10 --syn_num 140
# --netR_model_path ./out/CUB/zerodiff_DRG_10percent_att:sent_b:64_lr:0.0001_n_T:4_betas:0.1,20_gamma:ADV:10.0_VAE:1.0_x0:1.0_xt:1.0_dist:0.0_num:40_gzsl.tar \