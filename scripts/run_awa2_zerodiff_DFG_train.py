#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ZihanYe
"""
import os
os.system('''CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 python zerodiff_DFG_train.py \
--gzsl --encoded_noise --manualSeed 9182 --preprocessing --cuda --image_embedding res101 \
--class_embedding att --class_embedding_norm --nepoch 300 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 \
--nclass_all 50 --dataroot YourAccount/datasets/xlsa17/data --dataset AWA2 --eval_interval 5 \
--batch_size 64 --noiseSize 85 --attSize 85 --resSize 2048 \
--lr 0.0005 --classifier_lr 0.001 --gamma_recons 1.0 --freeze_dec --dec_lr 0.0001 \
--gamma_ADV 10 --gamma_VAE 1.0 --embed_type VA \
--n_T 4 --dim_t 85 --gamma_x0 1.0 --gamma_xt 1.0 \
--split_percent 100 --syn_num 5400  --gamma_dist 5.0 --factor_dist 1.5 \
--netR_model_path ./out/AWA2/diffzero_pretrain_100percent_att:att_b:64_lr:0.0005_n_T:4_betas:0.1,20_gamma:ADV:10.0_VAE:1.0_x0:1.0_xt:1.0_dist:0.0_num:1800_gzsl.tar \
''')

# split_percent 100:
# --split_percent 100 --syn_num 5400  --gamma_dist 5.0 --factor_dist 1.5 \
# --netR_model_path ./out/AWA2/diffzero_pretrain_100percent_att:att_b:64_lr:0.0005_n_T:4_betas:0.1,20_gamma:ADV:10.0_VAE:1.0_x0:1.0_xt:1.0_dist:0.0_num:1800_gzsl.tar \

# split_percent 30:
# --split_percent 30 --syn_num 1800  --gamma_dist 5.0 --factor_dist 1.5 \
# --netR_model_path ./out/AWA2/diffzero_pretrain_30percent_att:att_b:64_lr:0.0005_n_T:4_betas:0.1,20_gamma:ADV:10.0_VAE:1.0_x0:1.0_xt:1.0_dist:0.0_num:600_gzsl.tar \

# split_percent 10:
# --split_percent 10 --syn_num 600  --gamma_dist 5.0 --factor_dist 1.5  \
# --netR_model_path ./out/AWA2/diffzero_pretrain_10percent_att:att_b:64_lr:0.0005_n_T:4_betas:0.1,20_gamma:ADV:10.0_VAE:1.0_x0:1.0_xt:1.0_dist:0.0_num:200_gzsl.tar \
