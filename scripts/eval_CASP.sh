gpus=0
data_name=LEVIR-CD+
net_G=CASP # This is the best version
split=test
pretrain=False
vis_root=/data/
project_name=CD_CASP_GZ_CD_b16_lr0.0001_adamw_train_test_200_linear_ce_multi_train_False_multi_infer_False_shuffle_AB_False_pretrain_True_setting_CASP
checkpoints_root=/data/ptdoge/A2024CD/BCD/20250208/
checkpoint_name=best_ckpt.pt
img_size=512
embed_dim=64 # Make sure to change the embedding dim (best and default = 256)

python eval_cd.py --split ${split} --pretrain ${pretrain} --net_G ${net_G} --embed_dim ${embed_dim} --img_size ${img_size} --vis_root ${vis_root} --checkpoints_root ${checkpoints_root} --checkpoint_name ${checkpoint_name} --gpu_ids ${gpus} --project_name ${project_name} --data_name ${data_name} 