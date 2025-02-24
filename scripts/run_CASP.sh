#!/usr/bin/env bash

#GPUs
gpus=0 # 难道是显卡的原因
#Set paths
checkpoint_root=/data/
vis_root=/data/
data_name=LEVIR-CD+

img_size=512
batch_size=4
lr=0.0001       
max_epochs=200

net_G=CASP
lr_policy=linear # linear 有些新的设置就需要修改 mlstep
optimizer=adamw           # Choices: sgd (set lr to 0.01), adam, adamw
loss=ce                        # Choices: ce, fl (Focal Loss), miou
multi_scale_train=False
multi_scale_infer=False
shuffle_AB=False
n_class=2
embed_dim=256

#Initializing from pretrained weights
pretrain=True
setting=CASP
#Train and Validation splits
split=train   #trainval
split_val=test #test
project_name=CD_${net_G}_${data_name}_b${batch_size}_lr${lr}_${optimizer}_${split}_${split_val}_${max_epochs}_${lr_policy}_${loss}_multi_train_${multi_scale_train}_multi_infer_${multi_scale_infer}_shuffle_AB_${shuffle_AB}_pretrain_${pretrain}_setting_${setting}

python main_cd.py --img_size ${img_size} --loss ${loss} --checkpoint_root ${checkpoint_root} --vis_root ${vis_root} --lr_policy ${lr_policy} --optimizer ${optimizer} --pretrain ${pretrain} --split ${split} --split_val ${split_val} --net_G ${net_G} --multi_scale_train ${multi_scale_train} --multi_scale_infer ${multi_scale_infer} --gpu_ids ${gpus} --max_epochs ${max_epochs} --project_name ${project_name} --batch_size ${batch_size} --shuffle_AB ${shuffle_AB} --data_name ${data_name}  --lr ${lr}  --n_class ${n_class}
