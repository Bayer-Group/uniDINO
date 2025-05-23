export CUDA_VISIBLE_DEVICES=0,1,2,3,4 && python -m torch.distributed.launch \
 --master_port 47990 --nproc_per_node=5 train_dino.py \
 --arch vit_small --lr 0.0004 \
 --epochs 201 --freeze_last_layer 3 \
 --warmup_teacher_temp_epochs 30 --warmup_teacher_temp 0.01 \
 --norm_last_layer true --use_bn_in_head false  --momentum_teacher 0.9995 \
 --warmup_epochs 20 --use_fp16 false --batch_size_per_gpu 102 \
 --num_workers 6 --embed_dim 384 \
 --output_dir ./experiments \
 --saveckp_freq 10 \
 --include_jumpcp false \
 --include_bbbc021 false \
 --include_bbbc037 false \
 --include_hpa false \
 --include_insect false \
 --use_example_data true
