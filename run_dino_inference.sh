python inference.py --model dino --arch vit_small_patch16 --embed_dim 384  \
 --ckpt ./checkpoints/uniDINO.pth \
 --batch_size 10 --num_workers 4 \
 -o ./embeddings \
 --valset example_data\
 --size 224 --stride 224 --gpus 0  --norm_method no_post_proc