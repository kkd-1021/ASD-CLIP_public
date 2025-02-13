python -m torch.distributed.launch --nproc_per_node=4 --master_port=25645 \main.py -cfg ./configs/pretrain.yaml --output path_to_save_dir --pretrained path_to_ViT-L-14.pt
