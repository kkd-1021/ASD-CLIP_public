python -m torch.distributed.launch --nproc_per_node=1 --master_port=25645 \main.py -cfg ./configs/train.yaml --output path_to_save_dir --pretrained ckp/ViT-L-14.pt --resume ckp/k600_14_8.pth
