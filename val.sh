# change --resume path_to_save_dir/ckpt_epoch_9.pth to actual fine-tuned ckp
python -m torch.distributed.launch --nproc_per_node=1 --master_port=25645 \main.py -cfg ./configs/val.yaml --output path_to_save_dir --pretrained ckp/ViT-L-14.pt --resume path_to_save_dir/ckpt_epoch_9.pth
