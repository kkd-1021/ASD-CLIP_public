mkdir processed_video_path
mkdir video_labels
python video_preprocess.py
python generate_label/label_1Fold.py
mkdir path_to_save_dir
python -m torch.distributed.launch --nproc_per_node=1 --master_port=25645 \main.py -cfg ./configs/train.yaml --output path_to_save_dir --pretrained ckp/ViT-L-14.pt --resume ckp/k600_14_8.pth
python -m torch.distributed.launch --nproc_per_node=1 --master_port=25645 \main.py -cfg ./configs/val.yaml --output path_to_save_dir --pretrained ckp/ViT-L-14.pt --resume path_to_save_dir/ckpt_epoch_9.pth
