
# Automated Early Detection of Autism Spectrum Disorder via a Pretrained Vision-Language Model in Naturalistic Caregiver-Child Interactions




ASD-CLIP is a novel machine learning model designed to classify 3-minute videos into Autism Spectrum Disorder (ASD) or non-ASD categories.

# Environment Setup

To set up the required environment, execute the following commands:
```
git clone https://github.com/kkd-1021/ASD-CLIP_public
cd ASD-CLIP_public
conda create -n ASD-CLIP python=3.8
conda init bash && source /root/.bashrc
conda activate ASD-CLIP
bash env.sh
```
Installing all packages takes approximately 10 min, depending on the network
# Run demo
We provide an automated script to directly run the demo. Before running this script, please download two CKP files and place them in the ckp folder (you need to mkdir first) of this project
```
ASD-CLIP_public
  ckp
   ViT-L-14.pth
   k600_14_8.pth
```
download URL:
[k600_14_8.pth](https://drive.google.com/file/d/109PXerbP3vAFaNj1zy82uKrUQ1CUKTdU/view?usp=drive_link) 
[ViT-L-14.pth](https://drive.google.com/file/d/1kMB2Naa3IvTA8Yxc-2P48Gsp7upDvupx/view?usp=drive_link)
Then you can run the demo by executing the following command:
```
bash demo.sh
```

# Run step by step：
# Data Preparation

- **Step \#1:prepare video set**

You need to prepare a collection of videos, each with a duration of at least 3 minutes. Place all these videos in folder ./org_video_path.
There are 4 videos in org_video_path as demos.(The demo video was generated statically from a single image frame and does not contain actual video data from real participants. This approach is taken to protect the privacy of study participants.)


-  **Step \#2:prepare clinical information**

Prepare two files, totaldata.xlsx and total_video_list.txt, and place them in the ./clinical_data directory.
The clinical information for each child should be recorded in the following format and saved as ./clinical_data/totaldata.xlsx. The XLSX file must follow this specific format:
```
name,diagnosis,gender,age(month),mullen:gross motor,mullen:visual reception,mullen:fine motor,mullen:receptive language,mullen:expressive language,SA CSS,RRB CSS
```
Refer to ./clinical_data/totaldata.xlsx for an example.
The list of video names should be saved as ./clinical_data/total_video_list.txt. The video names should follow this format:
```
'child's name'+'_'+'index(if more than one video for a child)'+'.mp4'
```
Check ./clinical_data/total_video_list.txt for an example.


-  **Step \#3:video preprocess and generate train and val labels**  

After preparing the clinical data, run the following script to preprocess videos and generate video labels:
```
bash preprocess.sh
```
Preprocess step of demo dataset takes approximately 2 minutes.
This script retrieves the labels for each video from totaldata.xlsx and generates corresponding numerical labels. The ./video_labels directory should have the following structure, containing labels required for 1fold training and validation:
```
video_labels
    train_label.txt
    val_label.txt
    val_person_Id.txt
```
It should be noted that in the demo, it is assumed that DemoChildA and B will be used for training, and DemoChildC and D will be used for validation. If using one's own dataset, it is necessary to modify lines 81-89 of label_1Fold.py to reflect the desired training and validation sets.
# Train——pretrain on Kinetics
The training time is relatively long, it is recommended to use the checkpoints already trained
download [ViT-L/14](https://drive.google.com/file/d/1kMB2Naa3IvTA8Yxc-2P48Gsp7upDvupx/view?usp=drive_link)
pretrained model from [pre-trained-ckp](https://drive.google.com/file/d/109PXerbP3vAFaNj1zy82uKrUQ1CUKTdU/view?usp=drive_link) 
 and place these in ./ckp 

If you want to train on your own, please organize the dataset of [Kinetics-600/400](https://arxiv.org/abs/1705.06950 ) Convert to the same format as trainlabel.txt in the videolabels folder.
There is an example configuration file in configs/pretrain. yaml. Modify the root path in this YAML file to point to the folder containing processed videos. Then, run the training script:
```
bash pretrain.sh
```
# Train——finetune
before finetune, please ensure that you have pretrained model ./ckp/ViT-L-14.pth and ./ckp/k600_14_8.pth
There is an example configuration file in configs/train.yaml. Modify the root path in this YAML file to point to the folder containing the processed videos. Then, run the training script:
```
bash train.sh
```
- **Note:**
- remember to replace path_to_save_dir to your own path
- --nproc_per_node = the number of gpu in your server
- This script will train the model on the videos listed in train_label.txt, the corresponding validation videos are listed in val_label.txt

It takes about 3-5 minutes to finish demo's training on one nvidia 3090.
# Test
There is an example configuration file in configs/val.yaml. Update the root path in this YAML file to the folder with the processed videos. Then, run the testing script:
```
bash val.sh
```
**model's output:**
- The model's prediction results for each video can be obtained from the command line output. The printed format is as follows (example):
```
person name_id  
person_ID: 1  
ground truth diagnosis: 1.0  
2 out of 2 predictions matched  
model prediction: [0.61181640625, 0.61181640625]  
```
(Note: model prediction represents the model's predicted probability distribution for person_ID 1. The values in the list correspond to confidence scores for different classes or attributes.)
- The correspondence between person IDs and video names can be found in val_person_Id.txt, with the format "video_name personID".

**Note:**
- --nproc_per_node = the number of gpu in your server
- --resume the path to the checkpoint generated in #Train


It takes about 2 minutes to finish demo's validation on one nvidia 3090.


# Acknowledgements
Parts of the codes are borrowed from [X-CLIP](https://github.com/microsoft/VideoX/tree/master/X-CLIP),[mmaction2](https://github.com/open-mmlab/mmaction2), [Swin](https://github.com/microsoft/Swin-Transformer) and [CLIP](https://github.com/openai/CLIP). Sincere thanks to their wonderful works.
