
# Early Detection of Autism Spectrum Disorder from Caregiver-Child Interaction Videos Using a Pretrained Vision-Language Model




ASD-CLIP is a novel machine learning model designed to classify 3-minute videos into Autism Spectrum Disorder (ASD) or non-ASD categories.

# Environment Setup
To set up the required environment, execute the following commands:
```
conda create -n ASD-CLIP python=3.7
conda activate ASD-CLIP
pip install -r requirements.txt
```

and install Apex as follows
```
git clone https://github.com/NVIDIA/apex
cd apex
git checkout 23.08
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./

```
Installing all packages takes approximately 1 hour, depending on the network
# Data Preparation

- **Step \#1:prepare video set**

You need to prepare a collection of videos, each with a duration of at least 3 minutes. Place all these videos in folder ./org_video_path.
There are 4 videos in org_video_path as demos.

- **Step \#2:video pre-process**  

Use YOLO v8 to remove useless frames from the videos. Run the following command:
```
python video_preprocess.py
```
The videos ready for subsequent training and validation will be stored in ./processed_video_path.
4 videos in processed_video_path are the processed videos.

-  **Step \#3:prepare clinical information**

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

-  **Step \#4:generate train and val labels**  

After preparing the clinical data, run the following script to generate video labels:
```
python generate_label/label_1Fold.py
```
This script retrieves the labels for each video from totaldata.xlsx and generates corresponding numerical labels. The ./video_labels directory should have the following structure, containing labels required for 1fold training and validation:
```
video_labels
    train_label.txt
    val_label.txt
```
It should be noted that in the demo, it is assumed that DemoChildA and B will be used for training, and DemoChildC and D will be used for validation. If using one's own dataset, it is necessary to modify lines 81-89 of label_1Fold.py to reflect the desired training and validation sets.
# Train——pretrain on Kinetics
The training time is relatively long, it is recommended to use the checkpoints we have already trained

If you want to train on your own, please organize the dataset of [Kinetics-600/400](https://arxiv.org/abs/1705.06950 ) Convert to the same format as trainlabel.txt in the videolabels folder.
There is an example configuration file in configs/pretrain. yaml. Modify the root path in this YAML file to point to the folder containing processed videos. Then, run the training script:
```
bash pretrain.sh
```
# Train——finetune
There is an example configuration file in configs/train.yaml. Modify the root path in this YAML file to point to the folder containing the processed videos. Then, run the training script:
```
bash train.sh
```
- **Note:**
- --nproc_per_node = the number of gpu in your server
- This script will train the model on the videos listed in train_label.txt, the corresponding validation videos are listed in val_label.txt

It takes about 3-5 minutes to finish demo's training on one nvidia 3090.
# Test
There is an example configuration file in configs/val.yaml. Update the root path in this YAML file to the folder with the processed videos. Then, run the testing script:
```
bash val.sh
```
**Note:**
- --nproc_per_node = the number of gpu in your server
- --resume the path to the checkpoint generated in #Train
- 
It takes about 2 minutes to finish demo's validation on one nvidia 3090.


# Acknowledgements
Parts of the codes are borrowed from [X-CLIP](https://github.com/microsoft/VideoX/tree/master/X-CLIP),[mmaction2](https://github.com/open-mmlab/mmaction2), [Swin](https://github.com/microsoft/Swin-Transformer) and [CLIP](https://github.com/openai/CLIP). Sincere thanks to their wonderful works.
