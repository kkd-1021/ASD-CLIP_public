import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import argparse
import datetime
import shutil
from pathlib import Path
from utils.config import get_config
from utils.optimizer import build_optimizer, build_scheduler
from utils.tools import AverageMeter, reduce_tensor, epoch_saving, load_checkpoint, generate_text, auto_resume_helper
from datasets.build import build_dataloader
from utils.logger import create_logger
import time
import numpy as np
import random
from apex import amp
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from datasets.blending import CutmixMixupBlending
from utils.config import get_config
from models import xclip
import torch.nn.functional as F
from utils.asd_utils import get_asd_label_index, get_name_id, label_length, get_asd_label_index_with_id
from torch.utils.tensorboard import SummaryWriter  # Create an instance of the object
import sklearn.metrics
from utils.tools import reduce_tensor
import shap
from clip.model import CLIP
import matplotlib.pyplot as plt
import cv2
from utils.labelEncode import labelDecode,Feat2Idx

save_dir = 'path_to_dict/ASD-CLIP/new-ver/video_labels_explain/explain_result/'

@torch.no_grad()
def explain(val_loader, model, config):
    model.eval()

    # if os.path.exists("path_to_dict/ASD-CLIP/new-ver/video_labels_explain/result.txt"):
    #     print("file already exists")
    #     os.remove("path_to_dict/ASD-CLIP/new-ver/video_labels_explain/result.txt")


    for idx, batch_data in enumerate(val_loader):
        _image = batch_data["imgs"]
        label = batch_data["label"].cuda(non_blocking=True)
        label = label.reshape(-1)
        assert label.shape[0] == 1
        label_feat = labelDecode(label)
        name_id = label_feat[8]
        name_id=name_id.detach().item()
        # label_id = Feat2Idx(label_feat[:8]).to(label.device)
        diagnosis = label_feat[0].to(label.device).float()

        b, tn, c, h, w = _image.size()
        t = config.DATA.NUM_FRAMES
        n = tn // t
        _image = _image.view(b, n, t, c, h, w)
        assert b == 1 and n == 1


        image = _image[:, 0, :, :, :, :]  # [b,t,c,h,w]
        image_input = image.cuda(non_blocking=True)
        if config.TRAIN.OPT_LEVEL == 'O2':
            image_input = image_input.half()

        exp_input=image_input
        cls_set = model(image=exp_input, val=True)

        baseline_similarity = cls_set

        with open(save_dir + str(name_id) + ".txt", 'a') as f:
            f.write(str(name_id)+"###"+"prediction:"+str(baseline_similarity))
            f.write("\n")
        f.close()

        differ_maps = torch.zeros(exp_input.shape)

        step_len = 4

        topx=config.TEST.TOP_X
        top_num=config.TEST.TOP_X_PIXEL_NUM

        with torch.no_grad():
            for image_idx in range(exp_input.shape[1]):

                print("\033[0;31;40mexplaining image: \033[0m"+str(image_idx))
                #break
                for x_idx in range(int(exp_input.shape[3] / step_len)):
                    for y_idx in range(int(exp_input.shape[4] / step_len)):
                        print(image_idx, x_idx, y_idx)

                        masked_imgs = mask_imgs(exp_input, image_idx, x_idx, y_idx, step_len)

                        cls_set = model(image=masked_imgs, val=True)
                        similarity = cls_set

                        mark_influence(name_id,differ_maps, similarity - baseline_similarity, image_idx, x_idx, y_idx,
                                       step_len)

            for image_idx in range(exp_input.shape[1]):
                bk_img = exp_input.cpu().numpy()[0][image_idx].transpose(1, 2, 0)
                differ_map = differ_maps.cpu().numpy()[0][image_idx].transpose(1, 2, 0)
                result_generate(name_id,differ_map,bk_img,image_idx)






def mask_imgs(exp_input, image_idx, x_idx, y_idx, step_len):
    ret_imgs = exp_input.clone()
    #print(ret_imgs.shape)
    ret_imgs[0, image_idx, 0:3, x_idx * step_len:x_idx * step_len + step_len,
    y_idx * step_len:y_idx * step_len + step_len] = 0
    # print(ret_imgs)
    return ret_imgs


def mark_influence(name_id,differ_maps, similarity_differ, image_idx, x_idx, y_idx, step_len):

    assert (differ_maps[0, image_idx, 0:3, x_idx * step_len:x_idx * step_len + step_len,     y_idx * step_len:y_idx * step_len + step_len]==0).min()==True
    differ_maps[0, image_idx, 0:3, x_idx * step_len:x_idx * step_len + step_len,
    y_idx * step_len:y_idx * step_len + step_len] = similarity_differ

    with open(save_dir+str(name_id)+".txt", 'a') as f:
        f.write(str(image_idx))
        f.write(",")
        f.write(str(x_idx * step_len+1/2*step_len))
        f.write(",")
        f.write(str(y_idx * step_len + 1 / 2 * step_len))
        f.write("/")
        f.write(str(similarity_differ.detach().item()))
        f.write("\n")
    f.close()


def get_normal_color(img):
    max = np.max(img)
    min = np.min(img)
    if max!=min:
        ret = (img - min) / (max - min) * 255
    else:
        ret = (img - min) / 0.000000001 * 255
    return ret

# def keep_top_x_unique(matrix,top_x,top_num):
#     topx=top_x
#     # 将矩阵展平并找到前5个不同的值
#
#     unique_values, counts = np.unique(matrix.flatten(), return_counts=True)
#     print("unique_values")
#     print(matrix)
#     print(unique_values)
#     #top5_values = unique_values[np.argsort(counts)][::-1][:topx]
#     sorted_list = sorted(unique_values, reverse=True)
#     top_values = sorted_list[:topx]
#     print("top value")
#     print(top_values)
#
#     matching_elements = np.isin(matrix, top_values)
#     count = np.count_nonzero(matching_elements)
#     print(count)
#     if count<2:
#         print("top x error")
#         exit(1)
#
#     while count>top_num*24*24 and topx>1:
#         topx=topx-1
#         top_values = sorted_list[:topx]
#         matching_elements = np.isin(matrix, top_values)
#         count = np.count_nonzero(matching_elements)
#         print(count)
#     print("top_value_count")
#     print(count)
#
#     # 将不是前5个不同值的元素设置为0
#     mask = np.isin(matrix, top_values, invert=True)
#     #print(mask)
#     matrix[mask] = 0
#
#     return matrix



def result_generate(name_id,differ_map,bk_img,image_idx):
    bk_img = get_normal_color(bk_img)
    result_save(name_id,differ_map,bk_img,image_idx,"")







def result_save(name_id,differ_map,bk_img,image_idx,name:str):

    differ_map = get_normal_color(differ_map)
    bk_img = get_normal_color(bk_img)

    differ_map = differ_map[:, :, 0]
    differ_map = cv2.convertScaleAbs(differ_map)
    #print(differ_map.shape)
    heatmap = cv2.applyColorMap(differ_map, cv2.COLORMAP_JET)  # 将cam的结果转成伪彩色图片
    heatmap = np.float32(heatmap) / 255.  # 缩放到[0,1]之间
    bk_img = bk_img[...,::-1] / 255.
    if np.max(bk_img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")
    cam = heatmap + bk_img
    cam = cam / np.max(cam)

    path = save_dir+str(name_id)
    if os.path.exists(path) is False:
        os.makedirs(path)
    cv2.imwrite(path+'/hm_' + str(image_idx) +name+ '.jpg', np.uint8(255 * cam))

    if os.path.exists(path) is False:
        os.makedirs(path)
    cv2.imwrite(path+'/bk_' + str(image_idx) + name + '.jpg', np.uint8(255 *bk_img))




# def top_analyse(name_id,differ_map,bk_img,image_idx,topx,top_num,name):
#
#     print("img_idx:"+str(image_idx))
#     differ_map = get_normal_color(differ_map)
#     bk_img = get_normal_color(bk_img)
#
#     differ_map = keep_top_x_unique(differ_map, topx,top_num)
#     differ_map = get_normal_color(differ_map)
#     differ_map = differ_map[:, :, 0]
#     differ_map = cv2.convertScaleAbs(differ_map)
#     heatmap = cv2.applyColorMap(differ_map, cv2.COLORMAP_JET)  # 将cam的结果转成伪彩色图片
#     heatmap = np.float32(heatmap) / 255.  # 缩放到[0,1]之间
#     bk_img = bk_img / 255.
#     #print(bk_img)
#
#     if np.max(bk_img) > 1:
#         raise Exception(
#             "The input image should np.float32 in the range [0, 1]")
#
#     cam = heatmap + bk_img
#     cam = cam / np.max(cam)
#
#     path = "new_explain718/hm_" + str(name_id)
#     if os.path.exists(path) is False:
#         os.makedirs(path)
#
#     if name=="asd":
#         cv2.imwrite(path+'/heatmap_top_' + str(image_idx) + name+ '.jpg', np.uint8(255 * cam)[...,::-1])
#     else:
#         cv2.imwrite(path+'/heatmap_top_' + str(image_idx) + name + '.jpg', np.uint8(255 * cam)[...,::-1])
#
#     path = "new_explain718/bk_" + str(name_id)
#     if os.path.exists(path) is False:
#         os.makedirs(path)
#     cv2.imwrite(path+'/bk_' + str(image_idx) + name + '.jpg', np.uint8(255 *bk_img)[...,::-1])
#
