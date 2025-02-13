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
from utils.asd_utils import get_asd_label_index,get_name_id,label_length,get_asd_label_index_with_id
from torch.utils.tensorboard import SummaryWriter  # Create an instance of the object
import sklearn.metrics
from utils.tools import reduce_tensor

import matplotlib.pyplot as plt
from utils.labelEncode import labelDecode,Feat2Idx

def convert_to_scalar(array_or_scalar):
    if isinstance(array_or_scalar, np.ndarray) and array_or_scalar.size == 1:
        return array_or_scalar.item()
    else:
        return array_or_scalar

class Pred_storage():
    def __init__(self):
        super().__init__()
        self.person_pred_list=[]

    def add_pred(self,name_id, similarity,gt_label_index):
        name_id=name_id.cpu().numpy()
        similarity=similarity.cpu().numpy()
        gt_label_index=gt_label_index.cpu().numpy()

        name_id=convert_to_scalar(name_id)
        similarity=convert_to_scalar(similarity)
        gt_label_index=convert_to_scalar(gt_label_index)


        print("add similarity")
        print(similarity)
        print(name_id)
        print(gt_label_index)
        find_person=False

        for person in self.person_pred_list:
            if int(person.name_id)==int(name_id):
                find_person=True
                print("\033[0;31;40mfind person\033[0m")
                person.add_pred(similarity)
                if person.gt_label_index!=gt_label_index:

                    print(person.name_id)
                    print(gt_label_index)
                    print(person.gt_label_index)
                    print("\033[0;31;40mstorage error\033[0m")
                    exit(1)
                break

        if find_person is False:
            new_person=Person_pred(name_id,gt_label_index)
            new_person.add_pred(similarity)
            self.person_pred_list.append(new_person)



class Person_pred():
    def __init__(self,name_id,gt_label_index):
        super().__init__()
        self.name_id=name_id
        self.pred_list=[]
        self.gt_label_index=gt_label_index
        self.average_similarity=0
        self.count=0
        self.match_number=0
        print("\033[0;31;40mnew person\033[0m")
        print("name: "+str(self.name_id))
        print("gt label: "+str(gt_label_index))
    def add_pred(self, similarity):
        self.pred_list.append(similarity)
        self.average_similarity=np.mean(self.pred_list)
        self.count+=1
        if similarity>=0.5 and self.gt_label_index==1\
            or similarity<0.5 and self.gt_label_index==0:
            self.match_number+=1






@torch.no_grad()
def validate_personwise(val_loader, text_labels, model, config,logger,writer,full_text_tokens_set,val_data,epoch):
    model.eval()


    local_gt = []
    local_pred = []
    local_sa=[]
    local_rbb=[]
    local_sa_gt=[]
    local_rbb_gt=[]
    pred_storage=Pred_storage()


    with torch.no_grad():

        logger.info(f"{config.TEST.NUM_CLIP * config.TEST.NUM_CROP} views inference")
        for idx, batch_data in enumerate(val_loader):
            _image = batch_data["imgs"]
            # label_id = batch_data["label"]
            # label_id_asd = get_asd_label_index_with_id(label_id)
            # print("\033[0;31;40m label asd\033[0m")
            # print(label_id_asd)
            # name_id=get_name_id(label_id)

            label = batch_data["label"].cuda(non_blocking=True)
            label = label.reshape(-1)
            assert label.shape[0] == 1
            label_feat = labelDecode(label)
            name_id=label_feat[8]
            #label_id = Feat2Idx(label_feat[:8]).to(label.device)
            diagnosis = label_feat[0].to(label.device).float()
            SAgt = torch.tensor([label_feat[6]], dtype=torch.float, device=label.device)
            RBBgt = torch.tensor([label_feat[7]], dtype=torch.float, device=label.device)

            b, tn, c, h, w = _image.size()
            t = config.DATA.NUM_FRAMES
            n = tn // t
            _image = _image.view(b, n, t, c, h, w)

            #tot_similarity = torch.zeros((b, 1)).cuda()

            assert b==1 and n==1

            image = _image[:, 0, :, :, :, :]  # [b,t,c,h,w]
            image_input = image.cuda(non_blocking=True)
            if config.TRAIN.OPT_LEVEL == 'O2':
                image_input = image_input.half()

            #cls_set,sa,rbb = model(image=image_input,val=True)
            cls_set = model(image=image_input, val=True)

            sig_func=torch.nn.Sigmoid()
            similarity=sig_func(cls_set)
            # print("\033[0;31;40msimilarty\033[0m")
            # print(similarity)
            # tot_similarity += similarity
            # print(tot_similarity)
            pred_storage.add_pred(name_id,similarity,diagnosis)
            # local_sa.append(sa)
            # local_rbb.append(rbb)
            # local_sa_gt.append(SAgt)
            # local_rbb_gt.append(RBBgt)


        '''单个gpu中计算'''
        local_total_match=0
        local_total_count=0
        local_person_match=0
        local_person_count=0
        # print("\033[0;31;40mperson number\033[0m")
        # print(len(pred_storage.person_pred_list))


        for person in pred_storage.person_pred_list:
            print("\033[0;31;40mperson name_id\033[0m")
            print("person name"+str(person.name_id))

            if config.TEST.GET_BEST==True:
                if person.gt_label_index>0.5:
                    best_pred=max(person.pred_list)
                else:
                    best_pred = min(person.pred_list)
                local_pred.append(best_pred)
                local_gt.append(person.gt_label_index)
            else:
                local_pred.append(person.average_similarity)
                local_gt.append(person.gt_label_index)
            #print("\033[0;31;40mperson name_id\033[0m")
            print(person.average_similarity,person.gt_label_index)
            print(person.match_number,person.count)
            print(person.pred_list)
            local_total_count+=person.count
            local_total_match+=person.match_number

            local_person_count+=1
            if person.average_similarity>=0.5 and person.gt_label_index==1 \
                or person.average_similarity<0.5 and person.gt_label_index==0:
                local_person_match+=1



        print("\033[0;31;40mvideowise acc\033[0m")
        print(local_total_match,local_total_count)
        print("acc="+str(local_total_match/local_total_count*100.))

        print("\033[0;31;40mpersonwise acc\033[0m")
        print(local_person_match, local_person_count)
        print("acc=" + str(local_person_match / local_person_count * 100.))



        '''绘制单独roc图'''
        from sklearn.metrics import roc_curve, auc
        print("\033[0;31;40mlocal pred gt\033[0m")
        print(local_pred)
        print(local_gt)

        fpr, tpr, thersholds = roc_curve(local_gt, local_pred,pos_label=1)
        for i, value in enumerate(thersholds):
            print("%f %f %f" % (fpr[i], tpr[i], value))
        roc_auc = auc(fpr, tpr)
        print("\033[0;31;40mauc\033[0m")
        print(roc_auc)
        # plt.plot(fpr, tpr, 'k--', label='ROC (area = {0:.2f})'.format(roc_auc), lw=2)
        # plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
        # plt.ylim([-0.05, 1.05])
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')  # 可以使用中文，但需要导入一些库即字体
        # plt.title('ROC Curve')
        # plt.legend(loc="lower right")
        # rank=dist.get_rank()
        # plt.savefig('result_roc/local_'+str(rank)+'person-wise.png')
        # plt.cla()
        # #exit(1)
        #
        # from sklearn.calibration import calibration_curve
        # prob_true, prob_pred = calibration_curve(local_gt,local_pred, n_bins=5)
        #
        # # 绘制校准曲线
        # plt.figure(figsize=(7, 4), dpi=128)
        # plt.plot(prob_pred, prob_true, marker='o', label='uncalibrated')
        # plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='perfectly calibrated')
        # plt.xlabel('Mean predicted probability')
        # plt.ylabel('Fraction of positives')
        # plt.title('Calibration Curve (Uncalibrated)')
        # plt.legend()
        # plt.savefig('result_calibration/local_new_' + str(rank) + 'video-wise.png')
        # plt.cla()

        def average_tensor(tensor):
            """计算张量的全局平均值"""
            size = float(dist.get_world_size())
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            tensor /= size
            return tensor

        avg_auc=average_tensor(torch.tensor(roc_auc).to(similarity.device))
        avg_auc=avg_auc.cpu().numpy()
        writer.add_scalar('auc', avg_auc, epoch)
        print("\033[0;31;40m global auc\033[0m")
        print(avg_auc)


        # def Change2Three(inp):
        #     result=[]
        #     for item in inp:
        #         if item.shape[0]==11:
        #             num=torch.argmax(item, dim = 0)
        #             num+=1
        #         else:
        #             num=item
        #         if 0 <= num <= 3:
        #             value = 0
        #         elif 4 <= num <= 6:
        #             value = 1
        #         elif 7 <= num <= 10:
        #             value = 2
        #         else:
        #             print("css error")
        #             exit(1)
        #         result.append(value)
        #     return result
        #
        # local_sa=Change2Three(local_sa)
        # local_rbb=Change2Three(local_rbb)
        # local_rbb_gt=Change2Three(local_rbb_gt)
        # local_sa_gt=Change2Three(local_sa_gt)
        #
        # import numpy as np
        #
        # def calculate_metrics(pred, gt):
        #     assert len(pred) == len(gt), "预测标签和真实标签长度不一致"
        #
        #     # 计算准确率
        #     correct_num = sum([1 for p, g in zip(pred, gt) if p == g])
        #     accuracy = correct_num / len(pred)
        #
        #     # 计算每个类别的精确率、召回率和F1 - score
        #     classes = np.unique(gt)
        #     precision_per_class = {}
        #     recall_per_class = {}
        #     f1_per_class = {}
        #     for c in classes:
        #         true_positives = sum([1 for p, g in zip(pred, gt) if p == c and g == c])
        #         predicted_positives = sum([1 for p in pred if p == c])
        #         actual_positives = sum([1 for g in gt if g == c])
        #         precision = true_positives / predicted_positives if predicted_positives else 0
        #         recall = true_positives / actual_positives if actual_positives else 0
        #         f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
        #         precision_per_class[c] = precision
        #         recall_per_class[c] = recall
        #         f1_per_class[c] = f1
        #
        #     return accuracy, precision_per_class, recall_per_class, f1_per_class
        #
        # acc_SA,_,_,_=calculate_metrics(local_sa,local_sa_gt)
        # acc_RBB,_,_,_=calculate_metrics(local_rbb,local_rbb_gt)
        # print("acc of SA: ",acc_SA)
        # print("acc of RBB: ",acc_RBB)
        # writer.add_scalar('test-acc SA', acc_SA, epoch)
        # writer.add_scalar('test-acc RBB', acc_RBB, epoch)

        return avg_auc








