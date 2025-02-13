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
from utils.asd_utils import get_name_id,label_length,get_asd_label_index
from torch.utils.tensorboard import SummaryWriter  # Create an instance of the object
from utils.asd_utils import get_text_id
import clip
from utils.asd_utils import get_full_describe_text
from utils.labelEncode import labelDecode,Feat2Idx

# def int_to_tensor(num):
#     num = num.long().item() if isinstance(num, torch.Tensor) and num.dtype.is_floating_point else num
#     tensor = torch.zeros(11)
#     assert 0 <= num <= 10
#     tensor[num] = 1
#     return tensor

def train_one_epoch(epoch, model, criterion, optimizer, lr_scheduler, train_loader, text_labels, config,
                    mixup_fn,logger,writer,full_text_tokens_set,cls_optimizer,cls_lr_scheduler):
    tmp=None
    model.train()
    optimizer.zero_grad()
    cls_optimizer.zero_grad()

    num_steps = len(train_loader)
    batch_time = AverageMeter()
    tot_loss_meter = AverageMeter()
    start = time.time()
    end = time.time()


    for idx, batch_data in enumerate(train_loader):


        images = batch_data["imgs"].cuda(non_blocking=True)
        images = images.view((-1, config.DATA.NUM_FRAMES, 3) + images.size()[-2:])

        label = batch_data["label"].cuda(non_blocking=True)
        label = label.reshape(-1)
        assert label.shape[0]==1
        label_feat = labelDecode(label)[:8]
        label_id=Feat2Idx(label_feat.copy(),orgCss=True).to(label.device)
        assert label_id<576
        diagnosis=label_feat[0].clone().to(label.device).float()
        SAgt = torch.tensor([label_feat[6]], dtype=torch.float, device=label.device)
        RBBgt = torch.tensor([label_feat[7]], dtype=torch.float, device=label.device)

        # #获得每个label对应的gt——label（即对应full_text_tokens_set的哪一个）
        # gt_label_index=get_text_id(label_id)#index
        # gt_label_index=gt_label_index.to(label_id.device)
        # print("\033[0;31;40mgt label\033[0m")
        # print(gt_label_index)#256个标签中的一个index
        # label_id_asd_ind = get_asd_label_index(label_id)
        # label_id_asd_ind = label_id_asd_ind.to(label_id.device)


        if full_text_tokens_set.shape[0] == 1:
            full_text_tokens_set = full_text_tokens_set.view(1, -1)


        images, gt_label_vec = mixup_fn(images, label_id)

        output,cls_output,sa,rbb = model(images, full_text_tokens_set,val=False)

        #对比学习
        contrastive_loss = criterion(output, gt_label_vec)

        # #分类
        # print("\033[0;31;40mcls output\033[0m")
        # print(cls_output)
        # print("\033[0;31;40mlabel id asd ind\033[0m")
        # print(label_id_asd_ind)
        cls_criterion=nn.BCEWithLogitsLoss(reduction='sum')
        cls_loss = cls_criterion(cls_output, diagnosis)


        rbb_criterion = nn.BCEWithLogitsLoss(reduction='sum')
        sa_loss = rbb_criterion(sa, SAgt*0.1)
        sa_criterion = nn.BCEWithLogitsLoss(reduction='sum')
        rbb_loss = sa_criterion(rbb, RBBgt*0.1)
        assert RBBgt*0.1<=1. and SAgt*0.1<=1.
        # RBBgt=int_to_tensor(RBBgt).float().to(label.device)
        # SAgt = int_to_tensor(SAgt).float().to(label.device)
        # rbb_criterion = SoftTargetCrossEntropy()
        # rbb_loss = rbb_criterion(rbb, RBBgt)
        # sa_criterion = SoftTargetCrossEntropy()
        # sa_loss = sa_criterion(sa, SAgt)



        print("\033[0;31;40mcls loss total\033[0m")
        print(cls_loss)
        print(f"\033[0;31;40mcontrastive loss total\033[0m")
        print(contrastive_loss)

        writer.add_scalar('cls loss', cls_loss, idx + epoch * len(train_loader))
        writer.add_scalar('contrastive loss', contrastive_loss, idx + epoch * len(train_loader))
        writer.add_scalar('sa loss', sa_loss, idx + epoch * len(train_loader))
        writer.add_scalar('rbb loss', rbb_loss, idx + epoch * len(train_loader))

        if config.TRAIN.NO_CLASSIFIER:
            print(f"\033[0;31;40m NO_CLASS\033[0m")
            total_loss=contrastive_loss+(cls_loss+rbb_loss+sa_loss)*1e-100
        elif config.TRAIN.NO_TEXT:
            total_loss = contrastive_loss* 1e-100 + (cls_loss + rbb_loss + sa_loss)
        else:
            #total_loss=contrastive_loss+cls_loss*0.5
            total_loss = contrastive_loss * 0.2 + cls_loss + rbb_loss+ sa_loss
            #total_loss = contrastive_loss * 0. + cls_loss + rbb_loss* 0. + sa_loss* 0.
            writer.add_scalar('loss', total_loss, idx + epoch * len(train_loader))

        print("\033[0;31;40mloss\033[0m")
        print(total_loss)

        total_loss = total_loss / config.TRAIN.ACCUMULATION_STEPS

        # if config.TRAIN.ACCUMULATION_STEPS == 1:
        #     optimizer.zero_grad()
        #     #cls_optimizer.zero_grad()
        assert config.TRAIN.ACCUMULATION_STEPS > 1


        if config.TRAIN.OPT_LEVEL != 'O0':
            with amp.scale_loss(total_loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            total_loss.backward()

        lr = optimizer.state_dict()['param_groups'][0]['lr']
        print("\033[0;31;40mlr=\033[0m" + str(lr))
        writer.add_scalar('xclip_lr', optimizer.state_dict()['param_groups'][0]['lr'], idx + epoch * len(train_loader))
        writer.add_scalar('cls_lr', optimizer.param_groups[8]['lr'], idx + epoch * len(train_loader))

        if not config.TRAIN.ONLY_FINETUNE:
            print("\033[0;31;40mclip optimizing\033[0m" + str(lr))
            if config.TRAIN.ACCUMULATION_STEPS > 1:
                if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                    optimizer.step()
                    lr_scheduler.step_update(epoch * num_steps + idx)
                    # cls_optimizer.step()
                    # cls_lr_scheduler.step()
                    # cls_optimizer.zero_grad()
                    optimizer.zero_grad()
                    # for name, param in model.named_parameters():
                    #     if  'classi'in name:
                    #         print(tmp)
                    #         print(param.data)
                    #         tmp=param.data.clone()
                    #         break
                    writer.add_histogram(list(model.named_parameters())[-10][0], list(model.named_parameters())[-10][1].clone().cpu().data.numpy(),idx + epoch * len(train_loader))
            else:
                print("error")
                exit(1)
        else:
            print("\033[0;31;40m only cls optimizing\033[0m" + str(lr))
            exit(1)

        # '''classify优化'''
        # if (idx + 1) % (config.TRAIN.ACCUMULATION_STEPS/2) == 0:
        #     cls_optimizer.step()
        #     #cls_lr_scheduler.step()
        #     cls_optimizer.zero_grad()
        #     writer.add_scalar('cls_lr', cls_optimizer.param_groups[0]['lr'], idx + epoch * len(train_loader))

        # if config.TRAIN.ACCUMULATION_STEPS > 1:
        #     print("\033[0;31;40macc step\033[0m" )
        #     print(config.TRAIN.ACCUMULATION_STEPS)
        #     if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
        #         optimizer.zero_grad()



        torch.cuda.synchronize()

        tot_loss_meter.update(total_loss.item(), len(label_id))
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']

            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.15f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'tot_loss {tot_loss_meter.val:.4f} ({tot_loss_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
            writer.add_scalar('val', tot_loss_meter.val, idx + epoch * len(train_loader))
            writer.add_scalar('avg', tot_loss_meter.avg, idx + epoch * len(train_loader))

    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


