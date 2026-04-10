import torch
import torch.nn as nn
import torch.distributed as dist
import datetime
from utils.tools import AverageMeter
import time
import numpy as np
from apex import amp
from utils.labelEncode import labelDecode, Feat2Idx

def train_one_epoch(epoch, model, criterion, optimizer, lr_scheduler, train_loader, text_labels, config,
                    mixup_fn,logger,writer,full_text_tokens_set,cls_optimizer,cls_lr_scheduler):
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

        if full_text_tokens_set.shape[0] == 1:
            full_text_tokens_set = full_text_tokens_set.view(1, -1)


        images, gt_label_vec = mixup_fn(images, label_id)

        output,cls_output,sa,rbb = model(images, full_text_tokens_set,val=False)

        contrastive_loss = criterion(output, gt_label_vec)

        cls_criterion=nn.BCEWithLogitsLoss(reduction='sum')
        cls_loss = cls_criterion(cls_output, diagnosis)


        rbb_criterion = nn.BCEWithLogitsLoss(reduction='sum')
        sa_loss = rbb_criterion(sa, SAgt*0.1)
        sa_criterion = nn.BCEWithLogitsLoss(reduction='sum')
        rbb_loss = sa_criterion(rbb, RBBgt*0.1)
        assert RBBgt*0.1<=1. and SAgt*0.1<=1.

        writer.add_scalar('cls loss', cls_loss, idx + epoch * len(train_loader))
        writer.add_scalar('contrastive loss', contrastive_loss, idx + epoch * len(train_loader))
        writer.add_scalar('sa loss', sa_loss, idx + epoch * len(train_loader))
        writer.add_scalar('rbb loss', rbb_loss, idx + epoch * len(train_loader))

        total_loss = contrastive_loss * 0.2 + cls_loss + rbb_loss + sa_loss
        writer.add_scalar('loss', total_loss, idx + epoch * len(train_loader))

        logger.info(f"loss={total_loss}")

        total_loss = total_loss / config.TRAIN.ACCUMULATION_STEPS

        assert config.TRAIN.ACCUMULATION_STEPS > 1


        if config.TRAIN.OPT_LEVEL != 'O0':
            with amp.scale_loss(total_loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            total_loss.backward()

        writer.add_scalar('xclip_lr', optimizer.state_dict()['param_groups'][0]['lr'], idx + epoch * len(train_loader))
        writer.add_scalar('cls_lr', optimizer.param_groups[8]['lr'], idx + epoch * len(train_loader))

        if not config.TRAIN.ONLY_FINETUNE:
            if config.TRAIN.ACCUMULATION_STEPS > 1:
                if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                    optimizer.step()
                    lr_scheduler.step_update(epoch * num_steps + idx)
                    optimizer.zero_grad()
                    writer.add_histogram(list(model.named_parameters())[-10][0], list(model.named_parameters())[-10][1].clone().cpu().data.numpy(),idx + epoch * len(train_loader))
            else:
                raise RuntimeError("ACCUMULATION_STEPS must be > 1")
        else:
            raise RuntimeError("ONLY_FINETUNE mode is not supported in this training pipeline")

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


