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
from models import xclip
import torch.nn.functional as F
from utils.asd_utils import get_name_id, label_length
from torch.utils.tensorboard import SummaryWriter
from train_utils import train_one_epoch
from utils.asd_utils import get_full_describe_text, get_full_describe_text_NOGPT
import clip
from utils.asd_utils import build_cls_optimizer_scheduler
from explainer_utils import explain
from val_with_id_utils import validate_personwise



project_path = ("tf-logs")
name = "ASD-clip"
tb_writer_summary_path = os.path.join(project_path, "ASD-clip", name, "Logs")
current_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())
log_dir = os.path.join(tb_writer_summary_path, current_time)
writer = SummaryWriter(log_dir=log_dir, comment=name)



def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', required=True, type=str, default='configs/k400/32_8.yaml')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('--output', type=str, default="exp")
    parser.add_argument('--resume', type=str)
    parser.add_argument('--pretrained', type=str)
    parser.add_argument('--only_test', action='store_true')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--accumulation-steps', type=int)

    parser.add_argument("--local_rank", type=int, default=-1, help='local rank for DistributedDataParallel')
    args = parser.parse_args()

    config = get_config(args)

    return args, config




def main(config):



    train_data, val_data, train_loader, val_loader = build_dataloader(logger, config)
    model, _ = xclip.load(config.MODEL.PRETRAINED, config.MODEL.ARCH, 
                         device="cpu", jit=False, 
                         T=config.DATA.NUM_FRAMES, 
                         droppath=config.MODEL.DROP_PATH_RATE, 
                         use_checkpoint=config.TRAIN.USE_CHECKPOINT, 
                         use_cache=config.MODEL.FIX_TEXT,
                         logger=logger,
                        )
    model = model.cuda()




    mixup_fn = None

    if config.AUG.MIXUP > 0:
        criterion = SoftTargetCrossEntropy()
        mixup_fn = CutmixMixupBlending(num_classes=config.DATA.NUM_CLASSES, 
                                       smoothing=config.AUG.LABEL_SMOOTH, 
                                       mixup_alpha=config.AUG.MIXUP, 
                                       cutmix_alpha=config.AUG.CUTMIX, 
                                       switch_prob=config.AUG.MIXUP_SWITCH_PROB)
    elif config.AUG.LABEL_SMOOTH > 0:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.AUG.LABEL_SMOOTH)
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = build_optimizer(config, model)
    lr_scheduler = build_scheduler(config, optimizer, len(train_loader))

    cls_optimizer,cls_lr_scheduler=build_cls_optimizer_scheduler(model,config)


    if config.TRAIN.OPT_LEVEL != 'O0':
        model, optimizer = amp.initialize(models=model, optimizers=optimizer, opt_level=config.TRAIN.OPT_LEVEL)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False, find_unused_parameters=False)



    start_epoch, max_auc = 0, 0.0


    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        start_epoch, max_auc = load_checkpoint(config, model.module, optimizer, lr_scheduler, logger)




    text_labels = None
    if config.TRAIN.USE_GPT:
        full_text_tokens_set = get_full_describe_text(clip,config.DATA.NUM_CLASSES)
    else:
        full_text_tokens_set = get_full_describe_text_NOGPT(clip,config.DATA.NUM_CLASSES)

    if config.TEST.ONLY_TEST:
        validate_personwise(val_loader, text_labels, model, config, logger, writer, full_text_tokens_set, val_data, 0)
    if config.TEST.EXPLAIN:
        explain(val_loader, model, config)
        return
    if config.TEST.ONLY_TEST:
        return

    for epoch in range(start_epoch, config.TRAIN.EPOCHS):
        train_loader.sampler.set_epoch(epoch)
        train_one_epoch(epoch, model, criterion, optimizer, lr_scheduler, train_loader, text_labels, config, mixup_fn,logger,writer,
                       full_text_tokens_set,cls_optimizer,cls_lr_scheduler)

        auc=validate_personwise(val_loader, text_labels, model, config, logger, writer, full_text_tokens_set, val_data, epoch)
        writer.add_scalar('test-auc', auc,epoch)

        is_best = auc > max_auc
        max_auc = max(max_auc, auc)
        logger.info(f'Max AUC: {max_auc:.2f}%')

        if epoch == config.TRAIN.EPOCHS-1:
            epoch_saving(config, epoch, model.module, max_auc, optimizer, lr_scheduler, logger, config.OUTPUT, is_best)

    logger.info("Training finished.")
    writer.close()



if __name__ == '__main__':
                    
    args, config = parse_option()

                      
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier(device_ids=[args.local_rank])

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

                        
    Path(config.OUTPUT).mkdir(parents=True, exist_ok=True)
    
            
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.ARCH}")
    logger.info(f"working dir: {config.OUTPUT}")
    
                  
    if dist.get_rank() == 0:
        logger.info(config)
        shutil.copy(args.config, config.OUTPUT)

    main(config)


