import numpy
import torch.distributed as dist
import torch
import clip
import os


def reduce_tensor(tensor, n=None):
    if n is None:
        n = dist.get_world_size()
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt = rt / n
    return rt
   

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def sync(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        val = torch.tensor(self.val).cuda()
        sum_v = torch.tensor(self.sum).cuda()
        count = torch.tensor(self.count).cuda()
        self.val = reduce_tensor(val, world_size).item()
        self.sum = reduce_tensor(sum_v, 1).item()
        self.count = reduce_tensor(count, 1).item()
        self.avg = self.sum / self.count


def epoch_saving(config, epoch, model,  max_accuracy, optimizer, lr_scheduler, logger, working_dir, is_best):
    #print(model.state_dict())
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'epoch': epoch,
                  'config': config}
    
    save_path = os.path.join(working_dir, f'ckpt_epoch_{epoch}.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")
    if is_best:
        best_path = os.path.join(working_dir, f'best.pth')
        torch.save(save_state, best_path)
        logger.info(f"{best_path} saved !!!")


def load_checkpoint(config, model, optimizer, lr_scheduler, logger):
    if os.path.isfile(config.MODEL.RESUME): 
        logger.info(f"==============> Resuming from {config.MODEL.RESUME}....................")
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
        load_state_dict = checkpoint['model']


        #print("in load cp")
        #print(load_state_dict.keys())

        # assert has_classification_key(load_state_dict)
        # if config.TRAIN.CLASS_NET_STRUCT_CHANGE:
        #     for key in list(load_state_dict):
        #         if "classification_net" in key:
        #             del load_state_dict[key]
        #     assert not has_classification_key(load_state_dict)



        #msg = model.load_state_dict(load_state_dict, strict=False)
        msg = load_model_with_resize(model, load_state_dict)
        logger.info(f"resume model: {msg}")



        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

            start_epoch = checkpoint['epoch'] + 1
            max_accuracy = checkpoint['max_accuracy']

            logger.info(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")
            
            del checkpoint
            torch.cuda.empty_cache()

            return start_epoch, max_accuracy
        except:
            #print("checkpoint load fail")
            #exit(1)
            del checkpoint
            torch.cuda.empty_cache()
            return 0, 0.

    else:
        logger.info(("=> no checkpoint found at '{}'".format(config.MODEL.RESUME)))
        return 0, 0


def auto_resume_helper(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        print(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file


def generate_text(data):
    text_aug = f"{{}}"
    print("\033[0;31;40mtext=\033[0m")
    print(data.classes)
    classes = torch.cat([clip.tokenize(text_aug.format(c), context_length=77) for i, c in data.classes])
    print(classes)

    return classes

def has_classification_key(my_dict):
    for key in my_dict:
        if "classification_net" in key:
            return True
    return False

import torch


def load_model_with_resize(model, load_state_dict):
    own_state = model.state_dict()
    for name, param in load_state_dict.items():
        if name in own_state:
            if param.size()!= own_state[name].size():
                # 检查维度是否匹配
                #print("ckp param not match")
                param_dims = len(param.size())
                own_state_dims = len(own_state[name].size())
                if param_dims == 1:  # 处理一维张量
                    param_length = param.size(0)
                    required_length = own_state[name].size(0)
                    if required_length % param_length == 0:
                        repeat_times = required_length // param_length
                        new_param = param.repeat(repeat_times)
                        own_state[name].copy_(new_param)
                    else:
                        raise ValueError(f"Cannot resize param {name} from length {param_length} to {required_length}")
                elif param_dims == 2:  # 处理二维张量
                    param_rows, param_cols = param.size()
                    required_rows, required_cols = own_state[name].size()
                    # 计算在两个维度上需要重复的次数
                    repeat_times1 = required_rows // param_rows
                    repeat_times2 = required_cols // param_cols
                    new_param = param.repeat(repeat_times1, repeat_times2)
                    own_state[name].copy_(new_param)
                elif param_dims == 3:  # 处理三维张量
                    param_d1, param_d2, param_d3 = param.size()
                    required_d1, required_d2, required_d3 = own_state[name].size()
                    # 计算在三个维度上需要重复的次数
                    repeat_times1 = required_d1 // param_d1
                    repeat_times2 = required_d2 // param_d2
                    repeat_times3 = required_d3 // param_d3
                    new_param = param.repeat(repeat_times1, repeat_times2, repeat_times3)
                    own_state[name].copy_(new_param)
                elif param_dims == 4:  # 处理四维张量
                    param_d1, param_d2, param_d3, param_d4 = param.size()
                    required_d1, required_d2, required_d3, required_d4 = own_state[name].size()
                    # 计算在四个维度上需要重复的次数
                    repeat_times1 = required_d1 // param_d1
                    repeat_times2 = required_d2 // param_d2
                    repeat_times3 = required_d3 // param_d3
                    repeat_times4 = required_d4 // param_d4
                    new_param = param.repeat(repeat_times1, repeat_times2, repeat_times3, repeat_times4)
                    own_state[name].copy_(new_param)
                elif param_dims == 5:  # 处理五维张量
                    param_d1, param_d2, param_d3, param_d4, param_d5 = param.size()
                    required_d1, required_d2, required_d3, required_d4, required_d5 = own_state[name].size()
                    # 计算在五个维度上需要重复的次数
                    repeat_times1 = required_d1 // param_d1
                    repeat_times2 = required_d2 // param_d2
                    repeat_times3 = required_d3 // param_d3
                    repeat_times4 = required_d4 // param_d4
                    repeat_times5 = required_d5 // param_d5
                    new_param = param.repeat(repeat_times1, repeat_times2, repeat_times3, repeat_times4, repeat_times5)
                    own_state[name].copy_(new_param)
                else:
                    raise ValueError(f"Unsupported tensor dimension for param {name}: {param.size()}")
            else:
                own_state[name].copy_(param)
    # 加载调整后的状态字典
    msg=model.load_state_dict(own_state)
    return msg
