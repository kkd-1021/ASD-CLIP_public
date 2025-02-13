

import torch
import torch.distributed as dist
import matplotlib.pyplot as plt
import pandas as pd
import scipy
from utils.labelEncode import Idx2Feat

label_length=8


NAME_ID_RANGE=1000


def get_name_id(label_id):
    # print("\033[0;31;40m total id\033[0m")
    # print(label_id)
    # name_id=torch.zeros(label_id.shape).to(label_id.device)
    #
    # for idx in range(label_id.shape[0]):
    #     name_id[idx]=label_id[idx]%NAME_ID_RANGE
    # print("\033[0;31;40m name id\033[0m")
    # print(name_id)
    # return name_id
    print("已弃用")
    exit(1)



# describe_text864=[
#     ['is normal','is stunted in development','is autism'],
#     ['his body movements are flexible','his body movements are not flexible'],
#     ['is good at completing cognitive function tasks','has difficulty in completing cognitive function tasks'],
#     ['his hand movements are flexible','his hand movements are not flexible'],
#     ['can understand what others say','can not understand what others say'],
#     ['can say','can not say'],
#     ['has a good interaction with an adult','has a fair interaction with an adult','has a bad interaction with an adult'],
#     ['has no special interests','has fair special interests', 'has strong special interests']
# ]



def get_text_id(label_id):

    # gt_id_set=[]
    # for idx in range(label_id.shape[0]):
    #     label_num = label_id[idx].cpu().numpy()
    #     '''label_num 8位数映射到0-864'''
    #
    #     label_num_str=str(label_num)
    #     reversed_string = label_num_str[::-1]
    #
    #     gt_id=0
    #
    #     if NUM_CLASS==864:
    #         num_ut=[1,3,9,18,36,72,144,288]
    #     else:
    #         num_ut = [1, 2, 4, 8, 16, 32, 64, 128]
    #
    #     for char_idx,char in enumerate(reversed_string):
    #         gt_id+=num_ut[char_idx]*int(char)
    #
    #     if gt_id>=NUM_CLASS:
    #         print("gt id error")
    #         exit(1)
    #     gt_id_set.append(gt_id)
    #
    # print("\033[0;31;40mtext id of the video=\033[0m")
    # gt_id_set = torch.tensor(gt_id_set)
    # print(gt_id_set)
    # return gt_id_set
    print("已弃用")
    exit(1)



# def get_full_describe_text(clip):
#     describe_set = []
#     for i in range(0, NUM_CLASS):
#         if NUM_CLASS==864:
#             asd_label=i//((3**2)*(2**5))
#             mullen_label_list=[]
#             for idx in range(0,5):
#                 mullen_label=i//((3**2)*(2**idx))
#                 mullen_label=mullen_label%2
#                 mullen_label_list.append(mullen_label)
#
#             css_sa_label=i//3
#             css_sa_label=css_sa_label%3
#
#             css_rbb_label = css_sa_label % 3
#
#             if asd_label>2:
#                 print("asd_label error")
#                 exit(1)
#             describe = "a child " + describe_text864[0][asd_label]
#
#             for mullen_idx, mullen_label in enumerate(mullen_label_list):
#                 describe = describe + ' and ' + describe_text864[mullen_idx + 1][mullen_label]
#
#             describe = describe + ' and ' + describe_text864[6][css_sa_label]
#             describe = describe + ' and ' + describe_text864[7][css_rbb_label]
#
#             print(describe)
#
#             text_aug = f"{{}}"
#             describe_tokens = clip.tokenize(text_aug.format(describe), context_length=77)
#             # print(describe_tokens)
#             describe_set.append(describe_tokens[0])
#
#         else:
#             asd_label = i // ((2 ** 2) * (2 ** 5))
#             mullen_label_list = []
#             for idx in range(0, 5):
#                 mullen_label = i // ((2 ** 2) * (2 ** idx))
#                 mullen_label = mullen_label % 2
#                 mullen_label_list.append(mullen_label)
#
#             css_sa_label = i // 2
#             css_sa_label = css_sa_label % 2
#
#             css_rbb_label = css_sa_label % 2
#
#             if asd_label > 1:
#                 print("asd_label error")
#                 exit(1)
#
#             describe = "a child "  + describe_text256[0][asd_label]
#
#             for mullen_idx,mullen_label in enumerate(mullen_label_list):
#                 describe = describe + ' and ' + describe_text256[mullen_idx+1][mullen_label]
#
#             describe = describe + ' and ' + describe_text256[6][css_sa_label]
#             describe = describe + ' and ' + describe_text256[7][css_rbb_label]
#
#             print(describe)
#
#             text_aug = f"{{}}"
#             describe_tokens = clip.tokenize(text_aug.format(describe), context_length=77)
#             # print(describe_tokens)
#             describe_set.append(describe_tokens[0])
#
#
#
#     describe_set = torch.stack(describe_set)
#     print("\033[0;31;40mfull describe set=\033[0m")
#     #print(describe_set)
#     print(describe_set.shape)
#     return describe_set

def get_full_describe_text(clip,NUM_CLASS):
    df = pd.read_excel('./GPTexcel/GPT-2list_new.xlsx')
    describe_set = []
    assert NUM_CLASS==576
    for i in range(0, NUM_CLASS):
        #tokenSet=[]
        #feat=Idx2Feat(i)
        #describe='the child is typically developing, his/her body movements are flexible, is good at completing cognitive function tasks, his/her hand movements are flexible, can understand what others say and can say, has a good interaction with an adult, and has no special interests.'
        describe1 = df.iloc[i].tolist()[0]
        describe2 = df.iloc[i].tolist()[1]
        #describe1="The video shows an interaction between a 1-2-year-old child and a parent. The child has no autism. During the assessment, there are deficits in social communication like limited eye contact, few gestures, seldom showing or pointing, no nodding. Facial expressions are not rich. The child shares fun with the caregiver in a limited way, "
        #describe2="without guiding attention to toys or having joint attention on distal objects. Also, the child's gross motor, visual receptive, fine motor, language comprehension and expression abilities are significantly behind, and engages in repetitive sensory exploration and object manipulation behaviors"
        text_aug = f"{{}}"
        #describe_tokens = clip.tokenize(text_aug.format(describe), context_length=77)

        describe_tokens1 = clip.tokenize(text_aug.format(describe1), context_length=77)
        describe_tokens2 = clip.tokenize(text_aug.format(describe2), context_length=77)
        # try:
        #     describe_tokens1 = clip.tokenize(text_aug.format(describe1), context_length=77)
        #     #describe_tokens2 = clip.tokenize(text_aug.format(describe2), context_length=77)
        # except:
        #     print("ADOS",i+2)
        #     describe_tokens1 = [1]
        #     describe_tokens2 = [1]
        # try:
        #     #describe_tokens1 = clip.tokenize(text_aug.format(describe1), context_length=77)
        #     describe_tokens2 = clip.tokenize(text_aug.format(describe2), context_length=77)
        # except:
        #     print("MULLEN",i+2)
        #     describe_tokens1 = [1]
        #     describe_tokens2 = [1]
        #tokenSet=[describe_tokens1[0],describe_tokens2[0]]
        describe_set.append(describe_tokens1[0])
        describe_set.append(describe_tokens2[0])

    describe_set = torch.stack(describe_set)
    return describe_set


describe_text256=[
    ['is normal,','is autism,'],
    ['his body movements are not flexible,','his body movements are flexible,'],
    ['has difficulty in completing cognitive function tasks,','is good at completing cognitive function tasks,',],
    ['his hand movements are not flexible,','his hand movements are flexible,'],
    ['can not understand what others say,','can understand what others say,'],
    ['can not say,','can say,'],
    ['has a good interaction with an adult','has a bad interaction with an adult,'],
    ['has no special interests','has strong special interests,']
]
def get_full_describe_text_NOGPT(clip,NUM_CLASS):
    describe_set = []
    assert NUM_CLASS==576
    for i in range(0, 576):

        feat = Idx2Feat(i)
        prompt1 = 'The child'
        if feat[6]==1 or feat[6]==2:
            feat[6]=1
        if feat[7]==1 or feat[7]==2:
            feat[7]=1
        prompt1 += describe_text256[0][feat[0]]
        prompt1 += describe_text256[6][feat[6]]
        prompt1 += describe_text256[7][feat[7]]
        prompt2 = 'The child'
        prompt2 += describe_text256[1][feat[1]]
        prompt2 += describe_text256[2][feat[2]]
        prompt2 += describe_text256[3][feat[3]]
        prompt2 += describe_text256[4][feat[4]]
        prompt2 += describe_text256[5][feat[5]]

        describe1 = prompt1
        describe2 = prompt2
        text_aug = f"{{}}"
        describe_tokens1 = clip.tokenize(text_aug.format(describe1), context_length=77)
        describe_tokens2 = clip.tokenize(text_aug.format(describe2), context_length=77)

        describe_set.append(describe_tokens1[0])
        describe_set.append(describe_tokens2[0])

    describe_set = torch.stack(describe_set)
    return describe_set

def get_asd_label_index(label_id):
    # #只能用于val asd 3类
    # div=10**7
    # list=[]
    # for idx in range(label_id.shape[0]):
    #     label_id_asd=torch.zeros(1).to(label_id.device)
    #     if NUM_CLASS==864:
    #         if label_id[idx] / div >= 2:
    #             label_id_asd[0]=1#is asd
    #         else:
    #             label_id_asd[0]=0
    #     else:
    #         if label_id[idx] / div >= 1:
    #             label_id_asd[0]=1#is asd
    #         else:
    #             label_id_asd[0]=0
    #
    #     list.append(label_id_asd)
    # label_id_asd = torch.stack(list)
    # label_id_asd = torch.tensor(label_id_asd, dtype=torch.float32)
    # return label_id_asd
    print("已弃用")
    exit(1)



def get_asd_label_index_with_id(label_id):
    # #label_id_asd = torch.zeros(label_id.shape).to(label_id.device)
    # #div=10**7
    # div=10**10
    # list=[]
    # for idx in range(label_id.shape[0]):
    #     label_id_asd=torch.zeros(1).to(label_id.device)
    #     if label_id[idx] / div >= 1:
    #         label_id_asd[0]=1#is asd
    #     else:
    #         label_id_asd[0]=0
    #     list.append(label_id_asd)
    # label_id_asd = torch.stack(list)
    # label_id_asd = torch.tensor(label_id_asd, dtype=torch.float32)
    # return label_id_asd
    print("已弃用")
    exit(1)






def backward_hook(module, grad_in, grad_out):
    print("\033[0;31;40mgrad\033[0m")
    print(grad_out)
    print(grad_in)
def build_cls_optimizer_scheduler(model,config):
    model = model.module if hasattr(model, 'module') else model
    if not config.TRAIN.ONLY_FINETUNE:
        print("train together setting\n")
        optimizer = torch.optim.Adam(model.classification_net.parameters(), lr=0.1, eps=1e-08)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0.000001)
    else:
        print("ft_setting\n")
        optimizer = torch.optim.Adam(model.classification_net.parameters(), lr=0.005, eps=1e-08)
        scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.7, last_epoch=-1)

    return optimizer,scheduler



