import torch
import torch.distributed as dist
import pandas as pd
from utils.labelEncode import Idx2Feat

label_length=8


def get_name_id(label_id):
    raise DeprecationWarning("get_name_id is deprecated")


def get_text_id(label_id):
    raise DeprecationWarning("get_text_id is deprecated")


def get_full_describe_text(clip,NUM_CLASS):
    df = pd.read_excel('./GPTexcel/GPT-2list_new.xlsx')
    describe_set = []
    assert NUM_CLASS==576
    for i in range(0, NUM_CLASS):
        describe1 = df.iloc[i].tolist()[0]
        describe2 = df.iloc[i].tolist()[1]
        text_aug = f"{{}}"

        describe_tokens1 = clip.tokenize(text_aug.format(describe1), context_length=77)
        describe_tokens2 = clip.tokenize(text_aug.format(describe2), context_length=77)

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
    raise DeprecationWarning("get_asd_label_index is deprecated")


def get_asd_label_index_with_id(label_id):
    raise DeprecationWarning("get_asd_label_index_with_id is deprecated")




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



