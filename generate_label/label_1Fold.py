from copy import deepcopy
import pandas as pd
from label_utils import get_person_name, find_value_from_xlsx
import random
import os
from utils.labelEncode import labelEncode, labelDecode, Feat2Idx, Idx2Feat

child_map = []

def get_id(child_name):
    for id in range(len(child_map)):
        if child_name == child_map[id]:
            return id
    child_map.append(child_name)
    return len(child_map) - 1

class Person_storage():
    def __init__(self):
        self.person_list = []

    def add_video(self, video_name):
        person_name = get_person_name(video_name.split("_")[0])
        for person in self.person_list:
            if person.person_name == person_name:
                person.add_video(video_name)
                return
        print("new person_name: " + person_name)
        new_person = Person(person_name, len(self.person_list))
        new_person.add_video(video_name)
        self.person_list.append(new_person)

class Person():
    def __init__(self, person_name, person_id):
        self.person_name = person_name
        self.video_list = []
        self.label = 'TD'
        self.person_id = person_id

    def add_video(self, video_name):
        self.video_list.append(video_name)

    def set_label(self, label):
        if label not in ['ASD', 'TD', 'BAP']:
            print("label error")
            exit(1)
        self.label = label

f = open('clinical_data/total_video_list.txt', 'r')
person_storage = Person_storage()
for txt_data in f:
    video_name = txt_data.strip('\n')
    person_storage.add_video(video_name)
f.close()

print("person number= " + str(len(person_storage.person_list)))

io = r'clinical_data/totaldata.xlsx'
excel_data = pd.read_excel(io, sheet_name=0)

asd_person_list = []
bap_person_list = []
td_person_list = []

for person in person_storage.person_list:
    asd_ret = find_value_from_xlsx('diagnosis', excel_data, person.person_name)
    person.set_label(asd_ret)
    if asd_ret == 'ASD':
        asd_person_list.append(person)
    elif asd_ret == 'BAP':
        bap_person_list.append(person)
    elif asd_ret == 'TD':
        td_person_list.append(person)

dist_path = "video_labels/"

random.shuffle(asd_person_list)
random.shuffle(bap_person_list)
random.shuffle(td_person_list)


# val_ratio = 0.1
# asd_val_size = int(len(asd_person_list) * val_ratio)
# bap_val_size = int(len(bap_person_list) * val_ratio)
# td_val_size = int(len(td_person_list) * val_ratio)

# Assuming to use DemoChildA, B for training and DemoChildC for verification in the demo
asd_val_size = 1
bap_val_size = 0
td_val_size = 1

val_set = asd_person_list[:asd_val_size] + bap_person_list[:bap_val_size] + td_person_list[:td_val_size]
train_set = [element for element in person_storage.person_list if element not in val_set]

mullen_feat_set = ['mullen:gross motor', 'mullen:visual reception', 'mullen:fine motor', 'mullen:receptive language', 'mullen:expressive language']
css_feat_set = ['SA CSS', 'RRB CSS']

# 构建 train_label.txt
train_txt_file_name = dist_path + 'train_label.txt'
if os.path.exists(train_txt_file_name):
    print("file already exists")
    exit(1)

for itr_person in train_set:
    person = deepcopy(itr_person)
    while len(person.video_list) < 6:
        person.video_list.append(random.choice(person.video_list))

    total_feat = []
    asd_name = find_value_from_xlsx('diagnosis', excel_data, person.person_name)
    if asd_name == 'ASD':
        asd_ret = 1
    elif asd_name in ['BAP', 'TD']:
        asd_ret = 0
    else:
        print("diagnosis reading error")
        exit(1)
    total_feat.append(asd_ret)

    for mullen_feat in mullen_feat_set:
        mullen_ret = find_value_from_xlsx(mullen_feat, excel_data, person.person_name)
        mullen_ret = int(mullen_ret >= 35)
        total_feat.append(mullen_ret)

    for css_feat in css_feat_set:
        css_feat_ret = find_value_from_xlsx(css_feat, excel_data, person.person_name)
        total_feat.append(css_feat_ret)

    total_feat.append(person.person_id)
    label_id = labelEncode(total_feat)

    with open(train_txt_file_name, 'a') as f:
        for video_name in person.video_list:
            f.write(video_name)
            f.write(' ')
            f.write(str(label_id))
            f.write('\n')

# write val_label.txt
val_txt_file_name = dist_path + 'val_label.txt'
if os.path.exists(val_txt_file_name):
    print("file already exists")
    exit(1)

for person in val_set:
    total_feat = []
    asd_name = find_value_from_xlsx('diagnosis', excel_data, person.person_name)
    if asd_name == 'ASD':
        asd_ret = 1
    elif asd_name in ['BAP', 'TD']:
        asd_ret = 0
    else:
        print("diagnosis reading error")
        exit(1)
    total_feat.append(asd_ret)

    for mullen_feat in mullen_feat_set:
        mullen_ret = find_value_from_xlsx(mullen_feat, excel_data, person.person_name)
        mullen_ret = int(mullen_ret >= 35)
        total_feat.append(mullen_ret)

    for css_feat in css_feat_set:
        css_feat_ret = find_value_from_xlsx(css_feat, excel_data, person.person_name)
        total_feat.append(css_feat_ret)

    total_feat.append(person.person_id)
    label_id = labelEncode(total_feat)

    assert total_feat == labelDecode(label_id)

    with open(val_txt_file_name, 'a') as f:
        for video_name in person.video_list:
            f.write(video_name)
            f.write(' ')
            f.write(str(label_id))
            f.write('\n')