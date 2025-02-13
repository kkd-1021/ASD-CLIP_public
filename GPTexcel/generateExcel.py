from copy import deepcopy

import pandas as pd

import random
import os
from utils.labelEncode import labelEncode
from utils.labelEncode import labelDecode,Feat2Idx,Idx2Feat
import pandas as pd





import requests
import json

from volcenginesdkarkruntime import Ark

describe_text=[
    ['无自闭症','有自闭症'],
    ['儿童粗大运动能力明显落后,','儿童粗大运动能力与同龄儿相仿,'],
    ['儿童视觉接受能力明显落后,','儿童视觉接受能力与同龄儿相仿,'],
    ['儿童精细运动能力明显落后,','儿童精细运动能力与同龄儿相仿,'],
    ['儿童语言理解能力明显落后,','儿童语言理解能力与同龄儿相仿,'],
    ['儿童语言表达能力明显落后,','儿童语言表达能力与同龄儿相仿,'],
    ['几乎没有表现出社会互动和沟通的缺陷，尽管他/她有时沉浸在轻微的专注和重复使用物体中,',
     '在整个评估中，在社交交流方面表现出缺陷，包括有限的眼神接触，使用手势较少，很少展示，指点，不会点头示意，面部表情不丰富，他/她以有限的方式跟养育者分享乐趣，但没有引导养育者注意到玩具或共同关注远端物体,',
     '很少发起社会互动，很少发声，很少关注人，叫名不理，不回应别人的话，喜欢自己玩,'],
    ['观察到社会互动和反应的轻微缺陷，以及眼神接触和手势的不一致使用。',
     '他/她还参与了不寻常的感官兴趣，并表现出对物体的专注/重复使用,',
     '经常参与重复的感官探索和重复的运动行为或物品操作行为,']
]

prompt_list=[]
for i in range(370+139,576):
    feat=Idx2Feat(i)
    prompt='这是一段儿童与家长互动的视频。视频中的儿童'

    prompt += describe_text[0][feat[0]]
    prompt += describe_text[6][feat[6]]
    prompt += describe_text[7][feat[7]]

    prompt += '更多描述：'
    prompt += describe_text[1][feat[1]]
    prompt += describe_text[2][feat[2]]
    prompt += describe_text[3][feat[3]]
    prompt += describe_text[4][feat[4]]
    prompt += describe_text[5][feat[5]]
    prompt_list.append(prompt)

result_list=[]

client = Ark(
    base_url="https://ark.cn-beijing.volces.com/api/v3",
)
# Streaming:
print("----- streaming request -----")
error_count=0
for i in range(576):
    #prompt='这是一段对自闭症亲子互动筛查视频的描述，视频中的儿童为1-2岁。请用英文概括，不要漏掉细节，需要强调有无自闭症，尽量简洁，要求使用不超过200个字母（包含空格和标点）描述，不要出现汉字和缩写：'+prompt_list[i]
    # prompt = '这是一段对自闭症亲子互动筛查视频的描述，视频中的儿童为1-2岁。请用英文概括，不要漏掉细节，需要强调有无自闭症，尽量简洁，使用两句话概括，第一句话概括’更多描述‘之前的内容，第二句话概括之后的内容。要求每句话使用不超过100个字母（包含空格和标点）描述，不要出现汉字和缩写：' + \
    #          prompt_list[i]

    prompt0 = '这是一段对自闭症亲子互动筛查视频的描述，视频中的儿童为1-2岁。请用英文概括，不要漏掉细节，需要强调有无自闭症，尽量简洁，使用不超过150个字母（包含空格和标点）描述，不要出现汉字和缩写：' + \
             prompt_list[i].split('更多描述')[0]
    prompt1 = '这是一段对自闭症亲子互动筛查视频的描述，视频中的儿童为1-2岁。请用英文概括，不要漏掉细节，需要强调有无自闭症，尽量简洁，使用不超过150个字母（包含空格和标点）描述，不要出现汉字和缩写：' + \
             prompt_list[i].split('更多描述')[1]

    result=[]
    completion = client.chat.completions.create(
        model="ep-20250107213555-kpfcx",
        messages = [
            {"role": "system", "content": "你是一个自闭症领域医学专家"},
            {"role": "user", "content": prompt0},
            #{"role": "user", "content": prompt1},
        ],

        extra_headers={'x-is-encrypted': 'true'},
        #stream=True
    )
    print(completion.choices[0].message.content)
    result.append(completion.choices[0].message.content)
    completion = client.chat.completions.create(
        model="ep-20250107213555-kpfcx",
        messages=[
            {"role": "system", "content": "你是一个自闭症领域医学专家"},
            #{"role": "user", "content": prompt0},
            {"role": "user", "content": prompt1},
        ],
        extra_headers={'x-is-encrypted': 'true'},
        # stream=True
    )
    print(completion.choices[0].message.content)
    result.append(completion.choices[0].message.content)
    print(len(result[0]),len(result[1]) )
    result_list.append(result)
    print(i)




df = pd.DataFrame({
    'content': result_list
})

# 将DataFrame写入Excel文件
df.to_excel('GPTexcel/GPT-2list_new.xlsx', index=False)