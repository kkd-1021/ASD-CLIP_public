import os
import torch
import numpy as np
import cv2
from utils.labelEncode import labelDecode

save_dir = './explain_result/'

@torch.no_grad()
def explain(val_loader, model, config):
    model.eval()

    os.makedirs(save_dir, exist_ok=True)

    for idx, batch_data in enumerate(val_loader):
        _image = batch_data["imgs"]
        label = batch_data["label"].cuda(non_blocking=True)
        label = label.reshape(-1)
        assert label.shape[0] == 1
        label_feat = labelDecode(label)
        name_id = label_feat[8]
        name_id=name_id.detach().item()
        diagnosis = label_feat[0].to(label.device).float()

        b, tn, c, h, w = _image.size()
        t = config.DATA.NUM_FRAMES
        n = tn // t
        _image = _image.view(b, n, t, c, h, w)
        assert b == 1 and n == 1

        image = _image[:, 0, :, :, :, :]               
        image_input = image.cuda(non_blocking=True)
        if config.TRAIN.OPT_LEVEL == 'O2':
            image_input = image_input.half()

        exp_input=image_input
        cls_set = model(image=exp_input, val=True)

        baseline_similarity = cls_set

        with open(save_dir + str(name_id) + ".txt", 'a') as f:
            f.write(str(name_id)+"###"+"prediction:"+str(baseline_similarity))
            f.write("\n")

        differ_maps = torch.zeros(exp_input.shape)

        step_len = 4

        with torch.no_grad():
            for image_idx in range(exp_input.shape[1]):

                print(f"Explaining image: {image_idx}")
                for x_idx in range(int(exp_input.shape[3] / step_len)):
                    for y_idx in range(int(exp_input.shape[4] / step_len)):
                        print(image_idx, x_idx, y_idx)

                        masked_imgs = mask_imgs(exp_input, image_idx, x_idx, y_idx, step_len)

                        cls_set = model(image=masked_imgs, val=True)
                        similarity = cls_set

                        mark_influence(name_id,differ_maps, baseline_similarity - similarity, image_idx, x_idx, y_idx,
                                       step_len)

            for image_idx in range(exp_input.shape[1]):
                bk_img = exp_input.cpu().numpy()[0][image_idx].transpose(1, 2, 0)
                differ_map = differ_maps.cpu().numpy()[0][image_idx].transpose(1, 2, 0)
                result_generate(name_id,differ_map,bk_img,image_idx)


def mask_imgs(exp_input, image_idx, x_idx, y_idx, step_len):
    ret_imgs = exp_input.clone()
    ret_imgs[0, image_idx, 0:3, x_idx * step_len:x_idx * step_len + step_len,
    y_idx * step_len:y_idx * step_len + step_len] = 0
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


def get_normal_color(img):
    img_max = np.max(img)
    img_min = np.min(img)
    if img_max != img_min:
        ret = (img - img_min) / (img_max - img_min) * 255
    else:
        ret = np.zeros_like(img)
    return ret


def result_generate(name_id,differ_map,bk_img,image_idx):
    bk_img = get_normal_color(bk_img)
    result_save(name_id,differ_map,bk_img,image_idx,"")


def result_save(name_id,differ_map,bk_img,image_idx,name:str):

    differ_map = get_normal_color(differ_map)
    bk_img = get_normal_color(bk_img)

    differ_map = differ_map[:, :, 0]
    differ_map = cv2.convertScaleAbs(differ_map)
    heatmap = cv2.applyColorMap(differ_map, cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255.
    bk_img = bk_img[...,::-1] / 255.
    if np.max(bk_img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")
    cam = heatmap + bk_img
    cam = cam / np.max(cam)

    path = save_dir+str(name_id)
    os.makedirs(path, exist_ok=True)
    cv2.imwrite(path+'/hm_' + str(image_idx) +name+ '.jpg', np.uint8(255 * cam))
    cv2.imwrite(path+'/bk_' + str(image_idx) + name + '.jpg', np.uint8(255 *bk_img))
