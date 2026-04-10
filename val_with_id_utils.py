import torch
import torch.distributed as dist
import numpy as np
from sklearn.metrics import roc_curve, auc
from utils.labelEncode import labelDecode

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

        find_person=False

        for person in self.person_pred_list:
            if int(person.name_id)==int(name_id):
                find_person=True
                person.add_pred(similarity)
                if person.gt_label_index!=gt_label_index:

                    raise ValueError(
                        f"Label mismatch for person {person.name_id}: "
                        f"existing={person.gt_label_index}, new={gt_label_index}"
                    )
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
    def add_pred(self, similarity):
        self.pred_list.append(similarity)
        self.average_similarity=np.mean(self.pred_list)
        self.count+=1
        if (similarity>=0.5 and self.gt_label_index==1) \
                or (similarity<0.5 and self.gt_label_index==0):
            self.match_number+=1






@torch.no_grad()
def validate_personwise(val_loader, text_labels, model, config,logger,writer,full_text_tokens_set,val_data,epoch):
    model.eval()


    video_gt = []
    video_pred = []
    pred_storage=Pred_storage()


    with torch.no_grad():

        logger.info(f"{config.TEST.NUM_CLIP * config.TEST.NUM_CROP} views inference")
        for idx, batch_data in enumerate(val_loader):
            _image = batch_data["imgs"]

            label = batch_data["label"].cuda(non_blocking=True)
            label = label.reshape(-1)
            assert label.shape[0] == 1
            label_feat = labelDecode(label)
            name_id=label_feat[8]
            diagnosis = label_feat[0].to(label.device).float()
            SAgt = torch.tensor([label_feat[6]], dtype=torch.float, device=label.device)
            RBBgt = torch.tensor([label_feat[7]], dtype=torch.float, device=label.device)

            b, tn, c, h, w = _image.size()
            t = config.DATA.NUM_FRAMES
            n = tn // t
            _image = _image.view(b, n, t, c, h, w)

            assert b==1 and n==1

            image = _image[:, 0, :, :, :, :]               
            image_input = image.cuda(non_blocking=True)
            if config.TRAIN.OPT_LEVEL == 'O2':
                image_input = image_input.half()

            cls_set = model(image=image_input, val=True)

            sig_func=torch.nn.Sigmoid()
            similarity=sig_func(cls_set)
            pred_storage.add_pred(name_id,similarity,diagnosis)
            video_gt.append(diagnosis.item())
            video_pred.append(similarity.item())


        local_total_match=0
        local_total_count=0
        local_person_match=0
        local_person_count=0


        for person in pred_storage.person_pred_list:
            logger.info(f"person name_id")
            logger.info(f"person Id:{str(person.name_id)}")

            logger.info(f"ground truth diagnosis: {person.gt_label_index}")
            logger.info(f"{person.match_number} in {person.count} prediction match")
            logger.info(f"model prediction: {person.pred_list}")
            local_total_count+=person.count
            local_total_match+=person.match_number

            local_person_count+=1
            if (person.average_similarity>=0.5 and person.gt_label_index==1) \
                    or (person.average_similarity<0.5 and person.gt_label_index==0):
                local_person_match+=1



        logger.info(f"[Video-wise] {local_total_match}/{local_total_count} correct, "
                    f"acc={local_total_match/local_total_count*100.:.2f}%")

        logger.info(f"[Person-wise] {local_person_match}/{local_person_count} correct, "
                    f"acc={local_person_match / local_person_count * 100.:.2f}%")



        logger.info(f"video prediction list:{video_pred}")
        logger.info(f"video ground truth list:{video_gt}")

        fpr, tpr, thresholds = roc_curve(video_gt, video_pred, pos_label=1)
        roc_auc = auc(fpr, tpr)
        logger.info(f"AUC={roc_auc:.4f}")

        def average_tensor(tensor):
            size = float(dist.get_world_size())
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            tensor /= size
            return tensor

        avg_auc=average_tensor(torch.tensor(roc_auc).to(similarity.device))
        avg_auc=avg_auc.cpu().numpy()
        writer.add_scalar('auc', avg_auc, epoch)

        return avg_auc








