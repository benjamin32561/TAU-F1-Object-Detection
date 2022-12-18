import numpy as np
import torch
import torch.nn as nn

assert torch.__version__.split('.')[0] == '1'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def calc_iou(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])

    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)

    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih

    ua = torch.clamp(ua, min=1e-8)

    intersection = iw * ih

    IoU = intersection / ua

    return IoU

class FocalLoss(nn.Module):
    #def __init__(self):

    def forward(self, classifications, regressions, anchors, annotations):
        alpha = 0.25
        gamma = 2.0
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []

        anchor = anchors[0, :, :]

        anchor_widths  = anchor[:, 2] - anchor[:, 0]
        anchor_heights = anchor[:, 3] - anchor[:, 1]
        anchor_ctr_x   = anchor[:, 0] + 0.5 * anchor_widths
        anchor_ctr_y   = anchor[:, 1] + 0.5 * anchor_heights

        for j in range(batch_size):

            classification = classifications[j, :, :]
            regression = regressions[j, :, :]

            bbox_annotation = annotations[j, :, :]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]

            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)

            if bbox_annotation.shape[0] == 0:
                if torch.cuda.is_available():
                    alpha_factor = torch.ones(classification.shape).cuda() * alpha

                    alpha_factor = 1. - alpha_factor
                    focal_weight = classification
                    focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

                    bce = -(torch.log(1.0 - classification))

                    # cls_loss = focal_weight * torch.pow(bce, gamma)
                    cls_loss = focal_weight * bce
                    classification_losses.append(cls_loss.sum())
                    regression_losses.append(torch.tensor(0).float().cuda())

                else:
                    alpha_factor = torch.ones(classification.shape) * alpha

                    alpha_factor = 1. - alpha_factor
                    focal_weight = classification
                    focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

                    bce = -(torch.log(1.0 - classification))

                    # cls_loss = focal_weight * torch.pow(bce, gamma)
                    cls_loss = focal_weight * bce
                    classification_losses.append(cls_loss.sum())
                    regression_losses.append(torch.tensor(0).float())

                continue
            
            IoU = calc_iou(anchors[0, :, :], bbox_annotation[:, :4]) # num_anchors x num_annotations

            IoU_max, IoU_argmax = torch.max(IoU, dim=1) # num_anchors x 1

            #import pdb
            #pdb.set_trace()

            # compute the loss for classification
            targets = torch.ones(classification.shape) * -1

            targets = targets.to(DEVICE)

            targets[torch.lt(IoU_max, 0.4), :] = 0

            positive_indices = torch.ge(IoU_max, 0.5)

            num_positive_anchors = positive_indices.sum()

            assigned_annotations = bbox_annotation[IoU_argmax, :]

            targets[positive_indices, :] = 0
            targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1

            alpha_factor = torch.ones(targets.shape).to(DEVICE) * alpha

            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

            bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))

            # cls_loss = focal_weight * torch.pow(bce, gamma)
            cls_loss = focal_weight * bce

            tmp = torch.zeros(cls_loss.shape).to(DEVICE)
            cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, tmp)
            del tmp

            classification_losses.append(cls_loss.sum()/torch.clamp(num_positive_anchors.float(), min=1.0))

            # compute the loss for regression

            if positive_indices.sum() > 0:
                assigned_annotations = assigned_annotations[positive_indices, :]

                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

                gt_widths  = assigned_annotations[:, 2] - assigned_annotations[:, 0]
                gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                gt_ctr_x   = assigned_annotations[:, 0] + 0.5 * gt_widths
                gt_ctr_y   = assigned_annotations[:, 1] + 0.5 * gt_heights

                # clip widths to 1
                gt_widths  = torch.clamp(gt_widths, min=1)
                gt_heights = torch.clamp(gt_heights, min=1)

                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dw = torch.log(gt_widths / anchor_widths_pi)
                targets_dh = torch.log(gt_heights / anchor_heights_pi)

                targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh))
                targets = targets.t()

                if torch.cuda.is_available():
                    targets = targets/torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).cuda()
                else:
                    targets = targets/torch.Tensor([[0.1, 0.1, 0.2, 0.2]])

                negative_indices = 1 + (~positive_indices)

                regression_diff = torch.abs(targets - regression[positive_indices, :])

                regression_loss = torch.where(
                    torch.le(regression_diff, 1.0 / 9.0),
                    0.5 * 9.0 * torch.pow(regression_diff, 2),
                    regression_diff - 0.5 / 9.0
                )
                regression_losses.append(regression_loss.mean())
            else:
                regression_losses.append(torch.tensor(0).float().to(DEVICE))

        return torch.stack(classification_losses).mean(dim=0, keepdim=True), torch.stack(regression_losses).mean(dim=0, keepdim=True)


def Recall(tp,fn):
    return tp/(tp+fn)

def Precision(tp,fp):
    if (tp+fp)==0:
        return 0
    return tp/(tp+fp)

def ValidateModel(model,dataloader,loss_fun,IoU_thresh=0.5):
    model.training = False
    model.eval()

    n_images = len(dataloader)
    loss_data = []
    class_data = []
    bbx_data = []
    for idx, data in enumerate(dataloader):
        img = data['img'].to(torch.float32).to(DEVICE)
        clas,reg,anch,scores,class_pred,bbx_preds = model(img)
        annot = data['annot'].to(DEVICE)
        
        class_loss, reg_loss = loss_fun(clas,reg,anch,annot)

        loss_data.append([class_loss, reg_loss])
        n_pred_objects = class_pred.size()[0]

        annot = annot[0]
        bbx_label = annot[:,:-1]
        class_label = annot[:,-1]
        n_objects = bbx_label.size()[0]
        n_bbx_tp = 0
        n_bbx_fp = 0
        n_bbx_fn = 0
        n_class_tp = 0
        n_class_fp = 0
        n_class_fn = 0
        if n_pred_objects>0:
            #calculating class prediction fp,tp,fn
            for i in range(n_objects):
                bbx = bbx_label[i].repeat(n_pred_objects, 1)
                iou = calc_iou(bbx,bbx_preds)
                predicted_idx = iou>=IoU_thresh
                predicted = iou[predicted_idx]
                if predicted.size(0)==0:
                    n_bbx_fn+=1 #iou with every prediction gives iou<thresh
                    n_class_fn+=1 #failed to predict existing instance of class
                    continue
                bbx_class = class_label[i]
                rel_pred_class = class_pred[predicted_idx]
                n_current_class_tp=rel_pred_class[rel_pred_class==bbx_class].size(0)
                n_class_tp+=n_current_class_tp #iou>=thresh and same class
                del iou,bbx
            #calculating bbx prediction fp,tp,fn
            for predicted_bbx in bbx_preds:
                bbx = predicted_bbx.repeat(n_objects, 1)
                iou = calc_iou(bbx,bbx_label)
                rel_iou_idx = iou>=IoU_thresh
                n_rel_iou = iou[rel_iou_idx].size(0)
                if n_rel_iou==0:
                    n_bbx_fp+=1
                    n_class_fp+=1
                else:
                    n_bbx_tp+=1
                del iou,bbx
        else:
            n_bbx_fn+=n_objects
            n_class_fn+=n_objects
        
        class_data.append([Precision(n_class_tp,n_class_fp),Recall(n_class_fp,n_class_fn)])
        bbx_data.append([Precision(n_bbx_tp,n_bbx_fp),Recall(n_bbx_fp,n_bbx_fn)])
        print(f"\rValidating {idx+1}/{n_images}",end='')
        
        del img,clas,reg,anch,scores
        del bbx_preds,class_pred,annot
    
    class_data = np.array(class_data)
    bbx_data = np.array(bbx_data)
    loss_data = np.array(loss_data)

    print("\rValidation | c loss: {} | c precision: {} | c recall: {} |  bbx loss: {} | bbx precision: {} | bbx recall: {}".format(\
            loss_data[:,0].mean(),loss_data[:,1].mean(),class_data[:,0].mean(),class_data[:,1].mean(),bbx_data[:,0].mean(),bbx_data[:,1].mean()))

    #cls_loss,reg_loss,cls_pre,cls_rec,reg_pre,reg_rec
    return loss_data[:,0].mean(),loss_data[:,1].mean(), \
           class_data[:,0].mean(),class_data[:,1].mean(), \
           bbx_data[:,0].mean(),bbx_data[:,1].mean()