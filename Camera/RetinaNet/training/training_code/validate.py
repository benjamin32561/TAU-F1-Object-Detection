import argparse
import collections
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
from os import makedirs
from os.path import join, exists

from retinanet import model
from retinanet.dataloader import CocoDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, Normalizer
from torch.utils.data import DataLoader
from retinanet.losses import calc_iou

assert torch.__version__.split('.')[0] == '1'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def Validation(model,dataloader,IoU_thresh=0.5):
    n_images = len(dataloader)
    for idx, data in enumerate(dataloader):
        img = data['img'].to(torch.float32).to(DEVICE)
        annot = data['annot'][0]

        bbx_label = annot[:,:-1]
        class_label = annot[:,-1]
        n_objects = bbx_label.size()[0]

        scores, class_pred, bbx_preds = model(img)
        n_pred_objects = class_pred.size()[0]

        n_bbx_tp = 0
        n_bbx_fp = 0
        n_bbx_fn = 0
        n_class_tp = 0
        n_class_fp = 0
        n_class_fn = 0
        #calculating class prediction
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

        #calculating bbx prediction
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
        

        del img
        del scores
        del class_pred
        del bbx_preds
        break

def main(args=None):
    parser = argparse.ArgumentParser(description='Simple validation script for RetinaNet.')

    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--model_path', help='Path to model weights')

    parser = parser.parse_args(args)

    # Create the data loaders

    assert parser.coco_path is not None,'Must provide --coco_path when validating on COCO'
    assert parser.model_path is not None,'Must provide --model_path'

    dataset_val = CocoDataset(parser.coco_path, set_name='val2017',
                                transform=transforms.Compose([Normalizer(), Resizer()]))
    
    sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
    dataloader_val = DataLoader(dataset_val, num_workers=1, collate_fn=collater, batch_sampler=sampler_val)

    # Create the model
    retinanet = torch.load(parser.model_path)

    retinanet = torch.nn.DataParallel(retinanet).to(DEVICE)

    retinanet.training = False
    retinanet.eval()

    Validation(retinanet,dataloader_val)

if __name__ == '__main__':
    main()
