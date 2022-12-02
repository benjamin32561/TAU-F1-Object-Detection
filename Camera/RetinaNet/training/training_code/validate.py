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

        print(bbx_label.size())
        print(bbx_label)
        print(class_label.size())
        print(class_label)

        scores, class_pred, bbx_preds = model(img)

        print(bbx_preds.size())
        print(bbx_preds)
        print(class_pred.size())
        print(class_pred)

        n_objects = class_label.size()[0]
        for predicted_bbx in bbx_preds:
            bbx = predicted_bbx.repeat(n_objects, 1)
            iou = calc_iou(bbx,bbx_label)
            print(iou)
            current_bbx_tp_idx = iou>=IoU_thresh
            current_bbx_fp_idx = iou<IoU_thresh
        

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
