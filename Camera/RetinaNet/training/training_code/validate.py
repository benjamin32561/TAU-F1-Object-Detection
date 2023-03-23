import argparse
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms

from retinanet import model
from retinanet.dataloader import CocoDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, Normalizer
from torch.utils.data import DataLoader
from retinanet.losses import FocalLoss, ValidateModel

assert torch.__version__.split('.')[0] == '1'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main(args=None):
    parser = argparse.ArgumentParser(description='Simple validation script for RetinaNet.')

    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--model_path', help='Path to model weights')
    parser.add_argument('--batch_size', help='test batch size',type=int, default=4)

    parser = parser.parse_args(args)

    # Create the data loaders

    assert parser.coco_path is not None,'Must provide --coco_path when validating on COCO'
    assert parser.model_path is not None,'Must provide --model_path'

    dataset_val = CocoDataset(parser.coco_path, set_name='val2017',
                                transform=transforms.Compose([Normalizer(), Resizer()]))
    
    sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=parser.batch_size, drop_last=False)
    dataloader_val = DataLoader(dataset_val, num_workers=1, collate_fn=collater, batch_sampler=sampler_val,shuffle=False)

    # Create the model
    retinanet = torch.load(parser.model_path)

    retinanet = torch.nn.DataParallel(retinanet).to(DEVICE)

    retinanet.training = False
    retinanet.eval()

    loss_func = FocalLoss()
    ValidateModel(retinanet,dataloader_val,loss_func)

if __name__ == '__main__':
    main()
