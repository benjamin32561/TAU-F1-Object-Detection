import argparse
import numpy as np
import torch
import torch.optim as optim
import wandb
from torchvision import transforms
from os import makedirs
from os.path import join, exists

from retinanet import model
from retinanet.losses import FocalLoss
from retinanet.dataloader import CocoDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, Normalizer
from torch.utils.data import DataLoader
from retinanet.losses import ValidateModel
from copy import deepcopy

assert torch.__version__.split('.')[0] == '1'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--model_path', help='Path to model weights')
    parser.add_argument('--project_path', help='Path to folder to save models at',type=str, default='')
    parser.add_argument('--batch_size', help='Training batch size',type=int, default=4)
    parser.add_argument('--num_workers', help='Number of workers',type=int, default=3)
    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=100)
    parser.add_argument('--lr', help='Learning rate', type=float, default=1e-5)
    parser.add_argument('--start_from_epoch', help='Epoch to start training from (for resuming training)', type=int, default=0)
    parser.add_argument('--wandb_run_name', help='weights and biases run name')

    parser = parser.parse_args(args)

    # Create the data loaders

    # wandb.init(
    #         project="RetinaNet",
    #         name=parser.wandb_run_name,
    #         resume="allow")

    if parser.coco_path is None:
        raise ValueError('Must provide --coco_path when training on COCO,')

    if parser.project_path!='' and not exists(parser.project_path):
        makedirs(parser.project_path)

    dataset_train = CocoDataset(parser.coco_path, set_name='train2017',
                                transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
    dataset_val = CocoDataset(parser.coco_path, set_name='val2017',
                                transform=transforms.Compose([Normalizer(), Resizer()]))

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=parser.batch_size, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=parser.num_workers, collate_fn=collater, batch_sampler=sampler)

    if dataset_val is not None:
        sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
        dataloader_val = DataLoader(dataset_val, num_workers=parser.num_workers, collate_fn=collater, batch_sampler=sampler_val)

    # Create the model
    if parser.model_path is not None:
        retinanet = torch.load(parser.model_path)
    elif parser.depth == 18:
        retinanet = model.resnet18(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 34:
        retinanet = model.resnet34(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 50:
        retinanet = model.resnet50(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 101:
        retinanet = model.resnet101(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 152:
        retinanet = model.resnet152(num_classes=dataset_train.num_classes(), pretrained=True)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

    retinanet.to(DEVICE)

    if torch.cuda.is_available():
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet = torch.nn.DataParallel(retinanet)

    optimizer = optim.Adam(retinanet.parameters(), lr=parser.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    print('Num training images: {}'.format(len(dataset_train)))


    base_model_path = join(parser.project_path,'base.pt')
    torch.save(retinanet,base_model_path)
    best_model = -1
    best_loss = -1
    loss_func = FocalLoss()
    for epoch_num in range(parser.start_from_epoch,parser.epochs): 
        retinanet.training = True
        retinanet.train()
        retinanet.module.freeze_bn() #setting BN layers to eval()
        epoch_loss = []
        epoch_class_loss = []
        epoch_reg_loss = []
        n_iterations = len(dataloader_train)
        # for iter_num, data in enumerate(dataloader_train):
        #     optimizer.zero_grad()
            
        #     img_data = data['img'].to(torch.float32).to(DEVICE)
        #     classification_loss, regression_loss = retinanet([img_data, data['annot']])
                
        #     classification_loss = classification_loss.mean()
        #     regression_loss = regression_loss.mean()
        #     loss = classification_loss + regression_loss

        #     if bool(loss == 0):
        #         continue

        #     loss.backward()
        #     torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)
        #     optimizer.step()

        #     epoch_class_loss.append(float(classification_loss))
        #     epoch_reg_loss.append(float(regression_loss))
        #     epoch_loss.append(float(loss))

        #     del img_data
        #     del classification_loss
        #     del regression_loss

        #     print('\rEpoch: {} | Iteration: {}/{} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'
        #         .format(epoch_num,iter_num,n_iterations,np.mean(epoch_class_loss),np.mean(epoch_reg_loss),np.mean(epoch_loss)), end='')

        # scheduler.step(np.mean(epoch_loss))

        print('\nValidating model')
        # classification_val_loss = []
        # regression_val_loss = []
        # for iter_num, data in enumerate(dataloader_val):
        #     optimizer.zero_grad()

        #     img_data = data['img'].to(torch.float32).to(DEVICE)
        #     classification_loss, regression_loss = retinanet([img_data, data['annot']])
                
        #     classification_loss = classification_loss.mean()
        #     regression_loss = regression_loss.mean()
            
        #     classification_val_loss.append(float(classification_loss))
        #     regression_val_loss.append(float(regression_loss))
        #     del img_data
        #     del classification_loss
        #     del regression_loss
        #val_loss_sum = np.mean(regression_val_loss)+np.mean(classification_val_loss)
        
        retinanet.training = False
        retinanet.eval()
        val_cls_loss,val_reg_loss,val_cls_pre,val_cls_rec,val_reg_pre,val_reg_rec = ValidateModel(retinanet,dataloader_val,loss_func)

        print('Validation loss | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                val_cls_loss, val_reg_loss, val_reg_loss+val_cls_loss))

        if best_loss==-1 or best_loss>val_reg_loss+val_cls_loss:
            best_loss = val_reg_loss+val_cls_loss
            best_model = deepcopy(retinanet)
        print("Saving epoch data to wandb...\n")
        # wandb.log({
        #     "train loss":np.mean(epoch_loss),
        #     "train classification_loss":np.mean(epoch_class_loss),
        #     "train regression_loss":np.mean(epoch_reg_loss),
        #     "validation loss":val_reg_loss+val_cls_loss,
        #     "validation classification loss":val_cls_loss,
        #     "validation regression loss":val_reg_loss,
        #     "validation classification precision":val_cls_pre,
        #     "validation regression precision":val_reg_pre,
        #     "validation classification recall":val_cls_rec,
        #     "validation regression recall":val_reg_rec},
        #     step=epoch_num, commit=True)
        
        del epoch_loss
        del epoch_class_loss
        del epoch_reg_loss

    #saving epoch model
    best_model_path = join(parser.project_path,'best.pt')
    torch.save(best_model,best_model_path)
    final_model_path = join(parser.project_path,'last.pt')
    torch.save(retinanet, final_model_path)

if __name__ == '__main__':
    main()
