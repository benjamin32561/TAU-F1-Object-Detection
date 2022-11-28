import argparse
import collections
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms

from retinanet import model
from retinanet.dataloader import CocoDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, Normalizer
from torch.utils.data import DataLoader

assert torch.__version__.split('.')[0] == '1'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--model_path', help='Path to model weights')
    parser.add_argument('--save_model_path', help='Path to save model at /path/to/model.(pth/pt)')
    parser.add_argument('--batch_size', help='Training batch size',type=int, default=8)
    parser.add_argument('--val_batch_size', help='Validation batch size',type=int, default=8)
    parser.add_argument('--num_workers', help='Number of workers',type=int, default=3)
    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=100)
    parser.add_argument('--lr', help='Learning rate', type=float, default=1e-5)
    parser.add_argument('--start_from_epoch', help='Epoch to start training from (for resuming training)', type=int, default=0)

    parser = parser.parse_args(args)

    # Create the data loaders

    if parser.coco_path is None:
        raise ValueError('Must provide --coco_path when training on COCO,')

    dataset_train = CocoDataset(parser.coco_path, set_name='train2017',
                                transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
    dataset_val = CocoDataset(parser.coco_path, set_name='val2017',
                                transform=transforms.Compose([Normalizer(), Resizer()]))

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=parser.batch_size, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=parser.num_workers, collate_fn=collater, batch_sampler=sampler)

    if dataset_val is not None:
        sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=parser.val_batch_size, drop_last=False)
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
    loss_hist = collections.deque(maxlen=500)

    print('Num training images: {}'.format(len(dataset_train)))

    for epoch_num in range(parser.start_from_epoch,parser.epochs): 
        retinanet.training = True
        retinanet.train()
        retinanet.module.freeze_bn()

        epoch_loss = []
        n_iterations = len(dataloader_train)
        for iter_num, data in enumerate(dataloader_train):
            try:
                optimizer.zero_grad()
                
                img_data = data['img'].to(torch.float32).to(DEVICE)
                classification_loss, regression_loss = retinanet([img_data, data['annot']])
                    
                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()
                loss = classification_loss + regression_loss

                if bool(loss == 0):
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)
                optimizer.step()

                loss_hist.append(float(loss))
                epoch_loss.append(float(loss))

                print('\rEpoch: {} | Iteration: {}/{} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'
                    .format(epoch_num,iter_num,n_iterations,float(classification_loss),float(regression_loss),np.mean(loss_hist)), end='')

                del img_data
                del classification_loss
                del regression_loss
            except Exception as e:
                print(e)
                continue

        scheduler.step(np.mean(epoch_loss))

        print('\nEvaluating model...')
        retinanet.training = False
        retinanet.eval()
        retinanet.module.freeze_bn()
        classification_val_loss = []
        regression_val_loss = []
        for iter_num, data in enumerate(dataloader_train):
            optimizer.zero_grad()
            if torch.cuda.is_available():
                classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot']])
            else:
                classification_loss, regression_loss = retinanet([data['img'].float(), data['annot']])
                
            classification_loss = classification_loss.mean()
            regression_loss = regression_loss.mean()
            
            classification_val_loss.append(float(classification_loss))
            regression_val_loss.append(float(regression_loss))
            del classification_loss
            del regression_loss
        
        class_val_loss = np.mean(classification_val_loss)
        regression_val_loss = np.mean(regression_val_loss)
        print('validation data: Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}\n'.format(
                float(class_val_loss), float(regression_val_loss), regression_val_loss+class_val_loss))

        torch.save(retinanet, 'retinanet_epoch{}.pt'.format(epoch_num))

    retinanet.eval()

    torch.save(retinanet, parser.save_model_path)

if __name__ == '__main__':
    main()
