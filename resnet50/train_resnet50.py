import os
import math
import argparse

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler
from model import resnet50
import torch.nn as nn

import sys

sys.path.append("..")
from brain_dataset import MyDataSet
from data_tools import read_split_data, train_one_epoch, evaluate

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu") 

    print(args)
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    tb_writer = SummaryWriter()

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    train_data_set = MyDataSet(images_path=train_images_path,
                               images_class=train_images_label,
                               transform=data_transform["train"])

    val_data_set = MyDataSet(images_path=val_images_path,
                             images_class=val_images_label,
                             transform=data_transform["val"])
    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 32])
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_data_set,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_data_set.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_data_set,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_data_set.collate_fn)

    net = resnet50() 
    model_weight_path = args.weights 
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    net.load_state_dict(torch.load(model_weight_path, map_location=device)) 

    in_channel = net.fc.in_features 
    net.fc = nn.Linear(in_channel, args.num_classes) 
    net.to(device) 

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1E-4, nesterov=True)

    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    best_acc = 0.0
    for epoch in range(args.epochs):
        mean_loss, train_sum_num = train_one_epoch(model=net,
                                                   optimizer=optimizer,
                                                   data_loader=train_loader,
                                                   device=device,
                                                   epoch=epoch)

        scheduler.step() 

        sum_num = evaluate(model=net,
                           data_loader=val_loader,
                           device=device)
        val_acc = sum_num / len(val_data_set) 
        train_acc = train_sum_num / len(train_data_set)
        print("[epoch {}]  train_accuracy: {} val_accuracy: {}".format(epoch, round(train_acc, 4),
                                                                       round(val_acc, 4))) 
        tags = ["loss", "val_accuracy", "learning_rate", "train_accuracy"] 
        tb_writer.add_scalar(tags[0], mean_loss, epoch) 
        tb_writer.add_scalar(tags[1], val_acc, epoch) 
        tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch) 
        tb_writer.add_scalar(tags[3], train_acc, epoch) 

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(net.state_dict(), "./resnet50.pth")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=4)  
    parser.add_argument('--epochs', type=int, default=100)  
    parser.add_argument('--batch-size', type=int, default=16)  
    parser.add_argument('--lr', type=float, default=0.001) 
    parser.add_argument('--lrf', type=float, default=0.1)  
    parser.add_argument('--data-path', type=str,
                        default="../brain_dataset_24b")  
    parser.add_argument('--weights', type=str, default='./resnet50-pre.pth', 
                        help='initial weights path')
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)') 

    opt = parser.parse_args()

    main(opt)
