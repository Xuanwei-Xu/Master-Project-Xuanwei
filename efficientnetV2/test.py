import os
import argparse

import torch
from torchvision import transforms
from model import efficientnetv2_s as create_model
from utils import read_split_data

import sys

sys.path.append("..")
from brain_dataset import MyDataSet

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


# 此文件注释可参考 train_resnet.py


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(args)
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')

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

    val_data_set = MyDataSet(images_path=val_images_path,
                             images_class=val_images_label,
                             transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 32])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    val_loader = torch.utils.data.DataLoader(val_data_set,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_data_set.collate_fn)

    model = create_model(num_classes=4).to(device)
    # load model weights
    model_weight_path = "./weights/efficientnetv2.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    ############################################################################
    correct = list(0. for i in range(args.num_classes))
    total = list(0. for i in range(args.num_classes))
    with torch.no_grad():
        model.eval()
        for i, data in enumerate(val_loader):
            images, labels = data

            output = model(images.to(device))

            prediction = torch.argmax(output, 1)
            res = prediction == labels.to(device)
            for label_idx in range(len(labels)):
                label_single = labels[label_idx]
                correct[label_single] += res[label_idx].item()
                total[label_single] += 1
        acc_str = 'Accuracy: %.4f' % (sum(correct) / sum(total))
        for acc_idx in range(args.num_classes):
            try:
                acc = correct[acc_idx] / total[acc_idx]
            except:
                acc = 0
            finally:
                acc_str += '\tclassID:%d\tacc:%.4f\t' % (acc_idx + 1, acc)
        print(acc_str)


############################################################################


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=32)

    parser.add_argument('--data-path', type=str,
                        default="../brain_dataset_24b")

    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)