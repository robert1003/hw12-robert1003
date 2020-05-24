import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
from models import FeatureExtractor, LabelPredictor, DomainClassifier, Generator

# hyper params
parser = argparse.ArgumentParser()
parser.add_argument('--use_gpu', action='store_true', help='use gpu for training')
parser.add_argument('--gpu', type=str, default='0', help='num of gpu')
parser.add_argument('--n_epoch', type=int, default=500, help='training epochs')
parser.add_argument('--model_name', type=str, required=True, help='model name')
parser.add_argument('--log_name', type=str, required=True, help='log name')
parser.add_argument('--source', type=str, required=True, help='source dataroot')
parser.add_argument('--target', type=str, required=True, help='target dataroot')
parser.add_argument('--batch_size', type=int, default=64, help='batch size for train data')
parser.add_argument('--from_cycle', action='store_true', help='is data from cycle or not')
parser.add_argument('--adaptive_lamb', action='store_true', help='use adaptive lamb')
parser.add_argument('--resnet_type', type=int, default=18, help='which resnet to use')

args = parser.parse_args()

# logger
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[logging.FileHandler(args.log_name, 'w'), logging.StreamHandler(sys.stdout)]
)

logging.info(args)

device = 'cuda' if args.use_gpu else 'cpu'
if args.use_gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

# load dataset
transform_target = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15, fill=(0,)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
transform_test = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
if not args.from_cycle:
    transform_source = transforms.Compose([
        transforms.Grayscale(),
        transforms.Lambda(lambda x: cv2.Canny(np.array(x), 170, 300)),
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15, fill=(0,)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
else:
    transform_source = transforms.Compose([
        transforms.Grayscale(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15, fill=(0,)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])


source_dataset = ImageFolder(args.source, transform=transform_source)
target_dataset = ImageFolder(args.target, transform=transform_target)

source_loader = DataLoader(source_dataset, batch_size=args.batch_size, shuffle=True)
target_loader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=True)

# models
F = FeatureExtractor(resnet=args.resnet_type).to(device)
C = LabelPredictor(resnet=args.resnet_type).to(device)
D = DomainClassifier(resnet=args.resnet_type).to(device)

class_criterion = nn.CrossEntropyLoss()
domain_criterion = nn.BCEWithLogitsLoss()

opt_F = optim.Adam(F.parameters())
opt_C = optim.Adam(C.parameters())
opt_D = optim.Adam(D.parameters())

# train
F.train()
D.train()
C.train()
lamb, p, gamma, now, tot = 0, 0, 10, 0, len(source_loader) * args.n_epoch
if not args.adaptive_lamb:
    lamb = 0.1
for epoch in range(args.n_epoch):
    domain_loss, class_loss = 0, 0
    total_hit, total_num = 0, 0
    for i, ((source_data, source_label), (target_data, _)) in enumerate(zip(source_loader, target_loader)):

        source_data = source_data.to(device)
        source_label = source_label.to(device)
        target_data = target_data.to(device)

        mixed_data = torch.cat([source_data, target_data], dim=0)
        domain_label = torch.zeros((source_data.shape[0] + target_data.shape[0], 1)).to(device)
        domain_label[:source_data.shape[0]] = 1

        # train D
        feature = F(mixed_data)
        domain_logits = D(feature.detach())
        loss1 = domain_criterion(domain_logits, domain_label)
        domain_loss += loss1.item()
        opt_D.zero_grad()
        loss1.backward()
        opt_D.step()

        # train F, C
        class_logits = C(feature[:source_data.shape[0]])
        domain_logits = D(feature)
        loss2 = class_criterion(class_logits, source_label) - lamb * domain_criterion(domain_logits, domain_label)
        class_loss += loss2.item()
        opt_F.zero_grad()
        opt_C.zero_grad()
        loss2.backward()
        opt_F.step()
        opt_C.step()
        
        # accuracy
        hit, num = torch.sum(torch.argmax(class_logits, dim=1) == source_label).item(), source_data.shape[0]
        total_hit += hit
        total_num += num

        if args.adaptive_lamb:
            # adjust lamb, p
            p = now / tot
            lamb = 2.0 / (1.0 + np.exp(-gamma * p)) - 1.0
            now += 1

        print('Epoch {}/{} Iter {}/{} lamb {:.5f} domain_loss {:.5f} class_loss {:.5f} accuracy {:.5f}'.format(
                epoch, args.n_epoch, i + 1, len(source_loader), lamb, loss1, loss2, hit / num), end='\r')

    print('')
    logging.info('Epoch {}/{} domain_loss {:.5f} class_loss {:.5f} accuracy {:.5f}'.format(
            epoch, args.n_epoch, domain_loss / (i + 1), class_loss / (i + 1), total_hit / total_num))

    # save model
    torch.save({
        'feature_extractor': F.state_dict(),
        'label_predictor': C.state_dict()
    }, args.model_name)
