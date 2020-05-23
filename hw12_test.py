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
from models import FeatureExtractor, LabelPredictor, DomainClassifier

# hyper params
parser = argparse.ArgumentParser()
parser.add_argument('--use_gpu', action='store_true', help='use gpu for training')
parser.add_argument('--model_checkpoint', type=str, required=True, help='model checkpoint')
parser.add_argument('--dataroot', type=str, required=True, help='train/test dataroot')
parser.add_argument('--batch_size', type=int, default=256, help='batch size for train data')
parser.add_argument('--output_csv', type=str, required=True, help='predict file')

args = parser.parse_args()

device = 'cuda' if args.use_gpu else 'cpu'

# load dataset
transform_test = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

target_dataset = ImageFolder(f'{args.dataroot}/test_data', transform=transform_test)

target_loader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=False)

# models
F = FeatureExtractor().to(device)
C = LabelPredictor().to(device)
D = DomainClassifier().to(device)

checkpoint = torch.load(args.model_checkpoint)
F.load_state_dict(checkpoint['feature_extractor'])
C.load_state_dict(checkpoint['label_predictor'])

# predict
F.eval()
C.eval()
result = []
for i, (data, _) in enumerate(target_loader):
    print(i + 1, len(target_loader), end='\r')
    data = data.to(device)

    logits = C(F(data))

    x = torch.argmax(logits, dim=1).cpu().detach().numpy()
    result.append(x)
result = np.concatenate(result)

# Generate submission
df = pd.DataFrame({'id': np.arange(0,len(result)), 'label': result})
df.to_csv(f'{args.output_csv}', index=False)
