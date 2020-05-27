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
from scipy import stats
from joblib import Parallel, delayed, parallel_backend
from models import FeatureExtractor, LabelPredictor, DomainClassifier

# hyper params
parser = argparse.ArgumentParser()
parser.add_argument('--use_gpu', action='store_true', help='use gpu for training')
parser.add_argument('--model_checkpoint', type=str, nargs='+', required=True, help='model checkpoint')
parser.add_argument('--dataroot', type=str, required=True, help='train/test dataroot')
parser.add_argument('--batch_size', type=int, default=256, help='batch size for train data')
parser.add_argument('--output_csv', type=str, required=True, help='predict file')
parser.add_argument('--resnet_type', type=int, nargs='+', required=True, help='which resnet to use')

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

def get_prediction(model_checkpoint, resnet_type):
    # models
    F = FeatureExtractor(resnet=resnet_type).to(device)
    C = LabelPredictor(resnet=resnet_type).to(device)

    checkpoint = torch.load(model_checkpoint)
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

    # delete model
    del F
    del C
    torch.cuda.empty_cache()

    return np.concatenate(result)

# predict
with parallel_backend('loky', n_jobs=len(args.resnet_type)):
    results = np.array(
        Parallel()(delayed(get_prediction)(a, b) for a, b in zip(args.model_checkpoint, args.resnet_type))
    ).transpose()
results, counts = stats.mode(results, axis=1)

print('disagree: {}'.format((counts <= len(args.resnet_type) // 2).sum()))
print(np.unique(results.flatten(), return_counts=True))

# Generate submission
df = pd.DataFrame({'id': np.arange(0,len(results)), 'label': results.flatten()})
df.to_csv(f'{args.output_csv}', index=False)
