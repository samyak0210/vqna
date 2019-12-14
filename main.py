import argparse
import glob, os
import torch
import sys
import time
import torch.nn as nn
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import transforms, utils
from PIL import Image
from torch.utils.data import DataLoader
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from scipy.stats import multivariate_normal
# from dataloader import VqnaDataset

parser = argparse.ArgumentParser()
parser.add_argument('--no_epochs',default=20, type=int)
parser.add_argument('--lr',default=1e-4, type=float)
parser.add_argument('--num_classes',default=5, type=int)

args = parser.parse_args()

from model import Vqna

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.CrossEntropyLoss()

def train(model, optimizer, loader, epoch, device, args):
    model.train()
    tic = time.time()
    
    total_loss = 0.0
    for idx, (img, embeddings, labels) in enumerate(loader):
        img = img.to(device)
        embeddings = embeddings.to(device)
        
        optimizer.zero_grad()
        predictions = model(img)
        loss = criterion(predictions, labels)
        loss.backward()
        total_loss += loss.item()
        
        optimizer.step()
                
    print('[{:2d}, train] avg_loss : {:.5f}'.format(epoch, total_loss/len(loader)))
    sys.stdout.flush()

    return total_loss/len(loader)


def validate(model, loader, epoch, device, args):
    model.eval()
    tic = time.time()
    total_loss = 0.0
    for (img, embeddings, labels) in loader:
        img = img.to(device)
        embeddings = embeddings.to(device)
        
        predictions = model(img)
        loss = criterion(predictions, labels)
        total_loss += loss.item()
        
    print('[{:2d},   val] avg_loss : {:.5f}, time:{:3f} minutes'.format(epoch, total_loss/len(loader), (time.time()-tic)/60))
    sys.stdout.flush()
    
    return total_loss/len(loader)


model = Vqna(num_classes=args.num_classes)

if torch.cuda.device_count() > 1:
	print("Let's use", torch.cuda.device_count(), "GPUs!")
	model = nn.DataParallel(model)

model.to(device)


params = list(filter(lambda p: p.requires_grad, model.parameters())) 
optimizer = torch.optim.Adam(params, lr=args.lr)
print(device)

for (name, param) in model.named_parameters():
    if param.requires_grad==True:
        print(name, param.size())

sys.stdout.flush()

train_loss = []
val_loss = []

for epoch in range(0, args.no_epochs):
    loss = train(model, optimizer, train_loader, epoch, device, args)
    train_loss.append(loss)
    
    # with torch.no_grad():
    #     loss = validate(model, val_loader, epoch, device, args)
    #     val_loss.append(loss)
    #     if epoch == 0 :
    #         best_loss = loss
    #     if best_loss >= loss:
    #         best_loss = loss
    #         torch.save(model.state_dict(), args.model_val_path)
    #         print('[{:2d},  save, {}]'.format(epoch, args.model_val_path))
    #     print()