import subprocess
import os
import sys
import threading
import time
from datetime import datetime
import torch
import torchvision
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder
from mytransform import transform
from myutility import GPUtest, test, select_model,file,dest,apply_pruning
import torch.nn as nn
import torch.nn.utils.prune as prune
from torchsummary import summary
#path = dest(int(sys.argv[2]),sys.argv[1])
import torch.nn.functional as F



def prune_model(model, method_id: int, pruning_perc: float) -> None:
    # Check method id
    if method_id not in range(1, 4):
        print(f"Invalid method id {method_id}. Please provide an integer between 1 and 3.")
        return
    
    # Choose pruning method
    if method_id == 1:
        method = prune.l1_unstructured
    elif method_id == 2:
        method = prune.random_unstructured
    else:
        method = prune.ln_structured
    
    # Loop through model parameters
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            prune_amount = pruning_perc / 100.0
            if method_id == 3:
                prune.ln_structured(module, name='weight', amount=prune_amount, n=2, dim=0)
            else:
                method(module, name='weight', amount=prune_amount)
            prune.remove(module, 'weight')
            if prune.is_pruned(module):
                print(f"All parameters in module {name} are pruned.")
            else:
                print(f"Some parameters in module {name} are not pruned.")
                #print(module.weight.data)
    
    print('Dimensioni ridotte del modello:')
    print(torch.sum(torch.tensor([torch.numel(p) for p in model.parameters()])))




net=torch.load(file(int(sys.argv[1]))+"h")
net1=net
#summary(net1, (3, 100, 100))
print('Dimensioni ridotte del modello:')
print(torch.sum(torch.tensor([torch.numel(p) for p in net.parameters()])))
prune_model(net,2,70)
#summary(net, (3, 100, 100))