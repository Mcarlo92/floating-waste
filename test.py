import subprocess
import os
import sys
import threading
import time
from datetime import datetime
import torch
import torchvision
import torchvision.models as m
import torch.nn as nn
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder
from mytransform import transform
from myutility import GPUtest, test, select_model,file,dest,apply_pruning

stop_thread = False
path = dest(int(sys.argv[2]),sys.argv[1])
filename = "watt.txt"

def models(model_number, num_classes):
    if model_number == 1:
        model = m.mobilenet_v2()
        num_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_features, num_classes)
    elif model_number == 2:
        model = m.mobilenet_v3_small()
        num_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_features, num_classes)
    elif model_number == 3:
        model = m.mobilenet_v3_large()
        num_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_features, num_classes)
    elif model_number == 4:
        model = m.shufflenet_v2_x2_0()
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    elif model_number == 5:
        model = m.squeezenet1_1()
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
    elif model_number == 6:
        model = m.efficientnet_b0()
        num_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_features, num_classes)
    elif model_number == 7:
        model = m.efficientnet_b1()
        num_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_features, num_classes)
    elif model_number == 8:
        model = m.resnet18()
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    elif model_number == 9:
        model = m.resnet34()
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    elif model_number == 10:
        model = m.resnet50()
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    elif model_number == 11:
        model = m.vgg11()
        num_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_features, num_classes)
    elif model_number == 12:
        model = m.vgg16()
        num_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_features, num_classes)
    elif model_number == 13:
        model = m.vgg19()
        num_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_features, num_classes)
    else:
        raise ValueError("Invalid model number.")
    return model

def save_energy_rtx():
    import pynvml
    pynvml.nvmlInit()
    while not stop_thread:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        power_info = pynvml.nvmlDeviceGetPowerUsage(handle)
        # Ottieni il valore del consumo di energia
        watt = power_info / 1000.0
        # Ottieni il timestamp attuale nel formato "YYYY-MM-DD HH:MM:SS"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Scrivi il timestamp e il valore del consumo di energia nel file
        with open(path+"/"+filename, "a") as f:
            f.write(timestamp + " " + str(watt) + "\n")

        # Aspetta un secondo prima di eseguire il ciclo di nuovo
        time.sleep(0.1)

def save_energy_jetson():
    from jtop import jtop
    while not stop_thread:
        with jtop() as jetson:
            out = (jetson.power)
            power_info = out['rail']['POM_5V_GPU']['power']
            # Ottieni il valore del consumo di energia
            watt = power_info / 1000.0
        # Ottieni il timestamp attuale nel formato "YYYY-MM-DD HH:MM:SS"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Scrivi il timestamp e il valore del consumo di energia nel file
        with open(path+"/"+filename, "a") as f:
            f.write(timestamp + " " + str(watt) + "\n")

        # Aspetta un secondo prima di eseguire il ciclo di nuovo
        time.sleep(0.1)


GPUtest()
data_dir = "./dataset/testset"
dataset = ImageFolder(data_dir, transform())
input, label = dataset[0]
print("Follwing classes are there : \n", dataset.classes)

BATCH_SIZE = 1
test_dl = torch.utils.data.DataLoader(
    dataset, BATCH_SIZE, num_workers=1
)
print(f"\nLength of testing Data : {len(dataset)}")

# definisco le classi
classes = ("CONTAINERS", "PLASTIC_BAG", "PLASTIC_BOTTLE", "TIN_CAN", "UNKNOWN")

dataiter = iter(test_dl)
# images, labels = dataiter.__next__()

net = models(int(sys.argv[2]),5)
net.load_state_dict(torch.load(file(int(sys.argv[2]))))
net.cuda()
#net=torch.load(file(int(sys.argv[2]))+"h")
net.eval()

switch = sys.argv[1]
if switch == "RTX":
    energy_consumption_thread = threading.Thread(target=save_energy_rtx)
elif switch == "Jetson":
    energy_consumption_thread = threading.Thread(target=save_energy_jetson)

energy_consumption_thread.start()
test(test_dl, net,int(sys.argv[2]),switch)
stop_thread = True
energy_consumption_thread.join()

