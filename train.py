import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as m
import time
import sys
from matplotlib import pyplot as plt
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder
from mytransform import transform
from myutility import imshow, GPUtest, train, test, plot_accuracies, plot_losses,file,dest_train
from torchsummary import summary
from contextlib import redirect_stdout
import json

def get_pretrained_model(model_number, num_classes):
    if model_number == 1:
        model = m.mobilenet_v2(weights=m.MobileNet_V2_Weights.IMAGENET1K_V2)
        num_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_features, num_classes)
    elif model_number == 2:
        model = m.mobilenet_v3_small(weights=m.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        num_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_features, num_classes)
    elif model_number == 3:
        model = m.mobilenet_v3_large(weights=m.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        num_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_features, num_classes)
    elif model_number == 4:
        model = m.shufflenet_v2_x2_0(weights=m.ShuffleNet_V2_X2_0_Weights.IMAGENET1K_V1)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    elif model_number == 5:
        model = m.squeezenet1_1(weights=m.SqueezeNet1_1_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
    elif model_number == 6:
        model = m.efficientnet_b0(weights=m.EfficientNet_B0_Weights.IMAGENET1K_V1)
        num_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_features, num_classes)
    elif model_number == 7:
        model = m.efficientnet_b1(weights=m.EfficientNet_B1_Weights.IMAGENET1K_V1)
        num_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_features, num_classes)
    elif model_number == 8:
        model = m.resnet18(weights=m.ResNet18_Weights.IMAGENET1K_V1)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    elif model_number == 9:
        model = m.resnet34(weights=m.ResNet34_Weights.IMAGENET1K_V1)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    elif model_number == 10:
        model = m.resnet50(weights=m.ResNet50_Weights.IMAGENET1K_V2)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    elif model_number == 11:
        model = m.vgg11(weights=m.VGG11_Weights.IMAGENET1K_V1)
        num_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_features, num_classes)
    elif model_number == 12:
        model = m.vgg16(weights=m.VGG16_Weights.IMAGENET1K_V1)
        num_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_features, num_classes)
    elif model_number == 13:
        model = m.vgg19(weights=m.VGG19_Weights.IMAGENET1K_V1)
        num_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_features, num_classes)
    else:
        raise ValueError("Invalid model number.")
        
    return model

history = {
    'train_loss': [],
    'val_loss': [],
    'accuracy': []
    }


model=get_pretrained_model(int(sys.argv[1]),5)


# Congelare i pesi di tutti i layer tranne quelli del nuovo strato di classificazione


# controllo la gpu
GPUtest()

torch.cuda.empty_cache()
# train and test data directory
data_dir = "./dataset/trainset"
validation_dir = "./dataset/validation"

# load the train and test data
dataset = ImageFolder(data_dir, transform())
datasetV= ImageFolder(validation_dir, transform())

input, label = dataset[0]
input, label = datasetV[0]
print("Follwing classes are there : \n", dataset.classes)

BATCH_SIZE = 32
# Split data
persplit=20
val_size = (len(dataset))
train_size = (len(datasetV))


print(f"Length of Train Data : {len(dataset)}")
print(f"Length of Validation Data : {len(datasetV)}")

# eseguo i loader
train_dl = torch.utils.data.DataLoader(dataset, BATCH_SIZE, shuffle=True, num_workers=4)
val_dl = torch.utils.data.DataLoader(datasetV, BATCH_SIZE * 2, num_workers=4)


# definisco le classi
classes = ("CONTAINERS", "PLASTIC_BAG", "PLASTIC_BOTTLE", "TIN_CAN", "UNKNOWN")

dataiter = iter(train_dl)
images, labels = dataiter.__next__()

#imshow(torchvision.utils.make_grid(images))

# instanzio la rete
net = model
net.cuda()
# Numero di epoche
EPOCHS = 200

# uso la cross entropy per calcolare il coefficente di perdita vedi i calcolo
criterion = nn.CrossEntropyLoss().cuda()

#ottimizzatore uso Adam o optim.SGD(net.parameters(), lr=0.001, momentum=0.8)
optimizer = optim.Adam(net.parameters(), lr=0.00005, weight_decay=0.0001)

# Training Function
start_time = time.time()
train(EPOCHS, train_dl, val_dl, optimizer, net, criterion, history,file(int(sys.argv[1])),dest_train(int(sys.argv[1])))
end_time = time.time()
elapsed_time = end_time - start_time
#ottengo il riassunto della rete per vedere i parametri
#summary(net, (3, 100, 100))
print('Finished Training\n')
plot_accuracies(history,net,int(sys.argv[1]))
#plt.show()
plot_losses(history,net,int(sys.argv[1]))
#plt.show()
torch.cuda.empty_cache()
with open(dest_train(int(sys.argv[1]))+'/summary.txt', 'w') as f:
     with redirect_stdout(f):
        summary(net, (3, 100, 100))
with open(dest_train(int(sys.argv[1]))+'/time.txt', 'w') as file:
    file.write("Il tempo di esecuzione è: {} secondi".format(elapsed_time))
file_path = dest_train(int(sys.argv[1]))+'/history.txt'
with open(file_path, 'w') as f:
    json.dump(history, f)