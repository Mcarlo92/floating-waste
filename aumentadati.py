import torch
from torchvision.datasets import ImageFolder
from mytransform import data_augmentation
import numpy as np

#data directory
data_dir = "./dataset"
string = "data_augmentation_b1"

# switch per salvare le immagini 
def switch_case(case, img, i):
    if case == 0:
        Image.fromarray(img).save(data_dir+'/CONTAINERS/CONTAINERS'+string+str(i)+ '.jpg')
    elif case == 1:
        Image.fromarray(img).save(data_dir+'/PLASTIC_BAG/PLASTIC_BAG'+string+str(i)+ '.jpg')
    elif case == 2:
        Image.fromarray(img).save(data_dir+'/PLASTIC_BOTTLE/PLASTIC_BOTTLE'+string+str(i)+ '.jpg')
    elif case == 3:
        Image.fromarray(img).save(data_dir+'/TIN_CAN/TIN_CAN'+string+str(i)+ '.jpg')
    else:
        Image.fromarray(img).save(data_dir+'/UNKNOWN/UNKNOWN'+string+str(i)+ '.jpg')

# load data
dataset = ImageFolder(data_dir, data_augmentation())

input, label = dataset[0]
print("Following classes are there:\n", dataset.classes)

BATCH_SIZE = 32

# Split data
train_dl = torch.utils.data.DataLoader(dataset, BATCH_SIZE, shuffle=True, num_workers=4)

dataiter = iter(train_dl)
print(len(train_dl))

for i in range(len(train_dl)):
    images, labels = dataiter.__next__()
    img = (images.squeeze().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    etichetta = labels.numpy().tolist()[0]
    if etichetta != 0:
        if etichetta != 2:
            switch_case(etichetta, img, i)
