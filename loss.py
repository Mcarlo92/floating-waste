import os
import csv
import numpy as np
import locale
from matplotlib import pyplot as plt
import json
import sys
from myutility import plot_losses
def dest_train(model_number):
    if model_number == 1:
        path = "/info_train/mobilenetv2"
    elif model_number == 2:
        path = "/info_train/mobilenetv3_small"
    elif model_number == 3:
        path = "/info_train/mobilenetv3_large"
    elif model_number == 4:
        path = "/info_train/shufflenetv2"
    elif model_number == 5:
        path = "/info_train/squeezenet1_1"
    elif model_number == 6:
        path = "/info_train/efficientnet_b0"
    elif model_number == 7:
        path = "/info_train/efficientnet_b1"
    elif model_number == 8:
        path = "/info_train/resnet18"
    elif model_number == 9:
        path = "/info_train/resnet34"
    elif model_number == 10:
        path = "/info_train/resnet50"
    elif model_number == 11:
        path = "/info_train/vgg11"
    elif model_number == 12:
        path = "/info_train/vgg16"
    elif model_number == 13:
        path = "/info_train/vgg19"
    else:
        raise ValueError("Invalid model number.")
    return path



file_path = "."+dest_train(int(sys.argv[1]))+'/history.txt'
with open(file_path, 'r') as f:
    history = json.load(f)

plot_losses(history,"b",int(sys.argv[1]))
plt.show()