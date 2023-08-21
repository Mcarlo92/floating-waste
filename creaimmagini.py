import pandas as pd
import numpy as np
from PIL import Image
from datasets import load_dataset, concatenate_datasets
import os
from numpy import asarray

path_train="./dataset/"
path_test="./testset/"
def normalizza_bbox(image,bbox):

    width, height = image.size
    x_min_norm = bbox[0] * width 
    y_min_norm = bbox[1] * height
    x_max_norm = bbox[2] * width
    y_max_norm = bbox[3] * height

    #controllo corretta normalizzazione 
    if x_min_norm>=x_max_norm:
        x=x_min_norm-x_max_norm
        x_max_norm+=x
    if y_min_norm>=y_max_norm:
        y=y_min_norm-y_max_norm
        y_max_norm+=y

    #controllo valori fuori del range 
    if (x_max_norm+50)>=width:
        x_max_norm=width-50
    if (y_max_norm+50)>=height:
        y_max_norm=height-50
    if (x_min_norm-50)<0:
        x_min_norm=50
    if (y_min_norm-50)<0:
        y_min_norm=50

    bbox_norm = (x_min_norm -50 , y_min_norm-50, x_max_norm+50, y_max_norm+50)
    return bbox_norm


def creaimg(immagine1, bbox):
    k2 = immagine1.crop(normalizza_bbox(immagine1, bbox))
    new_size = (200, 200)
    k2 = k2.resize(new_size)
    return k2


def crerumore(immagine1):
    w, h = immagine1.size
    x1 = w / 2
    y1 = h / 2
    k2 = immagine1.crop((x1 + 100, y1 + 100, x1 + 200, y1 + 200))
    new_size = (200, 200)
    k2 = k2.resize(new_size)
    return k2


def estrai_immagini(immagine1, litter, cont, path):
    if len(litter["label"]) > 0:
        for i in range(len(litter["label"])):
            cont = cont + 1
            k = creaimg(immagine1, litter["bbox"][i])
            k = np.array(k, dtype="uint8")
            if litter["label"][i] == 1:
                Image.fromarray(k).save(
                    path +'PLASTIC_BOTTLE/PLASTIC_BOTTLE' + str(i) + str(cont) + '.jpg')
            if litter["label"][i] == 2:
                Image.fromarray(k).save(
                    path + 'PLASTIC_BAG/PLASTIC_BAG' + str(i) + str(cont) + '.jpg')
            if litter["label"][i] == 3:
                Image.fromarray(k).save(
                    path + 'OTHER_PLASTIC_WASTE/OTHER_PLASTIC_WASTE' + str(i) + str(cont) + '.jpg')
            if litter["label"][i] == 4:
                Image.fromarray(k).save(
                    path + 'NOT_PLASTIC_WASTE/NOT_PLASTIC_WASTE' + str(i) + str(cont) + '.jpg')

    if len(litter["label"]) == 0:
        cont = cont + 1
        k = crerumore(immagine1)
        k = np.array(k, dtype="uint8")
        Image.fromarray(k).save(path + 'NOT_PLASTIC_WASTE/NOT_PLASTIC_WASTE' + str(cont) + '.jpg')


def main():
    ds = load_dataset("Kili/plastic_in_river")
    print(ds)
    data = concatenate_datasets([ds['train'],ds['validation']],axis=0)
    immagini1 = data["image"]
    etichette1 = data["litter"]
    data2 = ds['test']
    immagini2 = data2["image"]
    etichette2 = data2["litter"]


    for i in range(len(immagini1)):
        estrai_immagini(immagini1[i], etichette1[i], i, path_train)

    for i in range(len(immagini2)):
        estrai_immagini(immagini2[i], etichette2[i], i, path_test)


if __name__ == "__main__":
    main()
