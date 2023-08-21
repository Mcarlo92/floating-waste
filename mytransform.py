import torchvision.transforms as transforms
from PIL import Image, ImageFilter


def transform():
    transformata = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
        transforms.RandomAffine(degrees=40, translate=None, scale=(1, 2), shear=15, fill=0),
        transforms.ToTensor(),
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    return transformata


def data_augmentation():
    transformata = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=180, translate=None, scale=None, shear=1.2, fill=0),
        transforms.Lambda(lambda img: img.filter(ImageFilter.UnsharpMask(radius=2, percent=150))),
        transforms.Lambda(lambda img: img.filter(ImageFilter.DETAIL)),
        transforms.ToTensor(),
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    return transformata
