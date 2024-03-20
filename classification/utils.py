import cv2
from PIL import Image

def read_class_names(path):
    classes = {}
    file1 = open(path, 'r')
    Lines = file1.readlines()
    
    count = 0
    for line in Lines:
        classes[count] = line.strip()
        count += 1

    return classes

def channel_4_to_3(img):
    img['image'] = img['image'].resize((256, 256))
    if img['image'].mode != "RGB":
        img['image'] = img['image'].convert("RGB")
    print(img['image'].mode)
    return img