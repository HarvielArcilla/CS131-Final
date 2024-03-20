import torch
import torchvision
import pandas as pd
import pickle 
from itertools import compress
from ultralytics import YOLO
from PIL import Image

def get_ingredients(rel_path, img):
    # 1) Import weights into YOLOv8
    trained = YOLO(rel_path + "weights/best.pt")

    # 2) Run prediction on chosen test image
    test_image = "images/veggies2.jpg"
    results = trained(img, conf=0.01)
    class_list = None
    im_rgb = None
    # 3) Print and save test image
    for i, r in enumerate(results):
        # Plot results image
        im_bgr = r.plot()  # BGR-order numpy array
        im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image
        im_rgb.save("_bound_box_result.jpg")
        class_list = r.boxes.cls.numpy()
            
    results_dict = results[0].names

    # 2) Convert into ingredient counts dictionary
    ingredients_dict = {}
    for i in class_list:
        index = int(i)
        ingred_name = results_dict[index]
        if ingred_name not in ingredients_dict:
            ingredients_dict[ingred_name] = 0
        ingredients_dict[ingred_name] += 1
        
    print(ingredients_dict)

    # 3) Convert ingredient counts dictionary into comma-separated string
    print('\n')
    food = ''
    first = True
    for ing in ingredients_dict.keys():
        if not first:
            food += ','
        food += ing.lower()
        first = False
    print(food)

    return food, im_rgb