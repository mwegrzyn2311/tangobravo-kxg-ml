import json
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

def CropImage(imgFolderPath = "D:/HackatonCrop/reference_images_part1/", outputPath = "D:/HackatonCrop/output/", labelFile = 'reference_images_part1.json'):
    f = open(labelFile)

    data = json.load(f)
    f.close()
    categories = data["categories"]
    images = data["images"]
    annotations = data["annotations"]

    categoriesNames = {}
    for item in categories:
        categoriesNames[item["id"]] = item["name"]
    # print(categoriesNames)

    fileNames = {}
    for item in images:
        fileNames[item["id"]] = item["file_name"]
    # print(fileNames)

    for item in annotations:
        # print(item)
        cat_id = int(item["category_id"])
        if cat_id in categoriesNames.keys():
            path = outputPath + categoriesNames[cat_id]
            if not os.path.exists(path):
                os.makedirs(path)
            img_id = int(item["image_id"])
            if img_id in fileNames.keys():
                img_path = imgFolderPath + fileNames[img_id]
                # print(img_path)
                img = cv2.imread(img_path)
                # Bboxes are in [top-left-x, top-left-y, width, height] format
                # print(item["bbox"])
                x = item["bbox"][0]
                y = item["bbox"][1]
                width = item["bbox"][2]
                height = item["bbox"][3]
                # cv2.imshow("cropped", img)
                print(img.shape)
                img = img[y:y + height, x:x + width]
                print(img.dtype)
                print(img.shape)
                output_path = outputPath + categoriesNames[cat_id] + "/" + str(item["id"]) + ".png"
                print(output_path)
                cv2.imwrite(output_path, img)

   # cv2.destroyAllWindows()

def forEachImg(path):
    subfolders = [ f.path for f in os.scandir(path) if f.is_dir() ]
    for categoryDir in subfolders:
        category = os.path.basename(categoryDir)
        for imagePath in os.listdir(categoryDir):
            img = cv2.imread(categoryDir + "/" + imagePath)
            train(category, img)



def train(category,img):
    #tu wrzuc swoj kod

    print(category)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

if __name__ == "__main__":
   #CropImage()
   forEachImg("D:/HackatonCrop/output/")