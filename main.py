import json
import os
import cv2

def CropImage():
    imgFolderPath = "public_dataset/reference_images_part1/"
    outputPath = "output/"
    # label json
    f = open('reference_images_part1.json')

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
                img_path = os.path.abspath(imgFolderPath + fileNames[img_id])
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
                print(img.shape)
                output_path = outputPath + categoriesNames[cat_id] + "/" + str(item["id"]) + ".png"
                print(output_path)
                cv2.imwrite(output_path, img)

if __name__ == "__main__":
   CropImage()