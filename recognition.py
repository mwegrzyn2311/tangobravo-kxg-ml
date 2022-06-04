import cv2
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

# TODO: Switch to parametrized file
filepath = "nomagic_boxers/62.jpg"

width = 1028
height = 1028

# Load image by Opencv2
img = cv2.imread(filepath)
# TODO: Is resize needed/even correct?
# Resize to respect the input_shape
inp = cv2.resize(img, (width, height))

# Convert img to HSV
hsv = cv2.cvtColor(inp, cv2.COLOR_BGR2HSV)

# Converting to uint8
hsv_tensor = tf.convert_to_tensor(hsv, dtype=tf.uint8)

# Add dims to rgb_tensor
hsv_tensor = tf.expand_dims(hsv_tensor, 0)

# TODO: Replace csv filepath
csv_filepath = "labels.csv"

# Loading csv with labels of classes
labels = pd.read_csv(csv_filepath, sep=';', index_col='ID')
labels = labels['OBJECT (2017 REL.)']

# TODO: Load model
detector = "xd"

# Creating prediction
boxes, scores, classes, num_detections = detector(hsv_tensor)

# Processing outputs
pred_labels = classes.numpy().astype('int')[0]
pred_labels = [labels[i] for i in pred_labels]
pred_boxes = boxes.numpy()[0].astype('int')
pred_scores = scores.numpy()[0]

# Putting the boxes and labels on the image
for score, (y_min, x_min, y_max, x_max), label in zip(pred_scores, pred_boxes, pred_labels):
    if score < 0.5:
        continue

    score_txt = f'{100 * round(score)}%'
    img_boxes = cv2.rectangle(hsv, (x_min, y_max), (x_max, y_min), (0, 255, 0), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img_boxes, label, (x_min, y_max - 10), font, 1.5, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img_boxes, score_txt, (x_max, y_max - 10), font, 1.5, (255, 0, 0), 2, cv2.LINE_AA)

plt.imshow(img_boxes)
