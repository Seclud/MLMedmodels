from keras import layers, models
import numpy as np
import open3d as o3d
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
import plaidml
import matplotlib.pyplot as plt
import pandas as pd
import keras

filename='0197.ply'
filename_without_ext, file_ext=os.path.splitext(filename)
cloud = o3d.io.read_point_cloud(f'Point clouds/{filename}')
# o3d.visualization.draw_geometries([cloud])
cloudPoints = np.asarray(cloud.points)
print(cloudPoints.shape)

LABELS = ['inside', 'outside']
COLORS = ['green', 'red']
point_clouds, test_point_clouds = [], []
point_cloud_labels, all_labels = [], []


def visualize_data(point_cloud, labels):
    df = pd.DataFrame(
        data={
            "x": point_cloud[:, 0],
            "y": point_cloud[:, 1],
            "z": point_cloud[:, 2],
            "label": labels,
        }
    )
    fig = plt.figure(figsize=(15, 10))
    ax = plt.axes(projection="3d")
    for index, label in enumerate(LABELS):
        c_df = df[df["label"] == label]
        try:
            ax.scatter(
                c_df["x"], c_df["y"], c_df["z"], label=label, alpha=0.5, c=COLORS[index]
            )
        except IndexError:
            pass
    ax.legend()
    plt.show()

label_data, num_labels = {}, 0
for label in LABELS:
    label_file = os.path.join('labels', filename_without_ext,f'labels_{label}{filename_without_ext}.txt')
    label_data[label] = np.loadtxt(label_file ).astype("float32")
    num_labels = len(label_data[label])

print(label_data)
print(label_data['inside'].tolist())

print(num_labels)

try:
    label_map = ["none"] * num_labels
    for label in LABELS:
        print(label)
        for i, data in enumerate(label_data[label]):
            if label == 'inside':
                print(data, 'inside')
            if label == 'outside':
                print(data, 'outside')
            label_map[i] = label if data == 1.0 else label_map[i]
            if data == 1.0:
                print(label_map[i])
    print(label_map)
    label_data = [
        LABELS.index(label) if label != "none" else len(LABELS)
        for label in label_map
    ]
    # Apply one-hot encoding to the dense label representation.
    label_data = keras.utils.to_categorical(label_data, num_classes=len(LABELS) + 1)

    point_clouds.append(cloud)
    point_cloud_labels.append(label_data)
    all_labels.append(label_map)
except KeyError :
    test_point_clouds.append(cloud)


all_labels=all_labels[0]
print(all_labels)
print(cloudPoints[:, 0])
print(cloudPoints[:, 1])
print(cloudPoints[:, 2])
print(len(all_labels))
print(len(cloudPoints[:, 0]))
print(len(cloudPoints[:, 1]))
print(len(cloudPoints[:, 2]))

visualize_data(cloudPoints, all_labels)
