from keras import layers, models
import numpy as np
import open3d as o3d
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split


def create_pointnet_model(num_points):
    input_points = layers.Input(shape=(num_points, 3))

    x = layers.Conv1D(64, 1, activation='relu')(input_points)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(128, 1, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(1024, 1, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=num_points)(x)

    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=input_points, outputs=x, name='pointnet')

    return model


def extract_labels(point_cloud):
    green = [0, 255, 0]
    red = [255, 0, 0]

    colors = np.asarray(point_cloud.colors)

    labels = np.zeros(len(colors))

    labels[np.all(colors == green, axis=1)] = 1
    labels[np.all(colors == red, axis=1)] = 0

    return labels


def main():
    point_clouds = []
    labels = []
    num_points = 10000
    print(os.listdir("Point clouds"))
    for filename in os.listdir("Point clouds"):
        filepath = os.path.join("Point clouds", filename)
        print(f"Reading file: {filepath}")
        point_cloud = o3d.io.read_point_cloud(filepath)
        points = np.asarray(point_cloud.points)
        point_clouds.append(points)
        labels.append(extract_labels(point_cloud))

    point_clouds = np.array(point_clouds)
    labels = np.array(labels)

    labels = labels[:, np.newaxis]

    x_train, x_test, y_train, y_test = train_test_split(point_clouds, labels, test_size=0.2, random_state=42)

    x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
    x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
    y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)

    model = create_pointnet_model(num_points)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=10)

    loss, accuracy = model.evaluate(x_test, y_test)

    print(f"Test set accuracy: {accuracy * 100}%")


if __name__ == '__main__':
    main()
