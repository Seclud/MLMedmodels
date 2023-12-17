# import os
#
# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
#
# import plaidml.keras
#
# plaidml.keras.install_backend()
# unfortunately paidml is old and i need to downgrade everything to make it work, so going to use cpu

from keras import layers, models
import numpy as np
import open3d as o3d
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split


def create_pointnet_model(num_points):
    input_points = layers.Input(shape=(num_points, 3))

    x = layers.Conv1D(64, 1, activation='relu')(input_points)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(128, 1, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(1024, 1, activation='relu')(x)
    x = layers.BatchNormalization()(x)

    # Global features
    global_features = layers.MaxPooling1D(pool_size=num_points)(x)
    global_features = layers.Flatten()(global_features)

    # Repeat the global features for each point
    global_features = layers.RepeatVector(num_points)(global_features)

    # Concatenate the global features with the original features
    x = layers.Concatenate(axis=-1)([x, global_features])

    x = layers.Conv1D(512, 1, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(256, 1, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(128, 1, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(1, 1, activation='sigmoid')(x)

    model = models.Model(inputs=input_points, outputs=x, name='pointnet')

    return model


def extract_labels(point_cloud):
    green = [0, 1, 0]
    red = [1, 0, 0]

    colors = np.asarray(point_cloud.colors)
    labels = np.zeros((len(colors), 1))

    labels[np.all(colors == green, axis=1)] = 1
    labels[np.all(colors == red, axis=1)] = 0

    return labels


def main():
    point_clouds = []
    labels = []
    num_points = 3000
    print(os.listdir("Point clouds"))
    for filename in os.listdir("Point clouds"):
        filepath = os.path.join("Point clouds", filename)
        point_cloud = o3d.io.read_point_cloud(filepath)
        points = np.asarray(point_cloud.points)
        point_clouds.append(points)
        labels.append(extract_labels(point_cloud))
    print('Read all files')
    point_clouds = np.array(point_clouds)
    labels = np.stack(labels, axis=0)

    # Split the data into training and validation sets
    point_clouds_train, point_clouds_val, labels_train, labels_val = train_test_split(point_clouds, labels,
                                                                                      test_size=0.2, random_state=42)
    # Create the PointNet model
    model = create_pointnet_model(num_points)
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    print(model.summary())
    # Train the model
    model.fit(point_clouds_train, labels_train, validation_data=(point_clouds_val, labels_val), epochs=10)

    loss, accuracy = model.evaluate(point_clouds_train, labels_train)

    # Save the model
    model.save('my_model.keras')

    print(f"Test set accuracy: {accuracy * 100}%")



if __name__ == '__main__':
    main()

