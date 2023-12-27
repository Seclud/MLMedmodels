# import os
#
# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
#
# import plaidml.keras
#
# plaidml.keras.install_backend()
# unfortunately paidml is old and i need to downgrade everything to make it work, so going to use cpu
import pandas as pd
from keras import layers, models
import numpy as np
import open3d as o3d
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from keras.models import load_model

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


def extract_labels(point_cloud,filename):
    green = [0, 1, 0]
    red = [1, 0, 0]

    colors = np.asarray(point_cloud.colors)
    labels_inside = np.zeros((len(colors), 1))
    labels_outside = np.zeros((len(colors), 1))

    print(colors)
    labels_inside[np.all(colors == green, axis=1)] = 1
    labels_outside[np.all(colors == red, axis=1)] = 1

    labels_inside1 = [str(x[0]) + '\n' for x in labels_inside]
    labels_outside1= [str(x[0]) + '\n' for x in labels_outside]
    file_name_without_extension, file_extension = os.path.splitext(filename)
    completeName_inside = os.path.join("labels", file_name_without_extension)
    completeName_outside = os.path.join("labels", file_name_without_extension)
    if not os.path.exists(completeName_inside):
        os.makedirs(completeName_inside)
    if not os.path.exists(completeName_outside):
        os.makedirs(completeName_outside)
    with open(os.path.join(completeName_inside, f'labels_inside{file_name_without_extension}.txt'), 'w') as f:
        f.writelines(labels_inside1)
    with open(os.path.join(completeName_outside, f'labels_outside{file_name_without_extension}.txt'), 'w') as f:
        f.writelines(labels_outside1)

    print(type(labels_inside1[1]))
    print(labels_inside.tolist())
    print(labels_inside[1])
    print(type(labels_inside[0][0]))
    return labels_inside

def visualize_predictions(model, point_clouds, labels):
    # Select the first 8 point clouds and labels
    point_clouds = point_clouds[50:]
    labels = labels[50:]
    print(labels.tolist())

    # Use the model to make predictions
    predictions = model.predict(point_clouds)
    predictions = np.round(predictions).astype(int)
    print(predictions)

    # Plot the point clouds with their predicted and actual labels
    fig = plt.figure(figsize=(15, 10))
    for i in range(8):
        ax = fig.add_subplot(2, 4, i + 1, projection='3d')
        ax.scatter(point_clouds[i, :, 0], point_clouds[i, :, 1], point_clouds[i, :, 2], c=predictions[i, :, 0], cmap='RdYlGn')
        ax.set_title(f"True: {np.mean(labels[i]) > 0.5}, Pred: {np.mean(predictions[i]) > 0.5}")
        ax.set_axis_off()
    plt.show()

    # Create new point clouds with the predicted colors
    for i in range(8):
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(point_clouds[i])
        point_cloud.colors = o3d.utility.Vector3dVector(predictions[i] * [0, 1, 0] + (1 - predictions[i]) * [1, 0, 0])
        o3d.io.write_point_cloud(f'Predicted clouds/predicted_{i}.ply', point_cloud)

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
        labels.append(extract_labels(point_cloud,filename))
    print('Read all files')
    point_clouds = np.array(point_clouds)
    labels = np.stack(labels, axis=0)

    # # Split the data into training and validation sets
    # point_clouds_train, point_clouds_val, labels_train, labels_val = train_test_split(point_clouds, labels,
    #                                                                                   test_size=0.2, random_state=42)
    # model_path = 'my_model.h5'
    # if os.path.exists(model_path):
    #     model = load_model(model_path)
    # else:
    #     model = create_pointnet_model(num_points)
    #     # Compile the model
    #     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    #     # Train the model
    #     model.fit(point_clouds_train, labels_train, validation_data=(point_clouds_val, labels_val), epochs=10)
    #     # Save the model
    #     model.save(model_path)
    #
    # loss, accuracy = model.evaluate(point_clouds_train, labels_train)
    #
    #
    # print(f"Test set accuracy: {accuracy * 100}%")
    #
    # # Use the function to visualize the predictions
    # visualize_predictions(model, point_clouds_val, labels_val)


if __name__ == '__main__':
    main()

