from keras import layers, models
import numpy as np
import open3d as o3d
import os
from sklearn.model_selection import train_test_split


cloud = o3d.io.read_point_cloud('Point clouds/0004.ply')
o3d.visualization.draw_geometries([cloud])
cloudPoints = np.asarray(cloud.points)