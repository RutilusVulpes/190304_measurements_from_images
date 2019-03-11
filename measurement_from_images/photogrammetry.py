from Camera import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
world_coords_1 = read_csv('gcp_stereo_1.txt', header=None)
world_coords_2 = read_csv('gcp_stereo_2.txt', header=None)

img_1 = plt.imread('campus_stereo_1.jpg')
img_2 = plt.imread('campus_stereo_2.jpg')

focal_length = 27 # mm

pose_guess = [272470.0,5193991.0, 985.0, 1.97, 0.214, 0.01]

camera_1 = Camera(pose_guess, focal_length, img_1.shape[0], img_1.shape[1])
camera_2 = Camera(pose_guess, focal_length, img_2.shape[0], img_2.shape[1])

X_world_1 = world_coords_1.iloc[:,2:5] #real world coordinates
X_cam_1 = world_coords_1.iloc[:,0:2] # pixel coordinates OR OBSERVED VALUES

X_world_2 = world_coords_2.iloc[:,2:5] #real world coordinatess
X_cam_2 = world_coords_2.iloc[:,0:2] # pixel coordinates OR OBSERVED VALUES

pose_1 = camera_1.estimate_pose(X_world_1, X_cam_1)
pose_2 = camera_2.estimate_pose(X_world_2, X_cam_2)

print (pose_1)

X_cam_1_pred = camera_1.convert_world_to_cam_coords(X_world_1)
X_cam_2_pred = camera_2.convert_world_to_cam_coords(X_world_2)

plt.imshow(img_1)
plt.scatter(X_cam_1.iloc[:,0], X_cam_1.iloc[:,1], color='r')
plt.scatter(X_cam_1_pred[:,1], X_cam_1_pred[:,0], color='g')
plt.show()
