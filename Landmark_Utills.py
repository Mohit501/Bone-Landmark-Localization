# Required Libraries

import sys
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import affines
import os
import warnings
from scipy import ndimage
warnings.filterwarnings(action='ignore')
import pickle
import pickle
import open3d as o3d
from probreg import cpd
import pickle
import copy
import time
from stl import mesh
from scipy.spatial import KDTree
from threading import Thread

def load_reference_points(mode):
    if mode == 1:
        with open ('D:\\Bone_Landmark_Detection_And_Segmentation\\notebooks\\Femur_Mesh\\107_LEFT_img.pickle','rb') as f:
            reference_points1 = pickle.load(f)
        with open ('D:\\Bone_Landmark_Detection_And_Segmentation\\notebooks\\Femur_Mesh\\107_RIGHT_img.pickle','rb') as f:
            reference_points2 = pickle.load(f)
        with open ('D:\\Bone_Landmark_Detection_And_Segmentation\\notebooks\\Femur_Mesh\\154_RIGHT_img.pickle','rb') as f:
            reference_points3 = pickle.load(f)  
        with open ('D:\\Bone_Landmark_Detection_And_Segmentation\\notebooks\\Femur_Mesh\\14941_LEFT_img.pickle','rb') as f:
            reference_points4 = pickle.load(f)  
    return reference_points1,reference_points2,reference_points3,reference_points4

# Loading Reference Images:
def load_reference_images(mode,number_of_point_clouds = 1000):
    if mode == 1:
        # Loading meshs
        ref_img1 = o3d.io.read_triangle_mesh('D:\\Bone_Landmark_Detection_And_Segmentation\\notebooks\\Femur_Mesh\\Femur_mesh_107_LEFT_img.stl')  
        ref_img2 = o3d.io.read_triangle_mesh('D:\\Bone_Landmark_Detection_And_Segmentation\\notebooks\\Femur_Mesh\\Femur_mesh_107_RIGHT_img.stl') 
        ref_img3 = o3d.io.read_triangle_mesh('D:\\Bone_Landmark_Detection_And_Segmentation\\notebooks\\Femur_Mesh\\Femur_mesh_154_RIGHT_img.stl')
        ref_img4 = o3d.io.read_triangle_mesh('D:\\Bone_Landmark_Detection_And_Segmentation\\notebooks\\Femur_Mesh\\Femur_mesh_14941_LEFT_img.stl')
        
        # Converting to point clouds
        Ref_img1 =ref_img1.sample_points_poisson_disk(number_of_point_clouds)
        Ref_img2 =ref_img2.sample_points_poisson_disk(number_of_point_clouds)
        Ref_img3 =ref_img3.sample_points_poisson_disk(number_of_point_clouds)
        Ref_img4 =ref_img4.sample_points_poisson_disk(number_of_point_clouds)

    return Ref_img1,Ref_img2,Ref_img3,Ref_img4

def load_moving_image(path,number_of_point_clouds):
    full_path = r"{}".format(path)
    moving_image = o3d.io.read_triangle_mesh(full_path)
    Moving_Image = moving_image.sample_points_poisson_disk(number_of_point_clouds)
    return Moving_Image

        



# Threading Class
class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args,
                                                **self._kwargs)
    def join(self, *args):
        Thread.join(self, *args)
        return self._return  

# Function for adding weights
def add_weights(image,centroid):
    points = np.asarray(image.points)
    if add_weights:
        distances = np.linalg.norm(points - centroid, axis=1)
        weights = distances/np.max(distances)
        return weights
    
# Function for cropping image
def image_partition(image,add_weights= False):
    # Takes a 3d point cloud as input and partitions it from its centroid and returns the top and bottom point cloud from the centroid
    points = np.asarray(image.points)
    centroid = np.mean(points, axis=0)
    
    threshold = centroid[2] 
    upper_points = points[points[:, 2] > threshold]
    lower_points = points[points[:, 2] < threshold]

    top_cloud = o3d.geometry.PointCloud()
    top_cloud.points = o3d.utility.Vector3dVector(upper_points)

    bottom_cloud = o3d.geometry.PointCloud()
    bottom_cloud.points = o3d.utility.Vector3dVector(lower_points)
    return top_cloud,bottom_cloud,centroid



# cpd registration function
def cpd_registration_fn(source,target,flag = True):
    if flag == True:
        tf_type_name = "affine"
    else:
        tf_type_name = "rigid"
      
    # Takes source and target variables as input and returns tf_param and result
    source = source

    target = target
    tick = time.time()
    # compute cpd registration
    print("Applying registration")
    if tf_type_name == 'affine':
        tf_param, sigma2, q = cpd.registration_cpd(source,target,maxiter=100,tf_type_name=tf_type_name)
    else:
        tf_param, sigma2, q = cpd.registration_cpd(source,target,maxiter=100,tf_type_name=tf_type_name,update_scale = flag)
    result = copy.deepcopy(source)
    result.points = tf_param.transform(result.points)
    
    print("Registration is done")
    tock = time.time()
    print("time taken:",tock-tick)
    return tf_param,result

# ICP Registration function

def weighted_icp(source, target, weights, max_iterations=50, threshold=1e-6):
    # Convert point clouds to numpy arrays
    source_points = np.asarray(source.points)
    target_points = np.asarray(target.points)

    # Initialize transformation matrix
    transformation = np.identity(4)

    for iteration in range(max_iterations):
        # Find closest point correspondences using KDTree
        tree = KDTree(target_points)
        distances, indices = tree.query(source_points)

        # Construct the weighted correspondence pairs
        correspondences = np.zeros((len(indices), 3), dtype=int)
        correspondences[:, 0] = np.arange(len(indices))
        correspondences[:, 1] = indices
        correspondences[:, 2] = weights

        # Estimate transformation using weighted correspondence pairs
        reg_result = o3d.pipelines.registration.registration_icp(
            source, target,
            max_correspondence_distance=np.inf,
            init=transformation,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=1)
        )

        # Update transformation matrix
        transformation = reg_result.transformation

        # Check convergence
        if reg_result.inlier_rmse < threshold:
            break

    return transformation

def apply_icp(source,target,weights,points):
    transformation = weighted_icp(source,target,weights)
    transform_mat = copy.deepcopy(target)
    result_image = transform_mat.transform(transformation)
    predicted_points = affines.apply_affine(transformation,points)

    return transformation,result_image,predicted_points

