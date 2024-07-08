# Loading Required Libraries
import sys
sys.path.insert(1, 'D:\\Bone_Landmark_Detection_And_Segmentation\\notebooks\\helper_code')
import IMAGE_3D_PROCESS_LIB_V2
import TRANS3D
import pandas as pd
from IMAGE_3D_PROCESS_LIB_V3 import *
from IMAGE_3D_PROCESS_LIB_V3 import SELECT_REGION_FROM_IMAGE
import numpy as np
import warnings
warnings.filterwarnings(action="ignore")
import nibabel as nib
from dipy.viz import regtools
from dipy.data import fetch_stanford_hardi, read_stanford_hardi
from dipy.data.fetcher import fetch_syn_data, read_syn_data
from dipy.align.imaffine import (transform_centers_of_mass,
                                 AffineMap,
                                 MutualInformationMetric,
                                 AffineRegistration)
from dipy.align.transforms import (TranslationTransform3D,
                                   RigidTransform3D,
                                   AffineTransform3D)
import matplotlib.pyplot as plt
import nrrd
import affines

import scipy as sp
import cv2

import os
import threading
import warnings
from scipy import ndimage
warnings.filterwarnings(action='ignore')
import DataGenerator
import pickle
import New_config as config
import pickle
from threading import Thread
from tensorflow.keras.models import load_model


# setting paths
data_dir = 'D:\\Bone_Landmark_Detection_And_Segmentation\\data\\CP_ARTHRO3D_DATA_100CT_1MM'
img_dir_list = os.listdir("D:\\Bone_Landmark_Detection_And_Segmentation\\data\\CP_ARTHRO3D_DATA_100CT_1MM")
data_folder = "'D:\\Bone_Landmark_Detection_And_Segmentation\\data\\CP_ARTHRO3D_DATA_100CT_1MM"

# Helper Functions


def read_txt_line_to_list(file_name):
    with open(file_name, 'r') as file:
    # Create an empty list to store the lines
        lines = []
        # Iterate over the lines of the file
        for line in file:
           
            # Remove the newline character at the end of the line
            line = line.strip()
    
            # Append the line to the list
            lines.append(line)
    return lines

left_ct_id_list = read_txt_line_to_list(data_dir + '\\' + 'left_ct_id.txt')
print(left_ct_id_list)


# function to filter and get landmarks from dict to numpy format
def get_landmarks_DistalFemure(points):
    lst = np.array([points["KneeCenter"],points["AMEpi"],points["SMEpi"],points["LatEpi"],points["PostMed"],points['PostLat'],points['DistalMed'],points['DistalLat'],
          points["WhiteSide1"],points['WhiteSide2'],points['HipCenter']])
    #lst = lst-lst[0]+64
    return lst  


# Cropping and downsampling functions
INTERPOLATION_ORDER = 1
def CROP_3D_IMAGE_AROUND_CENTER(output_size, img, pnts=None):
    output_size = np.asanyarray(output_size)
    if output_size.size == 1:
        output_size = np.ones((1,3))*output_size[0]
    Cs = np.floor( (np.array(img.shape)-output_size)/2 )
    Ce = np.ceil( (np.array(img.shape)-output_size)/2 )    
    if pnts is not None:
        pnts = pnts - Cs    
    Cs = np.int32(Cs)
    Ce = np.int32(Ce)    
    img = img[Cs[0]:-Ce[0], Cs[1]:-Ce[1], Cs[2]:-Ce[2]]                  
    if pnts is not None:
        return img, pnts
    else:
        return img, None
    

def SHIFT_3D_IMAGE_CENTER(centered_at, img, pnts=None):
    centered_at=np.asanyarray(centered_at)
    img_center = (np.asanyarray(img.shape)-1)/2
    translate= img_center - centered_at   

    T = GENERATE_TRANSFORMATION_MATRIX_3D (translate=translate)  
    
    if pnts is not None:
        img, pnts = APPLY_AFFINE_TRANSFORM_TO_3D_IMAGE(T, img, pnts)
        return img, pnts
    else:
        img = APPLY_AFFINE_TRANSFORM_TO_3D_IMAGE(T, img)
        return img, None

def GENERATE_TRANSFORMATION_MATRIX_3D(translate=[0,0,0],angle_dg=[0,0,0], scale=[1,1,1],centered_at = [0,0,0]):    
    #centered_at is correspond to image voxel location image.
    centered_at = np.asarray(centered_at)
    #print('translate:--',translate)
    t_init=TRANS3D.compose_matrix(translate=-1*centered_at)
    t_trans= TRANS3D.compose_matrix(translate=translate,scale=scale,angles=[np.deg2rad(angle_dg[0]),-1*np.deg2rad(angle_dg[1]),np.deg2rad(angle_dg[2])])
    t_end=TRANS3D.compose_matrix(translate=centered_at) 
    AffineTransMattrix_3D = t_end@t_trans@t_init   
    return AffineTransMattrix_3D
def APPLY_AFFINE_TRANSFORM_TO_3D_POINTS(AffineTransMattrix_3D, pnts): 
    # pnts : column is the dimension of points
    # for 3D, pnts : nX3
    pnts = affines.apply_affine(AffineTransMattrix_3D,pnts)  
    pnts=np.round(pnts)
    return pnts
def APPLY_AFFINE_TRANSFORM_TO_3D_IMAGE(AffineTransMattrix_3D, img, pnts=None ):
    #img=ndimage.affine_transform(img,np.linalg.inv(AffineTransMattrix_3D),order=0, prefilter=False)
    if pnts is not None:
        img=ndimage.affine_transform(img,np.linalg.inv(AffineTransMattrix_3D),order=INTERPOLATION_ORDER, prefilter=False) 
        pnts = APPLY_AFFINE_TRANSFORM_TO_3D_POINTS(AffineTransMattrix_3D, pnts)
        return img, pnts
    else:
        img=ndimage.affine_transform(img,np.linalg.inv(AffineTransMattrix_3D),order=INTERPOLATION_ORDER, prefilter=False)     
        return img

    
def SELECT_REGION_FROM_IMAGE( img, pnts=None, centered_at=None, crop_size=None, downsample_factor=None):   

     img, _= SHIFT_3D_IMAGE_CENTER(centered_at, img)
     if pnts is not None:
         pnts = pnts - centered_at # points with respect to image center
     img = CROP_3D_IMAGE_AROUND_CENTER(crop_size, img)
     img = img[0]
     
     img = DownUpsample_3dvol(downsample_factor, img)

     
     if pnts is not None:
         pnts = pnts*downsample_factor
     if pnts is not None:
         return img, pnts      
     return img
       
def DownUpsample_3dvol(sf, img, pnts=None):
    img = ndimage.zoom(img, zoom=sf, order=0)
    if pnts is not None:
        pnts = np.array(pnts)
        pnts = pnts*sf
        pnts = pnts.astype('int')
        return img,pnts
    return img

def Comput_Affine_Registration(reference_image, target_image):
    nbins = 32
    sampling_prop = None
    metric = MutualInformationMetric(nbins, sampling_prop)
    level_iters = [10000, 1000, 100]
    sigmas = [3.0, 1.0, 0.0]
    factors = [4, 2, 1]
    affreg = AffineRegistration(metric=metric,
                            level_iters=level_iters,
                            sigmas=sigmas,
                            factors=factors)
    transform = AffineTransform3D()
    params0 = None
    starting_affine=None

    affine_tranform_obj = affreg.optimize(reference_image,target_image,transform, params0)
    #identity = np.eye(4)
    #transformed = affine_tranform_obj.transform(image)
    #inv_transform = affine_tranform_obj.transform_inverse(reference_mask)

    return affine_tranform_obj


def do_registration(t_im, r_im, r_points):  
    #print(np.max(r_im)) 
    #r_im_crop, r_points_crop =SELECT_REGION_FROM_IMAGE(r_im,r_points,r_points[0],[256,256,256],0.5)        #r_points are in reference to image center
    #t_im_crop, t_points_crop =SELECT_REGION_FROM_IMAGE(t_im,t_points,t_crop_center_point,[256,256,256],0.5) #t_points are in reference to image center
    #r_im_crop, r_points_crop =SELECT_REGION_FROM_IMAGE(r_im,r_points,r_points[0],[256,256,256],0.5)        #r_points are in reference to image center
    affine_tranform_obj = Comput_Affine_Registration(r_im,t_im)
    Affin_Transform_Mat = affine_tranform_obj.affine
    estimated_t_points_crop = affines.apply_affine(Affin_Transform_Mat, r_points + 64) - 64
    return estimated_t_points_crop, Affin_Transform_Mat

# Deep Learning Model functions

def get_dice_loss(smoothing_factor):
   # def dice_coefficient(y_true, y_pred):
   #    flat_y_true = K.flatten(y_true)
   #    flat_y_pred = K.flatten(y_pred)
   #    return (2. * K.sum(flat_y_true * flat_y_pred) + smoothing_factor) / (K.sum(flat_y_true) + K.sum(flat_y_pred) + smoothing_factor)


   def dice_coefficient_loss(y_true, y_pred):
      # print(y_true.shape)
      # print(y_pred.shape)
      y_true = y_true[:,:,:,:,1:]
      y_pred = y_pred[:,:,:,:,1:]
      Ncl = y_pred.shape[-1]
      w = np.zeros((Ncl,))
      for l in range(0,Ncl): w[l] = np.sum( np.asarray(y_true[:,:,:,:,l]==1,np.int8) )
      # print(w)
      w = 1/(w**2+0.00001)
      # w = 1/(w+0.00001)
      
      flat_y_true = K.flatten(y_true)
      flat_y_pred = K.flatten(y_pred)
      return 1 - (2. * K.sum(flat_y_true * flat_y_pred) + smoothing_factor) / (K.sum(flat_y_true) + K.sum(flat_y_pred) + smoothing_factor)
   
   return dice_coefficient_loss


# knee_model = load_model(r'D:\Bone_Landmark_Detection_And_Segmentation\notebooks\Segmentation_models\model1_resunet\weights-improvement-100-0.06.h5', compile=False)
# knee_model.compile(optimizer = 'adam', loss=get_dice_loss(1),run_eagerly=True)

knee_models = []
for path in config.knee_segmentation_models:
    model = load_model(path, compile=False)
    model.compile(optimizer = 'adam', loss=get_dice_loss(1),run_eagerly=True)
    knee_models.append(model)


def get_dice_loss(smoothing_factor):
   # def dice_coefficient(y_true, y_pred):
   #    flat_y_true = K.flatten(y_true)
   #    flat_y_pred = K.flatten(y_pred)
   #    return (2. * K.sum(flat_y_true * flat_y_pred) + smoothing_factor) / (K.sum(flat_y_true) + K.sum(flat_y_pred) + smoothing_factor)


   def dice_coefficient_loss(y_true, y_pred):
      # print(y_true.shape)
      # print(y_pred.shape)
      flat_y_true = K.flatten(y_true)
      flat_y_pred = K.flatten(y_pred)
      return 1 - (2. * K.sum(flat_y_true * flat_y_pred) + smoothing_factor) / (K.sum(flat_y_true) + K.sum(flat_y_pred) + smoothing_factor)
   
   return dice_coefficient_loss

hip_models = []
for path in config.hip_segmentation_models:
    model = load_model(path, compile=False)
    model.compile(optimizer = 'adam', loss=get_dice_loss(1),run_eagerly=True)
    hip_models.append(model)


# Loading Reference images

#DistalFemure

idx = [4,7,10,12]
Good_Template_File_List = [left_ct_id_list[i] for i in idx ]


DistFemur_ref_landmarks_List=[]
r_im_crop_List =[]
r_points_crop_List =[]

for i in range(0,len(Good_Template_File_List)): 
    print("Reading _",i,"_reference data...",Good_Template_File_List[i])
    ref_image, ref_mask, ref_points_dict = DataGenerator.get_data_and_points(config.train_CP_ARTHRO3D_DATA_100CT_1MM,Good_Template_File_List[i]+ '_LEFT_nii_1mm')
    DistFemur_ref_landmarks = get_landmarks_DistalFemure(ref_points_dict)
    DistFemur_ref_image = np.asarray(ref_image*(ref_image>200)*(ref_mask==1),  dtype=np.float32)
    
    
    DistFemur_ref_landmarks_List.append(DistFemur_ref_landmarks)
    r_im_crop, r_points_crop =SELECT_REGION_FROM_IMAGE(DistFemur_ref_image, DistFemur_ref_landmarks, DistFemur_ref_landmarks[0],[256,256,256],0.5) 
    r_im_crop_List.append(r_im_crop)
    r_points_crop_List.append(r_points_crop)
    
    print("reference image data loaded...")


# Applying Registration

def apply_registration(sub_id):
     name = sub_id.split('\\')[-2]
     target_image, target_mask, target_points_dict = DataGenerator.get_data_and_points(config.train_CP_ARTHRO3D_DATA_100CT_1MM, name)
     DistFemur_target_landmarks = get_landmarks_DistalFemure(target_points_dict)
     DistFemur_target_image = np.asarray(target_image*(target_image>200),  dtype=np.float32)
     jitter = np.random.randint(-15,15,size=3)
     knee_center = DistFemur_target_landmarks[0]
     hip_center = DistFemur_target_landmarks[-1]
     x_size, y_size, z_size=target_image.shape

     # Flipping image if required
     if hip_center[2] > (z_size/2):
            target_image = target_image[:,:,::-1]
            DistFemur_target_landmarks[:,2] = z_size - DistFemur_target_landmarks[:,2]
            hip_center[2] = z_size - hip_center[2] 
            knee_center[2] = z_size - knee_center[2]

     #Cropping and downsampling image
     t_crop_center_point = np.array(knee_center-jitter)
     t_im_crop, t_points_crop = SELECT_REGION_FROM_IMAGE(DistFemur_target_image,DistFemur_target_landmarks,t_crop_center_point,[256,256,256],0.5)
     temp = np.expand_dims(t_im_crop[::2,::2,::2],0)
     temp = np.expand_dims(temp,-1)

     #Predicting Mask for the image using deep learning model and then applying it to the image
     pred_lower = None
     for model in knee_models:
        if pred_lower is not None:
            temp_pred =  model.predict(temp)
            temp_pred[temp_pred <0.5] = 0
            temp_pred[temp_pred >=0.5] = 1
            pred_lower += temp_pred
        else:
            pred_lower = model.predict(temp)
     pred_lower /= len(knee_models)
     pred_lower[pred_lower < 0.25] = 0
     pred_lower[pred_lower >= 0.25] = 1
     pred_lower = pred_lower[0,:,:,:,1]
     pred_lower = DownUpsample_3dvol(2,pred_lower)
     t_im_crop = t_im_crop*pred_lower
     print("target image data loaded...")

     #creating matrics to determine performance
     pred_diff=[]
     Pred_Landmarks = []
     temp_affine_arr = []

     # Applying Registration with reference images
     for k in range(0,4):
        print("K: ",k)
        estimated_t_points_crop, Affin_Transform_Mat = do_registration(t_im_crop,
                                                                    r_im_crop_List[k],
                                                                    r_points_crop_List[k])
        temp_affine_arr.append(Affin_Transform_Mat)
        DistFemur_predicted_landmarks = t_crop_center_point + estimated_t_points_crop*2
        print((DistFemur_target_landmarks-DistFemur_predicted_landmarks))
        pred_diff.append((DistFemur_target_landmarks-DistFemur_predicted_landmarks))
        Pred_Landmarks.append(DistFemur_predicted_landmarks)
     Pred_Pred_arr = np.mean(np.array(pred_diff),axis=0)
     Pred_Landmarks_arr = np.mean(np.array(Pred_Landmarks),axis=0)
     error = DistFemur_target_landmarks-Pred_Landmarks_arr
     predicted_landmarks = Pred_Landmarks_arr
     original_landmarks = DistFemur_target_landmarks

     return error,predicted_landmarks,original_landmarks



while True:

    img_path = str(input("Enter image path:"))
    if img_path == 'BREAK':
        break
    
    else:
        error,predictied_landmarks,original_landmarks = apply_registration(img_path)
       
        print("Predicted Landmarks are:")
        print(predictied_landmarks)
        predictied_landmarks = pd.DataFrame(predictied_landmarks)
        pred = predictied_landmarks.T
        pred.columns = ["KneeCenter","AMEpi","SMEpi","LatEpi","PostMed",'PostLat','DistalMed','DistalLat',
          "WhiteSide1",'WhiteSide2','HipCenter']
        pred = pred.T
        id = img_path.split('\\')[-2]
        pred.to_excel(f"Landmarks{id}.xlsx")
