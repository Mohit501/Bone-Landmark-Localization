import LANDMARK_DETECTION_CONFIG 
import DataGenerator_V1
import preprocess
from tensorflow.keras.models import load_model
import config
import numpy as np
import pandas as pd
from scipy import ndimage
import warnings
warnings.filterwarnings(action="ignore")

mode = "knee"

# Loading Reference Landmarks
ref_landmarks,reference_images,reference_images_points = LANDMARK_DETECTION_CONFIG.get_reference_images(landmark=mode)
#scale_factor,new_dir,name = preprocess.A3D_to_nii_1mm(img_dir,out_dir,img_name)

while True:
      input_image_path = str(input("Please Enter input image path:")) #D:\\Bone_Landmark_Detection_And_Segmentation\\data\\NRRD
      image_name = str(input("Please enter image name:")) #102_LEFT_nii_1mm

      if input_image_path == "BREAK" or input_image_path =="break":
            break
      else:
            print("Starting Registration Process:")

            # Loading  moving image
            moving_image = DataGenerator_V1.get_data_and_points(input_image_path,image_name)

            #Using Slice Prediction Model
            print("Slice Prediction Model is predicting slices:")
            x_upper, y_upper_name = DataGenerator_V1.data_generator_slice_prediction_v1_(moving_image, 5, (256,256), 'HipCenter')
            x_lower, y_lower_name = DataGenerator_V1.data_generator_slice_prediction_v1_(moving_image, 5, (256,256), 'KneeCenter')
                  
            hip_slice_model = load_model(config.slice_prediction_models['HipCenter']) #remaining to change
            knee_slice_model = load_model(config.slice_prediction_models['KneeCenter'])
            y_pred_upper = hip_slice_model.predict(x_upper)
            y_pred_lower = knee_slice_model.predict(x_lower)
            temp_points = list(ndimage.center_of_mass(moving_image[:,:,int(y_upper_name[np.argmax(y_pred_upper)][0])]+1024))
            temp_points.append(y_upper_name[np.argmax(y_pred_upper)][0])
            data_points_hip = np.array(temp_points).astype(int)
            temp_points = list(ndimage.center_of_mass(moving_image[:,:,int(y_lower_name[np.argmax(y_pred_lower)][0])]+1024))
            temp_points.append(y_lower_name[np.argmax(y_pred_lower)][0])
            data_points_knee = np.array(temp_points).astype(int)
            print("Slice Prediction done...")
            # Applying Registration
            print("Applying Registration...")
            Landmarks = LANDMARK_DETECTION_CONFIG.apply_registration(moving_image,data_points_hip,data_points_knee,ref_landmarks,reference_images,reference_images_points,mode)
            print("Saving Landmarks as CSV file")
            predictied_landmarks = pd.DataFrame(Landmarks)
            pred = predictied_landmarks.T
            pred.columns = ["KneeCenter","AMEpi","SMEpi","LatEpi","PostMed",'PostLat','DistalMed','DistalLat',
                        "WhiteSide1",'WhiteSide2','HipCenter']
            pred = pred.T
            pred.to_excel(f"Landmarks{image_name}.xlsx")
            print("Done")