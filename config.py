train_CP_ARTHRO3D_DATA_100CT_1MM = 'D:\\Bone_Landmark_Detection_And_Segmentation\\data\\CP_ARTHRO3D_DATA_100CT_1MM\\Orignal_train'
test_CP_ARTHRO3D_DATA_100CT_1MM = 'D:\\Bone_Landmark_Detection_And_Segmentation\\data\\CP_ARTHRO3D_DATA_100CT_1MM\\test'
val_CP_ARTHRO3D_DATA_100CT_1MM = 'D:\\Bone_Landmark_Detection_And_Segmentation\\data\\CP_ARTHRO3D_DATA_100CT_1MM\\val'

# knee_points = ['KneeCenter', 'AMEpi', 'SMEpi', 'PostMed', 'DistalMed', 'DistalLat', 'LatEpi', 'CanalSurface', 'kneeCenterTibia', 'c', 'Tibial_tuberosity_2', 'Medial_condyle_tibia',
#                'Lateral_condyle_tibia', 'postrolateraltibia']

# branch_to_take = {'KneeCenter':[128, 64, 16], 'AMEpi':[128, 64, 16], 'SMEpi':[128, 64, 16], 'PostMed':[128, 64, 16], 'DistalMed':[128, 64, 16], 
#                   'DistalLat':[128, 64, 16], 'LatEpi':[128, 64, 16], 'WhiteSide1':[128, 64, 16], 'WhiteSide2':[128, 64, 16], 'CanalSurface':[128, 64, 16], 
#                   'kneeCenterTibia':[128, 64, 16], 'Tibial_tuberosity_1':[128, 64, 16], 'Tibial_tuberosity_2':[128, 64, 16], 'Medial_condyle_tibia':[128, 64, 16],
#                 'Lateral_condyle_tibia':[128, 64, 16], 'postrolateraltibia':[128, 64, 16], 'tibiaMed':[128, 64, 16], 'TibiaCanalCenter2':[128, 64, 16], }

inverted_ct = ['15016_RIGHT_nii_1mm', '14996_LEFT_nii_1mm', '15029_RIGHT_nii_1mm', '18260_RIGHT_nii_1mm',
               '18414_LEFT_nii_1mm','19235_RIGHT_nii_1mm','19250_RIGHT_nii_1mm','19319_RIGHT_nii_1mm', '19340_LEFT_nii_1mm', '19628_LEFT_nii_1mm','19838_LEFT_nii_1mm','19968_LEFT_nii_1mm','19976_RIGHT_nii_1mm',
               '22385_LEFT_nii_1mm','22391_LEFT_nii_1mm', '22413_LEFT_nii_1mm', '22416_LEFT_nii_1mm']

problematic_ct = ['14946_LEFT_nii_1mm', '17794_LEFT_nii_1mm', '139_RIGHT_nii_1mm', 
                  '18828_LEFT_nii_1mm', '19968_LEFT_nii_1mm','19973_RIGHT_nii_1mm', '19976_RIGHT_nii_1mm','22330_LEFT_nii_1mm']

# {'KneeCenter':[128, 64, 16], 'AMEpi':[128, 64, 16], 'SMEpi':[128, 64, 16], 'PostMed':[128, 64, 16], 'DistalMed':[128, 64, 16], 
#                   'DistalLat':[128, 64, 16], 'LatEpi':[128, 64, 16], 'CanalSurface':[128, 64, 16], 
#                   'kneeCenterTibia':[128, 64, 16], 'Tibial_tuberosity_2':[128, 64, 16], 'Medial_condyle_tibia':[128, 64, 16],
#                 'Lateral_condyle_tibia':[128, 64, 16], 'postrolateraltibia':[128, 64, 16]}


branch_to_take = {'KneeCenter':[128, 64, 16], 'AMEpi':[128, 64, 16], 'SMEpi':[128, 64, 16], 'PostMed':[128, 64, 16], 'DistalMed':[128, 64, 16], 
                  'DistalLat':[128, 64, 16], 'LatEpi':[128, 64, 16], 'CanalSurface':[128, 64, 16], 
                  'kneeCenterTibia':[128, 64, 16], 'Tibial_tuberosity_2':[128, 64, 16], 'Medial_condyle_tibia':[128, 64, 16],
                'Lateral_condyle_tibia':[128, 64, 16], 'postrolateraltibia':[128, 64, 16]}


knee_points = ['Lateral_condyle_tibia']


knee_segmentation_models = [r'D:\Bone_Landmark_Detection_And_Segmentation\notebooks\Segmentation_models\model1_resunet\weights-improvement-100-0.06.h5',
                            r'D:\Bone_Landmark_Detection_And_Segmentation\notebooks\Segmentation_models\model2_resunet\weights-improvement-100-0.06.h5',
                            r'D:\Bone_Landmark_Detection_And_Segmentation\notebooks\Segmentation_models\model3_resunet\weights-improvement-100-0.06.h5',
                            r'D:\Bone_Landmark_Detection_And_Segmentation\notebooks\Segmentation_models\model4_unet\weights-improvement-100-0.07.h5',
                            r'D:\Bone_Landmark_Detection_And_Segmentation\notebooks\Segmentation_models\model5_unet\weights-improvement-100-0.06.h5']

hip_segmentation_models = [r'D:\Bone_Landmark_Detection_And_Segmentation\notebooks\Segmentation_models\hip_model1_resunet\weights-improvement-100-0.00.h5',
                           r'D:\Bone_Landmark_Detection_And_Segmentation\notebooks\Segmentation_models\hip_model2_resunet\weights-improvement-100-0.01.h5',
                           r'D:\Bone_Landmark_Detection_And_Segmentation\notebooks\Segmentation_models\hip_model3_resunet\weights-improvement-100-0.00.h5',
                           r'D:\Bone_Landmark_Detection_And_Segmentation\notebooks\Segmentation_models\hip_model4_unet\weights-improvement-100-0.04.h5',
                           r'D:\Bone_Landmark_Detection_And_Segmentation\notebooks\Segmentation_models\hip_model5_unet\weights-improvement-90-0.04.h5']


slice_prediction_models = {'HipCenter': r'D:\Bone_Landmark_Detection_And_Segmentation\notebooks\Slice_prediction_models\HipCenter_slice_prediction_model.h5',
                           'KneeCenter': r'D:\Bone_Landmark_Detection_And_Segmentation\notebooks\Slice_prediction_models\KneeCenter_slice_prediction_model.h5'}

final_segmented_mesh_path = r'D:\Bone_Landmark_Detection_And_Segmentation\notebooks\SEGMENTATION_COMBINED\GENERATED_DATA'
