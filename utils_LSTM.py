import pandas as pd
import numpy as np
import os
import utilsDataman
from utils import TRC2numpy
import tensorflow as tflow
import json
import copy
import time
from scipy.spatial.transform import Rotation as R
# import meshcat
# import meshcat.geometry as g
# import meshcat.transformations as tf

def modif_LSTM (trc_file_path,no_sujet,task):
    #Modification du fichier résultat pour compatibilité avec LSTM___________________________________________________________________________
    subject='sujet1'
    trial='exotique'
    # Chemins vers le fichier d'OpenCap
    csv_file_path = "/home/tbousquet/Documents/COSMIK/Donnees challenge markerless/a9fd6740-1c9d-40df-beca-15e6eecf08d7.csv"


    # Lire les fichiers
    csv_data = pd.read_csv(csv_file_path, header=None)
    trc_data = pd.read_csv(trc_file_path, header=None)
    # print(trc_data.shape)
    # print('FFFF',trc_data.columns)

    # Colonnes à garder dans le fichier .trc et leur nouvel ordre
    # joints_csv = ['Neck', 'RShoulder', 'RElbow', 'RWrist', 'LShoulder', 'LElbow', 'LWrist', 'midHip', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle', 'LBigToe', 'LSmallToe', 'LHeel', 'RBigToe', 'RSmallToe', 'RHeel']
    joints_trc = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'head', 'neck', 'hip', 'left_big_toe', 'right_big_toe', 'left_small_toe', 'right_small_toe', 'left_heel', 'right_heel','Left_hand_root', 'left_thumb1', 'left_thumb2', 'left_forefinger1', 'left_middle_finger1', 'left_ring_finger1', 'left_pinky_finger1','right_hand_root', 'right_thumb1', 'right_thumb2', 'right_forefinger1','right_middle_finger1', 'right_ring_finger1', 'right_pinky_finger1']

    # Créer un mapping de joints
    joint_mapping = {
        'neck': 'Neck',
        'right_shoulder': 'RShoulder',
        'right_elbow': 'RElbow',
        'right_wrist': 'RWrist',
        'left_shoulder': 'LShoulder',
        'left_elbow': 'LElbow',
        'left_wrist': 'LWrist',
        'hip': 'midHip',
        'right_hip': 'RHip',
        'right_knee': 'RKnee',
        'right_ankle': 'RAnkle',
        'left_hip': 'LHip',
        'left_knee': 'LKnee',
        'left_ankle': 'LAnkle',
        'left_big_toe': 'LBigToe',
        'left_small_toe': 'LSmallToe',
        'left_heel': 'LHeel',
        'right_big_toe': 'RBigToe',
        'right_small_toe': 'RSmallToe',
        'right_heel': 'RHeel'
    }

    # print('le trc a ', len(trc_data.columns), 'colonnes')

    # Indices des colonnes à garder
    columns_to_keep = []
    for joint in joint_mapping.keys():
        # print(joint)
        idx = (joints_trc.index(joint)* 3)
        # print(idx)
        columns_to_keep.extend([idx, idx + 1, idx + 2])
    # print ('colonne à garder = ', columns_to_keep)



    # Garder et réorganiser les colonnes dans le fichier .trc
    trc_filtered_data = trc_data.iloc[:, columns_to_keep]


    # Renommer les colonnes pour correspondre à celles du fichier .csv
    new_column_names = []
    for joint in joints_trc:
        if joint in joint_mapping:
            new_column_names.extend([joint_mapping[joint] + '_X', joint_mapping[joint] + '_Y', joint_mapping[joint] + '_Z'])

    trc_filtered_data.columns = new_column_names


    # Ajouter une colonne avec une numérotation des lignes
    trc_filtered_data.insert(0, 'Frame', range(1, len(trc_filtered_data) + 1))

    # Ajouter une colonne avec une indentation de 1/60 en commençant à 0
    trc_filtered_data.insert(1, 'Time', np.arange(0, len(trc_filtered_data))/60)




    # Ajouter des lignes supplémentaires pour correspondre au nombre de lignes du fichier .csv si nécessaire
    num_rows_to_add = len(csv_data) - len(trc_filtered_data)
    if num_rows_to_add > 0:
        additional_rows = pd.DataFrame(0.0, index=range(num_rows_to_add), columns=trc_filtered_data.columns)
        trc_filtered_data = pd.concat([trc_filtered_data, additional_rows], ignore_index=True)

    # Sauvegarder le fichier .trc transformé
    # Création du répertoire si nécessaire
    LSTM_dir = '/home/tbousquet/Documents/Donnees_cosmik/Data/sujet_0' + str(no_sujet) + '/' + task + '/LSTM'
    LSTM_output='/home/tbousquet/Documents/Donnees_cosmik/Data/sujet_0' + str(no_sujet) + '/' + task + '/LSTM/jcp_coordinates_ncameras_transformed_'+task+'_'+str(no_sujet)+'.trc'

    if not os.path.exists(LSTM_dir):
        os.makedirs(LSTM_dir)
        print(f'Le répertoire {LSTM_dir} a été créé. Et le fichier tranformé à donner au LSTM est enregistré dans ce dernier.')

    trc_filtered_data.to_csv(LSTM_output, header=False, index=False)

    # print(trc_filtered_data.head())
    # print('le trc transformé a ', len(trc_filtered_data.columns), 'colonnes')





# #Utilisation du LSTM __________________________________________________________________________


def augmentTRC(pathInputTRCFile, subject_mass, subject_height,
               pathOutputTRCFile, pathOutputCSVFile, augmenterDir, augmenterModelName="LSTM",
               augmenter_model='v0.3', offset=True):
               
    # This is by default - might need to be adjusted in the future.
    featureHeight = True
    featureWeight = True
    
    # Augmenter types
    if augmenter_model == 'v0.0':
        from utils import getOpenPoseMarkers_fullBody
        feature_markers_full, response_markers_full = getOpenPoseMarkers_fullBody()         
        augmenterModelType_all = [augmenter_model]
        feature_markers_all = [feature_markers_full]
        response_markers_all = [response_markers_full]            
    elif augmenter_model == 'v0.1' or augmenter_model == 'v0.2':
        # Lower body           
        augmenterModelType_lower = '{}_lower'.format(augmenter_model)
        from utils import getOpenPoseMarkers_lowerExtremity
        feature_markers_lower, response_markers_lower = getOpenPoseMarkers_lowerExtremity()
        # Upper body
        augmenterModelType_upper = '{}_upper'.format(augmenter_model)
        from utils import getMarkers_upperExtremity_noPelvis
        feature_markers_upper, response_markers_upper = getMarkers_upperExtremity_noPelvis()        
        augmenterModelType_all = [augmenterModelType_lower, augmenterModelType_upper]
        feature_markers_all = [feature_markers_lower, feature_markers_upper]
        response_markers_all = [response_markers_lower, response_markers_upper]
    else:
        # Lower body           
        augmenterModelType_lower = '{}_lower'.format(augmenter_model)
        from utils import getOpenPoseMarkers_lowerExtremity2
        feature_markers_lower, response_markers_lower = getOpenPoseMarkers_lowerExtremity2()
        # Upper body
        augmenterModelType_upper = '{}_upper'.format(augmenter_model)
        from utils import getMarkers_upperExtremity_noPelvis2
        feature_markers_upper, response_markers_upper = getMarkers_upperExtremity_noPelvis2()        
        augmenterModelType_all = [augmenterModelType_lower, augmenterModelType_upper]
        feature_markers_all = [feature_markers_lower, feature_markers_upper]
        response_markers_all = [response_markers_lower, response_markers_upper]
        
    print('Using augmenter model: {}'.format(augmenter_model))
    
    # %% Process data.
    # Import TRC file
    trc_file = utilsDataman.TRCFile(pathInputTRCFile)
    
    # Loop over augmenter types to handle separate augmenters for lower and
    # upper bodies.
    outputs_all = {}
    n_response_markers_all = 0
    for idx_augm, augmenterModelType in enumerate(augmenterModelType_all):
        outputs_all[idx_augm] = {}
        feature_markers = feature_markers_all[idx_augm]
        response_markers = response_markers_all[idx_augm]
        
        augmenterModelDir = os.path.join(augmenterDir, augmenterModelName, 
                                         augmenterModelType)
                                        
        
        # %% Pre-process inputs.
        # Step 1: import .trc file with OpenPose marker trajectories.  
        trc_data = TRC2numpy(pathInputTRCFile, feature_markers)
        trc_data_data = trc_data[:,1:]
        
        # Step 2: Normalize with reference marker position.
        with open(os.path.join(augmenterModelDir, "metadata.json"), 'r') as f:
            metadata = json.load(f)
        referenceMarker = metadata['reference_marker']
        referenceMarker_data = trc_file.marker(referenceMarker)
        norm_trc_data_data = np.zeros((trc_data_data.shape[0],
                                       trc_data_data.shape[1]))
        for i in range(0,trc_data_data.shape[1],3):
            norm_trc_data_data[:,i:i+3] = (trc_data_data[:,i:i+3] - 
                                           referenceMarker_data)
            
        # Step 3: Normalize with subject's height.
        norm2_trc_data_data = copy.deepcopy(norm_trc_data_data)
        norm2_trc_data_data = norm2_trc_data_data / subject_height
        
        # Step 4: Add remaining features.
        inputs = copy.deepcopy(norm2_trc_data_data)
        if featureHeight:    
            inputs = np.concatenate(
                (inputs, subject_height*np.ones((inputs.shape[0],1))), axis=1)
        if featureWeight:    
            inputs = np.concatenate(
                (inputs, subject_mass*np.ones((inputs.shape[0],1))), axis=1)
            
        # Step 5: Pre-process data
        pathMean = os.path.join(augmenterModelDir, "mean.npy")
        pathSTD = os.path.join(augmenterModelDir, "std.npy")
        if os.path.isfile(pathMean):
            trainFeatures_mean = np.load(pathMean, allow_pickle=True)
            inputs -= trainFeatures_mean
        if os.path.isfile(pathSTD):
            trainFeatures_std = np.load(pathSTD, allow_pickle=True)
            inputs /= trainFeatures_std 
            
        # Step 6: Reshape inputs if necessary (eg, LSTM)
        if augmenterModelName == "LSTM":
            inputs = np.reshape(inputs, (1, inputs.shape[0], inputs.shape[1]))
            
        # %% Load model and weights, and predict outputs.
        json_file = open(os.path.join(augmenterModelDir, "model.json"), 'r')
        pretrainedModel_json = json_file.read()
        json_file.close()
        model = tflow.keras.models.model_from_json(pretrainedModel_json)
        model.load_weights(os.path.join(augmenterModelDir, "weights.h5"))  
        outputs = model.predict(inputs)
        
        # %% Post-process outputs.
        # Step 1: Reshape if necessary (eg, LSTM)
        if augmenterModelName == "LSTM":
            outputs = np.reshape(outputs, (outputs.shape[1], outputs.shape[2]))
            
        # Step 2: Un-normalize with subject's height.
        unnorm_outputs = outputs * subject_height
        
        # Step 2: Un-normalize with reference marker position.
        unnorm2_outputs = np.zeros((unnorm_outputs.shape[0],
                                    unnorm_outputs.shape[1]))
        for i in range(0,unnorm_outputs.shape[1],3):
            unnorm2_outputs[:,i:i+3] = (unnorm_outputs[:,i:i+3] + 
                                        referenceMarker_data)
            
        # %% Add markers to .trc file.
        for c, marker in enumerate(response_markers):
            x = unnorm2_outputs[:,c*3]
            y = unnorm2_outputs[:,c*3+1]
            z = unnorm2_outputs[:,c*3+2]
            trc_file.add_marker(marker, x, y, z)
            
        # %% Gather data for computing minimum y-position.
        outputs_all[idx_augm]['response_markers'] = response_markers   
        outputs_all[idx_augm]['response_data'] = unnorm2_outputs
        n_response_markers_all += len(response_markers)
        
    # %% Extract minimum y-position across response markers. This is used
    # to align feet and floor when visualizing.
    responses_all_conc = np.zeros((unnorm2_outputs.shape[0],
                                   n_response_markers_all*3))
    idx_acc_res = 0
    for idx_augm in outputs_all:
        idx_acc_res_end = (idx_acc_res + 
                           (len(outputs_all[idx_augm]['response_markers']))*3)
        responses_all_conc[:,idx_acc_res:idx_acc_res_end] = (
            outputs_all[idx_augm]['response_data'])
        idx_acc_res = idx_acc_res_end
    # Minimum y-position across response markers.
    min_y_pos = np.min(responses_all_conc[:,1::3])
        
    # %% If offset
    if offset:
        trc_file.offset('y', -(min_y_pos-0.01))
        
    # %% Return augmented .trc file   
    trc_file.write(pathOutputTRCFile)
    
    # Lecture du fichier TRC en utilisant pandas
    donnees_trc_mks = pd.read_csv(pathOutputTRCFile, skiprows=3, delimiter='\t', header=None)

    # Affichage des premières lignes des données pour vérifier
    # Conversion des données en CSV
    donnees_trc_mks.to_csv(pathOutputCSVFile, index=False, header=True)


    print("Le LSTM a terminé.")
    return min_y_pos


