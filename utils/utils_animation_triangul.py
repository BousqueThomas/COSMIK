import pandas as pd
import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
import time 
import numpy as np


    
def affichage_triangul(file_triangul):

    jcp_data = pd.read_csv(file_triangul,header=None)
    row = jcp_data.iloc[1] #Récupération de la première ligne.

    jcp_list = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'head', 'neck', 'hip', 'left_big_toe', 'right_big_toe', 'left_small_toe', 'right_small_toe', 'left_heel', 'right_heel','Left_hand_root', 'left_thumb1', 'left_thumb2', 'left_forefinger1', 'left_middle_finger1', 'left_ring_finger1', 'left_pinky_finger1','right_hand_root', 'right_thumb1', 'right_thumb2', 'right_forefinger1','right_middle_finger1', 'right_ring_finger1', 'right_pinky_finger1']


    first_col = jcp_data.iloc[:, 0]
    nb_samples = int(len(first_col))
    print('nb_samples =', nb_samples)

    # Create a new visualizer
    vis = meshcat.Visualizer()
    vis.open()
    print("Vous pouvez ouvrir le visualiseur en visitant l'URL suivante :")
    print(vis.url())

    for i in range(len(jcp_list)):
        # print(jcp_list[i])
        vis[jcp_list[i]].set_object(g.Sphere([0.02]), g.MeshLambertMaterial(color=0xFF001E,reflectivity=0.8))



    for i in range(nb_samples):

        #Récupération de tous les JCP d'une frame
        row_jcps = jcp_data.iloc[i][:].tolist()
  
        jcps_positions = []
        num_points_jcps = len(row_jcps) // 3


        for i in range(num_points_jcps):
            start_index = i * 3
            jcps_positions.append([float(row_jcps[start_index]), float(row_jcps[start_index+1]), float(row_jcps[start_index+2])])

        for i in range(len(jcp_list)):

            p_arr = np.array(jcps_positions[i]).reshape(3,1)
            new_p = np.array([p_arr[0], p_arr[1], p_arr[2]]).flatten()
            vis[jcp_list[i]].set_transform(tf.translation_matrix(new_p))

        time.sleep(0.05)




def affichage_triangul_single_frame(file_triangul):

    jcp_data = pd.read_csv(file_triangul, header=None)
    
    # Définir la liste des noms de joints
    jcp_list = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'head', 'neck', 'hip', 'left_big_toe', 'right_big_toe', 'left_small_toe', 'right_small_toe', 'left_heel', 'right_heel']

    
    # Sélectionner la première ligne de données
    row_jcps = jcp_data.iloc[0].tolist()
    
    # Assurez-vous que le nombre de colonnes dans le fichier correspond au nombre attendu de points JCP
    expected_num_columns = len(jcp_list) * 3
    if jcp_data.shape[1] != expected_num_columns:
        raise ValueError(f"Le fichier de données a {jcp_data.shape[1]} colonnes, mais {expected_num_columns} étaient attendues.")
    

    # Calculer les positions des JCPs pour la première frame
    jcps_positions = []
    num_points_jcps = len(row_jcps) // 3
    for i in range(num_points_jcps):
        start_index = i * 3
        jcps_positions.append([float(row_jcps[start_index]), float(row_jcps[start_index+1]), float(row_jcps[start_index+2])])
    

    # Vérification pour le débogage
    print(f"Nombre de points JCP : {num_points_jcps}")
    print(f"Positions des JCPs : {jcps_positions}")

    
    # Créer un visualiseur Meshcat
    vis = meshcat.Visualizer()
    vis.open()
    print("You can open the visualizer by visiting the following URL:")
    print(vis.url())


    # Ajouter les objets au visualiseur
    for i in range(len(jcp_list)):
        vis[jcp_list[i]].set_object(g.Sphere([0.05]), g.MeshLambertMaterial(color=0xFF001E, reflectivity=0.8))
    
    # Mettre à jour les positions des objets pour la première frame
    for i in range(len(jcp_list)):
        if i < len(jcps_positions):
            p_arr = np.array(jcps_positions[i]).reshape(3,1)
            new_p = np.array([p_arr[0], p_arr[1], p_arr[2]]).flatten()
            vis[jcp_list[i]].set_transform(tf.translation_matrix(new_p))
        else: 
            print(f"Warning: No position data for {jcp_list[i]}.")

    time.sleep(10)


