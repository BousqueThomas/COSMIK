import pandas as pd
import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
import time 
import numpy as np

def affichage_triangul(file_triangul):

    jcp_data = pd.read_csv(file_triangul,header=None)

    row = jcp_data.iloc[1] #Récupération de la première ligne.
    # jcp_liste = row[:].tolist() #on enlève les deux premieres valeurs

    # jcp_list = [mot for mot in jcp_liste if pd.notna(mot)] #On enlève les nan correspondant aux cases vides du fichier csv
    jcp_list = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'head', 'neck', 'hip', 'left_big_toe', 'right_big_toe', 'left_small_toe', 'right_small_toe', 'left_heel', 'right_heel','Left_hand_root', 'left_thumb1', 'left_thumb2', 'left_forefinger1', 'left_middle_finger1', 'left_ring_finger1', 'left_pinky_finger1','right_hand_root', 'right_thumb1', 'right_thumb2', 'right_forefinger1','right_middle_finger1', 'right_ring_finger1', 'right_pinky_finger1']

    #print(jcp_list)

    first_col = jcp_data.iloc[:, 0]
    nb_samples = int(len(first_col))
    print('nb_samples =', nb_samples)

    # Create a new visualizer
    vis = meshcat.Visualizer()
    vis.open()
    # print("Vous pouvez ouvrir le visualiseur en visitant l'URL suivante :")
    print(vis.url())

    for i in range(len(jcp_list)):
        # print(jcp_list[i])
        vis[jcp_list[i]].set_object(g.Sphere([0.02]), g.MeshLambertMaterial(color=0xFF001E,reflectivity=0.8))



    for i in range(nb_samples):
        #Récupération de tous les JCP d'une frame

        row_jcps = jcp_data.iloc[i][:].tolist()
        # print('row_jcp=', row_jcps)


        jcps_positions = []
        num_points_jcps = len(row_jcps) // 3
        # print('num_points_jcps =', num_points_jcps)



        # row_mks = mk_data.iloc[i+2][2:].tolist()
        # row_mks.pop()
        # # print('row_mks=',row_mks)
        # # print('len __________ ',len(row_mks))



        # mks_positions = []
        # num_points_mks = len(row_mks) // 3
        # # print('num_points_mks =', num_points_mks)


        for i in range(num_points_jcps):
            start_index = i * 3
            jcps_positions.append([float(row_jcps[start_index]), float(row_jcps[start_index+1]), float(row_jcps[start_index+2])])
        # print('jcps_positions = ', jcps_positions)
        # print("JCPs Positions:", jcps_positions)

        # print("len jcps:", len(jcps_positions))
        for i in range(len(jcp_list)):
            # print(i)
            # print(jcp_list[i])
            p_arr = np.array(jcps_positions[i]).reshape(3,1)
            # print('p_arr',p_arr)

            # new_p_arr = np.matmul(rot.as_matrix(), p_arr)
            # new_p = np.array([new_p_arr[0], new_p_arr[1], new_p_arr[2]]).flatten()
            new_p = np.array([p_arr[0], p_arr[1], p_arr[2]]).flatten()
            # print('new_p',new_p)
            # print(f"New position for {jcp_list[i]}: {new_p}")
            
            vis[jcp_list[i]].set_transform(tf.translation_matrix(new_p))
        # print('p_array', p_arr)

        time.sleep(0.05)
