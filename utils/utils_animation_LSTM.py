import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
import pandas as pd
import numpy as np
import time


#AFFICHAGE ____________________________________________________

def affichage_LSTM(fichier_csv_mks):


    mk_data = pd.read_csv(fichier_csv_mks)
    row = mk_data.iloc[0] #Récupération de la deuxième ligne.
    first_col = mk_data.iloc[:, 0]
    nb_samples = int(len(first_col))
    mk_liste = row[2:].tolist() #on enlève les deux premieres valeurs

    mk_list = [mot for mot in mk_liste if pd.notna(mot)] #On enlève les nan correspondant aux cases vides du fichier csv

    # print('mk_list=',mk_list, len(mk_list))

    # Create a new visualizer
    vis = meshcat.Visualizer()
    vis.open()
    # print("Vous pouvez ouvrir le visualiseur en visitant l'URL suivante :")
    print(vis.url())

    for i in range(len(mk_list)):
        # print(mk_list[i])
        vis[mk_list[i]].set_object(g.Sphere([0.02]), g.MeshLambertMaterial(color=0x0000FF,reflectivity=0.8))



    for i in range(nb_samples):
        #Récupération de tous les JCP d'une frame

    
        row_mks = mk_data.iloc[i+2][2:].tolist()
        row_mks.pop()
        # print('row_mks=',row_mks)
        # print('len __________ ',len(row_mks))



        mks_positions = []
        num_points_mks = len(row_mks) // 3
        # print('num_points_mks =', num_points_mks)


        for i in range(num_points_mks):
            start_index = i * 3
            mks_positions.append([float(row_mks[start_index]), float(row_mks[start_index+1]), float(row_mks[start_index+2])])

        # print('mks_positions',mks_positions)


        for i in range(len(mk_list)):
            #     # print(mks_positions[i])

            p_arr = np.array(mks_positions[i]).reshape(3,1)
            #     # new_p_arr = np.matmul(rot.as_matrix(), p_arr)
            new_p = np.array([p_arr[0], p_arr[1], p_arr[2]]).flatten()

            vis[mk_list[i]].set_transform(tf.translation_matrix(new_p))


        time.sleep(0.05)
