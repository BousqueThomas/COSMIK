
# import eigenpy
from tqdm import tqdm
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from read_write_utils import remove_nans_from_list_of_dicts, read_lstm_data, get_lstm_mks_names, read_mocap_data, convert_to_list_of_dicts, write_joint_angle_results
from ik_utils import IK_Casadi
import pinocchio as pin 
from model_utils import get_subset_challenge_mks_names, get_segments_lstm_mks_dict_challenge, build_model_challenge
from pinocchio.visualize import GepettoVisualizer
from viz_utils import place
import numpy as np 
import time 


def run_ik (no_sujet, task, data_path):

    subject='sujet_0'+str(no_sujet)
    fichier_csv_lstm_mks_calib = f"{data_path}sujet_0"+str(no_sujet)+"/marche/LSTM/jcp_coordinates_ncameras_augmented_marche_"+str(no_sujet)+".csv"
    fichier_csv_lstm_mks = f"{data_path}"+subject+"/"+task+"/LSTM/jcp_coordinates_ncameras_augmented_"+task+"_"+str(no_sujet)+".csv"
    fichier_csv_mocap_mks = f"{data_path}mocap_mks_recup_"+subject+".trc"
    dir_courant=os.getcwd()
    meshes_folder_path = f"{dir_courant}/meshes/" #Changes le par ton folder de meshes

    #Read data
    lstm_mks_dict, mapping = read_lstm_data(fichier_csv_lstm_mks)
    lstm_mks_dict_calib, mapping_calib = read_lstm_data(fichier_csv_lstm_mks_calib)
    lstm_mks_names = get_lstm_mks_names(fichier_csv_lstm_mks) #Liste des noms des mks du lstm (totalité des mks)
    subset_challenge_mks_names = get_subset_challenge_mks_names() #Cette fonction te retourne les noms des markers dont on a besoin pour le challenge
    mocap_mks_dict = read_mocap_data(fichier_csv_mocap_mks) #Markers mocap, pas utilisés ici car merdiques pour le moment
    lstm_mks_dict_calib = convert_to_list_of_dicts(lstm_mks_dict_calib)
    lstm_mks_dict = convert_to_list_of_dicts(lstm_mks_dict) #Je convertis ton dictionnaire de trajectoires (arrays) en une "trajectoire de dictionnaires", c'est plus facile à manipuler pour la calib
    lstm_mks_positions_calib = lstm_mks_dict_calib[0] #Je prends la première frame de la trajectoire pour construire le modèle
    seg_names_mks = get_segments_lstm_mks_dict_challenge() #Dictionnaire contenant les noms des segments + les mks correspondnat à chaque segment


    #C'est normal qu'il y ait deux fois le même argument, normalement le 1er argument c'est les mks mocap. 
    model, geom_model, visuals_dict = build_model_challenge(lstm_mks_positions_calib, lstm_mks_positions_calib, meshes_folder_path)

    q0 = pin.neutral(model)
    q0[7:]=0.0001*np.ones(model.nq-7)

    ### IK 

    ###### FOR MANUTENTION ONLY TRY TO MITIGATE MOONWALK ######

    # for jj in range(1,len(lstm_mks_dict)):
    #     lstm_mks_dict[jj]["r_toe_study"]=lstm_mks_dict[0]["r_toe_study"]
    #     lstm_mks_dict[jj]["r_ankle_study"]=lstm_mks_dict[0]["r_ankle_study"]
    #     lstm_mks_dict[jj]["r_mankle_study"]=lstm_mks_dict[0]["r_mankle_study"]
    #     lstm_mks_dict[jj]["r_5meta_study"]=lstm_mks_dict[0]["r_5meta_study"]
    #     lstm_mks_dict[jj]["r_calc_study"]=lstm_mks_dict[0]["r_calc_study"]

    


    ik_problem = IK_Casadi(model, lstm_mks_dict, q0)

    q = ik_problem.solve_ik()

    q=np.array(q)

    directory_name = f"{dir_courant}/results_IK/"+subject+"/"+task
    if not os.path.exists(f'{directory_name}/'):
        os.makedirs(f'{directory_name}/')
        print(f"Le répertoire pour l'enregistrement dees résultats a été créé.")

    write_joint_angle_results(directory_name,q)

    ### Visualisation of the obtained trajectory 

    visual_model = geom_model
    viz = GepettoVisualizer(model, geom_model, visual_model)

    try:
        viz.initViewer()
    except ImportError as err:
        print("Error while initializing the viewer. It seems you should install gepetto-viewer")
        print(err)
        sys.exit(0)

    try:
        viz.loadViewerModel("pinocchio")
    except AttributeError as err:
        print("Error while loading the viewer model. It seems you should start gepetto-viewer")
        print(err)
        sys.exit(0)

    for name, visual in visuals_dict.items():
        viz.viewer.gui.setColor(viz.getViewerNodeName(visual, pin.GeometryType.VISUAL), [0, 1, 1, 0.5])

    for seg_name, mks in seg_names_mks.items():
        viz.viewer.gui.addXYZaxis(f'world/{seg_name}', [0, 255., 0, 1.], 0.008, 0.08)
        for mk_name in mks:
            sphere_name_model = f'world/{mk_name}_model'
            sphere_name_raw = f'world/{mk_name}_raw'
            viz.viewer.gui.addSphere(sphere_name_model, 0.01, [0, 0., 255, 1.])
            viz.viewer.gui.addSphere(sphere_name_raw, 0.01, [255, 0., 0, 1.])

    # Set color for other visual objects similarly
    data = model.createData()

    for i in range(len(q)):
        q_i = q[i]
        viz.display(q_i)

        pin.forwardKinematics(model, data, q_i)
        pin.updateFramePlacements(model, data)

        for seg_name, mks in seg_names_mks.items():
            #Display markers from model
            for mk_name in mks:
                sphere_name_model = f'world/{mk_name}_model'
                sphere_name_raw = f'world/{mk_name}_raw'
                mk_position = data.oMf[model.getFrameId(mk_name)].translation
                place(viz, sphere_name_model, pin.SE3(np.eye(3), np.matrix(mk_position.reshape(3,)).T))
                place(viz, sphere_name_raw, pin.SE3(np.eye(3), np.matrix(lstm_mks_dict[i][mk_name].reshape(3,)).T))
            
            #Display frames from model
            frame_name = f'world/{seg_name}'
            frame_se3= data.oMf[model.getFrameId(seg_name)]
            place(viz, frame_name, frame_se3)
        
        if i == 0:
            input("Ready?")
        else:
            time.sleep(0.016)

no_sujet = int(input("Entrez le numéro du sujet (ex: 2): "))
task = input("Entrez la tâche (ex: 'assis-debout'): ")
data_path='/home/tbousquet/Documents/Donnees_cosmik/Data/'
run_ik (no_sujet, task, data_path)