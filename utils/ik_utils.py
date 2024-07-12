import pinocchio as pin 
import casadi 
import pinocchio.casadi as cpin 
from typing import Dict, List
import numpy as np 
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import math
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from utils.read_write_utils import remove_nans_from_list_of_dicts, read_lstm_data, get_lstm_mks_names, read_mocap_data, convert_to_list_of_dicts, write_joint_angle_results
import pinocchio as pin 
from utils.model_utils import get_subset_challenge_mks_names, get_segments_lstm_mks_dict_challenge, build_model_challenge
from pinocchio.visualize import GepettoVisualizer
from utils.viz_utils import place
import time 
import cyipopt
from scipy.optimize import approx_fprime
import quadprog


def ik_pipe_qp(no_sujet, task, data_path):


    subject='sujet_0'+str(no_sujet)
    fichier_csv_lstm_mks= f"{data_path}"+subject+"/"+task+"/LSTM/jcp_coordinates_ncameras_augmented_"+task+"_"+str(no_sujet)+".csv"

    dir_courant=os.getcwd()
    meshes_folder_path = f"{dir_courant}/meshes/"


    #Read data
    lstm_mks_dict, mapping =read_lstm_data(fichier_csv_lstm_mks)
    lstm_mks_dict = convert_to_list_of_dicts(lstm_mks_dict)
    
    lstm_mks_positions_calib = lstm_mks_dict[0] 


    seg_names_mks = get_segments_lstm_mks_dict_challenge() #Dictionnaire contenant les noms des segments + les mks correspondnat à chaque segment



    #C'est normal qu'il y ait deux fois le même argument, normalement le 1er argument c'est les mks mocap. 
    model, geom_model, visuals_dict = build_model_challenge(lstm_mks_positions_calib, lstm_mks_positions_calib, meshes_folder_path)


    q0 = pin.neutral(model)
    # q0[7:]=0.0001*np.ones(model.nq-7)

    ### IK 

    ik_problem = IK_Quadprog(model, lstm_mks_dict, q0)

    q = ik_problem.solve_ik()

    q=np.array(q)
    directory_name =  f"{dir_courant}/results_IK/"+subject+"/"+task
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






def quadprog_solve_qp(P: np.ndarray, q: np.ndarray, G: np.ndarray=None, h: np.ndarray=None, A: np.ndarray=None, b: np.ndarray=None):
    """_Set up the qp solver using quadprog API_

    Args:
        P (np.ndarray): _Hessian matrix of the qp_
        q (np.ndarray): _Gradient vector of the qp_
        G (np.ndarray, optional): _Inequality constraints matrix_. Defaults to None.
        h (np.ndarray, optional): _Vector for inequality constraints_. Defaults to None.
        A (np.ndarray, optional): _Equality constraints matrix_. Defaults to None.
        b (np.ndarray, optional): _Vector for equality constraints_. Defaults to None.

    Returns:
        _launch solve_qp of quadprog solver_
    """
    qp_G = .5 * (P + P.T) #+ np.eye(P.shape[0])*(1e-5)   # make sure P is symmetric, pos,def
    qp_a = -q
    if A is not None:
        qp_C = -np.vstack([A, G]).T
        qp_b = -np.hstack([b, h])
        meq = A.shape[0]
    else:  # no equality constraint
        qp_C = -G.T
        qp_b = -h
        meq = 0
    return quadprog.solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0] #première solution de dq

class Ipopt_warm_start(object):

    def __init__(self,  model,meas,keys):
        self._meas=meas
        self._keys=keys
        self._model=model
        self._data=self._model.createData()
        
    def objective(self, x):
        # callback for objective 
        pin.forwardKinematics(self._model, self._data, x)
        pin.updateFramePlacements(self._model,self._data) 
      
        Goal=np.empty(shape=[0,3])
        markers_pos=[]

        for key in self._keys:
            Goal=np.concatenate((Goal,np.reshape(np.array(self._meas[key]),(1,3))),axis=0)
            markers_pos.append(self._data.oMf[self._model.getFrameId(key)].translation)           
        
        J=np.sum((Goal-markers_pos)**2)

        return  J 

    def constraints(self, x):
        """Returns the constraints."""
        return np.linalg.norm([x[3],x[4],x[5],x[6]]) # norm of the freeflyer quaternion equal to 1

    def gradient(self, x):
        # callback for gradient

        G=approx_fprime(x, self.objective, 1e-5)

        return G

    def jacobian(self, x):
        # callback for jacobian of constraints
        jac=approx_fprime(x, self.constraints, 1e-5)

        return jac

class IK_Quadprog:
    """_Class to manage multi body IK problem using qp solver quadprog_
    """
    def __init__(self,model: pin.Model, dict_m: Dict, q0: np.ndarray) -> None:
        """_Init of the class _

        Args:
            model (pin.Model): _Pinocchio biomechanical model_
            dict_m (Dict): _a dictionnary containing the measures of the landmarks_
            q0 (np.ndarray): _initial configuration_
        """
        self._model = model
        self._data = self._model.createData()
        self._dict_m = dict_m
        self._q0 = q0

        self._dt = 0.01 # TO SET UP : FRAMERATE OF THE DATA

        # Create a list of keys excluding the specified key
        self._keys_list = [key for key in self._dict_m[0].keys() if key !='Time']

        self._nq = self._model.nq
        self._nv = self._model.nv

        self._keys_to_track_list = ['C7_study',
                                    'r.ASIS_study', 'L.ASIS_study', 
                                    'r.PSIS_study', 'L.PSIS_study', 
                                    
                                    'r_shoulder_study',
                                    'r_lelbow_study', 'r_melbow_study',
                                    'r_lwrist_study', 'r_mwrist_study',
                                    'r_ankle_study', 'r_mankle_study',
                                    'r_toe_study','r_5meta_study', 'r_calc_study',
                                    'r_knee_study', 'r_mknee_study',
                                    'r_thigh1_study', 'r_thigh2_study', 'r_thigh3_study',
                                    'r_sh1_study', 'r_sh2_study', 'r_sh3_study',
                                    
                                    'L_shoulder_study', 
                                    'L_lelbow_study', 'L_melbow_study',
                                    'L_lwrist_study','L_mwrist_study',
                                    'L_ankle_study', 'L_mankle_study', 
                                    'L_toe_study','L_5meta_study', 'L_calc_study',
                                    'L_knee_study', 'L_mknee_study',
                                    'L_thigh1_study', 'L_thigh2_study', 'L_thigh3_study',
                                    'L_sh1_study', 'L_sh2_study', 'L_sh3_study']
        
        # for lower_body
        # ['LIAS', 'RIAS', 'LIPS', 'RIPS', 'LFLE', 'LFAL', 'LFCC', 'RFLE', 'RFAL', 'RFCC', 'LFME', 'LTAM', 'LFM5', 'LFM1', 'RFM5', 'RFM1', 'RTAM', 'RFME']

        pin.forwardKinematics(self._model, self._data, self._q0)
        pin.updateFramePlacements(self._model,self._data)
        
        markers_est_pos = []
        for ii in self._keys_to_track_list:
            markers_est_pos.append(self._data.oMf[self._model.getFrameId(ii)].translation.reshape((3,1)))
        self._dict_m_est = dict(zip(self._keys_to_track_list,markers_est_pos))

        # Quadprog and qp settings
        self._K_ii=1
        self._K_lim=1
        self._damping=1e-3



    def create_meas_list(self)-> List[Dict]:
        """_Create a list with each element is a dictionnary of measurements referencing a given sample_

        Returns:
            List[Dict]: _List of dictionnary of measures_
        """
        d_list = []
        for i in range(len(self._dict_m)):
            meas_i = []
            for key in self._keys_list:
                meas_i.append(self._dict_m[i][key])
            d_i = dict(zip(self._keys_list,meas_i))
            d_list.append(d_i)
        return d_list
    
    # def calculate_RMSE_dicts(self, meas:Dict, est:Dict)->float:
    #     """_Calculate the RMSE between a dictionnary of markers measurements and markers estimations_

    #     Args:
    #         meas (Dict): _Measured markers_
    #         est (Dict): _Estimated markers_

    #     Returns:
    #         float: _RMSE value for all the markers_
    #     """

    #     # Initialize lists to store all the marker positions
    #     all_est_pos = []
    #     all_meas_pos = []

    #     # Concatenate all marker positions and measurements
    #     for key in self._keys_to_track_list:
    #         all_est_pos.append(est[key])
    #         all_meas_pos.append(meas[key])

    #     # Convert lists to numpy arrays
    #     all_est_pos = np.concatenate(all_est_pos)
    #     all_meas_pos = np.concatenate(all_meas_pos)

    #     # Calculate the global RMSE
    #     rmse = np.sqrt(np.mean((all_meas_pos - all_est_pos) ** 2))

    #     return rmse
    
    def solve_ik_sample(self, ii: int, meas: Dict)->None:
        """_Solve the ik optimisation problem : q* = argmin(||P_m - P_e||^2 + lambda|q_init - q|) st to q_min <= q <= q_max for a given sample _

        Args:
            ii (int): _number of sample_
            meas (Dict): _Dictionnary of landmark measurements_

        """
        if ii == 0 : # Init to be done with ipopt
            lb = self._model.lowerPositionLimit # lower joint limits
            ub = self._model.upperPositionLimit # upper joint limits
            cl=cu=[1]  # Constraint lists (single constraint here)
            
            # Setting up the IPOPT problem
            nlp = cyipopt.Problem(
                n=len(self._q0),
                m=len(cl),
                problem_obj=Ipopt_warm_start(self._model,meas,self._keys_to_track_list),
                lb=lb,
                ub=ub,
                cl=cl,
                cu=cu,
                )

            # IPOPT options
            nlp.add_option('tol',1e-3)
            nlp.add_option('print_level',0)

            # Solving the IPOPT problem
            q_opt, info = nlp.solve(self._q0)

            return q_opt

        else : # Running QP for subsequent samples
            q0=self._q0  # Initial joint configuration
                    
            # Forward kinematics to update frame placements based on current joint configuration = Reset estimated markers dict 
            pin.forwardKinematics(self._model, self._data, q0)
            pin.updateFramePlacements(self._model,self._data)

            # Estimate marker positions based on current joint configuration
            markers_est_pos = []
            for el in self._keys_to_track_list:
                markers_est_pos.append(self._data.oMf[self._model.getFrameId(el)].translation.reshape((3,1)))
            self._dict_m_est = dict(zip(self._keys_to_track_list,markers_est_pos))

            # Initialize QP matrices
            P=np.zeros((self._nv,self._nv)) # Hessian matrix size nv \times nv
            q=np.zeros((self._nv,)) # Gradient vector size nv

            # Inequality matrix G and vector h for joint limit constraints
            G=np.concatenate((np.zeros((2*(self._nv-6),6)),np.concatenate((np.identity(self._nv-6),-np.identity(self._nv-6)),axis=0)),axis=1) # Inequality matrix size number of inequalities (=nv) \times nv

            q_max_n=self._K_lim*(self._model.upperPositionLimit[7:]-q0[7:])/self._dt
            q_min_n=self._K_lim*(-self._model.lowerPositionLimit[7:]+q0[7:])/self._dt
            h=np.reshape((np.concatenate((q_max_n,q_min_n),axis=0)),(2*len(q_max_n),))

            pin.forwardKinematics(self._model, self._data, q0)
            pin.updateFramePlacements(self._model,self._data)
            
            for marker_name in self._keys_to_track_list:

                self._dict_m_est[marker_name]=self._data.oMf[self._model.getFrameId(marker_name)].translation.reshape((3,1)) #Position du keypoints après FK
                
                if math.isnan(meas[marker_name].flatten()[0]):
                    v_ii = np.zeros((3, 1))
                else:
                    v_ii=(meas[marker_name].reshape(3,1)-self._dict_m_est[marker_name])/self._dt #cf bas page cours scaron

                mu_ii=self._damping*np.dot(v_ii.T,v_ii)
                
                #
                J_ii=pin.computeFrameJacobian(self._model,self._data,q0,self._model.getFrameId(marker_name),pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
                J_ii_reduced=J_ii[:3,:]

                P_ii=np.matmul(J_ii_reduced.T,J_ii_reduced)+mu_ii*np.eye(self._nv)
                P+=P_ii

                q_ii=np.matmul(-self._K_ii*v_ii.T,J_ii_reduced)
                q+=q_ii.flatten()

            print('Solving for ' + str(ii) +'...')
            dq=quadprog_solve_qp(P,q,G,h)
            q0=pin.integrate(self._model,q0,dq*self._dt)

            return q0

    
    def solve_ik(self)->List:
        """_Returns a list of joint angle configuration over the whole trajectory _

        Returns:
            List: _List of q_i_
        """
        q_list = []
        meas_list = self.create_meas_list()
        for i in range(len(meas_list)):
            self._q0=self.solve_ik_sample(i, meas_list[i])
            q_list.append(self._q0)
        return q_list
        






def ik_pipe (no_sujet, task, data_path):

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


class IK_Casadi:
    """ Class to manage multi body IK problem using pinocchio casadi 
    """
    def __init__(self,model: pin.Model, dict_m: Dict, q0: np.ndarray):
        """_Init of the class _

        Args:
            model (pin.Model): _Pinocchio biomechanical model_
            dict_m (Dict): _a dictionnary containing the measures of the landmarks_
            q0 (np.ndarray): _initial configuration_
        """
        self._model = model
        self._dict_m = dict_m
        self._q0 = q0
        self._cmodel = cpin.Model(self._model)
        self._cdata = self._cmodel.createData()


        # Create a list of keys excluding the specified key
        self._keys_list = [key for key in self._dict_m[0].keys() if key !='Time']

        ### CASADI FRAMEWORK
        self._nq = self._cmodel.nq
        self._nv = self._cmodel.nv

        cq = casadi.SX.sym("q",self._nq,1)
        cdq = casadi.SX.sym("dq",self._nv,1)

        cpin.framesForwardKinematics(self._cmodel, self._cdata, cq)
        self._integrate = casadi.Function('integrate',[ cq,cdq ],[cpin.integrate(self._cmodel,cq,cdq) ])

        cfunction_list = []
        self._new_key_list = [] # Only take the frames that are in the model 


        
        #Partie IK à modifier si on veut changer le model
        #Lister les points LSTM de référence. 
        
        self._keys_to_track_list = ['C7_study',
                                    'r.ASIS_study', 'L.ASIS_study', 
                                    'r.PSIS_study', 'L.PSIS_study', 
                                    
                                    'r_shoulder_study',
                                    'r_lelbow_study', 'r_melbow_study',
                                    'r_lwrist_study', 'r_mwrist_study',
                                    'r_ankle_study', 'r_mankle_study',
                                    'r_toe_study','r_5meta_study', 'r_calc_study',
                                    'r_knee_study', 'r_mknee_study',
                                    'r_thigh1_study', 'r_thigh2_study', 'r_thigh3_study',
                                    'r_sh1_study', 'r_sh2_study', 'r_sh3_study',
                                    
                                    'L_shoulder_study', 
                                    'L_lelbow_study', 'L_melbow_study',
                                    'L_lwrist_study','L_mwrist_study',
                                    'L_ankle_study', 'L_mankle_study', 
                                    'L_toe_study','L_5meta_study', 'L_calc_study',
                                    'L_knee_study', 'L_mknee_study',
                                    'L_thigh1_study', 'L_thigh2_study', 'L_thigh3_study',
                                    'L_sh1_study', 'L_sh2_study', 'L_sh3_study']
        



        for key in self._keys_to_track_list:
            index_mk = self._cmodel.getFrameId(key)
            if index_mk < len(self._model.frames.tolist()): # Check that the frame is in the model
                new_key = key.replace('.','')
                self._new_key_list.append(key)
                function_mk = casadi.Function(f'f_{new_key}',[cq],[self._cdata.oMf[index_mk].translation])
                cfunction_list.append(function_mk)

        self._cfunction_dict=dict(zip(self._new_key_list,cfunction_list))

        #'FF_TX','FF_TY','FF_TZ','FF_Rquat0','FF_Rquat1','FF_Rquat2','FF_Rquat3' => sont les joints du pelvis
        # Ensuite, il faut avoir les mêmes joint que lorsque l'on fait le check_build_model
        self._mapping_joint_angle = dict(zip(['FF_TX','FF_TY','FF_TZ','FF_Rquat0','FF_Rquat1','FF_Rquat2','FF_Rquat3','L5S1_FE','L5S1_R_EXT_INT','Shoulder_Z_R','Shoulder_X_R','Shoulder_Y_R','Elbow_Z_R','Elbow_Y_R','Shoulder_Z_L','Shoulder_X_L','Shoulder_Y_L','Elbow_Z_L','Elbow_Y_L','Hip_Z_R','Hip_X_R','Hip_Y_R','Knee_Z_R','Ankle_Z_R','Hip_Z_L','Hip_X_L','Hip_Y_L','Knee_Z_L','Ankle_Z_L'],np.arange(0,self._nq,1)))

    def create_meas_list(self)-> List[Dict]:
        """_Create a list with each element is a dictionnary of measurements referencing a given sample_

        Returns:
            List[Dict]: _List of dictionnary of measures_
        """
        d_list = []
        for i in range(len(self._dict_m)):
            meas_i = []
            for key in self._keys_list:
                meas_i.append(self._dict_m[i][key])
            d_i = dict(zip(self._keys_list,meas_i))
            d_list.append(d_i)
        return d_list

    def solve_ik_sample(self, ii: int, meas: Dict)->np.ndarray:
        """_Solve the ik optimisation problem : q* = argmin(||P_m - P_e||^2 + lambda|q_init - q|) st to q_min <= q <= q_max for a given sample _

        Args:
            ii (int): _number of sample_
            meas (Dict): _Dictionnary of landmark measurements_

        Returns:
            np.ndarray: _q_i joint angle at the i-th sample_
        """

        joint_to_regularize = [] #['RElbow_FE','RElbow_PS','RHip_RIE']
        value_to_regul = 0.001

        # Casadi optimization class
        opti = casadi.Opti()

        # Variables MX type
        DQ = opti.variable(self._nv)
        Q = self._integrate(self._q0,DQ)

        omega = 1e-6*np.ones(self._nq)

        for name in joint_to_regularize :
            if name in self._mapping_joint_angle:
                omega[self._mapping_joint_angle[name]] = value_to_regul # Adapt the weight for given joints, for instance the hip Y
            else :
                raise ValueError("Joint to regulate not in the model")

        cost = 0

        if ii == 0:
            for key in self._cfunction_dict.keys():
                cost+=1*casadi.sumsqr(meas[key]-self._cfunction_dict[key](Q))
        else : 
            for key in self._cfunction_dict.keys():
                cost+=1*casadi.sumsqr(meas[key]-self._cfunction_dict[key](Q))  + 0.001*casadi.sumsqr(casadi.dot(omega,self._q0-Q))

        # Set the constraint for the joint limits
        for i in range(7,self._nq):
            opti.subject_to(opti.bounded(self._model.lowerPositionLimit[i],Q[i],self._model.upperPositionLimit[i]))
        
        opti.minimize(cost)

        # Set Ipopt options to suppress output
        opts = {
            "ipopt.print_level": 0,
            "ipopt.sb": "yes",
            "ipopt.max_iter": 1000,
            "ipopt.linear_solver": "mumps"
        }

        opti.solver("ipopt", opts)

        print('Solving for ' + str(ii) +'...')
        sol = opti.solve()
        
        q_i = sol.value(Q)
        return q_i 

    def solve_ik(self)->List:
        """_Returns a list of joint angle configuration over the whole trajectory _

        Returns:
            List: _List of q_i_
        """
        q_list = []
        meas_list = self.create_meas_list()
        for i in range(len(meas_list)):
            self._q0 = self.solve_ik_sample(i, meas_list[i])
            q_list.append(self._q0)
        return q_list

        

class IK_Singlebody:
    """ Class to manage a single IK problem using pinocchio casadi 
    """
    def __init__(self):
        pass

    def rotation_matrix_to_euler(self, R: np.ndarray, convention: str)->np.ndarray:
        r = R.from_matrix(R)
        return r.as_euler(convention)

    def quaternion_to_euler(self, q: np.ndarray, convention: str)->np.ndarray:
        r = R.from_quat(q)
        return r.as_euler(convention)

    def rodrigues_to_euler(self, r_vec: np.ndarray, convention: str)->np.ndarray:
        theta = np.linalg.norm(r_vec)
        if theta == 0:
            return np.zeros(3)
        axis = r_vec / theta
        r = R.from_rotvec(axis * theta)
        return r.as_euler(convention)

    def to_rotation_matrix(self, orientation: np.ndarray)->np.ndarray:
        if isinstance(orientation, np.ndarray) and orientation.shape == (3, 3):
            return orientation
        elif isinstance(orientation, np.ndarray) and orientation.shape == (4,):
            return R.from_quat(orientation).as_matrix()
        elif isinstance(orientation, np.ndarray) and orientation.shape == (3,):
            return R.from_rotvec(orientation).as_matrix()
        else:
            raise ValueError("Invalid orientation format")

    def solve_ik(self, parent_orientation: np.ndarray, child_orientation: np.ndarray, convention: str):
        # Convert parent_orientation and child_orientation to rotation matrices if they are not already
        parent_matrix = self.to_rotation_matrix(parent_orientation)
        child_matrix = self.to_rotation_matrix(child_orientation)

        # Calculate the relative rotation matrix
        relative_matrix = np.linalg.inv(parent_matrix).dot(child_matrix)

        # Convert the relative rotation matrix to euler angles with the given convention
        joint_angles = self.rotation_matrix_to_euler(relative_matrix, convention)
        
        return joint_angles





